"""
Phase 4 feature matrix assembly.

Merges whale transactions, hourly sentiment, and hourly ETH prices into a
single feature matrix ready for the price-direction classifier.

Each row represents one whale transaction with:
  - Transaction features (size, gas, category, sender history)
  - Sentiment context (the most recent hourly sentiment)
  - Price context (recent returns, volatility)
  - Target variable(s): did ETH go up or down at t+1h, t+6h, t+24h?

IMPORTANT: all features use only data available at or before transaction time.
Target variables use future prices — they are labels, not features.
"""

from typing import Optional

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Prediction horizons in hours
HORIZONS = [1, 6, 24, 48, 72]

# Number of hours to look back for rolling price features
ROLLING_WINDOW = 24


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(
    whale_df: pd.DataFrame,
    price_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    fng_df: Optional[pd.DataFrame] = None,
    funding_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Assemble the full Phase 4 feature matrix.

    Parameters
    ----------
    whale_df : pd.DataFrame
        Whale transactions with features from Phase 2 (must have
        timestamp_utc, log_usd_value, log_gas_used, etc.).
    price_df : pd.DataFrame
        Hourly ETH prices (timestamp_utc, close, open, high, low, volume).
    sentiment_df : pd.DataFrame, optional
        Hourly sentiment (hour_utc, sentiment_mean, sentiment_std,
        article_count, positive_ratio, negative_ratio). If None, sentiment
        features are filled with 0 (neutral).
    fng_df : pd.DataFrame, optional
        Daily Fear & Greed Index (date, fng_value, fng_classification).
    funding_df : pd.DataFrame, optional
        Binance funding rate (timestamp_utc, funding_rate, mark_price).

    Returns
    -------
    pd.DataFrame
        One row per whale transaction, with all features and target columns.
        Rows where target variables cannot be computed (too close to the end
        of the price data) are dropped.
    """
    # --- Ensure timestamps are datetime ---
    whale = whale_df.copy()
    whale["timestamp_utc"] = pd.to_datetime(whale["timestamp_utc"], utc=True)

    price = price_df.copy()
    price["timestamp_utc"] = pd.to_datetime(price["timestamp_utc"], utc=True)

    # --- Add rolling price features to the hourly price table ---
    price = _add_price_features(price)

    # --- Floor whale timestamps to the hour for merging ---
    # A transaction at 14:37 gets matched to the 14:00 price bar
    whale["hour_utc"] = whale["timestamp_utc"].dt.floor("h")

    # --- Merge price features onto whale transactions ---
    # merge_asof finds the most recent price bar at or before each tx.
    # This avoids look-ahead: if price data has a gap, we use the last
    # known price, not a future one.
    whale = whale.sort_values("hour_utc")
    price = price.sort_values("timestamp_utc")

    whale = pd.merge_asof(
        whale,
        price.rename(columns={"timestamp_utc": "hour_utc"}),
        on="hour_utc",
        direction="backward",  # use most recent price at or before tx time
    )

    # --- Merge sentiment ---
    if sentiment_df is not None:
        sentiment = sentiment_df.copy()
        sentiment["hour_utc"] = pd.to_datetime(sentiment["hour_utc"], utc=True)
        sentiment = sentiment.sort_values("hour_utc")

        # merge_asof with backward direction: if no sentiment this hour,
        # use the most recent hour that had articles
        whale = pd.merge_asof(
            whale.sort_values("hour_utc"),
            sentiment,
            on="hour_utc",
            direction="backward",
        )

        # Fill NaN sentiment (hours before any articles exist) with neutral
        _fill_sentiment_nans(whale)
    else:
        # No sentiment data — fill with neutral values
        whale["sentiment_mean"] = 0.0
        whale["sentiment_std"] = 0.0
        whale["article_count"] = 0
        whale["positive_ratio"] = 0.0
        whale["negative_ratio"] = 0.0

    # --- Merge Fear & Greed Index ---
    if fng_df is not None:
        fng = fng_df.copy()
        fng["date"] = pd.to_datetime(fng["date"], utc=True)
        # Floor to date for daily merge
        whale["_date"] = whale["timestamp_utc"].dt.floor("D")
        fng = fng.rename(columns={"date": "_date"}).sort_values("_date")

        whale = pd.merge_asof(
            whale.sort_values("_date"),
            fng[["_date", "fng_value"]],
            on="_date",
            direction="backward",
        )
        whale.drop(columns="_date", inplace=True)
        whale["fng_value"] = whale["fng_value"].fillna(50)  # neutral default
        print(f"  Merged Fear & Greed Index")
    else:
        whale["fng_value"] = 50  # neutral

    # --- Merge Binance funding rate ---
    if funding_df is not None:
        funding = funding_df.copy()
        funding["timestamp_utc"] = pd.to_datetime(
            funding["timestamp_utc"], utc=True, format="ISO8601"
        )
        funding = funding.sort_values("timestamp_utc")

        # merge_asof: use the most recent funding rate at or before tx time
        whale = pd.merge_asof(
            whale.sort_values("timestamp_utc"),
            funding[["timestamp_utc", "funding_rate"]].rename(
                columns={"timestamp_utc": "hour_utc"}
            ),
            on="hour_utc",
            direction="backward",
        )
        whale["funding_rate"] = whale["funding_rate"].fillna(0.0)
        print(f"  Merged Binance funding rates")
    else:
        whale["funding_rate"] = 0.0

    # --- Encode transaction categories as binary features ---
    # One-hot encode tx_category so the model can learn that exchange_deposit
    # during negative sentiment is a different signal than wallet_to_wallet.
    # If tx_category is missing (raw data without labels), skip this step.
    if "tx_category" in whale.columns:
        # pd.get_dummies creates one boolean column per category value.
        # prefix="cat" gives columns like cat_exchange_deposit, cat_wallet_to_wallet.
        cat_dummies = pd.get_dummies(
            whale["tx_category"], prefix="cat", dtype=int
        )
        whale = pd.concat([whale, cat_dummies], axis=1)
        print(f"  Encoded {len(cat_dummies.columns)} transaction categories: "
              f"{list(cat_dummies.columns)}")

    # --- Compute target variables ---
    # Build a lookup from hour -> close price for fast future price retrieval
    price_lookup = price.set_index("timestamp_utc")["close"].to_dict()

    for h in HORIZONS:
        whale = _add_target(whale, price_lookup, horizon_hours=h)

    # --- Drop rows where any target is NaN (too close to end of data) ---
    target_cols = [f"target_{h}h" for h in HORIZONS]
    before = len(whale)
    whale = whale.dropna(subset=target_cols).reset_index(drop=True)
    dropped = before - len(whale)
    if dropped > 0:
        print(f"  Dropped {dropped:,} rows with missing targets "
              f"(near end of price data)")

    # --- Convert boolean targets to int for sklearn ---
    for col in target_cols:
        whale[col] = whale[col].astype(int)

    return whale


def get_phase4_feature_columns() -> list[str]:
    """Return column names used as model inputs in Phase 4."""
    return [
        # Transaction features (from Phase 2)
        "log_usd_value",
        "log_gas_used",
        "gas_price_gwei",
        "is_contract_call",
        "hour_of_day",
        "day_of_week",
        "sender_prior_tx_count",
        # Transaction category (one-hot encoded from Phase 2 classifier)
        "cat_exchange_deposit",
        "cat_exchange_withdrawal",
        "cat_defi_interaction",
        "cat_wallet_to_wallet",
        # Price context features
        "return_1h",
        "return_6h",
        "return_24h",
        "volatility_24h",
        "price_vs_24h_mean",
        # Sentiment features (news)
        "sentiment_mean",
        "sentiment_std",
        "article_count",
        "positive_ratio",
        "negative_ratio",
        # Market-derived sentiment
        "fng_value",
        "funding_rate",
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_price_features(price: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling return and volatility features to the hourly price table.

    All features are backward-looking (no look-ahead):
      - return_1h:  % change from 1 hour ago
      - return_6h:  % change from 6 hours ago
      - return_24h: % change from 24 hours ago
      - volatility_24h: std dev of hourly returns over last 24 hours
      - price_vs_24h_mean: current price / 24h rolling mean - 1
                           (positive = above recent average)
    """
    price = price.sort_values("timestamp_utc").reset_index(drop=True)

    # pct_change(n) computes (current - n_periods_ago) / n_periods_ago
    # This is a backward-looking return — safe from look-ahead
    price["return_1h"] = price["close"].pct_change(1)
    price["return_6h"] = price["close"].pct_change(6)
    price["return_24h"] = price["close"].pct_change(24)

    # Rolling volatility: std of 1h returns over the past 24 hours
    # .rolling(24) creates a 24-period sliding window. min_periods=12
    # allows computation even if we have fewer than 24 hours of data.
    price["volatility_24h"] = (
        price["return_1h"]
        .rolling(window=ROLLING_WINDOW, min_periods=12)
        .std()
    )

    # Price relative to its recent mean — captures momentum/mean-reversion
    rolling_mean = price["close"].rolling(window=ROLLING_WINDOW, min_periods=12).mean()
    price["price_vs_24h_mean"] = (price["close"] / rolling_mean) - 1

    return price


def _add_target(
    df: pd.DataFrame,
    price_lookup: dict,
    horizon_hours: int,
) -> pd.DataFrame:
    """
    Add a binary target column: 1 if ETH price is higher at t+horizon, else 0.

    Uses the price_lookup dict (hour -> close price) for fast retrieval.
    Rows where the future price is unavailable get NaN.
    """
    col_name = f"target_{horizon_hours}h"

    # pd.Timedelta creates a time offset we can add to timestamps
    offset = pd.Timedelta(hours=horizon_hours)

    def _get_future_direction(row):
        current_hour = row["hour_utc"]
        future_hour = current_hour + offset
        current_price = price_lookup.get(current_hour)
        future_price = price_lookup.get(future_hour)

        if current_price is None or future_price is None:
            return np.nan

        # 1 = price went up, 0 = price went down or stayed flat
        return 1.0 if future_price > current_price else 0.0

    # .apply(func, axis=1) runs the function on each row — slow but
    # necessary here because we need to look up arbitrary future hours
    df[col_name] = df.apply(_get_future_direction, axis=1)

    return df


def _fill_sentiment_nans(df: pd.DataFrame) -> None:
    """Fill NaN sentiment columns with neutral defaults (in-place)."""
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)
    df["sentiment_std"] = df["sentiment_std"].fillna(0.0)
    df["article_count"] = df["article_count"].fillna(0)
    df["positive_ratio"] = df["positive_ratio"].fillna(0.0)
    df["negative_ratio"] = df["negative_ratio"].fillna(0.0)
