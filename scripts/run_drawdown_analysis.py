"""
Maximum adverse excursion (MAE): how much pain would a trader following the
whale deposit (sell) signal have endured before it paid off?

Context: the long-horizon results (1 week to 6 months) report the return at
the END of the holding period only. A position that finishes +5% may have
been -20% at some point along the way -- most traders cannot tolerate that,
even if the signal is "eventually right". This was Limitation #6 in the
README; this script turns that caveat into an actual measurement.

Methodology
-----------
A whale deposit is a SELL signal: the implied position is short ETH (or "out
of the market") from the deposit's timestamp t0 for the holding horizon h.
The worst point for that position is the HIGHEST price reached at any hour
between t0 and t0+h -- that is the maximum adverse excursion (MAE):

    MAE = (max(price[t0 : t0+h]) - price[t0]) / price[t0]

This requires the full price PATH over the holding window, not just the
price at t0+h (which is all the existing forward-return code computes). We
get it with a "forward rolling max": reverse the hourly price series, take a
standard (backward-looking) rolling max, then reverse back -- a standard
pandas idiom for turning a trailing window into a leading one.

Using future price data to compute this is not look-ahead bias in the
predictive sense: MAE is a purely descriptive, ex-post statistic about what
the historical path looked like, computed for the same backtest purpose as
the existing forward-return columns. It is not used as a model input.

Usage:
    python scripts/run_drawdown_analysis.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.features.feature_engineer import assign_transaction_label  # noqa: E402

# Long-horizon holding periods, matching the README's existing long-horizon table.
HORIZONS_H = [168, 336, 720, 2160, 4320]
HORIZON_LABELS = ["1 week", "2 weeks", "1 month", "3 months", "6 months"]

MIN_N = 30  # minimum group size before reporting a percentile


def load_data():
    """Load whale transactions and hourly prices, with a gap-filled price index.

    The rolling-max window below is sized in ROW COUNT (h+1 hourly rows), which
    only equals h HOURS of calendar time if the price series has no missing
    hours. The raw feed has exactly one gap (2h instead of 1h) across 30,801
    rows -- negligible here, but silently wrong if a future data pull has more.
    Reindexing to a complete hourly grid and forward-filling makes this exact
    regardless of gaps, rather than relying on today's data happening to be
    almost gap-free.
    """
    whale = pd.read_csv(config.PROCESSED_DATA_DIR / "whale_txs.csv")
    prices = pd.read_csv(config.PROCESSED_DATA_DIR / "eth_prices_hourly.csv")
    fng = pd.read_csv(config.PROCESSED_DATA_DIR / "fear_greed_daily.csv")

    whale["timestamp_utc"] = pd.to_datetime(whale["timestamp_utc"], utc=True)
    prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
    fng["date"] = pd.to_datetime(fng["date"], utc=True)

    whale = assign_transaction_label(whale)
    whale["hour_utc"] = whale["timestamp_utc"].dt.floor("h")

    prices = prices.sort_values("timestamp_utc").reset_index(drop=True)
    full_index = pd.date_range(
        prices["timestamp_utc"].min(), prices["timestamp_utc"].max(),
        freq="h", tz="UTC",
    )
    price_series = (
        prices.set_index("timestamp_utc")["close"]
        .reindex(full_index)
        .ffill()  # fills the single known gap; a no-op everywhere else
    )

    return whale, price_series, fng


def compute_mae_table(whale: pd.DataFrame, price_series: pd.Series, fng: pd.DataFrame) -> pd.DataFrame:
    """For each horizon and condition, compute MAE stats split by eventual outcome."""
    deposits = whale[whale["tx_category"] == "exchange_deposit"].copy()
    deposits = deposits.merge(
        price_series.rename("price_t0"), left_on="hour_utc", right_index=True, how="left",
    )

    # Merge Fear & Greed for the extreme-greed conditioned subset
    fng_m = fng.rename(columns={"date": "_date"}).sort_values("_date")
    deposits["_date"] = deposits["timestamp_utc"].dt.floor("D")
    deposits = pd.merge_asof(
        deposits.sort_values("_date"), fng_m[["_date", "fng_value"]],
        on="_date", direction="backward",
    )
    deposits["fng_value"] = deposits["fng_value"].fillna(50)
    deposits.drop(columns="_date", inplace=True)

    conditions = {
        "unconditional": pd.Series(True, index=deposits.index),
        "extreme_greed": deposits["fng_value"] > 75,
    }

    rows = []
    for h, label in zip(HORIZONS_H, HORIZON_LABELS):
        # Final price at t0+h (for the hit/miss classification, as elsewhere)
        fut_hour = deposits["hour_utc"] + pd.Timedelta(hours=h)
        fut_price = fut_hour.map(price_series)
        ret = (fut_price - deposits["price_t0"]) / deposits["price_t0"]

        # Highest price reached at ANY point in [t0, t0+h] -- the forward rolling max.
        # Reverse -> trailing rolling max -> reverse back = leading (forward) max.
        fwd_max_series = price_series[::-1].rolling(window=h + 1, min_periods=1).max()[::-1]
        fwd_max_price = deposits["hour_utc"].map(fwd_max_series)
        mae = (fwd_max_price - deposits["price_t0"]) / deposits["price_t0"]

        d = deposits.assign(ret=ret, mae=mae).dropna(subset=["ret", "mae"])

        for cond_name, cond_mask in conditions.items():
            subset = d[cond_mask[d.index]]
            hit = subset[subset["ret"] < 0]      # signal eventually correct
            miss = subset[subset["ret"] >= 0]    # signal eventually wrong

            for outcome_name, outcome_df in [("hit", hit), ("miss", miss)]:
                n = len(outcome_df)
                if n < MIN_N:
                    continue
                rows.append({
                    "horizon": label, "horizon_h": h, "condition": cond_name,
                    "outcome": outcome_name, "n": n,
                    "mean_mae_pct": round(outcome_df["mae"].mean() * 100, 2),
                    "median_mae_pct": round(outcome_df["mae"].median() * 100, 2),
                    "p90_mae_pct": round(outcome_df["mae"].quantile(0.90) * 100, 2),
                    "mean_final_return_pct": round(outcome_df["ret"].mean() * 100, 2),
                })

    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame) -> None:
    print(f"\n{'='*100}")
    print("MAXIMUM ADVERSE EXCURSION: pain endured before the deposit (sell) signal paid off")
    print(f"{'='*100}")

    for cond in ["unconditional", "extreme_greed"]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        print(f"\n  DEPOSITS -- {cond.upper()}")
        print(f"  {'Horizon':>10} {'Outcome':>7} {'N':>8} {'Mean MAE':>10} "
              f"{'Median MAE':>11} {'P90 MAE':>9} {'Mean Final Ret':>15}")
        for _, r in sub.iterrows():
            print(f"  {r['horizon']:>10} {r['outcome']:>7} {r['n']:>8,} "
                  f"{r['mean_mae_pct']:>9.2f}% {r['median_mae_pct']:>10.2f}% "
                  f"{r['p90_mae_pct']:>8.2f}% {r['mean_final_return_pct']:>14.2f}%")


def main() -> None:
    print("Loading data...")
    whale, price_series, fng = load_data()
    print(f"Loaded {len(whale):,} whale transactions, "
          f"{len(price_series):,} hourly candles (gap-filled).")

    print("Computing maximum adverse excursion...")
    table = compute_mae_table(whale, price_series, fng)

    print_table(table)

    out_path = config.ROOT_DIR / "results" / "drawdown_analysis.csv"
    table.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
