"""Build the canonical event and market panels used by the research analyses.

The event panel contains one row per whale transaction. The market panel
contains one row per hourly ETH candle. Both receive the same forward-return
columns and the same backward-looking sentiment context, so a whale hit rate
and its market base rate are always measured from aligned data.
"""

from collections.abc import Sequence

import pandas as pd

from src.features.feature_engineer import assign_transaction_label


def forward_return_column(horizon_hours: int) -> str:
    """Return the canonical column name for a forward-return horizon."""
    return f"fwd_return_{horizon_hours}h"


def add_event_forward_returns(
    events_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizons: Sequence[int],
    *,
    drop_incomplete: bool = False,
    keep_entry_price: bool = False,
) -> pd.DataFrame:
    """Attach forward ETH returns to timestamped events.

    Event timestamps are floored to the hour because the market input is an
    hourly close series. Future prices are looked up by timestamp rather than
    row position, so a missing candle does not silently change the horizon.

    Parameters
    ----------
    events_df:
        Events containing ``timestamp_utc``.
    price_df:
        Hourly market data containing ``timestamp_utc`` and ``close``.
    horizons:
        Forward horizons measured in hours.
    drop_incomplete:
        If true, retain only events with a return at every requested horizon.
    keep_entry_price:
        If true, retain the temporary ``price_t0`` lookup column.
    """
    events = events_df.copy()
    events["timestamp_utc"] = pd.to_datetime(
        events["timestamp_utc"], utc=True
    )  # normalise mixed timestamp inputs before hourly alignment
    events["hour_utc"] = events["timestamp_utc"].dt.floor(
        "h"
    )  # match each transaction to its containing hourly candle

    prices = _prepare_prices(price_df)
    price_series = prices.set_index(
        "timestamp_utc"
    )["close"]  # timestamp index makes future-price lookup explicit

    events = events.merge(
        price_series.rename("price_t0"),
        left_on="hour_utc",
        right_index=True,
        how="left",
    )  # preserve events even when their entry candle is unavailable

    return_columns: list[str] = []
    for horizon in horizons:
        column = forward_return_column(horizon)
        future_hour = events["hour_utc"] + pd.Timedelta(
            hours=horizon
        )  # construct the exact future timestamp, not a row offset
        future_price = future_hour.map(
            price_series
        )  # missing future candles remain missing rather than shifting the horizon
        events[column] = (
            future_price - events["price_t0"]
        ) / events["price_t0"]
        return_columns.append(column)

    if not keep_entry_price:
        events = events.drop(
            columns="price_t0"
        )  # entry price is an implementation detail for most callers

    if drop_incomplete:
        events = events.dropna(
            subset=return_columns
        ).reset_index(drop=True)  # preserve the legacy event-study sample rule

    return events


def build_analysis_panel(
    whale_df: pd.DataFrame,
    price_df: pd.DataFrame,
    fear_greed_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    horizons: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned whale-event and hourly-market analysis panels.

    The same historical Fear & Greed and funding observations are attached to
    both panels with backward as-of joins. This prevents future sentiment from
    entering an event row and keeps conditioned whale rates comparable with
    their conditioned market base rates.
    """
    labelled_whales = assign_transaction_label(
        whale_df.copy()
    )  # derive the deposit/withdrawal labels before analysis
    events = add_event_forward_returns(
        labelled_whales,
        price_df,
        horizons,
        keep_entry_price=True,
    )
    events["year"] = events["timestamp_utc"].dt.year

    market = _add_market_forward_returns(price_df, horizons)

    events = _attach_funding(
        events,
        funding_df,
        timestamp_column="hour_utc",
    )
    market = _attach_funding(
        market,
        funding_df,
        timestamp_column="timestamp_utc",
    )

    events = _attach_fear_greed(
        events,
        fear_greed_df,
        timestamp_column="timestamp_utc",
    )
    market = _attach_fear_greed(
        market,
        fear_greed_df,
        timestamp_column="timestamp_utc",
    )
    market["year"] = market["timestamp_utc"].dt.year

    return events, market


def _prepare_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    """Return hourly prices in deterministic UTC timestamp order."""
    prices = price_df.copy()
    prices["timestamp_utc"] = pd.to_datetime(
        prices["timestamp_utc"], utc=True
    )  # ensure comparisons and joins use one timezone
    return prices.sort_values("timestamp_utc").reset_index(
        drop=True
    )  # merge_asof and timestamp lookup both require ordered data


def _add_market_forward_returns(
    price_df: pd.DataFrame,
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Attach forward returns to every hourly candle in the market universe."""
    market = _prepare_prices(price_df)
    price_series = market.set_index(
        "timestamp_utc"
    )["close"]  # timestamp-based lookup preserves calendar-hour horizons

    for horizon in horizons:
        future_hour = market["timestamp_utc"] + pd.Timedelta(hours=horizon)
        future_price = future_hour.map(price_series)
        market[forward_return_column(horizon)] = (
            future_price - market["close"]
        ) / market["close"]

    return market


def _attach_funding(
    frame: pd.DataFrame,
    funding_df: pd.DataFrame,
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    """Attach the latest known funding rate at or before each row."""
    funding = funding_df.copy()
    funding["timestamp_utc"] = pd.to_datetime(
        funding["timestamp_utc"], utc=True, format="ISO8601"
    )  # accept the ISO variants returned by the Binance source
    funding = funding.sort_values(
        "timestamp_utc"
    )  # backward as-of joins require ordered keys

    right = funding[["timestamp_utc", "funding_rate"]].rename(
        columns={"timestamp_utc": timestamp_column}
    )
    enriched = pd.merge_asof(
        frame.sort_values(timestamp_column),
        right,
        on=timestamp_column,
        direction="backward",
    )  # backward direction prevents future funding observations leaking in
    enriched["funding_rate"] = enriched["funding_rate"].fillna(
        0.0
    )  # retain the project's established neutral fallback
    return enriched


def _attach_fear_greed(
    frame: pd.DataFrame,
    fear_greed_df: pd.DataFrame,
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    """Attach the latest daily Fear & Greed value at or before each row."""
    fear_greed = fear_greed_df.copy()
    fear_greed["date"] = pd.to_datetime(
        fear_greed["date"], utc=True
    )  # daily observations need the same timezone as event timestamps
    fear_greed = fear_greed.rename(columns={"date": "_date"}).sort_values(
        "_date"
    )  # merge_asof requires an ordered right-hand key

    enriched = frame.copy()
    enriched["_date"] = enriched[timestamp_column].dt.floor(
        "D"
    )  # compare every row with the daily observation available that day
    enriched = pd.merge_asof(
        enriched.sort_values("_date"),
        fear_greed[["_date", "fng_value"]],
        on="_date",
        direction="backward",
    )  # backward direction prevents a later daily value entering an earlier row
    enriched["fng_value"] = enriched["fng_value"].fillna(
        50
    )  # retain the project's established neutral fallback
    return enriched.drop(columns="_date")
