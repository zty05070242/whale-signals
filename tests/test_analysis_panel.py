"""Tests for the canonical event and market analysis panels."""

import numpy as np
import pandas as pd

from src.analysis.panel import (
    add_event_forward_returns,
    build_analysis_panel,
    forward_return_column,
)


def make_prices() -> pd.DataFrame:
    """Return a deliberately unsorted hourly price series."""
    return pd.DataFrame({
        "timestamp_utc": [
            "2024-01-01 03:00:00+00:00",
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 02:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
        ],
        "close": [80.0, 100.0, 90.0, 110.0],
    })


def make_whales() -> pd.DataFrame:
    """Return minimal transactions covering deposit and withdrawal labels."""
    return pd.DataFrame({
        "timestamp_utc": [
            "2024-01-01 00:30:00+00:00",
            "2024-01-01 01:45:00+00:00",
            "2024-01-01 02:15:00+00:00",
        ],
        "from_category": ["unknown", "exchange", "unknown"],
        "to_category": ["exchange", "unknown", "exchange"],
    })


def test_event_returns_use_exact_future_hours() -> None:
    """Events use their containing hour and never a neighbouring price row."""
    events = add_event_forward_returns(
        make_whales(),
        make_prices(),
        horizons=[1, 2],
    )

    assert events.loc[0, forward_return_column(1)] == 0.10
    assert events.loc[0, forward_return_column(2)] == -0.10
    assert events.loc[1, forward_return_column(1)] == (90.0 - 110.0) / 110.0


def test_each_horizon_keeps_its_own_available_sample() -> None:
    """The canonical panel retains shorter returns when longer ones are absent."""
    events = add_event_forward_returns(
        make_whales(),
        make_prices(),
        horizons=[1, 2],
    )

    assert events[forward_return_column(1)].notna().sum() == 3
    assert events[forward_return_column(2)].notna().sum() == 2
    assert np.isnan(events.loc[2, forward_return_column(2)])


def test_legacy_complete_case_mode_is_preserved() -> None:
    """Existing event-study callers can retain their all-horizon sample rule."""
    events = add_event_forward_returns(
        make_whales(),
        make_prices(),
        horizons=[1, 2],
        drop_incomplete=True,
    )

    assert len(events) == 2
    assert events[[forward_return_column(1), forward_return_column(2)]].notna().all().all()


def test_analysis_panel_uses_only_available_sentiment() -> None:
    """Backward joins cannot attach later funding or Fear & Greed values."""
    funding = pd.DataFrame({
        "timestamp_utc": [
            "2023-12-31 23:00:00+00:00",
            "2024-01-01 02:00:00+00:00",
        ],
        "funding_rate": [-0.001, 0.002],
    })
    fear_greed = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "fng_value": [20, 80],
    })

    events, market = build_analysis_panel(
        make_whales(),
        make_prices(),
        fear_greed,
        funding,
        horizons=[1],
    )

    first_event = events.loc[events["timestamp_utc"].idxmin()]
    last_event = events.loc[events["timestamp_utc"].idxmax()]
    assert first_event["funding_rate"] == -0.001
    assert last_event["funding_rate"] == 0.002
    assert set(events["fng_value"]) == {20}
    assert set(market["fng_value"]) == {20}


def test_analysis_panel_aligns_whale_and_market_columns() -> None:
    """Whale and base-rate panels expose one shared return-column contract."""
    funding = pd.DataFrame({
        "timestamp_utc": ["2023-12-31 23:00:00+00:00"],
        "funding_rate": [0.0],
    })
    fear_greed = pd.DataFrame({
        "date": ["2024-01-01"],
        "fng_value": [50],
    })

    events, market = build_analysis_panel(
        make_whales(),
        make_prices(),
        fear_greed,
        funding,
        horizons=[1, 2],
    )

    expected = {forward_return_column(1), forward_return_column(2)}
    assert expected.issubset(events.columns)
    assert expected.issubset(market.columns)
    assert list(events["tx_category"]) == [
        "exchange_deposit",
        "exchange_withdrawal",
        "exchange_deposit",
    ]
