"""Characterisation tests for the core event-study calculations."""

import pandas as pd
import pytest

from src.analysis.event_study import (
    HORIZONS,
    compute_base_rate,
    compute_event_returns,
    compute_hit_rates,
)
from src.analysis.panel import forward_return_column


def test_compute_event_returns_preserves_public_column_contract() -> None:
    """The public helper keeps its three established return columns."""
    prices = pd.DataFrame({
        "timestamp_utc": pd.date_range(
            "2024-01-01", periods=26, freq="h", tz="UTC"
        ),
        "close": list(range(100, 126)),
    })
    whales = pd.DataFrame({
        "timestamp_utc": ["2024-01-01 00:30:00+00:00"],
        "tx_category": ["exchange_withdrawal"],
    })

    events = compute_event_returns(whales, prices)

    assert len(events) == 1
    assert {forward_return_column(horizon) for horizon in HORIZONS}.issubset(
        events.columns
    )
    assert events.loc[0, forward_return_column(1)] == pytest.approx(0.01)
    assert events.loc[0, forward_return_column(24)] == pytest.approx(0.24)


def test_hit_rates_apply_opposite_deposit_and_withdrawal_directions() -> None:
    """Falling prices hit for deposits while rising prices hit for withdrawals."""
    rows: list[dict] = []
    for category, value in (
        ("exchange_deposit", -0.01),
        ("exchange_withdrawal", 0.01),
    ):
        for _ in range(30):
            row = {"tx_category": category}
            row.update({forward_return_column(horizon): value for horizon in HORIZONS})
            rows.append(row)

    results = compute_hit_rates(pd.DataFrame(rows))

    for horizon in HORIZONS:
        assert results["exchange_deposit"][horizon]["hit_rate"] == 1.0
        assert results["exchange_withdrawal"][horizon]["hit_rate"] == 1.0


def test_base_rate_uses_the_requested_calendar_horizon() -> None:
    """The base rate is calculated from exact future timestamps."""
    prices = pd.DataFrame({
        "timestamp_utc": pd.date_range(
            "2024-01-01", periods=4, freq="h", tz="UTC"
        ),
        "close": [100.0, 90.0, 100.0, 80.0],
    })

    rate = compute_base_rate(prices, direction="down", horizon=1)

    assert rate == pytest.approx(2 / 3)
