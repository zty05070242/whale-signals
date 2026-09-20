"""Tests for the shared bull/bear market state machine."""

import pandas as pd

from src.analysis.regimes import compute_regime


def test_regime_changes_only_after_twenty_percent_moves() -> None:
    """A 20% drawdown starts a bear and a 20% trough rally restarts a bull."""
    prices = pd.DataFrame({"close": [100.0, 120.0, 97.0, 96.0, 80.0, 95.0, 96.0]})

    regimes = compute_regime(prices)

    assert regimes.tolist() == [
        "bull",
        "bull",
        "bull",
        "bear",
        "bear",
        "bear",
        "bull",
    ]
