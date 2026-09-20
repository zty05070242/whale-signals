"""Market-regime classifications shared by research and presentation code."""

import numpy as np
import pandas as pd


def compute_regime(prices: pd.DataFrame) -> pd.Series:
    """Classify each price row as a classic 20% bull or bear regime.

    The series begins in a bull state. A 20% fall from the running peak starts
    a bear regime; a 20% rise from the subsequent running trough starts the
    next bull regime.
    """
    close = prices["close"].to_numpy()
    regime = np.empty(len(close), dtype=object)
    state = "bull"
    peak = close[0]
    trough = close[0]

    for index, price in enumerate(close):
        if state == "bull":
            peak = max(peak, price)
            if price <= peak * 0.80:
                state = "bear"
                trough = price
        else:
            trough = min(trough, price)
            if price >= trough * 1.20:
                state = "bull"
                peak = price
        regime[index] = state

    return pd.Series(regime, index=prices.index)
