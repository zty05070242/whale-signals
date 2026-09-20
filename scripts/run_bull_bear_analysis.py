"""
Does the whale deposit/withdrawal signal mean something different depending
on whether ETH is in a bull market or a bear market?

Motivation
----------
Every conditioning variable used elsewhere in this project (Fear & Greed,
funding rate) measures short-term sentiment. This tests a different, more
structural condition: the prevailing multi-month price TREND itself. A whale
deposit during a bear market and a whale deposit during a bull market could
carry very different information, and nothing so far has tested that split
directly.

Regime definition (standard, not tuned)
-----------------------------------------
The classic textbook definition: a bear market begins when price falls 20%
from the most recent peak; a bull market begins when price rises 20% from
the most recent trough since entering the bear. This is a state machine, not
a fixed window -- no parameter was chosen to make results look a particular
way, it is the standard definition used in financial media and textbooks.

Why this reuses the SAME simple methodology as the rest of the project
--------------------------------------------------------------------------
This is not a new statistical technique. It is the identical hit-rate vs
base-rate comparison already used for Fear & Greed and funding-rate
conditioning (see README Section 2), just with a new conditioning variable.
Restricted to SHORT horizons (24h, 1 week) specifically, since those have
far more independent observations than the long horizons that failed
scrutiny earlier in this project (see Limitations and Discussion).

Usage:
    python scripts/run_bull_bear_analysis.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import config  # noqa: E402
from src.analysis.panel import (  # noqa: E402
    build_analysis_panel,
    forward_return_column,
)
from src.analysis.regimes import compute_regime  # noqa: E402

HORIZONS_H = [24, 72, 168]
HORIZON_LABELS = ["24h", "3d", "1w"]
HORIZON_HOURS = dict(zip(HORIZON_LABELS, HORIZONS_H))
# The sentiment-conditioned test (regime x sentiment x category) showed sign
# flips across horizons -- a symptom of noise from stacking three conditions
# on already-smaller samples. Testing shorter horizons only (1d/3d/5d, no 1w)
# to see whether that instability is specific to the longer end or persists
# throughout.
SENTIMENT_HORIZONS_H = [24, 72, 120]
SENTIMENT_HORIZON_LABELS = ["24h", "3d", "5d"]
SENTIMENT_HORIZON_HOURS = dict(
    zip(SENTIMENT_HORIZON_LABELS, SENTIMENT_HORIZONS_H)
)
THRESHOLDS = [1_000_000, 2_000_000, 5_000_000, 10_000_000]
MIN_N = 30


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load whale transactions and hourly prices, with regime, sentiment, and
    forward returns attached.
    """
    whale = pd.read_csv(config.PROCESSED_DATA_DIR / "whale_txs.csv")
    prices = pd.read_csv(config.PROCESSED_DATA_DIR / "eth_prices_hourly.csv")
    fng = pd.read_csv(config.PROCESSED_DATA_DIR / "fear_greed_daily.csv")
    funding = pd.read_csv(config.PROCESSED_DATA_DIR / "eth_funding_rate.csv")

    prices["timestamp_utc"] = pd.to_datetime(
        prices["timestamp_utc"], utc=True
    )  # create a complete calendar-hour grid before row-based regime analysis
    prices = prices.sort_values("timestamp_utc").reset_index(drop=True)
    full_index = pd.date_range(
        prices["timestamp_utc"].min(), prices["timestamp_utc"].max(), freq="h", tz="UTC",
    )
    price_series = (
        prices.set_index("timestamp_utc")["close"]
        .reindex(full_index)
        .ffill()
    )  # the known missing candle must not change a row-count-based horizon
    filled_prices = pd.DataFrame({"close": price_series}).reset_index(
        names="timestamp_utc"
    )

    all_horizons = sorted(set(HORIZONS_H + SENTIMENT_HORIZONS_H))
    whale, price_df = build_analysis_panel(
        whale,
        filled_prices,
        fng,
        funding,
        all_horizons,
    )
    price_df = price_df.sort_values("timestamp_utc").reset_index(
        drop=True
    )  # the state machine must see candles in strict chronological order
    price_df["regime"] = compute_regime(price_df)
    regime_by_hour = price_df.set_index("timestamp_utc")["regime"]
    whale["regime"] = whale["hour_utc"].map(regime_by_hour)

    return whale, price_df


def run_test(whale: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Hit rate vs base rate, deposits and withdrawals, split by bull/bear regime
    and by minimum transaction size, matching the threshold set used in
    Section 4 elsewhere in this project.
    """
    rows = []
    for threshold in THRESHOLDS:
        sized = whale[whale["usd_value"] >= threshold]
        deposits = sized[sized["tx_category"] == "exchange_deposit"]
        withdrawals = sized[sized["tx_category"] == "exchange_withdrawal"]

        for label in HORIZON_LABELS:
            col = forward_return_column(HORIZON_HOURS[label])
            for regime in ["bull", "bear"]:
                base_returns = price_df[price_df["regime"] == regime][col].dropna()

                for cat_name, cat_df, direction in [
                    ("deposit", deposits, "down"),
                    ("withdrawal", withdrawals, "up"),
                ]:
                    subset = cat_df[cat_df["regime"] == regime][col].dropna()
                    if len(subset) < MIN_N or len(base_returns) < MIN_N:
                        continue

                    if direction == "down":
                        hits = int((subset < 0).sum())
                        base_rate = (base_returns < 0).mean()
                    else:
                        hits = int((subset > 0).sum())
                        base_rate = (base_returns > 0).mean()

                    n = len(subset)
                    hit_rate = hits / n
                    p_value = stats.binomtest(hits, n, p=0.5).pvalue

                    rows.append({
                        "threshold": threshold, "category": cat_name,
                        "horizon": label, "regime": regime,
                        "n": n, "hit_rate": round(hit_rate * 100, 1),
                        "base_rate": round(base_rate * 100, 1),
                        "edge": round((hit_rate - base_rate) * 100, 1),
                        "pvalue": round(p_value, 4),
                    })
    return pd.DataFrame(rows)


def run_sentiment_conditioned_test(whale: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Bull vs bear, WITHIN each of two sentiment regimes, at the base $1M+
    threshold only (no threshold breakdown here, matching how Section 2's
    sentiment-conditioned tables work elsewhere in this project).
    """
    deposits = whale[whale["tx_category"] == "exchange_deposit"]
    withdrawals = whale[whale["tx_category"] == "exchange_withdrawal"]

    conditions = {
        "extreme_greed": (lambda d: d["fng_value"] > 75, lambda p: p["fng_value"] > 75),
        "negative_funding": (lambda d: d["funding_rate"] < 0, lambda p: p["funding_rate"] < 0),
    }

    rows = []
    for cond_name, (whale_mask_fn, price_mask_fn) in conditions.items():
        for label in SENTIMENT_HORIZON_LABELS:
            col = forward_return_column(SENTIMENT_HORIZON_HOURS[label])
            for regime in ["bull", "bear"]:
                base_pool = price_df[(price_df["regime"] == regime) & price_mask_fn(price_df)][col].dropna()

                for cat_name, cat_df, direction in [
                    ("deposit", deposits, "down"),
                    ("withdrawal", withdrawals, "up"),
                ]:
                    pool = cat_df[(cat_df["regime"] == regime) & whale_mask_fn(cat_df)]
                    subset = pool[col].dropna()
                    if len(subset) < MIN_N or len(base_pool) < MIN_N:
                        continue

                    if direction == "down":
                        hits = int((subset < 0).sum())
                        base_rate = (base_pool < 0).mean()
                    else:
                        hits = int((subset > 0).sum())
                        base_rate = (base_pool > 0).mean()

                    n = len(subset)
                    hit_rate = hits / n
                    p_value = stats.binomtest(hits, n, p=0.5).pvalue

                    rows.append({
                        "condition": cond_name, "category": cat_name,
                        "horizon": label, "regime": regime,
                        "n": n, "hit_rate": round(hit_rate * 100, 1),
                        "base_rate": round(base_rate * 100, 1),
                        "edge": round((hit_rate - base_rate) * 100, 1),
                        "pvalue": round(p_value, 4),
                    })
    return pd.DataFrame(rows)


def print_sentiment_results(results: pd.DataFrame) -> None:
    print(f"\n{'='*100}")
    print("BULL vs BEAR, WITHIN EXTREME GREED / NEGATIVE FUNDING (base $1M+ threshold)")
    print(f"{'='*100}")

    for cond in ["extreme_greed", "negative_funding"]:
        print(f"\n{'#'*100}\nCONDITION: {cond}\n{'#'*100}")
        c_results = results[results["condition"] == cond]

        for cat in ["deposit", "withdrawal"]:
            print(f"\n  {cat.upper()}")
            print(f"  {'Horizon':>8} {'Regime':>6} {'N':>8} {'Hit Rate':>9} {'Base Rate':>10} {'Edge':>7} {'p-value':>8}")
            sub = c_results[c_results["category"] == cat]
            for _, r in sub.iterrows():
                sig = "***" if r["pvalue"] < 0.001 else "**" if r["pvalue"] < 0.01 else "*" if r["pvalue"] < 0.05 else ""
                print(f"  {r['horizon']:>8} {r['regime']:>6} {r['n']:>8,} {r['hit_rate']:>8.1f}% "
                      f"{r['base_rate']:>9.1f}% {r['edge']:>+6.1f}% {r['pvalue']:>7.4f}{sig}")


def print_results(results: pd.DataFrame) -> None:
    print(f"\n{'='*100}")
    print("BULL vs BEAR MARKET: does the whale signal differ by price regime, and by size?")
    print(f"{'='*100}")

    for threshold in THRESHOLDS:
        print(f"\n{'#'*100}\nTHRESHOLD: ${threshold/1e6:.0f}M+\n{'#'*100}")
        t_results = results[results["threshold"] == threshold]

        for cat in ["deposit", "withdrawal"]:
            print(f"\n  {cat.upper()} ({'sell signal, hit = price fell' if cat == 'deposit' else 'buy signal, hit = price rose'})")
            print(f"  {'Horizon':>8} {'Regime':>6} {'N':>8} {'Hit Rate':>9} {'Base Rate':>10} {'Edge':>7} {'p-value':>8}")
            sub = t_results[t_results["category"] == cat]
            for _, r in sub.iterrows():
                sig = "***" if r["pvalue"] < 0.001 else "**" if r["pvalue"] < 0.01 else "*" if r["pvalue"] < 0.05 else ""
                print(f"  {r['horizon']:>8} {r['regime']:>6} {r['n']:>8,} {r['hit_rate']:>8.1f}% "
                      f"{r['base_rate']:>9.1f}% {r['edge']:>+6.1f}% {r['pvalue']:>7.4f}{sig}")


def main() -> None:
    print("Loading data and computing bull/bear regimes...")
    whale, price_df = load_data()
    print(f"Regime hours: {price_df['regime'].value_counts().to_dict()}")

    results = run_test(whale, price_df)
    print_results(results)

    out_path = config.ROOT_DIR / "results" / "bull_bear_analysis.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    sentiment_results = run_sentiment_conditioned_test(whale, price_df)
    print_sentiment_results(sentiment_results)

    sentiment_out_path = config.ROOT_DIR / "results" / "bull_bear_sentiment_analysis.csv"
    sentiment_results.to_csv(sentiment_out_path, index=False)
    print(f"\nResults saved to {sentiment_out_path}")


if __name__ == "__main__":
    main()
