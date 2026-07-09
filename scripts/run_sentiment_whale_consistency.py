"""
Does whale behaviour track public news sentiment, or does it look independent
of (or contrarian to) what the news is saying?

Motivation
----------
News sentiment, scored with VADER, was tested earlier in this project as a
predictor of PRICE and found uninformative (see README, Sentiment Data
section). This is a different question: not "does news predict price", but
"does news sentiment line up with what whales are doing that same day".

Two honest outcomes are both informative:
  - If whale selling/buying tracks news sentiment (bearish news, more
    deposits), whales may just be reacting to the same public information as
    everyone else, not acting on anything special.
  - If whale activity looks unrelated or contrarian to news sentiment, that
    is a point in favour of whales acting on information not reflected in
    public reporting.

Data coverage (checked before building this)
---------------------------------------------
News sentiment covers Jan 2023 to Sep 2024 only (5,906 articles), not the
full 2023-2026 whale dataset. Of the ~619 calendar days in that span, 484
have at least one article (median 12/day). Any finding from this script is
scoped to that ~20-month window, not the full whale dataset.

Methodology
-----------
Unit of observation: the DAY (not the individual whale transaction, and not
an event with a forward-looking window). This is a same-day comparison, not
a prediction, so it does NOT have the overlapping-forward-window problem this
project spent a lot of effort on elsewhere. It has a different, milder
version of a related issue: news sentiment and whale activity can each have
their own day-to-day momentum, so a raw correlation could partly reflect two
trending series moving together rather than a real same-day relationship.
This script reports the lag-1 autocorrelation of both series so that risk is
visible, rather than hidden.

For each day with at least one news article:
    sentiment_t = mean VADER compound score across that day's articles
    net_flow_usd_t = total deposit USD - total withdrawal USD that day
    net_flow_count_t = deposit count - withdrawal count that day

Two simple, complementary checks, both standard and easy to explain:
  1. Pearson correlation between sentiment_t and net_flow_t (both USD and
     count versions).
  2. Two-group comparison: split days into "bearish news" (sentiment < 0)
     and "bullish news" (sentiment > 0), compare mean net flow between the
     groups with a two-sample t-test.

Usage:
    python scripts/run_sentiment_whale_consistency.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import config  # noqa: E402
from src.features.feature_engineer import assign_transaction_label  # noqa: E402


def build_daily_panel() -> pd.DataFrame:
    """One row per day-with-news: sentiment, deposit/withdrawal volume and count."""
    news = pd.read_csv(config.PROCESSED_DATA_DIR / "bitcoin_news_scored.csv")
    news["Date"] = pd.to_datetime(news["Date"])
    news["day"] = news["Date"].dt.floor("D")
    daily_sentiment = news.groupby("day")["vader_compound"].mean().rename("sentiment")

    whale = pd.read_csv(config.PROCESSED_DATA_DIR / "whale_txs.csv")
    whale["timestamp_utc"] = pd.to_datetime(whale["timestamp_utc"], utc=True).dt.tz_localize(None)
    whale = assign_transaction_label(whale)
    whale["day"] = whale["timestamp_utc"].dt.floor("D")

    deposits = whale[whale["tx_category"] == "exchange_deposit"]
    withdrawals = whale[whale["tx_category"] == "exchange_withdrawal"]

    dep_usd = deposits.groupby("day")["usd_value"].sum().rename("dep_usd")
    wd_usd = withdrawals.groupby("day")["usd_value"].sum().rename("wd_usd")
    dep_count = deposits.groupby("day").size().rename("dep_count")
    wd_count = withdrawals.groupby("day").size().rename("wd_count")

    panel = pd.concat([daily_sentiment, dep_usd, wd_usd, dep_count, wd_count], axis=1, sort=False)
    # Restrict to days that actually have news coverage -- do not impute
    # sentiment for the 135 gap days, and do not extend past the news window.
    panel = panel.dropna(subset=["sentiment"])
    panel[["dep_usd", "wd_usd", "dep_count", "wd_count"]] = (
        panel[["dep_usd", "wd_usd", "dep_count", "wd_count"]].fillna(0)
    )

    panel["net_flow_usd"] = panel["dep_usd"] - panel["wd_usd"]
    panel["net_flow_count"] = panel["dep_count"] - panel["wd_count"]
    return panel.sort_index()


def run_checks(panel: pd.DataFrame) -> dict:
    """Correlation and two-group comparison, plus the autocorrelation caveat check."""
    results = {}

    for col in ["net_flow_usd", "net_flow_count"]:
        r, p = stats.pearsonr(panel["sentiment"], panel[col])
        results[f"corr_{col}"] = {"r": round(r, 4), "pvalue": round(p, 4), "n": len(panel)}

    bearish = panel[panel["sentiment"] < 0]
    bullish = panel[panel["sentiment"] > 0]
    for col in ["net_flow_usd", "net_flow_count"]:
        t, p = stats.ttest_ind(bearish[col], bullish[col], equal_var=False)
        results[f"ttest_{col}"] = {
            "bearish_n": len(bearish), "bullish_n": len(bullish),
            "bearish_mean": round(bearish[col].mean(), 2),
            "bullish_mean": round(bullish[col].mean(), 2),
            "tstat": round(t, 4), "pvalue": round(p, 4),
        }

    # Lag-1 autocorrelation: how much does each series carry over day to day?
    # High values here mean a raw correlation could partly reflect two
    # trending series, not a genuine same-day relationship.
    results["autocorr_sentiment"] = round(panel["sentiment"].autocorr(lag=1), 4)
    results["autocorr_net_flow_usd"] = round(panel["net_flow_usd"].autocorr(lag=1), 4)

    return results


def print_results(panel: pd.DataFrame, results: dict) -> None:
    print(f"\n{'='*90}")
    print(f"NEWS SENTIMENT vs WHALE ACTIVITY: {len(panel)} days with news coverage")
    print(f"Date range: {panel.index.min().date()} to {panel.index.max().date()}")
    print(f"{'='*90}")

    print("\n  CORRELATION (daily sentiment vs daily net flow)")
    for col, label in [("net_flow_usd", "Net flow ($)"), ("net_flow_count", "Net flow (count)")]:
        r = results[f"corr_{col}"]
        print(f"  {label:>20}: r={r['r']:+.4f}  p={r['pvalue']:.4f}  n={r['n']}")

    print("\n  BEARISH NEWS DAYS vs BULLISH NEWS DAYS (mean net flow)")
    for col, label in [("net_flow_usd", "Net flow ($)"), ("net_flow_count", "Net flow (count)")]:
        t = results[f"ttest_{col}"]
        print(f"  {label:>20}: bearish(n={t['bearish_n']}) mean={t['bearish_mean']:,.0f}  "
              f"bullish(n={t['bullish_n']}) mean={t['bullish_mean']:,.0f}  "
              f"t={t['tstat']:+.3f}  p={t['pvalue']:.4f}")

    print("\n  AUTOCORRELATION CHECK (day t vs day t-1, both series)")
    print(f"  Sentiment lag-1 autocorrelation:  {results['autocorr_sentiment']:+.4f}")
    print(f"  Net flow ($) lag-1 autocorrelation: {results['autocorr_net_flow_usd']:+.4f}")
    print("  High values on either line mean any correlation above could partly")
    print("  reflect two persistent, trending series rather than a same-day link.")

    print(f"\n  Positive correlation = bearish news days see MORE net selling (deposits).")
    print(f"  Negative correlation = bearish news days see MORE net buying (withdrawals),")
    print(f"  i.e. whales moving opposite to the public narrative.")


def main() -> None:
    print("Building daily panel...")
    panel = build_daily_panel()
    results = run_checks(panel)
    print_results(panel, results)

    out_path = config.ROOT_DIR / "results" / "sentiment_whale_consistency.csv"
    panel.to_csv(out_path)
    print(f"\nDaily panel saved to {out_path}")


if __name__ == "__main__":
    main()
