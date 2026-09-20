"""Run the shortest auditable path through the project's core research.

This entry point deliberately covers the central event study and yearly
walk-forward check. Specialist analyses such as maximum adverse excursion and
the separate machine-learning prototypes remain independent because they ask
different methodological questions.

Usage:
    python scripts/run_research.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.analysis.event_study import (  # noqa: E402
    HORIZONS,
    compute_conditioned_hit_rates,
    compute_hit_rates,
    print_conditioned_hit_rates,
    print_hit_rate_results,
    walk_forward_by_year,
)
from src.analysis.panel import (  # noqa: E402
    build_analysis_panel,
    forward_return_column,
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four processed datasets needed by the core event study."""
    processed = config.PROCESSED_DATA_DIR
    whales = pd.read_csv(
        processed / "whale_txs.csv"
    )  # labelled on-chain transaction inputs
    prices = pd.read_csv(
        processed / "eth_prices_hourly.csv"
    )  # hourly ETH base-rate universe
    fear_greed = pd.read_csv(
        processed / "fear_greed_daily.csv"
    )  # daily sentiment regime
    funding = pd.read_csv(
        processed / "eth_funding_rate.csv"
    )  # eight-hourly derivatives positioning
    return whales, prices, fear_greed, funding


def main() -> None:
    """Build one canonical panel and run the core research checks from it."""
    print("Loading processed research inputs...")
    whales, prices, fear_greed, funding = load_inputs()

    print("Building the aligned event and market panels...")
    events, market = build_analysis_panel(
        whales,
        prices,
        fear_greed,
        funding,
        HORIZONS,
    )

    complete_events = events.dropna(
        subset=[forward_return_column(horizon) for horizon in HORIZONS]
    ).reset_index(drop=True)  # match the original event study's sample boundary
    print(f"  {len(complete_events):,} events have every core forward return.")

    print("\nRunning the unconditional hit-rate study...")
    hit_rates = compute_hit_rates(complete_events)
    print_hit_rate_results(hit_rates)

    print("\nRunning the sentiment-conditioned study...")
    conditioned = compute_conditioned_hit_rates(complete_events)
    print_conditioned_hit_rates(conditioned)

    print("\nRunning the yearly walk-forward check...")
    walk_forward = walk_forward_by_year(complete_events, market)
    output_path = config.ROOT_DIR / "results" / "walk_forward_results.csv"
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # allow the runner to work in a fresh checkout
    walk_forward.to_csv(
        output_path, index=False
    )  # retain the existing machine-readable result contract
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
