"""Terminal reporting helpers for event-study result dictionaries."""


def print_hit_rate_results(results: dict) -> None:
    """Print unconditional whale hit-rate results."""
    print(f"\n{'='*80}")
    print("WHALE SMART MONEY TEST: Hit Rates")
    print("(Hit = whale's action correctly predicted price direction)")
    print("(50% = random, >50% = smart money, <50% = dumb money)")
    print(f"{'='*80}")

    for category, horizons in results.items():
        if not horizons:
            continue

        first_horizon = min(horizons)
        direction = horizons[first_horizon]["expected_direction"]
        action = "sold before drop" if direction == "down" else "bought before rise"

        print(f"\n  {category.upper().replace('_', ' ')}  (hit = {action})")
        print(
            f"  {'Horizon':>8}  {'N':>8}  {'Hits':>8}  {'Hit Rate':>9}  "
            f"{'p-value':>8}  {'Sig':>5}  {'Verdict':>12}"
        )

        for horizon, statistics in sorted(horizons.items()):
            significance = _significance_marker(statistics["pvalue"])
            verdict = _verdict(statistics["hit_rate"], statistics["pvalue"])
            print(
                f"  {horizon:>6}h  {statistics['n']:>8,}  "
                f"{statistics['hits']:>8,}  {statistics['hit_rate']:>8.1%}  "
                f"{statistics['pvalue']:>8.4f}  {significance:>5}  "
                f"{verdict:>12}"
            )


def print_conditioned_hit_rates(results: dict) -> None:
    """Print sentiment-conditioned whale hit-rate results."""
    print(f"\n{'='*80}")
    print("CONDITIONED HIT RATES: Are Whales Smarter in Certain Sentiment Regimes?")
    print(f"{'='*80}")

    for category, conditions in results.items():
        if not conditions:
            continue

        print(f"\n  {category.upper().replace('_', ' ')}")

        for condition_name, horizons in sorted(conditions.items()):
            print(f"\n    {condition_name}")
            print(
                f"    {'Horizon':>8}  {'N':>8}  {'Hit Rate':>9}  "
                f"{'p-value':>8}  {'Sig':>5}  {'Verdict':>12}"
            )

            for horizon, statistics in sorted(horizons.items()):
                significance = _significance_marker(statistics["pvalue"])
                verdict = _verdict(statistics["hit_rate"], statistics["pvalue"])
                print(
                    f"    {horizon:>6}h  {statistics['n']:>8,}  "
                    f"{statistics['hit_rate']:>8.1%}  "
                    f"{statistics['pvalue']:>8.4f}  {significance:>5}  "
                    f"{verdict:>12}"
                )


def _significance_marker(pvalue: float) -> str:
    """Return the established star marker for a p-value."""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def _verdict(hit_rate: float, pvalue: float) -> str:
    """Return the established plain-English hit-rate verdict."""
    if hit_rate > 0.50 and pvalue < 0.05:
        return "SMART"
    if hit_rate < 0.50 and pvalue < 0.05:
        return "WRONG"
    return "random"
