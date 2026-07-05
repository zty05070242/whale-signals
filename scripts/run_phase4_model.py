"""
Phase 4 Step 2: Train and evaluate the price direction predictor.

Runs walk-forward validation on all three prediction horizons (1h, 6h, 24h)
and reports results.

Usage:
    python scripts/run_phase4_model.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.features.phase4_features import get_phase4_feature_columns, HORIZONS  # noqa: E402
from src.models.price_predictor import walk_forward_validation, print_results  # noqa: E402

# ---------------------------------------------------------------------------
# Load feature matrix
# ---------------------------------------------------------------------------

matrix_path = config.PROCESSED_DATA_DIR / "phase4_feature_matrix.csv"
print(f"Loading feature matrix from {matrix_path} ...")
df = pd.read_csv(matrix_path)
print(f"  {len(df):,} rows loaded")

# Subsample to reduce CPU load in Codespace (stratified by month)
# 50k rows is enough for statistically meaningful results
MAX_ROWS = 50_000
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42).sort_values("timestamp_utc").reset_index(drop=True)
    print(f"  Subsampled to {len(df):,} rows (Codespace CPU limit)")

feature_cols = get_phase4_feature_columns()
# Only use features that exist in the matrix
available = [c for c in feature_cols if c in df.columns]
print(f"  Using {len(available)} features: {available}")

# ---------------------------------------------------------------------------
# Run walk-forward validation for each horizon
# ---------------------------------------------------------------------------

all_results = {}

for horizon in HORIZONS:
    target = f"target_{horizon}h"
    print(f"\n{'='*60}")
    print(f"HORIZON: {horizon}h (predicting ETH direction {horizon} hours ahead)")
    print(f"{'='*60}")

    results = walk_forward_validation(
        df,
        feature_cols=available,
        target_col=target,
        train_months=6,
        test_months=1,
    )
    print_results(results)
    all_results[horizon] = results

# ---------------------------------------------------------------------------
# Cross-horizon comparison
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"CROSS-HORIZON COMPARISON")
print(f"{'='*60}")
print(f"  {'Horizon':>8}  {'Baseline':>8}  {'LR':>8}  {'RF':>8}  {'LR edge':>8}  {'RF edge':>8}")
for h in HORIZONS:
    r = all_results[h]
    lr_edge = r["overall_accuracy_lr"] - r["overall_baseline"]
    rf_edge = r["overall_accuracy_rf"] - r["overall_baseline"]
    print(f"  {h:>6}h  {r['overall_baseline']:>8.3f}  "
          f"{r['overall_accuracy_lr']:>8.3f}  {r['overall_accuracy_rf']:>8.3f}  "
          f"{lr_edge:>+8.3f}  {rf_edge:>+8.3f}")
