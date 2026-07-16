"""
Phase 2 Step 2: Evaluate the transaction classifier on a time-based hold-out.

The 71% accuracy figure quoted in the README was measured once, back when
the dataset had ~292k rows (see commit e8fbe72). The dataset has since grown
to 646k+ rows, and the classifier is retrained on the full current dataset
every time run_phase4_features.py runs, but its accuracy has never been
re-measured at the current scale. This script exists so that number has a
script backing it going forward, the same way every other figure in this
project does.

Usage:
    python scripts/run_phase2_classifier_eval.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.features.feature_engineer import (  # noqa: E402
    assign_transaction_label,
    build_features,
    get_feature_columns,
)
from src.models.transaction_classifier import (  # noqa: E402
    prepare_labelled_data,
    time_based_split,
    train_classifier,
    evaluate_classifier,
)

print("Loading processed whale data...")
whale = pd.read_csv(config.PROCESSED_DATA_DIR / "whale_txs.csv")
print(f"  {len(whale):,} transactions loaded")

whale = assign_transaction_label(whale)
whale = build_features(whale)

labelled, unlabelled = prepare_labelled_data(whale)
print(f"  Labelled: {len(labelled):,}, Unlabelled: {len(unlabelled):,} "
      f"({len(unlabelled) / len(whale):.1%})")

# Same 80/20 time-based split used at Phase 2 completion: sort by time, train
# on the earliest 80%, test on the most recent 20% the model has never seen.
train_df, test_df = time_based_split(labelled, train_fraction=0.8)
print(f"  Train: {len(train_df):,} rows ({train_df['timestamp_utc'].min()} to "
      f"{train_df['timestamp_utc'].max()})")
print(f"  Test:  {len(test_df):,} rows ({test_df['timestamp_utc'].min()} to "
      f"{test_df['timestamp_utc'].max()})")

feature_cols = get_feature_columns()
clf = train_classifier(train_df, feature_columns=feature_cols)
results = evaluate_classifier(clf, test_df, feature_columns=feature_cols)

print(f"\n{'='*60}")
print(f"ACCURACY ON TIME-BASED HOLD-OUT: {results['accuracy']:.1%}")
print(f"(292k-row measurement at Phase 2 completion was 71%)")
print(f"{'='*60}")
