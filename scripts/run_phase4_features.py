"""
Phase 4 Step 1: Assemble the feature matrix.

Loads whale transactions, ETH prices, and sentiment data, merges them
into a single feature matrix, and saves the result.

Usage:
    python scripts/run_phase4_features.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.features.feature_engineer import (  # noqa: E402
    build_features,
    assign_transaction_label,
)
from src.models.transaction_classifier import (  # noqa: E402
    prepare_labelled_data,
    train_classifier,
    predict_unlabelled,
)
from src.features.phase4_features import (  # noqa: E402
    build_feature_matrix,
    get_phase4_feature_columns,
    HORIZONS,
)

# ---------------------------------------------------------------------------
# 1. Load whale transactions and run Phase 2 classifier
# ---------------------------------------------------------------------------

print("Loading processed whale data (with wallet labels)...")
processed_path = config.PROCESSED_DATA_DIR / "whale_txs.csv"
whale = pd.read_csv(processed_path)
print(f"  {len(whale):,} transactions loaded")

# Assign rule-based labels (exchange_deposit, withdrawal, etc.)
print("  Assigning rule-based transaction labels...")
whale = assign_transaction_label(whale)
print(f"  Category distribution:")
for cat, count in whale["tx_category"].value_counts().items():
    print(f"    {cat}: {count:,} ({count/len(whale):.1%})")

# Build Phase 2 features (log values, time features, sender history)
print("  Building Phase 2 features...")
whale = build_features(whale)

# Train classifier on labelled data and predict unlabelled
print("  Training Phase 2 classifier on labelled transactions...")
labelled, unlabelled = prepare_labelled_data(whale)
print(f"    Labelled: {len(labelled):,}, Unlabelled: {len(unlabelled):,}")

clf = train_classifier(labelled)

# Predict categories for unknown->unknown transactions
predicted = predict_unlabelled(clf, unlabelled)
print(f"  Predicted category distribution (unlabelled):")
for cat, count in predicted["predicted_category"].value_counts().items():
    print(f"    {cat}: {count:,} ({count/len(predicted):.1%})")

# Merge: labelled rows keep their rule-based category,
# unlabelled rows get their predicted category
labelled_out = labelled.copy()
predicted_out = predicted.copy()
predicted_out["tx_category"] = predicted_out["predicted_category"]
whale = pd.concat([labelled_out, predicted_out], ignore_index=True)
whale = whale.sort_values("timestamp_utc").reset_index(drop=True)
print(f"\n  Final category distribution (all {len(whale):,} transactions):")
for cat, count in whale["tx_category"].value_counts().items():
    print(f"    {cat}: {count:,} ({count/len(whale):.1%})")

# ---------------------------------------------------------------------------
# 2. Load ETH prices
# ---------------------------------------------------------------------------

print("\nLoading ETH hourly prices...")
prices = pd.read_csv(config.PROCESSED_DATA_DIR / "eth_prices_hourly.csv")
print(f"  {len(prices):,} hourly price candles loaded")

# ---------------------------------------------------------------------------
# 3. Load sentiment
# ---------------------------------------------------------------------------

print("\nLoading hourly sentiment...")
sentiment_path = config.PROCESSED_DATA_DIR / "hourly_sentiment.csv"
if sentiment_path.exists():
    sentiment = pd.read_csv(sentiment_path)
    print(f"  {len(sentiment):,} hourly sentiment bins loaded")
else:
    sentiment = None
    print("  No sentiment data found. Proceeding without sentiment features.")

# ---------------------------------------------------------------------------
# 3b. Load Fear & Greed Index
# ---------------------------------------------------------------------------

print("\nLoading Fear & Greed Index...")
fng_path = config.PROCESSED_DATA_DIR / "fear_greed_daily.csv"
if fng_path.exists():
    fng = pd.read_csv(fng_path)
    print(f"  {len(fng):,} daily Fear & Greed records loaded")
else:
    fng = None
    print("  No Fear & Greed data found.")

# ---------------------------------------------------------------------------
# 3c. Load Binance funding rate
# ---------------------------------------------------------------------------

print("\nLoading Binance funding rates...")
funding_path = config.PROCESSED_DATA_DIR / "eth_funding_rate.csv"
if funding_path.exists():
    funding = pd.read_csv(funding_path)
    print(f"  {len(funding):,} funding rate records loaded")
else:
    funding = None
    print("  No funding rate data found.")

# ---------------------------------------------------------------------------
# 4. Build feature matrix
# ---------------------------------------------------------------------------

print("\nBuilding Phase 4 feature matrix...")
matrix = build_feature_matrix(whale, prices, sentiment, fng_df=fng, funding_df=funding)

# ---------------------------------------------------------------------------
# 5. Summary stats
# ---------------------------------------------------------------------------

feature_cols = get_phase4_feature_columns()
# Only include features that actually exist in the matrix
available_features = [c for c in feature_cols if c in matrix.columns]

print(f"\n{'='*60}")
print(f"FEATURE MATRIX SUMMARY")
print(f"{'='*60}")
print(f"Total rows:     {len(matrix):,}")
print(f"Date range:     {matrix['timestamp_utc'].min()} to {matrix['timestamp_utc'].max()}")
print(f"Features:       {len(available_features)}")
print(f"Targets:        {[f'target_{h}h' for h in HORIZONS]}")

print(f"\nTarget class balance:")
for h in HORIZONS:
    col = f"target_{h}h"
    up = matrix[col].mean()
    print(f"  {col}: {up:.1%} up / {1-up:.1%} down")

print(f"\nFeature availability (non-null %):")
for col in available_features:
    pct = matrix[col].notna().mean()
    print(f"  {col}: {pct:.1%}")

print(f"\nSentiment coverage:")
has_sentiment = (matrix["article_count"] > 0).sum()
print(f"  Rows with sentiment data: {has_sentiment:,} / {len(matrix):,} "
      f"({has_sentiment/len(matrix):.1%})")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------

output = config.PROCESSED_DATA_DIR / "phase4_feature_matrix.csv"
output.parent.mkdir(parents=True, exist_ok=True)
matrix.to_csv(output, index=False)
print(f"\nSaved feature matrix to {output}")
