"""
Phase 4 price direction predictor with walk-forward validation.

Walk-forward validation simulates real-world deployment:
  1. Train on data from months 1-N
  2. Predict month N+1
  3. Slide window forward and repeat

This prevents look-ahead bias because the model never sees future data
during training. It also reveals how performance changes over time —
does the signal decay, or is it stable?

Two models are tested:
  - Logistic Regression: simple baseline (linear decision boundary)
  - Random Forest: captures non-linear interactions between features

If neither beats 50% consistently, the honest conclusion is that whale
transactions + sentiment do not predict short-term ETH price direction.
That is a valid research finding.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_24h",
    train_months: int = 6,
    test_months: int = 1,
) -> dict:
    """
    Run walk-forward validation on the feature matrix.

    Slides a window through time: train on train_months of data,
    predict the next test_months, then advance by test_months.

    Parameters
    ----------
    df : pd.DataFrame
        Phase 4 feature matrix (output of build_feature_matrix).
        Must have timestamp_utc and the specified feature/target columns.
    feature_cols : list[str]
        Column names to use as model inputs.
    target_col : str
        Column name of the binary target (0 or 1).
    train_months : int
        Number of months of history to train on in each fold.
    test_months : int
        Number of months to predict in each fold.

    Returns
    -------
    dict with keys:
        - folds: list of per-fold results (accuracy, dates, predictions)
        - overall_accuracy_lr: weighted mean accuracy for logistic regression
        - overall_accuracy_rf: weighted mean accuracy for random forest
        - fold_details: summary DataFrame
    """
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Validate columns exist
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # Drop rows with NaN in features or target
    df = df.dropna(subset=feature_cols + [target_col])

    # Generate fold boundaries using pd.DateOffset for month-level stepping
    min_date = df["timestamp_utc"].min()
    max_date = df["timestamp_utc"].max()

    folds = []
    fold_num = 0
    train_start = min_date

    while True:
        # Training window: train_start to train_start + train_months
        train_end = train_start + pd.DateOffset(months=train_months)
        # Test window: train_end to train_end + test_months
        test_end = train_end + pd.DateOffset(months=test_months)

        if train_end >= max_date:
            break  # not enough data for a test set

        # Split data by timestamp
        train_mask = (
            (df["timestamp_utc"] >= train_start)
            & (df["timestamp_utc"] < train_end)
        )
        test_mask = (
            (df["timestamp_utc"] >= train_end)
            & (df["timestamp_utc"] < test_end)
        )

        train_data = df[train_mask]
        test_data = df[test_mask]

        # Skip folds with too little data
        if len(train_data) < 100 or len(test_data) < 50:
            train_start = train_start + pd.DateOffset(months=test_months)
            continue

        fold_num += 1
        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values

        # --- StandardScaler ---
        # Logistic regression is sensitive to feature scale. Fit the scaler
        # on training data only — transforming test data with train stats
        # prevents information leakage.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- Logistic Regression (baseline) ---
        lr = LogisticRegression(
            max_iter=1000,     # enough iterations for convergence
            random_state=42,
            class_weight="balanced",  # handle slight class imbalance
        )
        lr.fit(X_train_scaled, y_train)
        lr_preds = lr.predict(X_test_scaled)
        lr_acc = accuracy_score(y_test, lr_preds)

        # --- Random Forest ---
        rf = RandomForestClassifier(
            n_estimators=100,       # fewer trees to reduce CPU load
            max_depth=8,            # shallower trees = faster + less overfitting
            min_samples_leaf=50,    # larger leaves = faster + more regularised
            class_weight="balanced",
            random_state=42,
            n_jobs=2,               # limit to 2 cores (Codespace-friendly)
        )
        rf.fit(X_train, y_train)  # RF does not need scaling
        rf_preds = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_preds)

        # Feature importances from Random Forest
        importances = dict(zip(feature_cols, rf.feature_importances_))

        fold_result = {
            "fold": fold_num,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
            "train_size": len(train_data),
            "test_size": len(test_data),
            "lr_accuracy": lr_acc,
            "rf_accuracy": rf_acc,
            "baseline": y_test.mean(),  # fraction of class 1 (always-up accuracy)
            "feature_importances": importances,
        }
        folds.append(fold_result)

        print(f"  Fold {fold_num}: train {train_start.date()} to {train_end.date()}, "
              f"test {train_end.date()} to {test_end.date()} | "
              f"LR={lr_acc:.3f}  RF={rf_acc:.3f}  baseline={y_test.mean():.3f} "
              f"(n_test={len(test_data):,})")

        # Slide the window forward by test_months
        train_start = train_start + pd.DateOffset(months=test_months)

    if not folds:
        raise RuntimeError("No valid folds could be created. Check data size and date range.")

    # --- Aggregate results ---
    fold_df = pd.DataFrame([
        {k: v for k, v in f.items() if k != "feature_importances"}
        for f in folds
    ])

    # Weighted average accuracy (weighted by test set size)
    total_test = fold_df["test_size"].sum()
    overall_lr = (fold_df["lr_accuracy"] * fold_df["test_size"]).sum() / total_test
    overall_rf = (fold_df["rf_accuracy"] * fold_df["test_size"]).sum() / total_test
    overall_base = (fold_df["baseline"] * fold_df["test_size"]).sum() / total_test

    # Average feature importances across all folds
    avg_importances = {}
    for col in feature_cols:
        avg_importances[col] = np.mean([f["feature_importances"][col] for f in folds])

    return {
        "folds": folds,
        "fold_details": fold_df,
        "overall_accuracy_lr": overall_lr,
        "overall_accuracy_rf": overall_rf,
        "overall_baseline": overall_base,
        "avg_feature_importances": avg_importances,
        "target_col": target_col,
        "n_folds": len(folds),
        "total_test_samples": total_test,
    }


def print_results(results: dict) -> None:
    """Print a formatted summary of walk-forward validation results."""
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Target:          {results['target_col']}")
    print(f"Folds:           {results['n_folds']}")
    print(f"Total test rows: {results['total_test_samples']:,}")

    print(f"\nOverall accuracy (weighted by fold size):")
    print(f"  Always-up baseline: {results['overall_baseline']:.3f}")
    print(f"  Logistic Regression: {results['overall_accuracy_lr']:.3f}")
    print(f"  Random Forest:       {results['overall_accuracy_rf']:.3f}")

    edge_lr = results["overall_accuracy_lr"] - results["overall_baseline"]
    edge_rf = results["overall_accuracy_rf"] - results["overall_baseline"]
    print(f"\nEdge over baseline:")
    print(f"  LR: {edge_lr:+.3f} ({'better' if edge_lr > 0 else 'worse'})")
    print(f"  RF: {edge_rf:+.3f} ({'better' if edge_rf > 0 else 'worse'})")

    print(f"\nTop 10 feature importances (Random Forest, averaged):")
    sorted_imp = sorted(
        results["avg_feature_importances"].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, imp in sorted_imp[:10]:
        bar = "#" * int(imp * 100)
        print(f"  {name:25s} {imp:.3f}  {bar}")

    print(f"\nPer-fold accuracy:")
    print(f"  {'Fold':>4}  {'Period':>25}  {'LR':>6}  {'RF':>6}  {'Base':>6}  {'N':>7}")
    for _, row in results["fold_details"].iterrows():
        period = f"{row['train_end'].date()} to {row['test_end'].date()}"
        print(f"  {int(row['fold']):4d}  {period:>25}  "
              f"{row['lr_accuracy']:6.3f}  {row['rf_accuracy']:6.3f}  "
              f"{row['baseline']:6.3f}  {int(row['test_size']):7,}")
