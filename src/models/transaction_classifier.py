"""
Whale transaction classifier.

Trains a Random Forest on rule-labelled transactions (where at least one
address is a known exchange or DeFi protocol) and predicts the category
for unlabelled (unknown->unknown) transactions.

The classifier is a feature engineering step for Phase 4, not an end in
itself. Its outputs — predicted category and probability scores — become
features in the price impact prediction model.

Design decisions:
  - Time-based train/test split (not random). Random splits leak temporal
    patterns across the boundary. This is consistent with Phase 4's
    walk-forward requirement.
  - The classifier must be retrainable on temporal subsets so that Phase 4
    can retrain it at each walk-forward step.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.features.feature_engineer import (
    assign_transaction_label,
    build_features,
    get_feature_columns,
    WALLET_TO_WALLET,
)


def prepare_labelled_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into labelled rows (for training/evaluation) and
    unlabelled rows (for prediction).

    Labelled = at least one known address (tx_category != wallet_to_wallet).
    Unlabelled = both sides unknown (tx_category == wallet_to_wallet).

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_features() with tx_category column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (labelled_df, unlabelled_df)
    """
    labelled = df[df["tx_category"] != WALLET_TO_WALLET].copy()
    unlabelled = df[df["tx_category"] == WALLET_TO_WALLET].copy()
    return labelled, unlabelled


def time_based_split(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and test sets based on timestamp order.

    The first train_fraction of rows (by time) go to training, the rest
    to testing. This prevents future data from leaking into the training set.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a timestamp_utc column. Should already be sorted by time
        (build_features() does this).
    train_fraction : float
        Proportion of rows for training. Default 0.8 (80/20 split).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    # Sort by time to ensure the split is temporal, not arbitrary
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    split_idx = int(len(df) * train_fraction)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    return train, test


def train_classifier(
    train_df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    random_state: int = 42,
    n_estimators: int = 200,
    class_weight: Optional[str] = "balanced",
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on labelled whale transactions.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with feature columns and tx_category.
    feature_columns : list[str], optional
        Which columns to use as features. Defaults to get_feature_columns().
    random_state : int
        Seed for reproducibility. Default 42.
    n_estimators : int
        Number of trees in the forest. Default 200.
    class_weight : str or None
        'balanced' adjusts weights inversely proportional to class frequency.
        This helps when one category (e.g. exchange_deposit) dominates.
        Set to None for unweighted training.

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    if feature_columns is None:
        feature_columns = get_feature_columns()

    X_train = train_df[feature_columns]
    y_train = train_df["tx_category"]

    # RandomForestClassifier fits an ensemble of decision trees.
    # Each tree sees a random subset of rows (bootstrap) and features,
    # then the forest averages their votes — reducing overfitting vs a
    # single decision tree.
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,  # use all CPU cores for parallel tree fitting
    )

    # .fit() trains the model: learns decision boundaries from X_train -> y_train
    clf.fit(X_train, y_train)

    return clf


def evaluate_classifier(
    clf: RandomForestClassifier,
    test_df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
) -> dict:
    """
    Evaluate the classifier on a hold-out test set.

    Prints a classification report and returns structured results for
    programmatic use.

    Parameters
    ----------
    clf : RandomForestClassifier
        Fitted model from train_classifier().
    test_df : pd.DataFrame
        Test data (must not overlap with training data).
    feature_columns : list[str], optional
        Must match the columns used during training.

    Returns
    -------
    dict
        Keys: 'accuracy', 'classification_report' (str),
        'confusion_matrix' (np.ndarray), 'feature_importances' (dict),
        'predictions' (np.ndarray), 'probabilities' (np.ndarray).
    """
    if feature_columns is None:
        feature_columns = get_feature_columns()

    X_test = test_df[feature_columns]
    y_test = test_df["tx_category"]

    # .predict() returns the class with the highest vote across all trees
    y_pred = clf.predict(X_test)

    # .predict_proba() returns probability estimates for each class.
    # Each row sums to 1.0 — e.g. [0.1, 0.7, 0.05, 0.15] means the model
    # is 70% confident this is the second class.
    y_proba = clf.predict_proba(X_test)

    # clf.score() computes accuracy: fraction of predictions that match y_test
    accuracy = clf.score(X_test, y_test)

    # classification_report produces precision, recall, F1 per class — more
    # informative than accuracy alone, especially with imbalanced classes
    report = classification_report(y_test, y_pred, zero_division=0)

    # confusion_matrix[i, j] = number of samples with true label i predicted as j
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    # feature_importances_ measures how much each feature contributes to the
    # forest's decisions (Gini importance). Higher = more influential.
    importances = dict(zip(feature_columns, clf.feature_importances_))

    print(f"Accuracy: {accuracy:.3f}")
    print()
    print(report)
    print("Feature importances (Gini):")
    # sorted() with key= sorts by importance descending
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:30s} {imp:.4f}")

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importances": importances,
        "predictions": y_pred,
        "probabilities": y_proba,
    }


def predict_unlabelled(
    clf: RandomForestClassifier,
    unlabelled_df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Predict transaction categories for unlabelled (unknown->unknown) rows.

    Adds two columns:
      - predicted_category:  the classifier's best guess
      - prediction_confidence: probability of the predicted class (0.0-1.0)

    Parameters
    ----------
    clf : RandomForestClassifier
        Fitted model from train_classifier().
    unlabelled_df : pd.DataFrame
        Rows where tx_category == wallet_to_wallet.
    feature_columns : list[str], optional
        Must match the columns used during training.

    Returns
    -------
    pd.DataFrame
        Copy of unlabelled_df with predicted_category and prediction_confidence.
    """
    if feature_columns is None:
        feature_columns = get_feature_columns()

    df = unlabelled_df.copy()

    if len(df) == 0:
        df["predicted_category"] = pd.Series(dtype=str)
        df["prediction_confidence"] = pd.Series(dtype=float)
        return df

    X = df[feature_columns]

    df["predicted_category"] = clf.predict(X)

    # predict_proba returns an array of shape (n_samples, n_classes).
    # np.max across axis=1 gives the highest probability for each row —
    # i.e. the model's confidence in its best guess.
    proba = clf.predict_proba(X)
    df["prediction_confidence"] = np.max(proba, axis=1)

    return df
