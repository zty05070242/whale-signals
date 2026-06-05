"""
Tests for src/models/transaction_classifier.py.

Uses synthetic DataFrames with known distributions so tests are deterministic.
No real data or file I/O required.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineer import (
    assign_transaction_label,
    build_features,
    get_feature_columns,
    EXCHANGE_DEPOSIT,
    EXCHANGE_WITHDRAWAL,
    DEFI_INTERACTION,
    WALLET_TO_WALLET,
)
from src.models.transaction_classifier import (
    prepare_labelled_data,
    time_based_split,
    train_classifier,
    evaluate_classifier,
    predict_unlabelled,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_synthetic_dataset(n: int = 200, random_state: int = 42) -> pd.DataFrame:
    """
    Build a synthetic whale transaction DataFrame with clear feature
    separation between categories, so a Random Forest can learn them.

    The features are deliberately exaggerated:
      - exchange_deposit:    high usd_value, low gas, is_contract_call=False
      - exchange_withdrawal: medium usd_value, low gas, is_contract_call=False
      - defi_interaction:    any usd_value, high gas, is_contract_call=True
      - wallet_to_wallet:    low usd_value, low gas, is_contract_call=False
    """
    rng = np.random.RandomState(random_state)

    # Generate n rows split roughly evenly across 4 categories
    quarter = n // 4
    remainder = n - 4 * quarter

    categories_from = (
        ["unknown"] * quarter +          # exchange_deposit (to=exchange)
        ["exchange"] * quarter +         # exchange_withdrawal
        ["defi"] * quarter +             # defi_interaction
        ["unknown"] * (quarter + remainder)  # wallet_to_wallet
    )
    categories_to = (
        ["exchange"] * quarter +         # exchange_deposit
        ["unknown"] * quarter +          # exchange_withdrawal
        ["unknown"] * quarter +          # defi_interaction (from=defi)
        ["unknown"] * (quarter + remainder)  # wallet_to_wallet
    )

    # Generate timestamps spanning 2023
    base_ts = pd.Timestamp("2023-01-01", tz="UTC")
    timestamps = [base_ts + pd.Timedelta(hours=i * 4) for i in range(n)]

    # Features with clear separation per category
    usd_values = (
        rng.uniform(5_000_000, 20_000_000, quarter).tolist() +   # deposits: large
        rng.uniform(2_000_000, 8_000_000, quarter).tolist() +    # withdrawals: medium
        rng.uniform(1_000_000, 10_000_000, quarter).tolist() +   # defi: varied
        rng.uniform(1_000_000, 3_000_000, quarter + remainder).tolist()  # w2w: smaller
    )

    gas_used = (
        rng.uniform(21_000, 30_000, quarter).tolist() +          # deposits: simple
        rng.uniform(21_000, 30_000, quarter).tolist() +          # withdrawals: simple
        rng.uniform(150_000, 500_000, quarter).tolist() +        # defi: complex
        rng.uniform(21_000, 40_000, quarter + remainder).tolist()  # w2w: simple
    )

    is_contract = (
        [False] * quarter +             # deposits
        [False] * quarter +             # withdrawals
        [True] * quarter +              # defi
        [False] * (quarter + remainder)  # w2w
    )

    # Use unique-ish addresses so sender_prior_tx_count varies
    from_addresses = [f"addr_{i % 40}" for i in range(n)]

    df = pd.DataFrame({
        "from_category": categories_from,
        "to_category": categories_to,
        "from_address": from_addresses,
        "to_address": [f"dest_{i}" for i in range(n)],
        "timestamp_utc": timestamps,
        "usd_value": usd_values,
        "gas_used": gas_used,
        "gas_price_gwei": rng.uniform(10, 100, n).tolist(),
        "eth_usd_price": rng.uniform(1500, 2000, n).tolist(),
        "is_contract_call": is_contract,
    })

    # Run through the full pipeline
    df = assign_transaction_label(df)
    df = build_features(df)

    return df


# ---------------------------------------------------------------------------
# prepare_labelled_data
# ---------------------------------------------------------------------------

class TestPrepareLabelledData:
    def test_splits_correctly(self):
        """Labelled rows should have known categories, unlabelled should be w2w."""
        df = make_synthetic_dataset()
        labelled, unlabelled = prepare_labelled_data(df)

        assert (labelled["tx_category"] != WALLET_TO_WALLET).all()
        assert (unlabelled["tx_category"] == WALLET_TO_WALLET).all()

    def test_row_count_preserved(self):
        """Total rows should equal labelled + unlabelled."""
        df = make_synthetic_dataset()
        labelled, unlabelled = prepare_labelled_data(df)
        assert len(labelled) + len(unlabelled) == len(df)

    def test_no_shared_indices(self):
        """Labelled and unlabelled should not share any rows."""
        df = make_synthetic_dataset()
        labelled, unlabelled = prepare_labelled_data(df)
        shared = set(labelled.index) & set(unlabelled.index)
        assert len(shared) == 0


# ---------------------------------------------------------------------------
# time_based_split
# ---------------------------------------------------------------------------

class TestTimeBasedSplit:
    def test_split_sizes(self):
        """80/20 split on 150 labelled rows."""
        df = make_synthetic_dataset()
        labelled, _ = prepare_labelled_data(df)
        train, test = time_based_split(labelled, train_fraction=0.8)

        expected_train = int(len(labelled) * 0.8)
        assert len(train) == expected_train
        assert len(test) == len(labelled) - expected_train

    def test_no_temporal_leakage(self):
        """All training timestamps must be before all test timestamps."""
        df = make_synthetic_dataset()
        labelled, _ = prepare_labelled_data(df)
        train, test = time_based_split(labelled)

        train_max = train["timestamp_utc"].max()
        test_min = test["timestamp_utc"].min()
        assert train_max <= test_min

    def test_custom_fraction(self):
        """A 50/50 split should produce roughly equal halves."""
        df = make_synthetic_dataset(n=100)
        labelled, _ = prepare_labelled_data(df)
        train, test = time_based_split(labelled, train_fraction=0.5)
        assert len(train) == int(len(labelled) * 0.5)


# ---------------------------------------------------------------------------
# train_classifier
# ---------------------------------------------------------------------------

class TestTrainClassifier:
    def test_returns_fitted_model(self):
        """train_classifier should return a fitted RandomForestClassifier."""
        df = make_synthetic_dataset()
        labelled, _ = prepare_labelled_data(df)
        train, _ = time_based_split(labelled)
        clf = train_classifier(train)

        # A fitted sklearn estimator has the classes_ attribute
        assert hasattr(clf, "classes_")
        assert len(clf.classes_) > 0

    def test_learned_classes_match_labels(self):
        """The model should know all three labelled categories."""
        df = make_synthetic_dataset()
        labelled, _ = prepare_labelled_data(df)
        train, _ = time_based_split(labelled)
        clf = train_classifier(train)

        expected = {EXCHANGE_DEPOSIT, EXCHANGE_WITHDRAWAL, DEFI_INTERACTION}
        assert set(clf.classes_) == expected

    def test_reproducible_with_random_state(self):
        """Same random_state should produce identical predictions."""
        df = make_synthetic_dataset()
        labelled, _ = prepare_labelled_data(df)
        train, test = time_based_split(labelled)

        clf1 = train_classifier(train, random_state=42)
        clf2 = train_classifier(train, random_state=42)

        X_test = test[get_feature_columns()]
        pred1 = clf1.predict(X_test)
        pred2 = clf2.predict(X_test)

        # np.array_equal checks element-wise equality of two arrays
        assert np.array_equal(pred1, pred2)


# ---------------------------------------------------------------------------
# evaluate_classifier
# ---------------------------------------------------------------------------

class TestEvaluateClassifier:
    def test_returns_expected_keys(self):
        """Evaluation result dict should have all expected keys."""
        df = make_synthetic_dataset()
        labelled, _ = prepare_labelled_data(df)
        train, test = time_based_split(labelled)
        clf = train_classifier(train)
        results = evaluate_classifier(clf, test)

        expected_keys = {
            "accuracy", "classification_report", "confusion_matrix",
            "feature_importances", "predictions", "probabilities",
        }
        assert set(results.keys()) == expected_keys

    def test_accuracy_above_chance(self):
        """With exaggerated feature separation, accuracy should beat random (33%)."""
        df = make_synthetic_dataset(n=300)
        labelled, _ = prepare_labelled_data(df)
        train, test = time_based_split(labelled)
        clf = train_classifier(train)
        results = evaluate_classifier(clf, test)

        # 3-class random baseline is ~33%. With clear feature separation
        # the model should do much better.
        assert results["accuracy"] > 0.5

    def test_feature_importances_sum_to_one(self):
        """Gini importances across all features should sum to 1.0."""
        df = make_synthetic_dataset()
        labelled, _ = prepare_labelled_data(df)
        train, test = time_based_split(labelled)
        clf = train_classifier(train)
        results = evaluate_classifier(clf, test)

        total = sum(results["feature_importances"].values())
        assert abs(total - 1.0) < 0.001

    def test_predictions_shape(self):
        """Predictions array should have one entry per test row."""
        df = make_synthetic_dataset()
        labelled, _ = prepare_labelled_data(df)
        train, test = time_based_split(labelled)
        clf = train_classifier(train)
        results = evaluate_classifier(clf, test)

        assert len(results["predictions"]) == len(test)

    def test_probabilities_shape(self):
        """Probabilities should be (n_test, n_classes)."""
        df = make_synthetic_dataset()
        labelled, _ = prepare_labelled_data(df)
        train, test = time_based_split(labelled)
        clf = train_classifier(train)
        results = evaluate_classifier(clf, test)

        n_classes = len(clf.classes_)
        assert results["probabilities"].shape == (len(test), n_classes)


# ---------------------------------------------------------------------------
# predict_unlabelled
# ---------------------------------------------------------------------------

class TestPredictUnlabelled:
    def test_adds_predicted_columns(self):
        """Should add predicted_category and prediction_confidence."""
        df = make_synthetic_dataset()
        labelled, unlabelled = prepare_labelled_data(df)
        train, _ = time_based_split(labelled)
        clf = train_classifier(train)

        result = predict_unlabelled(clf, unlabelled)
        assert "predicted_category" in result.columns
        assert "prediction_confidence" in result.columns

    def test_predictions_are_valid_categories(self):
        """All predicted categories should be in the known set."""
        df = make_synthetic_dataset()
        labelled, unlabelled = prepare_labelled_data(df)
        train, _ = time_based_split(labelled)
        clf = train_classifier(train)

        result = predict_unlabelled(clf, unlabelled)
        valid = {EXCHANGE_DEPOSIT, EXCHANGE_WITHDRAWAL, DEFI_INTERACTION}
        # The classifier only knows 3 classes (not wallet_to_wallet)
        assert set(result["predicted_category"].unique()).issubset(valid)

    def test_confidence_in_valid_range(self):
        """Confidence scores should be between 0 and 1."""
        df = make_synthetic_dataset()
        labelled, unlabelled = prepare_labelled_data(df)
        train, _ = time_based_split(labelled)
        clf = train_classifier(train)

        result = predict_unlabelled(clf, unlabelled)
        assert (result["prediction_confidence"] >= 0.0).all()
        assert (result["prediction_confidence"] <= 1.0).all()

    def test_does_not_mutate_input(self):
        """Should return a copy, not modify the input."""
        df = make_synthetic_dataset()
        labelled, unlabelled = prepare_labelled_data(df)
        train, _ = time_based_split(labelled)
        clf = train_classifier(train)

        original_cols = set(unlabelled.columns)
        predict_unlabelled(clf, unlabelled)
        assert set(unlabelled.columns) == original_cols

    def test_empty_unlabelled(self):
        """Should handle an empty DataFrame without errors."""
        df = make_synthetic_dataset()
        labelled, _ = prepare_labelled_data(df)
        train, _ = time_based_split(labelled)
        clf = train_classifier(train)

        empty = labelled.head(0).copy()
        empty["tx_category"] = pd.Series(dtype=str)
        result = predict_unlabelled(clf, empty)
        assert len(result) == 0
        assert "predicted_category" in result.columns
        assert "prediction_confidence" in result.columns

    def test_row_count_preserved(self):
        """Output should have the same number of rows as input."""
        df = make_synthetic_dataset()
        labelled, unlabelled = prepare_labelled_data(df)
        train, _ = time_based_split(labelled)
        clf = train_classifier(train)

        result = predict_unlabelled(clf, unlabelled)
        assert len(result) == len(unlabelled)
