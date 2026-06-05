"""
Tests for src/features/feature_engineer.py.

All tests use in-memory DataFrames — no file I/O, no trained models.
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_labelled_df(
    from_categories: list[str],
    to_categories: list[str],
    **kwargs,
) -> pd.DataFrame:
    """
    Build a minimal DataFrame that assign_transaction_label() can consume.

    Extra keyword arguments are merged as columns — useful for adding
    fields needed by build_features() (usd_value, gas_used, etc.).
    """
    n = len(from_categories)
    assert len(to_categories) == n

    data = {
        "from_category": from_categories,
        "to_category": to_categories,
    }
    data.update(kwargs)
    return pd.DataFrame(data)


def make_full_df(n: int = 5) -> pd.DataFrame:
    """
    Build a DataFrame with all columns needed by both assign_transaction_label()
    and build_features(). Uses realistic but deterministic values.
    """
    return pd.DataFrame({
        "from_category": ["exchange", "unknown", "defi", "unknown", "unknown"],
        "to_category": ["unknown", "exchange", "unknown", "defi", "unknown"],
        "from_address": ["addr_a", "addr_b", "addr_c", "addr_d", "addr_a"],
        "to_address": ["addr_x", "addr_y", "addr_z", "addr_w", "addr_v"],
        "timestamp_utc": pd.to_datetime([
            "2023-06-01 08:00:00",
            "2023-06-01 14:30:00",
            "2023-06-02 03:15:00",
            "2023-06-02 22:00:00",
            "2023-06-03 08:00:00",
        ], utc=True),
        "usd_value": [2_000_000.0, 5_000_000.0, 1_500_000.0, 3_000_000.0, 1_200_000.0],
        "gas_used": [21_000, 150_000, 250_000, 180_000, 21_000],
        "gas_price_gwei": [30.0, 50.0, 120.0, 45.0, 35.0],
        "eth_usd_price": [1800.0, 1810.0, 1795.0, 1820.0, 1805.0],
        "is_contract_call": [False, True, True, True, False],
    })


# ---------------------------------------------------------------------------
# assign_transaction_label
# ---------------------------------------------------------------------------

class TestAssignTransactionLabel:
    def test_exchange_deposit(self):
        """to_category == 'exchange' -> exchange_deposit."""
        df = make_labelled_df(["unknown"], ["exchange"])
        result = assign_transaction_label(df)
        assert result["tx_category"].iloc[0] == EXCHANGE_DEPOSIT

    def test_exchange_withdrawal(self):
        """from_category == 'exchange' -> exchange_withdrawal."""
        df = make_labelled_df(["exchange"], ["unknown"])
        result = assign_transaction_label(df)
        assert result["tx_category"].iloc[0] == EXCHANGE_WITHDRAWAL

    def test_defi_interaction_to_side(self):
        """to_category == 'defi' -> defi_interaction."""
        df = make_labelled_df(["unknown"], ["defi"])
        result = assign_transaction_label(df)
        assert result["tx_category"].iloc[0] == DEFI_INTERACTION

    def test_defi_interaction_from_side(self):
        """from_category == 'defi' -> defi_interaction."""
        df = make_labelled_df(["defi"], ["unknown"])
        result = assign_transaction_label(df)
        assert result["tx_category"].iloc[0] == DEFI_INTERACTION

    def test_wallet_to_wallet(self):
        """Both sides unknown -> wallet_to_wallet."""
        df = make_labelled_df(["unknown"], ["unknown"])
        result = assign_transaction_label(df)
        assert result["tx_category"].iloc[0] == WALLET_TO_WALLET

    def test_exchange_to_exchange(self):
        """Exchange->exchange -> exchange_deposit (to-side priority)."""
        df = make_labelled_df(["exchange"], ["exchange"])
        result = assign_transaction_label(df)
        assert result["tx_category"].iloc[0] == EXCHANGE_DEPOSIT

    def test_exchange_to_defi(self):
        """exchange->defi: to_category is defi, but from is exchange.
        Neither is 'exchange' on the to-side, so rule 1 doesn't fire.
        Rule 2: from_category == exchange -> exchange_withdrawal."""
        df = make_labelled_df(["exchange"], ["defi"])
        result = assign_transaction_label(df)
        assert result["tx_category"].iloc[0] == EXCHANGE_WITHDRAWAL

    def test_defi_to_exchange(self):
        """defi->exchange: to_category == exchange -> exchange_deposit (rule 1 wins)."""
        df = make_labelled_df(["defi"], ["exchange"])
        result = assign_transaction_label(df)
        assert result["tx_category"].iloc[0] == EXCHANGE_DEPOSIT

    def test_defi_to_defi(self):
        """defi->defi -> defi_interaction."""
        df = make_labelled_df(["defi"], ["defi"])
        result = assign_transaction_label(df)
        assert result["tx_category"].iloc[0] == DEFI_INTERACTION

    def test_does_not_mutate_input(self):
        """The function should return a copy, not modify the input."""
        df = make_labelled_df(["unknown"], ["exchange"])
        assign_transaction_label(df)
        assert "tx_category" not in df.columns

    def test_missing_columns_raises(self):
        """Should raise ValueError if required columns are missing."""
        df = pd.DataFrame({"from_category": ["exchange"]})
        with pytest.raises(ValueError, match="missing required columns"):
            assign_transaction_label(df)

    def test_multiple_rows(self):
        """All four categories assigned correctly in a mixed DataFrame."""
        df = make_labelled_df(
            ["unknown", "exchange", "defi", "unknown"],
            ["exchange", "unknown", "unknown", "unknown"],
        )
        result = assign_transaction_label(df)
        expected = [EXCHANGE_DEPOSIT, EXCHANGE_WITHDRAWAL, DEFI_INTERACTION, WALLET_TO_WALLET]
        assert list(result["tx_category"]) == expected

    def test_empty_dataframe(self):
        """Should handle an empty DataFrame without errors."""
        df = make_labelled_df([], [])
        result = assign_transaction_label(df)
        assert len(result) == 0
        assert "tx_category" in result.columns


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_log_usd_value(self):
        """log10($2M) should be approximately 6.3."""
        df = make_full_df()
        result = build_features(df)
        # $2,000,000 -> log10(2_000_000) = 6.301...
        assert abs(result["log_usd_value"].iloc[0] - np.log10(2_000_000)) < 0.001

    def test_log_gas_used(self):
        """log10(21000) should be approximately 4.32."""
        df = make_full_df()
        result = build_features(df)
        assert abs(result["log_gas_used"].iloc[0] - np.log10(21_000)) < 0.001

    def test_hour_of_day(self):
        """First row timestamp is 08:00 UTC -> hour_of_day == 8."""
        df = make_full_df()
        result = build_features(df)
        assert result["hour_of_day"].iloc[0] == 8

    def test_day_of_week(self):
        """2023-06-01 is a Thursday -> day_of_week == 3."""
        df = make_full_df()
        result = build_features(df)
        assert result["day_of_week"].iloc[0] == 3

    def test_sender_prior_tx_count_first_appearance(self):
        """First transaction by a sender should have count 0."""
        df = make_full_df()
        result = build_features(df)
        # addr_a appears at row 0 (08:00) and row 4 (next day 08:00)
        # After sorting by time, addr_a's first appearance should be 0
        addr_a_rows = result[result["from_address"] == "addr_a"]
        first_count = addr_a_rows["sender_prior_tx_count"].iloc[0]
        assert first_count == 0

    def test_sender_prior_tx_count_second_appearance(self):
        """Second transaction by the same sender should have count 1."""
        df = make_full_df()
        result = build_features(df)
        addr_a_rows = result[result["from_address"] == "addr_a"]
        second_count = addr_a_rows["sender_prior_tx_count"].iloc[1]
        assert second_count == 1

    def test_sender_prior_tx_count_no_look_ahead(self):
        """
        The sender_prior_tx_count for a sender's first ever transaction
        must be 0, regardless of how many future transactions that sender
        makes. This is the core look-ahead test.
        """
        df = pd.DataFrame({
            "from_category": ["unknown"] * 4,
            "to_category": ["unknown"] * 4,
            "from_address": ["whale_x", "whale_x", "whale_x", "whale_x"],
            "to_address": ["dest_1", "dest_2", "dest_3", "dest_4"],
            "timestamp_utc": pd.to_datetime([
                "2023-01-01 00:00:00",
                "2023-03-01 00:00:00",
                "2023-06-01 00:00:00",
                "2023-12-01 00:00:00",
            ], utc=True),
            "usd_value": [1_000_000.0] * 4,
            "gas_used": [21_000] * 4,
            "gas_price_gwei": [30.0] * 4,
            "eth_usd_price": [1800.0] * 4,
            "is_contract_call": [False] * 4,
        })
        result = build_features(df)
        counts = list(result["sender_prior_tx_count"])
        # Must be [0, 1, 2, 3] — strictly cumulative, no future knowledge
        assert counts == [0, 1, 2, 3]

    def test_all_feature_columns_present(self):
        """Every column listed in get_feature_columns() must exist."""
        df = make_full_df()
        result = build_features(df)
        for col in get_feature_columns():
            assert col in result.columns, f"Missing feature column: {col}"

    def test_does_not_mutate_input(self):
        """The function should return a copy, not modify the input."""
        df = make_full_df()
        original_cols = set(df.columns)
        build_features(df)
        assert set(df.columns) == original_cols

    def test_missing_columns_raises(self):
        """Should raise ValueError if required columns are missing."""
        df = pd.DataFrame({"usd_value": [1_000_000.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            build_features(df)

    def test_passthrough_features_unchanged(self):
        """gas_price_gwei, eth_usd_price, is_contract_call should pass through."""
        df = make_full_df()
        result = build_features(df)
        # Values should be identical (just check first row)
        assert result["gas_price_gwei"].iloc[0] == 30.0
        assert result["eth_usd_price"].iloc[0] == 1800.0
        assert result["is_contract_call"].iloc[0] == False


class TestGetFeatureColumns:
    def test_returns_list_of_strings(self):
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert all(isinstance(c, str) for c in cols)

    def test_no_label_columns_in_features(self):
        """Feature list must not include the label or address columns."""
        cols = get_feature_columns()
        forbidden = {"tx_category", "from_category", "to_category",
                     "from_address", "to_address", "from_label", "to_label"}
        assert not forbidden.intersection(cols)
