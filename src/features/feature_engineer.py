"""
Feature engineering for the whale transaction classifier.

Two responsibilities:
  1. assign_transaction_label() — rule-based labelling from address categories
  2. build_features()           — compute ML features for the classifier

The rule-based labels act as "ground truth" for transactions where at least one
address is known (exchange or DeFi). The classifier then learns to predict
labels for unknown->unknown transactions using the features built here.

IMPORTANT: sender_prior_tx_count is computed using only rows with timestamp_utc
strictly before the current row. This prevents look-ahead bias from leaking
into Phase 4's walk-forward validation.
"""

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Transaction category constants
# ---------------------------------------------------------------------------

EXCHANGE_DEPOSIT = "exchange_deposit"
EXCHANGE_WITHDRAWAL = "exchange_withdrawal"
DEFI_INTERACTION = "defi_interaction"
WALLET_TO_WALLET = "wallet_to_wallet"

# All valid categories, for validation
VALID_CATEGORIES = {EXCHANGE_DEPOSIT, EXCHANGE_WITHDRAWAL, DEFI_INTERACTION, WALLET_TO_WALLET}


# ---------------------------------------------------------------------------
# Part 1: Rule-based labelling
# ---------------------------------------------------------------------------

def assign_transaction_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a tx_category label to each row based on from_category and to_category.

    Priority order (first match wins):
      1. to_category == 'exchange'   -> exchange_deposit
      2. from_category == 'exchange' -> exchange_withdrawal
      3. either side == 'defi'       -> defi_interaction
      4. both sides == 'unknown'     -> wallet_to_wallet

    Edge case: exchange->exchange (e.g. Binance to Coinbase) is treated as
    exchange_deposit because the destination is more informative for price
    impact — funds arriving at an exchange could be sold. This is a documented
    design decision, not an accident.

    Parameters
    ----------
    df : pd.DataFrame
        Output of flag_mev_candidates(). Must have from_category and
        to_category columns with values in {'exchange', 'defi', 'unknown'}.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new 'tx_category' column.
    """
    _validate_required_columns(df, ["from_category", "to_category"])

    df = df.copy()

    # np.select applies conditions in order and returns the first matching value.
    # This is the vectorised equivalent of an if-elif chain — much faster than
    # iterating row by row with .apply().
    conditions = [
        df["to_category"] == "exchange",                           # rule 1: deposit
        df["from_category"] == "exchange",                         # rule 2: withdrawal
        (df["from_category"] == "defi") | (df["to_category"] == "defi"),  # rule 3: DeFi
    ]
    choices = [EXCHANGE_DEPOSIT, EXCHANGE_WITHDRAWAL, DEFI_INTERACTION]

    # default= is returned when no condition matches (both sides unknown)
    df["tx_category"] = np.select(conditions, choices, default=WALLET_TO_WALLET)

    return df


# ---------------------------------------------------------------------------
# Part 2: Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features used by the transaction classifier.

    Features added:
      - log_usd_value:          log10 of USD transaction value (reduces skew)
      - log_gas_used:           log10 of gas used (separates simple transfers
                                from complex contract calls)
      - hour_of_day:            0-23, extracted from timestamp_utc
      - day_of_week:            0=Monday to 6=Sunday
      - sender_prior_tx_count:  how many whale transactions this sender has
                                made BEFORE this row's timestamp (no look-ahead)

    Features passed through unchanged:
      - gas_price_gwei, eth_usd_price, is_contract_call

    Parameters
    ----------
    df : pd.DataFrame
        Output of assign_transaction_label(). Must have timestamp_utc,
        usd_value, gas_used, gas_price_gwei, eth_usd_price, is_contract_call,
        and from_address columns.

    Returns
    -------
    pd.DataFrame
        Copy of df with additional feature columns.
    """
    required = [
        "timestamp_utc", "usd_value", "gas_used", "gas_price_gwei",
        "eth_usd_price", "is_contract_call", "from_address",
    ]
    _validate_required_columns(df, required)

    df = df.copy()

    # --- Ensure timestamp is datetime for time-based features ---
    # pd.to_datetime converts string timestamps to datetime objects
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # --- Log-transformed value features ---
    # np.log10 compresses the scale: $1M -> 6.0, $10M -> 7.0, $100M -> 8.0
    # np.clip prevents log(0) errors — our data has min $1M so this is defensive
    df["log_usd_value"] = np.log10(np.clip(df["usd_value"], a_min=1.0, a_max=None))

    # Gas used: 21,000 for a plain ETH transfer, 150,000+ for complex DeFi calls
    # Log-transform because the distribution is heavily right-skewed
    df["log_gas_used"] = np.log10(np.clip(df["gas_used"], a_min=1.0, a_max=None))

    # --- Time features ---
    # .dt accessor exposes datetime properties on a pandas Series
    df["hour_of_day"] = df["timestamp_utc"].dt.hour
    # Monday=0, Sunday=6 — matches Python's datetime.weekday() convention
    df["day_of_week"] = df["timestamp_utc"].dt.weekday

    # --- Sender history (look-ahead-safe) ---
    df = _compute_sender_prior_tx_count(df)

    return df


def get_feature_columns() -> list[str]:
    """Return the list of column names used as classifier inputs."""
    return [
        "log_usd_value",
        "log_gas_used",
        "gas_price_gwei",
        "eth_usd_price",
        "is_contract_call",
        "hour_of_day",
        "day_of_week",
        "sender_prior_tx_count",
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_sender_prior_tx_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, count how many whale transactions the same sender
    (from_address) has made with a strictly earlier timestamp.

    This is the look-ahead-safe version. A naive groupby().transform('count')
    would use the full dataset, leaking future information into early rows.

    Method: sort by time, then use groupby().cumcount() which assigns 0 to the
    first occurrence, 1 to the second, etc. — exactly the count of prior rows
    by the same sender.

    Parameters
    ----------
    df : pd.DataFrame
        Must have timestamp_utc (datetime) and from_address columns.

    Returns
    -------
    pd.DataFrame
        df with sender_prior_tx_count column added.
    """
    # Sort by timestamp so cumcount reflects temporal order
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # groupby('from_address').cumcount() assigns a running count within each group:
    #   first tx by address X -> 0 (no prior transactions)
    #   second tx by address X -> 1 (one prior transaction)
    #   third tx by address X -> 2 (two prior transactions)
    # This is inherently look-ahead-safe because it counts only preceding rows.
    df["sender_prior_tx_count"] = df.groupby("from_address").cumcount()

    return df


def _validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required columns are missing."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
