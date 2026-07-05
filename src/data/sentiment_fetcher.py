"""
Fetch market-derived sentiment data: Fear & Greed Index and Binance funding rates.

These are stronger sentiment signals than news headlines because they reflect
actual market behaviour (volatility, volume, futures positioning) rather than
journalist opinions published after the fact.

Fear & Greed Index:
  - Daily score 0 (extreme fear) to 100 (extreme greed)
  - Composite of: volatility, volume, social media, BTC dominance, Google Trends
  - Source: alternative.me (free, no API key)

Binance Funding Rate:
  - Every 8 hours, positive = longs pay shorts (bullish bias), negative = bearish
  - This is real-money sentiment -- traders putting capital behind their view
  - Source: Binance futures API (free, no API key)

No API keys required for either source.
"""

import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEAR_GREED_URL = "https://api.alternative.me/fng/"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

DEFAULT_FNG_OUTPUT = config.PROCESSED_DATA_DIR / "fear_greed_daily.csv"
DEFAULT_FUNDING_OUTPUT = config.PROCESSED_DATA_DIR / "eth_funding_rate.csv"


# ---------------------------------------------------------------------------
# Fear & Greed Index
# ---------------------------------------------------------------------------

def fetch_fear_greed(limit: int = 0) -> pd.DataFrame:
    """
    Fetch the full history of the Crypto Fear & Greed Index.

    Parameters
    ----------
    limit : int
        Number of days to fetch. 0 = all available history (since 2018).

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime, UTC), fng_value (int 0-100),
        fng_classification (str).
    """
    print("Fetching Crypto Fear & Greed Index...")
    response = requests.get(
        FEAR_GREED_URL,
        params={"limit": limit, "format": "json"},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()["data"]

    records = []
    for entry in data:
        records.append({
            # alternative.me returns UNIX timestamps (seconds)
            "date": pd.Timestamp(int(entry["timestamp"]), unit="s", tz="UTC"),
            "fng_value": int(entry["value"]),
            "fng_classification": entry["value_classification"],
        })

    df = pd.DataFrame(records)
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  {len(df):,} daily records")
    print(f"  Date range: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
    print(f"  Value range: {df['fng_value'].min()} to {df['fng_value'].max()}")

    return df


def save_fear_greed(df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Save Fear & Greed data to CSV."""
    path = output_path or DEFAULT_FNG_OUTPUT
    path = type(path) is str and __import__("pathlib").Path(path) or path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} Fear & Greed records to {path}")


# ---------------------------------------------------------------------------
# Binance Funding Rate
# ---------------------------------------------------------------------------

def fetch_funding_rate(
    symbol: str = "ETHUSDT",
    start_date: str = "2023-01-01",
    end_date: str = "2025-07-01",
) -> pd.DataFrame:
    """
    Fetch historical funding rates for ETH perpetual futures from Binance.

    Funding rates are settled every 8 hours (00:00, 08:00, 16:00 UTC).
    Positive rate = longs pay shorts (market is bullish).
    Negative rate = shorts pay longs (market is bearish).

    Parameters
    ----------
    symbol : str
        Futures symbol. Default: ETHUSDT.
    start_date : str
        ISO date string for the start of the range.
    end_date : str
        ISO date string for the end of the range.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp_utc, funding_rate, mark_price.
    """
    start_ms = _date_to_ms(start_date)
    end_ms = _date_to_ms(end_date)

    print(f"Fetching {symbol} funding rates: {start_date} to {end_date}")

    all_records = []
    current_ms = start_ms
    request_count = 0

    while current_ms < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_ms,
            "endTime": end_ms,
            "limit": 1000,  # Binance max per request
        }

        response = requests.get(BINANCE_FUNDING_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        for entry in data:
            all_records.append({
                "timestamp_utc": pd.Timestamp(
                    entry["fundingTime"], unit="ms", tz="UTC"
                ),
                "funding_rate": float(entry["fundingRate"]),
                # Some historical records have empty markPrice
                "mark_price": float(entry["markPrice"]) if entry.get("markPrice") else None,
            })

        # Move past the last record to avoid duplicates
        current_ms = data[-1]["fundingTime"] + 1
        request_count += 1

        if request_count % 5 == 0:
            print(f"  {len(all_records):,} records fetched...")

        time.sleep(0.5)  # be polite

    df = pd.DataFrame(all_records)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    print(f"  {len(df):,} funding rate records in {request_count} requests")
    print(f"  Date range: {df['timestamp_utc'].iloc[0]} to {df['timestamp_utc'].iloc[-1]}")
    print(f"  Rate range: {df['funding_rate'].min():.6f} to {df['funding_rate'].max():.6f}")

    return df


def save_funding_rate(df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Save funding rate data to CSV."""
    path = output_path or DEFAULT_FUNDING_OUTPUT
    path = type(path) is str and __import__("pathlib").Path(path) or path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} funding rate records to {path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _date_to_ms(date_str: str) -> int:
    """Convert an ISO date string to a UNIX timestamp in milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)