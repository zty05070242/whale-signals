# Data files

Large downloaded and generated datasets are excluded from Git. The original
snapshot behind the README and dashboard is not available in this checkout, so
the reported analysis cannot currently be reproduced from source rows.

## Sources

| Dataset | Source | Expected local file |
|---|---|---|
| Whale transactions and WETH/USD | [Dune Analytics](https://dune.com/), using [the saved SQL](../docs/dune_queries/whale_transactions.sql) | `data/raw/dune_whale_transactions.csv` |
| ETH/USDT hourly candles | [Binance Spot API](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data) | `data/processed/eth_prices_hourly.csv` |
| ETH perpetual funding | [Binance Futures API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History) | `data/processed/eth_funding_rate.csv` |
| Crypto Fear & Greed | [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/) | `data/processed/fear_greed_daily.csv` |
| Bitcoin news | Kaggle; the original dataset page was not recorded | `data/raw/bitcoin_sentiments_21_24.csv` |
| Address labels | brianleect/etherscan-labels and dawsbot/eth-labels | `data/reference/address_labels.csv` |

The Dune export should contain one row per successful top-level transaction
with these columns:

`timestamp_utc`, `block_number`, `tx_hash`, `from_address`, `to_address`,
`eth_value`, `usd_value`, `eth_usd_price`, `gas_price_gwei`, `gas_used`,
`tx_fee_eth`, `is_contract_call`.

The processed whale file is expected at `data/processed/whale_txs.csv`. It adds
address labels, MEV flags and rule-based transaction categories. Collection and
phase scripts under `scripts/` document the existing pipeline, but a fresh API
pull is a new snapshot and may not match the reported 646,442 rows.

## What is committed

The small reference tables in `data/reference/` and the aggregate dashboard
file `app/dashboard_data.json` are committed. The compact CSVs in `results/`
are extracts of that dashboard JSON, not an independent reproduction.

Known gaps are the missing raw snapshot, unrecorded upstream revisions for the
merged labels, and unrecorded provenance and licence details for the Kaggle
news file. These should be fixed before claiming full reproducibility.
