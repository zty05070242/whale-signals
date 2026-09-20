# Are Ethereum Whales Smart Money? An Event Study of On-Chain Signals and Sentiment

**Languages:** English | [繁體中文](README.zh-Hant.md) | [简体中文](README.zh-Hans.md)

I tested whether large Ethereum exchange deposits predict subsequent price
declines. The study covers 646,442 successful, top-level transactions above $1
million from January 2023 to July 2026. Using known address labels, I categorise
transactions as exchange deposits, exchange withdrawals, DeFi interactions or
other transfers, then compare their subsequent ETH price direction with a
matched market base rate.

Exchange deposits show a modest but repeatable downside association. At 24
hours, their hit rate is 50.45%, compared with a 49.12% matched downward base
rate: an edge of 1.33 percentage points. The edge remains between 1.23 and 1.42
points across thresholds from $1M to $10M and increases through the later yearly
samples, reaching 3.86 points in the partial 2026 sample.

The result is asymmetric. The earlier positive withdrawal association
disappears in the later sample, while extreme-greed conditions strengthen the
deposit edge. Longer horizons produce larger measured differences, although
overlapping return windows make those estimates less suitable for inference.

## What I found

- **Exchange deposits show a 24-hour downside edge.** Their 50.45% hit rate is
  1.33 percentage points above the matched downward base rate.
- **The deposit edge is stable across size thresholds.** It remains between
  +1.23 and +1.42 points from $1M to $10M, so the result is not concentrated at
  one selected cutoff.
- **The deposit edge is stronger in the later sample.** It rises from -0.17
  points in 2023 to +3.86 points in the partial 2026 sample.
- **Market conditions sharpen the result.** During extreme greed, deposits have
  a +2.65-point edge over the corresponding market base rate.
- **Withdrawals do not show the same persistence.** Their earlier positive
  association under negative funding disappears in 2025 and reverses in the
  partial 2026 sample.

This is an event study of association, not a claim that every deposit represents
a sale or an execution-ready trading strategy.

**Live dashboard:**
[crypto-whale-signals-and-sentiment.streamlit.app](https://crypto-whale-signals-and-sentiment-lkhygb3594bbrogn23qbps.streamlit.app/)

## Research question and scope

> Are Ethereum transactions above $1 million associated with the direction of
> subsequent ETH returns, and does market sentiment change that association?

I examine four minimum transaction sizes ($1M, $2M, $5M and $10M), eight
published return horizons (1 hour, 6 hours, 24 hours, 3 days, 1 week, 1 month,
3 months and 6 months), calendar-year subsamples, seven sentiment conditions,
and a rule-based bull/bear split. A two-week horizon is also used in the
drawdown calculation.

The event study uses categories determined directly from known address labels.
Unknown-to-unknown wallet transfers are excluded from the deposit and withdrawal
tests, so model-predicted labels do not enter the reported financial results.

## Data

| Source | Use | Published records | Published period |
|---|---|---:|---|
| [Dune Analytics](https://dune.com/) | Successful top-level Ethereum transactions and contemporaneous WETH/USD | 646,442 transactions | Jan 2023–Jul 2026 |
| [Binance Spot API](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data) | Hourly ETH/USDT candles | 30,801 hours | Jan 2023–Jul 2026 |
| [Binance Futures API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History) | ETH perpetual funding rate | 3,851 observations | Jan 2023–Jul 2026 |
| [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/) | Daily Crypto Fear & Greed Index | 3,075 observations | Feb 2018–Jul 2026 |
| Kaggle | Bitcoin news headlines used in a separate VADER exercise | 5,906 articles | Jan 2023–Sep 2024 |
| [brianleect/etherscan-labels](https://github.com/brianleect/etherscan-labels) and [dawsbot/eth-labels](https://github.com/dawsbot/eth-labels) | Exchange, DeFi and other address labels | 52,768 addresses | Local merged snapshot |

The Dune query is in
[`docs/dune_queries/whale_transactions.sql`](docs/dune_queries/whale_transactions.sql).
Ethereum's native asset is not present in Dune's token price table, so the query
uses WETH as the contemporaneous ETH/USD price proxy. It selects successful
top-level transactions with `usd_value > 1,000,000`; internal calls from
`ethereum.traces` are outside scope. For query efficiency, it first applies the
configured 200 ETH pre-filter and then the contemporaneous
`usd_value > 1,000,000` test. This could omit otherwise qualifying transactions
if ETH trades above $5,000, so the pre-filter should be lowered if that occurs.

The large source and processed files are not committed. Their expected names,
schemas and sources are documented in [`data/README.md`](data/README.md). The
analysed input snapshot is not available in this checkout, so the 646,442 source
rows and headline results cannot be independently regenerated from the
repository alone.

## Operational definitions

### Whale transaction

A whale transaction is a successful top-level Ethereum transaction whose
contemporaneous value is strictly greater than the selected USD threshold. The
base case is $1 million. “Whale” is therefore a value filter, not proof that an
address belongs to a wealthy individual; exchange operations, custodians,
contracts and automated actors can all pass it.

### Transaction categories

Rules are applied in the following order, with the first match taking
precedence:

1. Destination labelled `exchange` → **exchange deposit**.
2. Sender labelled `exchange` → **exchange withdrawal**.
3. Either side labelled `defi` → **DeFi interaction**.
4. No earlier rule matched → **wallet-to-wallet**.

An exchange-to-exchange transfer is therefore counted as a deposit because the
destination rule comes first. `wallet_to_wallet` includes unknown-to-unknown
transfers and other unmatched categories; it should not be read as verified
personal-wallet activity. The published deposit and withdrawal analyses use
only the first two rule-derived groups.

MEV candidates are flagged using a small known-address list and transaction
heuristics. They are retained so exclusion can be treated as a sensitivity
choice rather than silently changing the dataset. The list is not exhaustive.

### Signals, hit rates and matched base rates

For an exchange deposit, a “hit” means the ETH price is lower at the selected
future horizon. For an exchange withdrawal, a hit means it is higher. A deposit
is only a proxy for possible sale availability, and a withdrawal is only a
proxy for possible accumulation; neither observes an executed trade or the
owner's motive.

Raw hit rate alone is misleading because ETH can spend long periods trending in
one direction. I therefore report:

```text
edge (percentage points) = whale-event hit rate − matched hourly base rate
```

The base rate uses all hourly price observations at the same horizon. When an
event result is conditioned by year or sentiment, the base rate is matched to
that same year or condition and tested in the corresponding direction. For
example, a 54% deposit hit rate during extreme greed is compared with the
frequency of price declines after all extreme-greed hours, not with 50%.

This adjustment controls for the direction already common in that market
state. It does not make neighbouring whale events independent.

### Sentiment and market regimes

The Crypto Fear & Greed Index is divided into extreme fear (≤25), fear (25–45),
neutral (45–55), greed (55–75) and extreme greed (>75). Funding is split at
zero. These variables condition the event study; they are not presented as
standalone forecasts.

The index is daily. The project joins it by calendar date, so the value may not
have been known at the start of every UTC day. Sentiment-conditioned results
should therefore be treated cautiously until publication timing is modelled.

Bull and bear periods use a 20% peak/trough rule. A bear regime begins after a
20% decline from the latest peak, and a bull regime begins after a 20% rise from
the latest trough. Applied to this sample, the rule produces 27 alternating
segments rather than assigning one label to an entire calendar year.

News headlines were scored with VADER in a separate exercise. News sentiment
was weak as a price feature and is not part of the canonical result tables.
VADER is lexicon-based, not a model trained in this repository. A third-party
RoBERTa score supplied with the Kaggle data was used only as a sanity check.

## Method

The processing sequence is:

1. Retrieve or load the Dune export and filter it to an explicit half-open date
   interval and USD threshold.
2. Normalise addresses, attach the checked-in label table, flag possible MEV,
   and assign the rules above.
3. Retrieve hourly prices, funding and daily Fear & Greed values for the same
   interval.
4. Join each event to the latest information available at that time. Market
   joins are backward-looking to prevent future observations entering a row.
5. Calculate forward ETH returns at each horizon.
6. Calculate deposit and withdrawal hit rates and subtract the corresponding
   matched hourly base rates.
7. Repeat by year, threshold, sentiment condition and bull/bear regime.

The committed dashboard aggregate is the output used by the live dashboard.
Compact CSV extracts in [`results/`](results/) copy selected values from that
JSON so they are easier to inspect. They are supporting artefacts, not an
independent reproduction of the analysis.

The calendar-year tables are descriptive subsamples, not four independent
experiments in a strict statistical sense. I developed the main interpretation
while working with 2023–2025 and fetched 2026 later without changing the
parameters. That makes 2026 a useful later-sample check. It is only a partial
year, shares the same address-labelling and analysis choices, and does not
provide absolute confirmation.

## Results: an asymmetric exchange-flow signal

All values below come from the committed files described in
[`results/README.md`](results/README.md). Percentage-point differences are
shown as `pp`.

### Exchange deposits show a 24-hour downside edge

At the $1M threshold, labelled exchange deposits precede a lower ETH price 24
hours later in 50.45% of cases. The matched downward base rate across all
eligible hours is 49.12%, leaving a +1.33 percentage-point difference.

Within this sample, exchange deposits therefore precede short-term downside
slightly more often than the same price direction occurs in the wider market.

### The deposit edge is stable from $1M to $10M

Unconditional deposits have similar 24-hour differences at the four main size
cuts:

| Minimum size | Deposit hit rate | Edge |
|---|---:|---:|
| $1M | 50.45% | +1.33 pp |
| $2M | 50.54% | +1.42 pp |
| $5M | 50.35% | +1.23 pp |
| $10M | 50.48% | +1.35 pp |

The narrow 1.23–1.42 point range shows that the 24-hour result is not
concentrated at one selected USD cutoff. The complete conditioned threshold
table is in
[`results/published_threshold_sensitivity.csv`](results/published_threshold_sensitivity.csv).

The original Dune pull uses a strict `usd_value > $1M` filter, while the
threshold sensitivity code uses `usd_value >= threshold`; transactions exactly
on a boundary may therefore be treated differently.

### The deposit edge is stronger in the later yearly samples

The 24-hour year split compares unconditional deposits with withdrawals made
during negative funding, the withdrawal condition that was strongest early in
the sample.

| Year | Deposit edge | Negative-funding withdrawal edge |
|---|---:|---:|
| 2023 | −0.17 pp | +4.71 pp |
| 2024 | +0.88 pp | +10.12 pp |
| 2025 | +1.79 pp | +0.07 pp |
| 2026 (to July) | +3.86 pp | −3.92 pp |

Source: [`results/published_yearly_edges.csv`](results/published_yearly_edges.csv).

The deposit difference is positive from 2024 onward and reaches +3.86 points
in the partial 2026 sample, which was collected later and analysed without
changing the parameters. The negative-funding withdrawal association follows a
different path: it is positive in 2023–2024, approximately zero in 2025 and
negative in partial 2026.

Staking, liquidity provision, bridging and custody reorganisation could help
explain why withdrawals changed, but this remains a hypothesis because the
project does not trace post-withdrawal destinations.

### Extreme greed sharpens the deposit edge

At 24 hours and $1M+, deposits during extreme greed have a 54.43% hit rate. The
matched downward base rate during extreme greed is 51.78%, leaving a +2.65
point difference.

The matched comparison separates the deposit association from the downside
already common during extreme greed. Sentiment is used here as a market
condition rather than as a standalone forecast.

### The deposit difference is larger in rule-defined bear periods

At a one-week horizon, the deposit difference is larger in the rule-defined
bear periods at all four thresholds. The withdrawal difference is negative in
both regimes and more negative in bear periods.

| Minimum size | Deposit: bull | Deposit: bear | Withdrawal: bull | Withdrawal: bear |
|---|---:|---:|---:|---:|
| $1M | +0.46 pp | +3.92 pp | −1.21 pp | −3.95 pp |
| $2M | +0.66 pp | +4.83 pp | −1.78 pp | −4.30 pp |
| $5M | +1.53 pp | +4.92 pp | −4.19 pp | −6.14 pp |
| $10M | +2.17 pp | +4.61 pp | −3.14 pp | −7.69 pp |

Source: [`results/published_bull_bear.csv`](results/published_bull_bear.csv).

The rule-defined regimes show where the deposit association is most pronounced.
Because these regimes are constructed from the evaluated price path, this table
is a conditional description rather than a trading rule.

### Longer horizons show a larger descriptive difference

The deposit edge is similar at 24 hours, three days and one week, then becomes
larger across the longer measured horizons.

| Horizon | Deposit edge vs matched base rate |
|---|---:|
| 1 hour | +0.11 pp |
| 6 hours | −0.05 pp |
| 24 hours | +1.33 pp |
| 3 days | +1.36 pp |
| 1 week | +1.61 pp |
| 1 month | +4.79 pp |
| 3 months | +8.05 pp |
| 6 months | +12.44 pp |

Source: [`results/published_horizon_edges.csv`](results/published_horizon_edges.csv).

Long return windows repeatedly count the same broad market moves, so this table
describes how the measured difference changes with horizon; it is not evidence
that depositors forecast six months ahead. Dependence-robust validation is
needed before treating the long-horizon values as inferential results.

### Long-horizon deposit outcomes include substantial path risk

For deposit events that end with a negative return, maximum adverse excursion
(MAE) records the largest interim price rise before the horizon ends. It
therefore describes the path against the deposit-implied direction.

| Horizon | Correct deposit events | Mean MAE | Median MAE | 90th-percentile MAE | Mean final return |
|---|---:|---:|---:|---:|---:|
| 1 week | 91,764 | 2.69% | 2.04% | 6.18% | −6.42% |
| 2 weeks | 95,088 | 4.30% | 3.52% | 9.72% | −8.82% |
| 1 month | 99,620 | 6.02% | 5.16% | 12.90% | −12.84% |
| 3 months | 93,568 | 9.64% | 7.60% | 20.99% | −26.11% |
| 6 months | 92,904 | 20.11% | 13.25% | 54.38% | −30.38% |

Source: [`results/published_drawdown.csv`](results/published_drawdown.csv).

The result shows that endpoint accuracy can conceal substantial interim movement
in the opposite direction. It describes path risk among events with a negative
final return rather than simulating an executable strategy.

### Supporting classifier prototype

A separate Random Forest prototype tests whether transaction features can
recover the address-rule categories for otherwise unlabelled wallets. It
achieves 67.7% accuracy on a time-based hold-out whose categories were first
assigned by the address rules. The predicted categories remain exploratory and
do not feed into the event-study results above.

## Bottom line

The central result is a measured downside association after large exchange
deposits. At 24 hours, the +1.33-point edge holds across thresholds from $1M to
$10M, strengthens in the later yearly samples and reaches +2.65 points during
extreme greed. Withdrawals provide a contrasting result: their early positive
association under negative funding does not persist through the later sample.

The next research step is dependence-robust validation of the event study,
followed by a live test with fixed labels and execution rules.

## Research boundaries and next tests

1. **Dependence across time.** Transactions cluster during active markets, and
   long-horizon events can share most of the same return path. A previously
   reported 78.3% result for $10M+ deposits during extreme greed was concentrated
   in ten calendar days and is not treated as a finding. Calendar-time aggregation
   or another event-study correction is the next test for the longer horizons.

2. **Address coverage and transaction interpretation.** The present-day label
   snapshot identifies at least one endpoint for 62.8% of transactions. Deposits
   and withdrawals remain proxies for exchange flows rather than observed trades,
   while top-level calls, historical label changes and unidentified MEV activity
   define the current coverage boundary.

3. **Threshold and regime definitions.** A fixed USD threshold represents a
   changing amount of ETH over time. Fear & Greed is daily, funding represents
   one derivatives market, and the 20% bull/bear rule is path-dependent. Testing
   alternative definitions would show how sensitive the conditional results are
   to those choices.

4. **Reproduction and application.** The dashboard aggregates and compact result
   tables are committed, but the original transaction and market-data snapshot
   is not. Reproducing the exact published analysis requires that snapshot. A
   live strategy test would additionally need execution delay, fees, slippage,
   position sizing and risk-management rules.

## Related literature

The design follows two nearby strands rather than claiming a direct replication.
Hoang and Baur study Bitcoin private wallets, exchange reserves and returns at
an aggregate level; Ante and Fiedler use an event-study design for large Bitcoin
transfers. This project applies a transfer-level design to labelled Ethereum
exchange flows.

- Hoang, L. T., & Baur, D. G. (2022), [“Loaded for bear: Bitcoin private
  wallets, exchange reserves and prices”](https://doi.org/10.1016/j.jbankfin.2022.106622),
  *Journal of Banking & Finance*, 144.
- Ante, L., & Fiedler, I. (2021), [“Market reaction to large transfers on the
  Bitcoin blockchain—Do size and motive matter?”](https://doi.org/10.1016/j.frl.2020.101619),
  *Finance Research Letters*, 39.
- Chi, Y., Chu, Q., & Hao, W. (2024), [“Return and Volatility Forecasting Using
  On-Chain Flows in Cryptocurrency Markets”](https://arxiv.org/abs/2411.06327),
  arXiv:2411.06327.

The main statistical concern is clustered event dates and partially overlapping
windows:

- Kolari, J. W., & Pynnönen, S. (2010), [“Event Study Testing with
  Cross-Sectional Correlation of Abnormal Returns”](https://doi.org/10.1093/rfs/hhq072),
  *Review of Financial Studies*, 23(11), 3996–4025.
- Kolari, J. W., Pape, B., & Pynnönen, S. (2018), [“Event Study Testing with
  Cross-Sectional Correlation Due to Partially Overlapping Event
  Windows”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167271),
  working paper.

Other references used to frame the prototype classifier, market regimes and
drawdown measurement are:

- Harlev, M. A. et al. (2018), [“Breaking Bad: De-Anonymising Entity Types on
  the Bitcoin Blockchain Using Supervised Machine Learning”](https://doi.org/10.24251/HICSS.2018.443),
  *HICSS 51*.
- Guidolin, M., & Timmermann, A. (2008), [“Size and Value Anomalies under Regime
  Shifts”](https://doi.org/10.1093/jjfinec/nbm021), *Journal of Financial
  Econometrics*, 6(1), 1–48.
- Sweeney, J. L. (1997), *Maximum Adverse Excursion: Analyzing Price
  Fluctuations for Trading Management*, Wiley.

These papers motivate choices and cautions. They do not validate the specific
Ethereum results reported here.

## Running the project

Python 3.12 is declared in [`.python-version`](.python-version). Create an
environment and install the project dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -c constraints.txt
```

`constraints.txt` records known-working versions of direct dependencies, not a
complete transitive lock. Install the test dependencies and run the suite with:

```bash
.venv/bin/python -m pip install -r requirements-test.txt -c constraints.txt
.venv/bin/python -m pytest -q
```

Run the dashboard from its committed aggregate with:

```bash
.venv/bin/python -m pip install -r app/requirements.txt
.venv/bin/streamlit run app/dashboard.py
```

The dashboard reads only [`app/dashboard_data.json`](app/dashboard_data.json),
so it works without the raw dataset. The CSVs in [`results/`](results/) are
small extracts from that same JSON.

The full event study cannot currently be independently regenerated from this
checkout because the analysed transaction and market-data snapshot is not
included. The collection and analysis scripts can be used with a new dataset,
but a new API pull may differ from the one behind the reported results. See
[`data/README.md`](data/README.md) for the expected files and sources.

## Repository structure

```text
whale_signals/
├── app/
│   ├── dashboard.py                 # Streamlit UI
│   └── dashboard_data.json          # committed canonical aggregates
├── data/
│   ├── README.md                    # provenance and file contracts
│   ├── raw/                         # downloaded inputs (gitignored)
│   ├── processed/                   # prepared inputs (gitignored)
│   └── reference/                   # checked-in address tables
├── docs/
│   ├── dune_queries/                # transaction SQL
│   └── project_arc.md               # project history and hand-off notes
├── results/
│   ├── README.md                    # artefact definitions
│   └── published_*.csv              # compact result tables
├── scripts/
│   ├── build_dashboard_data.py      # aggregate analysis
│   └── run_*.py                     # phase and diagnostic scripts
├── src/
│   ├── analysis/                    # event-study calculations
│   ├── data/                        # provider clients and enrichment
│   ├── features/                    # rule labels and model features
│   ├── models/                      # separate classifier prototypes
│   └── sentiment/                   # VADER scoring and aggregation
└── tests/                            # transformation and pipeline tests
```

### Short code-review path

The core research can be reviewed without reading every acquisition,
dashboard, or prototype module:

1. [`src/analysis/panel.py`](src/analysis/panel.py) constructs the aligned
   whale-event and hourly-market panels. It owns timestamp normalisation,
   forward returns, and backward-looking sentiment joins.
2. [`src/analysis/event_study.py`](src/analysis/event_study.py) measures hit
   rates, market base rates, and yearly stability.
3. [`tests/test_analysis_panel.py`](tests/test_analysis_panel.py) and
   [`tests/test_event_study.py`](tests/test_event_study.py) demonstrate the
   alignment and directional rules on small, inspectable examples.
4. [`scripts/run_research.py`](scripts/run_research.py) is the concise
   end-to-end entry point. The dashboard-data builder consumes the same panel
   rather than maintaining a second implementation of the calculations.
