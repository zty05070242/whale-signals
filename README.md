# Are Ethereum Whales Smart Money? An Event Study of On-Chain Signals and Sentiment

**Languages:** English | [繁體中文](README.zh-Hant.md) | [简体中文](README.zh-Hans.md)

Large Ethereum transfers are weak signals when treated as one undifferentiated
group. Once I separate transfers by their direction relative to labelled
exchanges, however, a measurable asymmetry appears: deposits are followed by
slightly more downside than the matched market base rate, while withdrawals do
not retain a stable positive association through the later sample.

The analysed dataset contains 646,442 successful, top-level Ethereum
transactions above $1 million from January 2023 to July 2026. Transfers are
classified from known address labels, joined to hourly ETH prices and two
market-sentiment measures, and compared with the return direction from all
eligible hours under the same conditions.

## What I found

- **Exchange deposits contain a modest, repeatable signal.** At 24 hours they
  precede downside 50.45% of the time, a +1.33 percentage-point difference from
  the matched downward base rate. The result is similar from $1M to $10M and
  becomes larger in the 2024, 2025 and partial-2026 yearly samples.
- **The useful withdrawal condition did not persist.** Withdrawals during
  negative funding had a +4.71 point edge in 2023 and +10.12 points in 2024,
  then fell to +0.07 in 2025 and -3.92 in the partial 2026 sample.
- **The deposit association is larger at long horizons, but harder to infer
  from.** It reaches +12.44 points at six months, where overlapping event
  windows sharply reduce the independent information. Maximum adverse
  excursion also shows that deposits ending in downside often moved
  substantially in the opposite direction first.

These are associations, not proof that a transfer caused a price move or that
the owner intended to trade. Exchange direction is a proxy, and the analysis is
not a live or costed trading strategy. Within those limits, the result is not
that whales contain no information; it is that transaction size alone is less
useful than where the transaction is going.

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

The work has two distinct analytical components:

- The **canonical event study** uses categories determined directly from known
  address labels. Unknown-to-unknown wallet transfers are excluded from the
  deposit and withdrawal tests.
- A **Random Forest prototype** was evaluated separately to see whether simple
  transaction features could recover categories for unknown wallets. It
  achieved 67.7% accuracy on held-out rows whose categories were first assigned
  by the address rules. Applying it to unknown-to-unknown wallets is exploratory
  and has no independently validated ground truth. Its predictions do not feed
  the reported event-study results.

That separation matters. A predicted exchange label would introduce model
error into the financial result, whereas the present event study can be read as
a test of an explicit set of address rules.

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

### Deposits show a modest 24-hour association and a larger long-horizon difference

At the $1M threshold, labelled exchange deposits produce a +1.33 point
difference at 24 hours. The difference is similar at three days and one week,
then grows across the longer measured horizons.

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

**Measurement:** the 24-hour deposit hit rate is 50.45%, compared with a
49.12% matched downward base rate. The difference is 1.33 points.

**Interpretation:** within this sample, labelled exchange deposits precede
declines modestly more often than all eligible hours. The result is absent at 6
hours and becomes larger as the return window expands.

**Caution:** the increasing numbers do not show that depositors forecast six
months ahead. Long windows repeatedly count the same market moves, so the
event-row count overstates the independent information available. The table is
a descriptive horizon comparison, not evidence of monotonic causal impact.

### Withdrawals lose their early conditional association

The 24-hour year split contrasts unconditional deposits with withdrawals made
during negative funding, the withdrawal condition that looked strongest early
in the sample.

| Year | Deposit edge | Negative-funding withdrawal edge |
|---|---:|---:|
| 2023 | −0.17 pp | +4.71 pp |
| 2024 | +0.88 pp | +10.12 pp |
| 2025 | +1.79 pp | +0.07 pp |
| 2026 (to July) | +3.86 pp | −3.92 pp |

Source: [`results/published_yearly_edges.csv`](results/published_yearly_edges.csv).

**Measurement:** the deposit difference is positive in 2024–2026 and larger
in the partial 2026 sample. The negative-funding withdrawal difference is
positive in 2023–2024, approximately zero in 2025, and negative in 2026.

**Interpretation:** the negative-funding withdrawal condition identified a
sizeable positive association in 2023–2024, but it did not survive into the
later years. By comparison, the deposit result is positive and increases from
2024 through the partial 2026 sample. The sample does not establish that either
pattern will continue.

**Untested hypothesis:** staking, liquidity provision, bridging and custody
reorganisation may have changed what exchange withdrawals represent. This
project does not trace post-withdrawal destinations, so it cannot identify that
mechanism. The threshold results rule out only the narrow claim that the change
appears solely below $10M; they do not prove a DeFi explanation.

### The deposit result is similar from $1M to $10M

Unconditional deposits have similar 24-hour differences at the four main size
cuts:

| Minimum size | Deposit hit rate | Edge |
|---|---:|---:|
| $1M | 50.45% | +1.33 pp |
| $2M | 50.54% | +1.42 pp |
| $5M | 50.35% | +1.23 pp |
| $10M | 50.48% | +1.35 pp |

The original Dune pull uses a strict `usd_value > $1M` filter, while the later
threshold sensitivity code uses `usd_value >= threshold`. Values exactly on a
threshold are unlikely, but the boundary convention is not identical. Subject
to that caveat, the similarity suggests this 24-hour result is not created by
one chosen USD cutoff. It does not show that larger transactions have a larger effect. The
complete conditioned threshold table is in
[`results/published_threshold_sensitivity.csv`](results/published_threshold_sensitivity.csv).

The threshold itself changes economic meaning over time. A fixed $1M represents
fewer ETH when ETH is expensive, so yearly comparisons mix changes in market
behaviour with changes in the population admitted by the filter.

### Market conditions change the raw hit rate

At 24 hours and $1M+, deposits during extreme greed have a 54.43% hit rate. The
matched downward base rate during extreme greed is 51.78%, leaving a +2.65
point difference. Under negative funding, withdrawals have a 55.14% upward hit
rate, but the matched upward base rate is higher at 56.43%, producing a −1.29
point difference.

This is why a high raw hit rate is not enough. Conditioning can select hours
where the same price direction is already common. The market-derived variables
are useful here as controls and grouping variables; the results do not show
that sentiment causes the subsequent move.

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

One possible interpretation is that exchange flows differ between advancing
and declining markets. However, these regimes are constructed from the same
price path being evaluated, and event windows still overlap. The table should
be treated as a conditional description rather than a tradable regime model.

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

This table does not test a strategy. It selects events by their eventual
outcome, ignores transaction costs and execution delay, and does not simulate a
stop-loss. Its purpose is narrower: endpoint accuracy can conceal substantial
interim movement in the opposite direction.

### Why I withdrew the 78.3% result

An earlier version reported a 78.3% hit rate for $10M+ deposits during extreme
greed. The arithmetic was real, but the events were concentrated in ten
calendar days. Many rows were therefore measuring the same price moves. I no
longer treat 78.3% as a finding and retain it here because discovering the
problem changed the interpretation of the wider analysis.

This is the clearest example of why transaction count is not independent sample
size. A future version should collapse clustered events, use calendar-time
portfolios, or apply an event-study correction designed for cross-sectional
dependence before making inferential claims.

## Bottom line

Generic whale alerts are weak because transaction size does not say what the
transfer represents. The most repeatable signal in this study is the direction
of funds relative to labelled exchanges: deposits have a modest negative-price
association at 24 hours that is similar across size thresholds and stronger in
the later yearly samples. The early positive association for withdrawals during
negative funding disappears and reverses by the partial 2026 sample.

That is a useful narrowing of the original claim, not a claim that whales
reliably predict prices. The next step is dependence-robust validation of the
event study, followed by a live test with fixed labels and execution rules.

## Limitations

1. **Overlapping event windows.** Transactions cluster in time, especially
   during volatile periods. At one month and beyond, many events share most of
   their return window. Conventional row-level p-values overstate precision;
   this README emphasises measured differences instead.

2. **Address labels are incomplete and not historical.** The merged list is a
   present-day snapshot. Addresses can change purpose, labels can be wrong, and
   62.8% coverage of at least one identified endpoint still leaves a large
   unmatched group. The exact upstream revisions were not recorded.

3. **Transfer categories are proxies.** A deposit does not prove a sale, and a
   withdrawal does not prove a purchase. Internal exchange movements, custody
   changes, bridge activity and transfers between services may have no
   directional intent. Exchange-to-exchange transfers follow an explicit but
   contestable precedence rule.

4. **No internal calls.** The Dune query uses top-level transactions, not
   `ethereum.traces`. Some economically important contract-mediated flows are
   absent.

5. **The analysed inputs are unavailable.** The dashboard JSON and compact
   tables are committed, but the original 187 MB transaction file and market
   inputs are not. The tables show what the dashboard reports; they do not
   independently verify the underlying event study.

6. **The threshold is fixed in USD.** Changes in ETH price alter the amount of
   ETH and population of actors represented by $1M. ETH-denominated and
   inflation-adjusted thresholds would answer related but different questions.

7. **The 2026 sample is partial.** It was collected later and analysed without
   parameter changes, but ends in July and shares the same labels and analysis
   choices. It is a robustness check, not definitive confirmation.

8. **Regime definitions are simplified.** Fear & Greed is daily while returns
   are hourly; funding is one derivatives-market measure; and the 20% bull/bear
   rule is path-dependent. Results can change under other definitions.

9. **MEV filtering is incomplete.** Heuristics and four known addresses cannot
   identify the full MEV population. Automated transactions may remain.

10. **No implementable strategy is evaluated.** There is no slippage, fee,
    latency, position-sizing, portfolio construction or stop-loss simulation.
    The study asks an association question, not a profitability question.

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
