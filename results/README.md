# Result extracts

These CSV files copy selected values from
[`app/dashboard_data.json`](../app/dashboard_data.json) into a form that is easy
to inspect:

- `published_horizon_edges.csv`: $1M+ deposit edge by horizon;
- `published_yearly_edges.csv`: yearly deposit and withdrawal edges;
- `published_threshold_sensitivity.csv`: the four reported size cuts;
- `published_bull_bear.csv`: short-horizon market-regime results;
- `published_drawdown.csv`: maximum adverse excursion summaries.

They support the tables in the main README, but they are not results regenerated
from the raw transactions. Because the original input snapshot is not committed,
they cannot independently verify the empirical analysis. Hit rates are
percentages; edges are percentage-point differences. Long-horizon event windows
overlap substantially.
