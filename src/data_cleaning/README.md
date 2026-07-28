# data_cleaning & portfolio (Phase 2)

**Owner:** Lateefah Jamiu
**Purpose:** Take Phase 1's raw-cleaned prices and apply Phase 2 treatment
— align trading dates, remove stale prices, winsorize extreme returns —
then produce a portfolio-level return series.

## Inputs

| Module | Input | Type | Notes |
|---|---|---|---|
| `alignment.align_trading_dates` | `prices` | `pd.DataFrame` | Wide price matrix, may have per-ticker gaps |
| `cleaner.remove_stale_prices` | `prices`, `min_repeat_run` | `pd.DataFrame`, `int` | Default run length: 5 trading days |
| `cleaner.winsorize_returns` | `returns`, `lower_percentile`, `upper_percentile` | `pd.DataFrame`, `float`, `float` | Defaults: 1st/99th percentile, per ticker |
| `construction.equal_weighted_portfolio_returns` | `returns` | `pd.DataFrame` | Assumes 1/N weight per asset |

## Outputs

| Module | Output | Notes |
|---|---|---|
| `alignment.align_trading_dates` | `(pd.DataFrame, AlignmentReport)` | Report states dates dropped, tickers gap-filled |
| `cleaner.remove_stale_prices` | `(pd.DataFrame, StalePriceReport)` | Report states stale runs found per ticker |
| `cleaner.winsorize_returns` | `(pd.DataFrame, WinsorizeReport)` | Report states values clipped per ticker |
| `construction.equal_weighted_portfolio_returns` | `pd.Series` | Named `portfolio_return` |

All treatment is explicit and reported — nothing is changed silently
(per team guide 3.4).

Saved files: `data/processed/prices_clean_phase2.csv`,
`returns_clean_winsorized.csv`, `portfolio_returns_equal_weighted.csv`,
`rolling_volatility_21d.csv`.

## Why log returns, not simple returns

See `docs/why_log_returns.docx` for the full write-up. Short version:
log returns are time-additive (needed for 1-day vs 10-day VaR), closer
to the normality assumption GARCH-family models rely on, and more
numerically stable when chained across a multi-asset pipeline.

## Execution

```bash
pip install -r requirements.txt
jupyter notebook notebooks/phase2_data_cleaning.ipynb
```

Requires Phase 1 to have already run and populated
`data/processed/prices_adjusted_close.csv`.

## Tests

```bash
python -m pytest tests/ -v
```

34 tests total (Phase 1 + Phase 2), covering normal cases, known-answer
cases, edge cases, and invalid inputs for every module.

## Known limitations

- `construction.py` only implements equal-weighting; market-cap and
  user-defined weights are Phase 3 scope.
- Winsorization bounds (1st/99th percentile) are a reasonable default,
  not tuned against this specific universe — see the referenced papers
  in `docs/` for the reasoning behind percentile-based winsorizing of
  stock returns.
