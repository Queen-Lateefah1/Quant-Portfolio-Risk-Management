# Quant Portfolio Risk Management

Internship project for the Market Risk track at QFI's Quant Lab. Building
a Python-based risk analytics platform that calculates, scales, backtests,
and visualizes Value at Risk (VaR) for an equity portfolio.

**Author:** Lateefah Jamiu ([@Queen-Lateefah1](https://github.com/Queen-Lateefah1))

## What This Project Does

The end goal is a full risk analytics pipeline covering:
- Historical Simulation VaR
- EWMA / GARCH / GJR-GARCH volatility scaling
- VaR backtesting (Kupiec, Christoffersen, Basel Traffic Light)
- Stress testing against historical and hypothetical scenarios
- An interactive Streamlit dashboard

It's built in phases, each adding a new module to the pipeline rather than
one big script — see "Architecture" below for why.

## Progress

| Phase | Status | What it does |
|---|---|---|
| 1 — Data Collection | ✅ Done | Downloads adjusted close prices for a 10-stock universe + S&P 500 benchmark, computes log returns, looks up sector classifications |
| 2 — Data Cleaning | ✅ Done | Aligns trading dates across tickers, removes stale prices, winsorizes extreme returns, builds an equal-weighted portfolio return series |
| 3 — Portfolio Construction | ⏳ Not started | Market-cap weighting, user-defined weights, rebalancing |
| 4 — Historical Simulation VaR | ⏳ Not started | |
| 5–7 — Volatility Scaling | ⏳ Not started | EWMA, GARCH, GJR-GARCH |
| 8 — Backtesting | ⏳ Not started | |
| 9 — Model Comparison | ⏳ Not started | |
| 10 — Stress Testing | ⏳ Not started | |
| Dashboard | ⏳ Not started | Streamlit |

## Universe

AAPL, MSFT, NVDA, JPM, XOM, AMZN, META, GOOGL, TSLA, UNH — benchmarked
against the S&P 500 (^GSPC).

## Project Structure

```
├── notebooks/              # Thin notebooks — call modules, show results
│   ├── data_collection.ipynb
│   └── phase2_data_cleaning.ipynb
├── src/
│   ├── data_collection/    # Phase 1: download, validate, compute returns
│   ├── data_cleaning/      # Phase 2: align dates, remove stale prices, winsorize
│   └── portfolio/          # Portfolio return construction
├── tests/                  # pytest suite — normal, known-answer, edge, invalid-input cases
├── data/
│   ├── raw/                 # Untouched, as-downloaded data
│   └── processed/           # Cleaned/derived outputs
├── docs/                    # Supporting write-ups (e.g. why log returns)
└── requirements.txt
```

Each module has one responsibility — a teammate working on data collection
doesn't need to read cleaning or portfolio code, and vice versa. See
`docs/` for the fuller reasoning behind this structure.

## Setup

```bash
git clone https://github.com/Queen-Lateefah1/Quant-Portfolio-Risk-Management.git
cd Quant-Portfolio-Risk-Management
pip install -r requirements.txt
```

## Running It

```bash
jupyter notebook
```
Then open, in order: `notebooks/data_collection.ipynb` (Phase 1), followed
by `notebooks/phase2_data_cleaning.ipynb` (Phase 2). Phase 2 reads Phase 1's
output, so Phase 1 must be run first.

## Testing

```bash
python -m pytest tests/ -v
```
34 tests covering normal cases, known-answer cases, edge cases, and
invalid-input cases for every module.

## Data Sources

Yahoo Finance, via `yfinance`.

## Known Limitations

- Portfolio construction currently supports equal-weighting only —
  market-cap and user-defined weights are Phase 3 scope.
- No retry/backoff logic yet for Yahoo Finance rate-limiting.
- Winsorization bounds (1st/99th percentile) are a reasonable default,
  not tuned to this specific universe.
