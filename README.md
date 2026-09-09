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
| 3 — Portfolio Construction | ✅ Done | Three weighting schemes (Market-cap weighting, user-defined weights, rebalancing)  builds daily portfolio value/P&L/returns with optional rebalancing |
| 4 — Historical Simulation VaR | ✅ Done | Historical Simulation VaR across 250/500/750-day windows and 95%/99%/99.5% confidence levels, Expected Shortfall, rolling VaR time series |
| 5–7 — Volatility Scaling | ✅ Done | EWMA, GARCH(1,1), and GJR-GARCH volatility scaling — all three feed into scaled Historical VaR |
| 8 — Backtesting | ✅ Done | Kupiec POF, Christoffersen Independence + Conditional Coverage, Basel Traffic Light |
| 9 — Model Comparison | ✅ Done | Compares Historical, EWMA, GARCH, GJR-GARCH VaR side by side, with a calibration-based ranking |
| 10 — Stress Testing | ✅ Done | Historical scenario replay (2008, COVID, 2022, 2023 banking stress, tech selloff), volatility/correlation/sector shocks — correlation shock uses actual asset-level covariance matrices, not a simplified approximation |
| Dashboard | ⏳ In progress | Streamlit |

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
│   ├── portfolio/          # Phase 3: weighting schemes, portfolio value/P&L/returns
│   ├── var/                # Phase 4: Historical Simulation VaR, Expected Shortfall
│ │   ├── volatility/         # Phase 5-7: EWMA, GARCH(1,1), GJR-GARCH volatility scaling
│   ├── backtesting/        # Phase 8-9: Kupiec, Christoffersen, Basel, model comparison
│   ├── stress_testing/     # Phase 10: historical replay, volatility/correlation/sector shocks
│   └── visualization/      # Charts for every phase above
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
Then open, in order: `notebooks/data_collection.ipynb` (Phase 1),
`notebooks/phase2_data_cleaning.ipynb` (Phase 2),
`notebooks/phase3_portfolio_construction.ipynb` (Phase 3),
`notebooks/phase4_5_var_ewma.ipynb` (Phase 4 & 5),
`notebooks/phase6_garch.ipynb` (Phase 6),
`notebooks/phase7_gjr_garch.ipynb` (Phase 7),
`notebooks/phase8_9_backtesting_comparison.ipynb` (Phase 8 & 9), and
`notebooks/phase10_stress_testing.ipynb` (Phase 10). Each phase reads
the previous phase's saved output, so they must be run in order.

## Testing

```bash
python -m pytest tests/ -v
```
205 tests total (Phases 1-10), covering normal cases, known-answer
cases, edge cases, and invalid-input cases for every module.

## Data Sources

Yahoo Finance, via `yfinance`.

## Known Limitations

- Portfolio construction currently supports equal-weighting only —
  market-cap and user-defined weights are Phase 3 scope.
- No retry/backoff logic yet for Yahoo Finance rate-limiting.
- Winsorization bounds (1st/99th percentile) are a reasonable default,
  not tuned to this specific universe.
