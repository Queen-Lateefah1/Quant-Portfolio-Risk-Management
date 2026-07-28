# data_collection

**Owner:** Lateefah Jamiu
**Purpose:** Download, clean, and validate raw equity price data and derive
daily log returns for the equity risk platform (Phase 1 of the internship
project). Portfolio weighting (Phase 3) and sector classification are out
of scope here and stubbed in `weights.py` / `sectors.py`.

## Inputs

| Module | Input | Type | Units / Notes |
|---|---|---|---|
| `downloader.py` | `tickers` | `list[str]` | Ticker symbols |
| `downloader.py` | `start_date`, `end_date` | `str` (ISO date) | `end_date=None` defaults to today |
| `validator.clean_prices` | `prices` | `pd.DataFrame` | Raw wide price matrix, date index |
| `returns.calculate_log_returns` | `prices` | `pd.DataFrame` | Positive adjusted-close prices |

## Outputs

| Module | Output | Type | Units / Sign convention |
|---|---|---|---|
| `downloader.fetch_adjusted_close` | price matrix | `pd.DataFrame` | Adjusted close, currency units, one column per ticker |
| `downloader.fetch_index_prices` | index series | `pd.Series` | Adjusted close, currency units |
| `validator.clean_prices` | `(cleaned_df, CleaningReport)` | tuple | `CleaningReport` documents every change made |
| `validator.validate_prices` | `ValidationResult` | dataclass | `passed=False` if any hard check fails |
| `returns.calculate_log_returns` | daily log returns | `pd.DataFrame` | Decimal form (e.g. `0.05` = +5%) |

Saved files land in `data/raw/` (untouched, as-downloaded) and
`data/processed/` (cleaned prices, returns, aligned index) — never mixed,
per the team's raw/processed/final convention.

## Dependencies

See `requirements.txt` in the project root. Install with:
```bash
pip install -r requirements.txt
```

## Execution

```bash
# from project root
jupyter notebook notebooks/data_collection.ipynb
```

The notebook only calls the modules and displays results — all logic
lives in `src/data_collection/`.

## Tests

```bash
# from project root
python -m pytest tests/ -v
```

Downloader tests cover input validation only (no live network calls in
the automated suite, so tests run offline). Returns and validator tests
use synthetic fixed sample data and include known-answer, edge-case, and
invalid-input cases.

## Known limitations

- `sectors.py` and `weights.py` are placeholders (not implemented — sector
  data is optional for Phase 1; portfolio weighting is Phase 3).
- Live download correctness (actual Yahoo Finance connectivity) is not
  covered by the automated test suite — verify manually by running the
  notebook end to end.
- No retry/backoff logic yet for transient Yahoo Finance rate-limiting.

## Downstream consumers

Phase 2 (data cleaning refinement) and Phase 3 (portfolio construction)
consume `data/processed/prices_adjusted_close.csv` and
`data/processed/returns_daily_log.csv`.
