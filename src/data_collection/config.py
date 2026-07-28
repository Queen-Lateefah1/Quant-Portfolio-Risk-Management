"""
config.py

Shared constants for the data_collection module.

Per team guide 6.1 (Use Configuration Files) and 3.7 (Avoid Hard-Coded
Values and Paths): tickers, dates, and file locations live here so every
module and every team member uses the same assumptions.
"""

from pathlib import Path

# --- Universe ---------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "AMZN", "META", "GOOGL", "TSLA", "UNH"]
INDEX_TICKER = "^GSPC"  # S&P 500 benchmark

# --- Date window --------------------------------------------------------
# ~4.5 years of history so later phases (250/500/750-day rolling VaR windows)
# don't require re-collecting data.
START_DATE = "2021-01-01"
END_DATE = None  # None -> resolved to today() at runtime by downloader.py

# --- Paths (3.3: raw data is immutable; cleaned/derived data goes elsewhere) --
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_FINAL_DIR = PROJECT_ROOT / "data" / "final"

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_FINAL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Shared numeric conventions (used from Phase 4 onward, defined now) ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EWMA_LAMBDA = 0.94
DEFAULT_CONFIDENCE_LEVELS = [0.95, 0.99, 0.995]
DEFAULT_WINDOWS = [250, 500, 750]
