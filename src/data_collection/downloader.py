"""
downloader.py

Responsibility: download raw adjusted-close price data from Yahoo Finance.

This module ONLY downloads. It does not clean, validate, or transform data
(see validator.py and returns.py) — per guide 1.1 (one module, one
responsibility) and 3.3 (raw data must stay untouched).
"""

import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_adjusted_close(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Download daily adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers:
        List of ticker symbols to download.
    start_date:
        ISO date string (YYYY-MM-DD), inclusive start of the window.
    end_date:
        ISO date string (YYYY-MM-DD). If None, defaults to today.

    Returns
    -------
    pd.DataFrame
        Wide price matrix indexed by trading date, one column per ticker.
        Prices are already split/dividend-adjusted (auto_adjust=True).

    Raises
    ------
    ValueError
        If `tickers` is empty, or if no data could be downloaded for
        any ticker in the list.
    """
    if not tickers:
        raise ValueError("tickers cannot be empty")

    end_date = end_date or datetime.today().strftime("%Y-%m-%d")

    logger.info("Downloading adjusted close prices for %d tickers (%s to %s)",
                len(tickers), start_date, end_date)

    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    available = [t for t in tickers if t in raw.columns.get_level_values(0)]
    missing = set(tickers) - set(available)
    if missing:
        logger.warning("No data returned for tickers: %s", sorted(missing))

    if not available:
        raise ValueError(f"No data could be downloaded for any of: {tickers}")

    prices = pd.DataFrame({t: raw[t]["Close"] for t in available})
    prices.index.name = "date"
    return prices


def fetch_index_prices(
    index_ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.Series:
    """Download daily adjusted close prices for a single benchmark index.

    Parameters
    ----------
    index_ticker:
        Index ticker symbol, e.g. '^GSPC' for the S&P 500.
    start_date:
        ISO date string (YYYY-MM-DD), inclusive start of the window.
    end_date:
        ISO date string (YYYY-MM-DD). If None, defaults to today.

    Returns
    -------
    pd.Series
        Adjusted close price series indexed by trading date, named
        after `index_ticker`.

    Raises
    ------
    ValueError
        If no data could be downloaded for the index.
    """
    if not index_ticker:
        raise ValueError("index_ticker cannot be empty")

    end_date = end_date or datetime.today().strftime("%Y-%m-%d")

    logger.info("Downloading index prices for %s (%s to %s)", index_ticker, start_date, end_date)

    raw = yf.download(index_ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data could be downloaded for index: {index_ticker}")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    series = close.rename(index_ticker)
    series.index.name = "date"
    return series
