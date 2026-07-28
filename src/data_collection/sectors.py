"""
sectors.py

Responsibility: sector classification lookup for the ticker universe.

Uses yfinance's Ticker.info to look up each stock's GICS-style sector
(e.g. 'Technology', 'Financial Services'). This is a per-ticker network
call (yfinance has no bulk sector endpoint), so it's slower than the
price/index downloads in downloader.py — expect a few seconds for a
10-ticker universe.
"""

import logging
from typing import Dict, List

import yfinance as yf

logger = logging.getLogger(__name__)


def get_sector_map(tickers: List[str]) -> Dict[str, str]:
    """Look up the sector for each ticker in the universe.

    Parameters
    ----------
    tickers:
        List of ticker symbols to look up.

    Returns
    -------
    dict[str, str]
        Mapping of ticker -> sector name. If a ticker's sector cannot
        be determined (missing data, delisted, lookup failure), it is
        mapped to 'Unknown' rather than silently dropped, so every
        requested ticker always appears as a key.

    Raises
    ------
    ValueError
        If `tickers` is empty.
    """
    if not tickers:
        raise ValueError("tickers cannot be empty")

    sector_map: Dict[str, str] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector") or "Unknown"
        except Exception as exc:  # yfinance can raise various network/parsing errors
            logger.warning("Sector lookup failed for %s: %s", ticker, exc)
            sector = "Unknown"
        sector_map[ticker] = sector

    unknown = [t for t, s in sector_map.items() if s == "Unknown"]
    if unknown:
        logger.warning("Could not determine sector for: %s", unknown)

    return sector_map
