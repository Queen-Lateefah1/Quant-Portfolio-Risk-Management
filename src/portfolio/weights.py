"""
weights.py

Responsibility: compute a weight vector for the portfolio under one of
several schemes. This module ONLY computes weights — it doesn't build
portfolio value or returns (see construction.py for that).
"""

import logging
from typing import Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def equal_weights(tickers: List[str]) -> pd.Series:
    """Assign 1/N weight to each ticker.

    Parameters
    ----------
    tickers:
        List of ticker symbols.

    Returns
    -------
    pd.Series
        Weights indexed by ticker, summing to 1.0.

    Raises
    ------
    ValueError
        If `tickers` is empty.
    """
    if not tickers:
        raise ValueError("tickers cannot be empty")

    n = len(tickers)
    weights = pd.Series(1.0 / n, index=tickers, name="weight")
    logger.info("Computed equal weights for %d tickers", n)
    return weights


def user_defined_weights(weights: Dict[str, float]) -> pd.Series:
    """Validate and return a user-supplied weight dictionary as a Series.

    Weights must sum to 1.0 (within a small floating-point tolerance).
    This function does NOT silently normalize an invalid input — if the
    weights don't sum to 1, it raises, so a typo doesn't quietly produce
    a portfolio that isn't fully invested (or is over-invested).

    Parameters
    ----------
    weights:
        Mapping of ticker -> weight. Weights may be negative for a
        long-short book, but must still sum to 1.0 overall (net exposure).

    Returns
    -------
    pd.Series
        Weights indexed by ticker.

    Raises
    ------
    ValueError
        If `weights` is empty, or if the weights don't sum to 1.0
        (tolerance: 1e-6).
    """
    if not weights:
        raise ValueError("weights cannot be empty")

    series = pd.Series(weights, name="weight")
    total = series.sum()

    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"weights must sum to 1.0, got {total:.6f}. "
            "If you intended a partially-invested or long-short book, "
            "confirm the weights are correct before proceeding."
        )

    logger.info("Validated user-defined weights for %d tickers (sum=%.6f)", len(series), total)
    return series


def market_cap_weights(tickers: List[str]) -> pd.Series:
    """Weight each ticker proportionally to its current market capitalization.

    Tickers whose market cap can't be determined (lookup failure, missing
    data) are excluded and the remaining weights are renormalized to sum
    to 1.0 — this is reported via a warning log, not done silently.

    Parameters
    ----------
    tickers:
        List of ticker symbols.

    Returns
    -------
    pd.Series
        Weights indexed by ticker (only tickers with usable market cap
        data are included), summing to 1.0.

    Raises
    ------
    ValueError
        If `tickers` is empty, or if NO ticker's market cap could be
        determined.
    """
    if not tickers:
        raise ValueError("tickers cannot be empty")

    market_caps: Dict[str, float] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            cap = info.get("marketCap")
            if cap:
                market_caps[ticker] = float(cap)
        except Exception as exc:
            logger.warning("Market cap lookup failed for %s: %s", ticker, exc)

    excluded = set(tickers) - set(market_caps)
    if excluded:
        logger.warning("Excluded from market-cap weighting (no data): %s", sorted(excluded))

    if not market_caps:
        raise ValueError(f"Could not determine market cap for any of: {tickers}")

    caps = pd.Series(market_caps, name="weight")
    weights = caps / caps.sum()

    logger.info("Computed market-cap weights for %d/%d tickers", len(weights), len(tickers))
    return weights


def long_short_weights(long_tickers: List[str], short_tickers: List[str]) -> pd.Series:
    """Build weights for a long-short book (equal-weighted within each leg).

    Not yet implemented — the internship doc marks long-short as optional
    for Phase 3. Placeholder kept here so it has a clear home if the team
    decides to build it out later.
    """
    raise NotImplementedError(
        "weights.long_short_weights is a placeholder. Long-short "
        "portfolios are marked optional for Phase 3 and have not been built yet."
    )
