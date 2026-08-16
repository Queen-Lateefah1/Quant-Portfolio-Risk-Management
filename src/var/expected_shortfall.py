"""
expected_shortfall.py

Responsibility: Expected Shortfall (Conditional VaR) — the average loss
GIVEN that the loss exceeds the VaR threshold. Always >= VaR at the
same confidence level, per the internship doc's validation requirement
("Expected Shortfall > VaR").
"""

import logging

import pandas as pd

from src.var.historical import calculate_historical_var

logger = logging.getLogger(__name__)


def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate historical Expected Shortfall (average tail loss beyond VaR).

    Parameters
    ----------
    returns:
        Daily portfolio return series.
    confidence_level:
        Confidence level, matching the VaR it's conditioned on.

    Returns
    -------
    float
        Expected Shortfall as a positive loss magnitude. Always >= the
        VaR at the same confidence level.

    Raises
    ------
    ValueError
        If `returns` is empty, `confidence_level` is not in (0, 1), or
        no observations fall in the tail beyond VaR (can happen with
        very small samples — widen the sample or lower confidence).
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be strictly between 0 and 1")

    var = calculate_historical_var(returns, confidence_level)
    losses = -returns
    tail_losses = losses[losses >= var]

    if tail_losses.empty:
        raise ValueError(
            f"No observations in the tail beyond VaR ({var:.4f}) — "
            "sample may be too small for this confidence level."
        )

    es = float(tail_losses.mean())
    logger.info("Expected Shortfall at %.1f%%: %.4f (VaR was %.4f)", confidence_level * 100, es, var)
    return es


def rolling_expected_shortfall(
    returns: pd.Series,
    window: int = 250,
    confidence_level: float = 0.95,
) -> pd.Series:
    """Compute a rolling Expected Shortfall time series.

    Uses a rolling .apply() (ES has no closed-form vectorized rolling
    equivalent like VaR's quantile does) — fine for the window sizes
    this project uses (a few hundred to ~1000 rows).

    Parameters
    ----------
    returns:
        Daily portfolio return series.
    window:
        Rolling lookback window size in trading days.
    confidence_level:
        Confidence level.

    Returns
    -------
    pd.Series
        ES (positive loss magnitude) at each date, NaN for the first
        `window - 1` dates.

    Raises
    ------
    ValueError
        If `returns` is empty or `window` < 2.
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")
    if window < 2:
        raise ValueError("window must be >= 2")

    def _es(x):
        try:
            return calculate_expected_shortfall(pd.Series(x), confidence_level)
        except ValueError:
            return float("nan")

    result = returns.rolling(window).apply(_es, raw=False)
    result.name = f"es_{int(confidence_level*100)}pct_{window}d"
    return result
