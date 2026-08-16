"""
historical.py

Responsibility: Historical Simulation VaR.

Convention used throughout this module: VaR and losses are expressed as
POSITIVE numbers representing magnitude of loss (a VaR of 0.02 means a
2% loss threshold), even though the underlying returns are signed
(negative = loss). This matches the internship doc's validation
requirement: "VaR is positive loss number."

Historical Simulation VaR at confidence level c is defined as the loss
threshold such that only (1-c) of historical observations lost more:

    VaR_c = quantile(losses, c)   where losses = -returns

Since quantile is antitone under negation (quantile(-X, q) = -quantile(X, 1-q)),
this is computed directly and efficiently as:

    VaR_c = -quantile(returns, 1-c)
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate single-point Historical Simulation VaR from a return series.

    Parameters
    ----------
    returns:
        Daily portfolio return series (decimal form, e.g. -0.02 = -2%).
    confidence_level:
        VaR confidence level, e.g. 0.95, 0.99, 0.995.

    Returns
    -------
    float
        VaR as a positive loss magnitude (decimal). E.g. 0.023 means a
        2.3% loss threshold at the given confidence level.

    Raises
    ------
    ValueError
        If `returns` is empty, or `confidence_level` is not in (0, 1).
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be strictly between 0 and 1")

    var = -returns.quantile(1 - confidence_level)
    return float(var)


def calculate_historical_var_multi(
    returns: pd.Series,
    windows: List[int] = [250, 500, 750],
    confidence_levels: List[float] = [0.95, 0.99, 0.995],
) -> pd.DataFrame:
    """Build a VaR summary table across multiple lookback windows and
    confidence levels, using the most recent `window` observations
    trailing from the end of `returns` for each window size.

    Parameters
    ----------
    returns:
        Daily portfolio return series, most recent date last.
    windows:
        List of lookback window sizes in trading days.
    confidence_levels:
        List of confidence levels.

    Returns
    -------
    pd.DataFrame
        Rows indexed by window size, columns by confidence level,
        values are VaR (positive loss magnitude). A window larger than
        the available history is silently capped to the available data
        length and a warning is logged (not raised — a smaller sample
        is still informative, just less robust).

    Raises
    ------
    ValueError
        If `returns` is empty, or `windows`/`confidence_levels` is empty.
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")
    if not windows:
        raise ValueError("windows cannot be empty")
    if not confidence_levels:
        raise ValueError("confidence_levels cannot be empty")

    results = {}
    for window in windows:
        effective_window = min(window, len(returns))
        if effective_window < window:
            logger.warning(
                "Window %d exceeds available history (%d obs); using %d",
                window, len(returns), effective_window,
            )
        trailing = returns.iloc[-effective_window:]
        row = {cl: calculate_historical_var(trailing, cl) for cl in confidence_levels}
        results[window] = row

    table = pd.DataFrame(results).T
    table.index.name = "window"
    table.columns.name = "confidence_level"
    return table


def rolling_historical_var(
    returns: pd.Series,
    window: int = 250,
    confidence_level: float = 0.95,
) -> pd.Series:
    """Compute a rolling Historical Simulation VaR time series.

    Vectorized via pandas' rolling().quantile() rather than a rolling
    .apply() loop, for performance on longer histories.

    Parameters
    ----------
    returns:
        Daily portfolio return series.
    window:
        Rolling lookback window size in trading days.
    confidence_level:
        VaR confidence level.

    Returns
    -------
    pd.Series
        VaR (positive loss magnitude) at each date, NaN for the first
        `window - 1` dates where insufficient history exists.

    Raises
    ------
    ValueError
        If `returns` is empty, `window` < 2, or `confidence_level` is
        not in (0, 1).
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")
    if window < 2:
        raise ValueError("window must be >= 2")
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be strictly between 0 and 1")

    rolling_var = -returns.rolling(window).quantile(1 - confidence_level)
    rolling_var.name = f"var_{int(confidence_level*100)}pct_{window}d"
    return rolling_var
