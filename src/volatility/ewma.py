"""
ewma.py

Responsibility: EWMA (Exponentially Weighted Moving Average) volatility
estimation and volatility-scaled VaR.

The idea: historical returns from calm periods understate risk if
today's market is more volatile. EWMA scaling rescales each historical
return by the ratio of today's estimated volatility to that day's
estimated volatility, so the Historical Simulation VaR reacts to
current volatility regimes instead of just raw historical frequency.
"""

import logging

import numpy as np
import pandas as pd

from src.var.historical import calculate_historical_var

logger = logging.getLogger(__name__)

DEFAULT_LAMBDA = 0.94  # RiskMetrics standard for daily returns


def ewma_variance(returns: pd.Series, lambda_: float = DEFAULT_LAMBDA) -> pd.Series:
    """Calculate EWMA variance of a return series.

    Recursive definition: sigma2_t = lambda * sigma2_{t-1} + (1-lambda) * r_{t-1}^2
    Implemented via pandas' ewm(), which computes this efficiently.

    Parameters
    ----------
    returns:
        Daily return series.
    lambda_:
        Decay factor. RiskMetrics standard is 0.94 for daily returns
        (higher = slower-reacting, longer effective memory).

    Returns
    -------
    pd.Series
        EWMA variance at each date. The first value equals the first
        squared return (no prior history to decay from).

    Raises
    ------
    ValueError
        If `returns` is empty, or `lambda_` is not in (0, 1).
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")
    if not (0 < lambda_ < 1):
        raise ValueError("lambda_ must be strictly between 0 and 1")

    squared_returns = returns**2
    # alpha in pandas' ewm() = 1 - lambda, matching the RiskMetrics recursion
    variance = squared_returns.ewm(alpha=1 - lambda_, adjust=False).mean()
    variance.name = "ewma_variance"
    return variance


def ewma_volatility(returns: pd.Series, lambda_: float = DEFAULT_LAMBDA) -> pd.Series:
    """Calculate EWMA volatility (standard deviation) of a return series.

    Parameters
    ----------
    returns:
        Daily return series.
    lambda_:
        Decay factor (see ewma_variance).

    Returns
    -------
    pd.Series
        EWMA volatility (daily, decimal form) at each date.
    """
    volatility = np.sqrt(ewma_variance(returns, lambda_))
    volatility.name = "ewma_volatility"
    return volatility


def ewma_scaled_returns(returns: pd.Series, lambda_: float = DEFAULT_LAMBDA) -> pd.Series:
    """Rescale historical returns to today's (most recent) EWMA volatility.

    Each historical return is scaled by the ratio of the CURRENT
    (most recent date's) EWMA volatility to that day's own EWMA
    volatility. This means calm-period returns get amplified and
    volatile-period returns get dampened, so the resulting distribution
    reflects "what would this history have looked like if it all
    happened under today's volatility regime."

    Parameters
    ----------
    returns:
        Daily return series.
    lambda_:
        Decay factor.

    Returns
    -------
    pd.Series
        Volatility-scaled returns, same index as `returns`.

    Raises
    ------
    ValueError
        If `returns` is empty.
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")

    vol = ewma_volatility(returns, lambda_)
    current_vol = vol.iloc[-1]

    scaled = returns * (current_vol / vol)
    scaled.name = "scaled_return"

    logger.info(
        "Scaled %d returns to current EWMA volatility (%.4f)", len(returns), current_vol
    )
    return scaled


def ewma_scaled_historical_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    lambda_: float = DEFAULT_LAMBDA,
) -> float:
    """Calculate Historical Simulation VaR on EWMA volatility-scaled returns.

    This is the Phase 5 output: a VaR figure that reacts to the current
    volatility regime rather than treating all historical observations
    as equally representative of today's risk.

    Parameters
    ----------
    returns:
        Daily return series.
    confidence_level:
        VaR confidence level.
    lambda_:
        EWMA decay factor.

    Returns
    -------
    float
        EWMA-scaled VaR (positive loss magnitude).

    Raises
    ------
    ValueError
        If `returns` is empty or `confidence_level` is not in (0, 1).
    """
    scaled_returns = ewma_scaled_returns(returns, lambda_)
    return calculate_historical_var(scaled_returns, confidence_level)
