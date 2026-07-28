"""
returns.py

Responsibility: convert a clean price matrix into daily log returns.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate continuously compounded (log) daily returns.

    Parameters
    ----------
    prices:
        Positive adjusted-close prices indexed by trading date.

    Returns
    -------
    pd.DataFrame
        Daily log returns in decimal form, indexed by trading date.
        The first row of `prices` has no prior day to compare to, so
        the returned frame has one fewer row than `prices`.

    Raises
    ------
    ValueError
        If `prices` is empty, non-positive, or non-numeric.
    """
    if prices.empty:
        raise ValueError("prices cannot be empty")
    if not np.issubdtype(prices.to_numpy().dtype, np.number):
        raise ValueError("prices must be numeric")
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive to compute log returns")

    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    logger.info("Computed log returns: %d rows x %d tickers", *returns.shape)
    return returns
