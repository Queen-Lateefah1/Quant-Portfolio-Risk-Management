"""
construction.py

Responsibility: build a portfolio-level return series from per-asset
returns and weights.

Phase 2 requires a "portfolio return series" as an output, ahead of the
full Phase 3 portfolio construction (rebalancing, market-cap weights,
long-short). This module currently implements equal-weighting only —
the minimum needed to satisfy Phase 2 — and is designed to be extended
in Phase 3 without changing this function's signature.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def equal_weighted_portfolio_returns(returns: pd.DataFrame) -> pd.Series:
    """Combine per-asset returns into an equal-weighted portfolio return series.

    Parameters
    ----------
    returns:
        Daily return matrix (log or simple), one column per ticker.
        All tickers are assumed to carry weight 1/N.

    Returns
    -------
    pd.Series
        Daily portfolio return series, indexed the same as `returns`.

    Raises
    ------
    ValueError
        If `returns` is empty.
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")

    n_assets = returns.shape[1]
    weights = pd.Series(1.0 / n_assets, index=returns.columns)

    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    portfolio_returns.name = "portfolio_return"

    logger.info("Computed equal-weighted portfolio returns across %d assets", n_assets)
    return portfolio_returns
