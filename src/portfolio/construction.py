"""
construction.py

Responsibility: build a portfolio value/P&L/return series from prices
and a weight vector, with optional periodic rebalancing.

`equal_weighted_portfolio_returns` (built for Phase 2, before weighting
schemes existed) is kept unchanged for backward compatibility — it
operates directly on a return matrix. The functions below are the full
Phase 3 implementation: they operate on PRICES plus a weight vector,
and produce portfolio value, P&L, and returns together.
"""

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


def equal_weighted_portfolio_returns(returns: pd.DataFrame) -> pd.Series:
    """Combine per-asset returns into an equal-weighted portfolio return series.

    Kept from Phase 2 for backward compatibility — operates directly on
    a return matrix rather than on prices + a weight vector.

    Parameters
    ----------
    returns:
        Daily return matrix (log or simple), one column per ticker.

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


@dataclass
class PortfolioResult:
    """Structured output of build_portfolio (per guide 2.5)."""

    value: pd.Series
    pnl: pd.Series
    returns: pd.Series
    rebalance_dates: list


def build_portfolio(
    prices: pd.DataFrame,
    weights: pd.Series,
    initial_value: float = 1_000_000.0,
    rebalance_freq: str | None = None,
) -> PortfolioResult:
    """Build daily portfolio value, P&L, and returns from prices and weights.

    On the first date, `initial_value` is split across tickers according
    to `weights` to determine share counts. Between rebalance dates, share
    counts are held fixed (buy-and-hold) so portfolio weights drift with
    price moves — this is standard and intentional. On each rebalance
    date, share counts are recomputed so weights reset to the target.

    Parameters
    ----------
    prices:
        Wide price matrix (date index, one column per ticker). Must
        contain every ticker in `weights`.
    weights:
        Target weight per ticker (from weights.py), summing to 1.0.
    initial_value:
        Starting portfolio value in currency units. Default: 1,000,000.
    rebalance_freq:
        Pandas offset alias for rebalancing frequency, e.g. 'M' (monthly),
        'Q' (quarterly), 'W' (weekly). If None, the portfolio is never
        rebalanced after the initial allocation (pure buy-and-hold).

    Returns
    -------
    PortfolioResult
        `value`: daily portfolio value (currency units)
        `pnl`: daily portfolio P&L (currency units, first day is 0)
        `returns`: daily portfolio simple returns (decimal)
        `rebalance_dates`: list of dates rebalancing occurred, for auditing

    Raises
    ------
    ValueError
        If `prices` or `weights` is empty, if `weights` contains tickers
        not present in `prices`, or if `initial_value` <= 0.
    """
    if prices.empty:
        raise ValueError("prices cannot be empty")
    if weights.empty:
        raise ValueError("weights cannot be empty")
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")

    missing = set(weights.index) - set(prices.columns)
    if missing:
        raise ValueError(f"weights reference tickers not in prices: {sorted(missing)}")

    tickers = list(weights.index)
    asset_prices = prices[tickers].sort_index()

    rebalance_dates: list = []

    if rebalance_freq is None:
        # Pure buy-and-hold: shares fixed from day 1.
        shares = (initial_value * weights) / asset_prices.iloc[0]
        holdings_value = asset_prices.mul(shares, axis=1)
        portfolio_value = holdings_value.sum(axis=1)
    else:
        # Rebalance at the start of each period: reset shares to target weights
        # using that period's opening prices and the portfolio value carried
        # forward from the prior period's close.
        date_series = pd.Series(asset_prices.index, index=asset_prices.index)
        period_starts = date_series.resample(rebalance_freq).first()
        rebalance_dates = sorted(set(period_starts.dropna()) & set(asset_prices.index))

        portfolio_value = pd.Series(index=asset_prices.index, dtype=float)
        current_value = initial_value
        shares = (current_value * weights) / asset_prices.iloc[0]

        for date in asset_prices.index:
            if date in rebalance_dates and date != asset_prices.index[0]:
                current_value = (asset_prices.loc[date] * shares).sum()
                shares = (current_value * weights) / asset_prices.loc[date]
            portfolio_value.loc[date] = (asset_prices.loc[date] * shares).sum()

    pnl = portfolio_value.diff().fillna(0.0)
    pnl.iloc[0] = 0.0
    returns = portfolio_value.pct_change().fillna(0.0)

    portfolio_value.name = "portfolio_value"
    pnl.name = "portfolio_pnl"
    returns.name = "portfolio_return"

    logger.info(
        "Built portfolio: %d days, initial=%.2f, final=%.2f, rebalances=%d",
        len(portfolio_value), initial_value, portfolio_value.iloc[-1], len(rebalance_dates),
    )

    return PortfolioResult(value=portfolio_value, pnl=pnl, returns=returns, rebalance_dates=rebalance_dates)
