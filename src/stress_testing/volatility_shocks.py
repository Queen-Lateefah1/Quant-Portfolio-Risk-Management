"""
volatility_shocks.py

Responsibility: "Volatility doubles" and "Correlation spikes" hypothetical
scenarios. Both require re-deriving portfolio volatility under a
stressed assumption and re-scaling VaR/ES accordingly — this is
genuinely parametric risk modeling (asset-level covariance), not just
a shock applied to the portfolio return series directly, because
correlation is fundamentally a multi-asset concept that doesn't exist
in a single portfolio-level return series.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.var.historical import calculate_historical_var
from src.var.expected_shortfall import calculate_expected_shortfall

logger = logging.getLogger(__name__)


@dataclass
class ParametricStressResult:
    """Structured output for a parametric (covariance-based) stress scenario."""

    scenario: str
    base_portfolio_vol: float
    stressed_portfolio_vol: float
    vol_scale_factor: float
    base_var: float
    stressed_var: float
    base_es: float
    stressed_es: float


def apply_volatility_doubling_shock(
    portfolio_returns: pd.Series,
    vol_multiplier: float = 2.0,
    confidence_level: float = 0.95,
) -> ParametricStressResult:
    """Stress test: what if portfolio volatility suddenly doubled (or
    scaled by any multiplier)?

    Approach: VaR and Expected Shortfall both scale approximately
    linearly with volatility under a fixed distributional SHAPE
    assumption (only the scale changes, not the tail shape) — this is
    the standard, defensible simplification for a quick stress
    estimate; a full repricing would require re-simulating returns
    under the new volatility, which is a heavier Monte Carlo exercise
    outside this phase's scope.

    Parameters
    ----------
    portfolio_returns:
        Daily portfolio return series (the base/unstressed case).
    vol_multiplier:
        How much to scale volatility by. 2.0 = "volatility doubles".
    confidence_level:
        VaR/ES confidence level.

    Returns
    -------
    ParametricStressResult

    Raises
    ------
    ValueError
        If `portfolio_returns` is empty, or `vol_multiplier` <= 0.
    """
    if portfolio_returns.empty:
        raise ValueError("portfolio_returns cannot be empty")
    if vol_multiplier <= 0:
        raise ValueError("vol_multiplier must be positive")

    base_vol = float(portfolio_returns.std())
    stressed_vol = base_vol * vol_multiplier

    base_var = calculate_historical_var(portfolio_returns, confidence_level)
    base_es = calculate_expected_shortfall(portfolio_returns, confidence_level)

    stressed_var = base_var * vol_multiplier
    stressed_es = base_es * vol_multiplier

    logger.info(
        "Volatility shock (x%.1f): VaR %.4f -> %.4f, ES %.4f -> %.4f",
        vol_multiplier, base_var, stressed_var, base_es, stressed_es,
    )

    return ParametricStressResult(
        scenario=f"Volatility x{vol_multiplier:.1f}",
        base_portfolio_vol=base_vol, stressed_portfolio_vol=stressed_vol, vol_scale_factor=vol_multiplier,
        base_var=base_var, stressed_var=stressed_var, base_es=base_es, stressed_es=stressed_es,
    )


def apply_correlation_spike_shock(
    returns_matrix: pd.DataFrame,
    weights: pd.Series,
    stressed_correlation: float = 0.9,
    confidence_level: float = 0.95,
) -> ParametricStressResult:
    """Stress test: what if all assets suddenly became highly correlated
    (a classic "everything sells off together" crisis dynamic)?

    Approach: build the actual covariance matrix from historical
    per-asset returns and volatilities, compute portfolio variance
    under the REAL (estimated) correlation structure, then rebuild the
    covariance matrix with off-diagonal correlations replaced by
    `stressed_correlation` (diagonal/individual variances unchanged —
    only how assets move TOGETHER is stressed) and recompute portfolio
    variance. The ratio of stressed to base portfolio volatility is
    then used to rescale VaR/ES, same linear-scaling assumption as the
    volatility shock.

    Parameters
    ----------
    returns_matrix:
        Daily returns, one column per asset (NOT the portfolio-level
        series — this needs individual asset returns to compute a
        correlation matrix).
    weights:
        Portfolio weights, indexed by the same tickers as `returns_matrix`'s columns.
    stressed_correlation:
        The assumed correlation between every pair of assets under
        stress (e.g. 0.9 = near-perfect co-movement).
    confidence_level:
        VaR/ES confidence level.

    Returns
    -------
    ParametricStressResult

    Raises
    ------
    ValueError
        If `returns_matrix` is empty, `weights` references tickers not
        in `returns_matrix`, or `stressed_correlation` is not in [-1, 1].
    """
    if returns_matrix.empty:
        raise ValueError("returns_matrix cannot be empty")
    if not (-1 <= stressed_correlation <= 1):
        raise ValueError("stressed_correlation must be in [-1, 1]")

    missing = set(weights.index) - set(returns_matrix.columns)
    if missing:
        raise ValueError(f"weights reference tickers not in returns_matrix: {sorted(missing)}")

    tickers = list(weights.index)
    asset_returns = returns_matrix[tickers]
    w = weights.values

    vols = asset_returns.std().values
    base_corr = asset_returns.corr().values
    base_cov = np.outer(vols, vols) * base_corr
    base_portfolio_var = float(w @ base_cov @ w)
    base_portfolio_vol = np.sqrt(max(base_portfolio_var, 0))

    n = len(tickers)
    stressed_corr = np.full((n, n), stressed_correlation)
    np.fill_diagonal(stressed_corr, 1.0)
    stressed_cov = np.outer(vols, vols) * stressed_corr
    stressed_portfolio_var = float(w @ stressed_cov @ w)
    stressed_portfolio_vol = np.sqrt(max(stressed_portfolio_var, 0))

    scale_factor = stressed_portfolio_vol / base_portfolio_vol if base_portfolio_vol > 0 else 1.0

    portfolio_returns = asset_returns.mul(weights, axis=1).sum(axis=1)
    base_var = calculate_historical_var(portfolio_returns, confidence_level)
    base_es = calculate_expected_shortfall(portfolio_returns, confidence_level)
    stressed_var = base_var * scale_factor
    stressed_es = base_es * scale_factor

    logger.info(
        "Correlation shock (rho=%.2f): portfolio vol %.4f -> %.4f (x%.2f), VaR %.4f -> %.4f",
        stressed_correlation, base_portfolio_vol, stressed_portfolio_vol, scale_factor, base_var, stressed_var,
    )

    return ParametricStressResult(
        scenario=f"Correlation spike (rho={stressed_correlation:.2f})",
        base_portfolio_vol=base_portfolio_vol, stressed_portfolio_vol=stressed_portfolio_vol,
        vol_scale_factor=scale_factor, base_var=base_var, stressed_var=stressed_var,
        base_es=base_es, stressed_es=stressed_es,
    )
