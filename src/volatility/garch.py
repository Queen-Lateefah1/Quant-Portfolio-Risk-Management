"""
garch.py

Responsibility: GARCH(1,1) volatility modeling and GARCH-scaled Historical VaR.

Where EWMA (Phase 5) uses a fixed decay factor (lambda=0.94) chosen in
advance, GARCH estimates its own persistence and reaction parameters
from the data itself via maximum likelihood — it can capture volatility
clustering (calm periods followed by calm periods, volatile followed by
volatile) more flexibly than EWMA's single fixed decay rate.

Uses the `arch` library (Kevin Sheppard's ARCH/GARCH toolkit), the
standard Python package for this — reimplementing GARCH MLE fitting by
hand would be reinventing a well-tested wheel and a real risk of subtle
numerical bugs.
"""

import logging

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from src.var.historical import calculate_historical_var

logger = logging.getLogger(__name__)


def fit_garch_model(returns: pd.Series, p: int = 1, q: int = 1) -> ARCHModelResult:
    """Fit a GARCH(p, q) model to a return series.

    Returns are rescaled by 100 internally (the `arch` library's
    optimizer is numerically better-behaved on returns expressed in
    percent rather than raw decimals) and rescaled back out wherever
    this module reports a result — callers of this module's other
    functions never need to think about this detail.

    Parameters
    ----------
    returns:
        Daily return series (decimal form, e.g. -0.02 = -2%).
    p:
        GARCH lag order for the conditional variance's own past values.
    q:
        ARCH lag order for past squared residuals. Together (p=1, q=1)
        is the standard GARCH(1,1) the doc specifies.

    Returns
    -------
    ARCHModelResult
        The fitted model, from which conditional volatility, forecasts,
        and residuals can all be extracted.

    Raises
    ------
    ValueError
        If `returns` is empty or has fewer than 30 observations (GARCH
        MLE is unreliable on very short samples).
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")
    if len(returns) < 30:
        raise ValueError("returns must have at least 30 observations for reliable GARCH fitting")

    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol="Garch", p=p, q=q, dist="normal")
    fitted = model.fit(disp="off")

    logger.info(
        "Fitted GARCH(%d,%d): omega=%.6f, alpha=%.4f, beta=%.4f, persistence=%.4f",
        p, q,
        fitted.params.get("omega", float("nan")),
        fitted.params.get("alpha[1]", float("nan")),
        fitted.params.get("beta[1]", float("nan")),
        fitted.params.get("alpha[1]", 0) + fitted.params.get("beta[1]", 0),
    )
    return fitted


def get_conditional_volatility(fitted_model: ARCHModelResult) -> pd.Series:
    """Extract the in-sample conditional volatility path from a fitted GARCH model.

    Parameters
    ----------
    fitted_model:
        Result from `fit_garch_model`.

    Returns
    -------
    pd.Series
        Daily conditional volatility (decimal form, rescaled back down
        from the internal x100 fitting scale), same index as the
        original returns.
    """
    conditional_vol = fitted_model.conditional_volatility / 100
    conditional_vol.name = "garch_conditional_volatility"
    return conditional_vol


def forecast_volatility(fitted_model: ARCHModelResult, horizon: int = 1) -> pd.Series:
    """Forecast conditional volatility forward from a fitted GARCH model.

    Parameters
    ----------
    fitted_model:
        Result from `fit_garch_model`.
    horizon:
        Number of days ahead to forecast.

    Returns
    -------
    pd.Series
        Forecasted volatility (decimal form) for each of the next
        `horizon` days, indexed 1..horizon.

    Raises
    ------
    ValueError
        If `horizon` < 1.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    forecast = fitted_model.forecast(horizon=horizon, reindex=False)
    variance_forecast = forecast.variance.iloc[-1]
    vol_forecast = np.sqrt(variance_forecast) / 100
    vol_forecast.index = range(1, horizon + 1)
    vol_forecast.name = "garch_forecast_volatility"
    return vol_forecast


def get_standardized_residuals(fitted_model: ARCHModelResult) -> pd.Series:
    """Extract standardized residuals (residual / conditional volatility)
    from a fitted GARCH model.

    A well-specified GARCH model should produce standardized residuals
    that look roughly like white noise (no leftover volatility
    clustering) — this is the standard diagnostic check for model fit.

    Parameters
    ----------
    fitted_model:
        Result from `fit_garch_model`.

    Returns
    -------
    pd.Series
        Standardized residuals, same index as the original returns.
    """
    residuals = fitted_model.resid / fitted_model.conditional_volatility
    residuals.name = "standardized_residuals"
    return residuals


def garch_scaled_returns(returns: pd.Series, fitted_model: ARCHModelResult) -> pd.Series:
    """Rescale historical returns to the current (most recent) GARCH
    conditional volatility, the same idea as EWMA scaling (Phase 5) but
    using GARCH's estimated volatility path instead of a fixed-decay one.

    Parameters
    ----------
    returns:
        The same return series the GARCH model was fitted on.
    fitted_model:
        Result from `fit_garch_model`.

    Returns
    -------
    pd.Series
        Volatility-scaled returns, same index as `returns`.
    """
    conditional_vol = get_conditional_volatility(fitted_model)
    current_vol = conditional_vol.iloc[-1]
    scaled = returns * (current_vol / conditional_vol)
    scaled.name = "garch_scaled_return"
    return scaled


def garch_scaled_historical_var(returns: pd.Series, confidence_level: float = 0.95, p: int = 1, q: int = 1) -> float:
    """Calculate Historical Simulation VaR on GARCH volatility-scaled returns.

    This is the Phase 6 headline output: a VaR figure that reacts to the
    GARCH model's estimate of the current volatility regime.

    Parameters
    ----------
    returns:
        Daily return series.
    confidence_level:
        VaR confidence level.
    p, q:
        GARCH lag orders (see fit_garch_model).

    Returns
    -------
    float
        GARCH-scaled VaR (positive loss magnitude).
    """
    fitted_model = fit_garch_model(returns, p=p, q=q)
    scaled_returns = garch_scaled_returns(returns, fitted_model)
    return calculate_historical_var(scaled_returns, confidence_level)
