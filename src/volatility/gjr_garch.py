"""
gjr_garch.py

Responsibility: GJR-GARCH volatility modeling and GJR-GARCH-scaled
Historical VaR.

Plain GARCH (Phase 6) treats a positive and a negative shock of the
same size as equally impactful on future volatility. In equities this
isn't realistic: a large drop tends to increase future volatility more
than an equally large rally does (the "leverage effect" — named because
a falling stock price mechanically raises a firm's leverage ratio,
though the effect is now understood more broadly as asymmetric risk
appetite/panic). GJR-GARCH adds one extra term (gamma) that only
activates when the shock is negative, capturing this asymmetry
directly.

Conditional variance recursion:
    sigma2_t = omega + alpha * eps_{t-1}^2 + gamma * eps_{t-1}^2 * I(eps_{t-1} < 0) + beta * sigma2_{t-1}

gamma > 0 means negative shocks add extra variance beyond what a
symmetric GARCH would predict — this is the "leverage effect
parameter" the doc asks for.
"""

import logging

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from src.var.historical import calculate_historical_var

logger = logging.getLogger(__name__)


def fit_gjr_garch_model(returns: pd.Series, p: int = 1, o: int = 1, q: int = 1) -> ARCHModelResult:
    """Fit a GJR-GARCH(p, o, q) model to a return series.

    Same x100 internal rescaling as garch.py, for the same numerical
    reason (the optimizer behaves better on percent-scale returns).

    Parameters
    ----------
    returns:
        Daily return series (decimal form).
    p:
        GARCH lag order.
    o:
        Asymmetry (leverage) lag order. o=1 is what makes this
        GJR-GARCH rather than plain GARCH.
    q:
        ARCH lag order.

    Returns
    -------
    ARCHModelResult
        The fitted model.

    Raises
    ------
    ValueError
        If `returns` is empty or has fewer than 30 observations.
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")
    if len(returns) < 30:
        raise ValueError("returns must have at least 30 observations for reliable GJR-GARCH fitting")

    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol="Garch", p=p, o=o, q=q, dist="normal")
    fitted = model.fit(disp="off")

    logger.info(
        "Fitted GJR-GARCH(%d,%d,%d): omega=%.6f, alpha=%.4f, gamma=%.4f, beta=%.4f",
        p, o, q,
        fitted.params.get("omega", float("nan")),
        fitted.params.get("alpha[1]", float("nan")),
        fitted.params.get("gamma[1]", float("nan")),
        fitted.params.get("beta[1]", float("nan")),
    )
    return fitted


def get_conditional_volatility(fitted_model: ARCHModelResult) -> pd.Series:
    """Extract the in-sample conditional volatility path from a fitted
    GJR-GARCH model. Same shape/convention as garch.get_conditional_volatility.
    """
    conditional_vol = fitted_model.conditional_volatility / 100
    conditional_vol.name = "gjr_garch_conditional_volatility"
    return conditional_vol


def get_leverage_effect_parameter(fitted_model: ARCHModelResult) -> float:
    """Extract the leverage effect (asymmetry) parameter, gamma.

    Parameters
    ----------
    fitted_model:
        Result from `fit_gjr_garch_model`.

    Returns
    -------
    float
        The fitted gamma[1] coefficient. gamma > 0 means negative
        shocks increase future volatility more than positive shocks of
        the same magnitude (the expected sign for equities). gamma
        close to 0 means little/no asymmetry was detected — in that
        case plain GARCH would fit about as well.

    Raises
    ------
    KeyError
        If the fitted model has no gamma parameter (i.e. it wasn't
        actually fit with o >= 1).
    """
    if "gamma[1]" not in fitted_model.params.index:
        raise KeyError(
            "fitted_model has no gamma[1] parameter — was it fit with o=0? "
            "GJR-GARCH requires o >= 1."
        )
    return float(fitted_model.params["gamma[1]"])


def forecast_volatility(fitted_model: ARCHModelResult, horizon: int = 1) -> pd.Series:
    """Forecast conditional volatility forward from a fitted GJR-GARCH model.

    Same interface as garch.forecast_volatility.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    forecast = fitted_model.forecast(horizon=horizon, reindex=False)
    variance_forecast = forecast.variance.iloc[-1]
    vol_forecast = np.sqrt(variance_forecast) / 100
    vol_forecast.index = range(1, horizon + 1)
    vol_forecast.name = "gjr_garch_forecast_volatility"
    return vol_forecast


def gjr_garch_scaled_returns(returns: pd.Series, fitted_model: ARCHModelResult) -> pd.Series:
    """Rescale historical returns to the current GJR-GARCH conditional volatility.

    Same idea as garch_scaled_returns, using the asymmetric volatility
    path instead of the symmetric GARCH one.
    """
    conditional_vol = get_conditional_volatility(fitted_model)
    current_vol = conditional_vol.iloc[-1]
    scaled = returns * (current_vol / conditional_vol)
    scaled.name = "gjr_garch_scaled_return"
    return scaled


def gjr_garch_scaled_historical_var(
    returns: pd.Series, confidence_level: float = 0.95, p: int = 1, o: int = 1, q: int = 1
) -> float:
    """Calculate Historical Simulation VaR on GJR-GARCH volatility-scaled returns.

    This is the Phase 7 headline output.

    Parameters
    ----------
    returns:
        Daily return series.
    confidence_level:
        VaR confidence level.
    p, o, q:
        GJR-GARCH lag orders (see fit_gjr_garch_model).

    Returns
    -------
    float
        GJR-GARCH-scaled VaR (positive loss magnitude).
    """
    fitted_model = fit_gjr_garch_model(returns, p=p, o=o, q=q)
    scaled_returns = gjr_garch_scaled_returns(returns, fitted_model)
    return calculate_historical_var(scaled_returns, confidence_level)
