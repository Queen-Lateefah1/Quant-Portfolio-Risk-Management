import numpy as np
import pandas as pd
import pytest

from src.volatility.gjr_garch import (
    fit_gjr_garch_model,
    get_conditional_volatility,
    get_leverage_effect_parameter,
    forecast_volatility,
    gjr_garch_scaled_returns,
    gjr_garch_scaled_historical_var,
)
from src.volatility import gjr_garch as gjr_module
from src.var.historical import calculate_historical_var


def _simulated_gjr_returns(n=600, seed=42, omega=0.00001, alpha=0.03, gamma=0.10, beta=0.88):
    """Simulate a genuine GJR-GARCH process with a known positive
    leverage effect (gamma > 0), so tests check the fitter recovers the
    right SIGN and rough scale — not just that it runs without error."""
    np.random.seed(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - gamma / 2 - beta)
    for t in range(1, n):
        shock_sq = returns[t - 1] ** 2
        asym = shock_sq if returns[t - 1] < 0 else 0.0
        sigma2[t] = omega + alpha * shock_sq + gamma * asym + beta * sigma2[t - 1]
        returns[t] = np.random.normal(0, np.sqrt(sigma2[t]))
    dates = pd.date_range("2023-01-01", periods=n)
    return pd.Series(returns, index=dates)


@pytest.fixture
def gjr_returns():
    return _simulated_gjr_returns()


# --- fit_gjr_garch_model ---

def test_normal_case_fits_without_error(gjr_returns):
    fitted = fit_gjr_garch_model(gjr_returns)
    assert fitted is not None
    assert "gamma[1]" in fitted.params.index


def test_edge_case_minimum_sample_size():
    returns = _simulated_gjr_returns(n=30)
    fitted = fit_gjr_garch_model(returns)
    assert fitted is not None


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        fit_gjr_garch_model(pd.Series(dtype=float))


def test_invalid_input_too_short_raises():
    returns = pd.Series(np.random.normal(0, 0.02, 10))
    with pytest.raises(ValueError, match="at least 30 observations"):
        fit_gjr_garch_model(returns)


# --- get_conditional_volatility ---

def test_normal_case_conditional_volatility_positive(gjr_returns):
    fitted = fit_gjr_garch_model(gjr_returns)
    vol = get_conditional_volatility(fitted)
    assert (vol > 0).all()
    assert len(vol) == len(gjr_returns)


# --- get_leverage_effect_parameter ---

def test_known_answer_recovers_positive_leverage_effect(gjr_returns):
    """The simulated data has a genuine positive leverage effect
    (gamma=0.10 baked in) — the fitted model should recover a positive
    gamma, confirming asymmetry is actually detected, not just present
    as an unused parameter."""
    fitted = fit_gjr_garch_model(gjr_returns)
    leverage = get_leverage_effect_parameter(fitted)
    assert leverage > 0


def test_edge_case_no_asymmetry_gives_small_gamma():
    """Data simulated from plain (symmetric) GARCH, with no leverage
    effect at all, should fit a gamma close to zero."""
    np.random.seed(2)
    n = 500
    omega, alpha, beta = 0.00001, 0.08, 0.90
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = np.random.normal(0, np.sqrt(sigma2[t]))
    returns_series = pd.Series(returns, index=pd.date_range("2023-01-01", periods=n))

    fitted = fit_gjr_garch_model(returns_series)
    leverage = get_leverage_effect_parameter(fitted)
    assert abs(leverage) < 0.15  # should be small/insignificant, not necessarily exactly 0


def test_invalid_input_missing_gamma_raises():
    """A plain GARCH fit (o=0, no asymmetry term) should not have a
    gamma parameter to extract."""
    returns = _simulated_gjr_returns()
    from arch import arch_model
    plain_fit = arch_model(returns * 100, vol="Garch", p=1, o=0, q=1, dist="normal").fit(disp="off")
    with pytest.raises(KeyError, match="gamma"):
        get_leverage_effect_parameter(plain_fit)


# --- forecast_volatility ---

def test_normal_case_forecast_shape(gjr_returns):
    fitted = fit_gjr_garch_model(gjr_returns)
    forecast = forecast_volatility(fitted, horizon=5)
    assert len(forecast) == 5
    assert (forecast > 0).all()


def test_invalid_input_bad_horizon_raises(gjr_returns):
    fitted = fit_gjr_garch_model(gjr_returns)
    with pytest.raises(ValueError, match="horizon must be"):
        forecast_volatility(fitted, horizon=0)


# --- gjr_garch_scaled_returns ---

def test_known_answer_scaling_last_point_unchanged(gjr_returns):
    fitted = fit_gjr_garch_model(gjr_returns)
    scaled = gjr_garch_scaled_returns(gjr_returns, fitted)
    assert scaled.iloc[-1] == pytest.approx(gjr_returns.iloc[-1])


# --- gjr_garch_scaled_historical_var ---

def test_normal_case_var_positive(gjr_returns):
    var = gjr_garch_scaled_historical_var(gjr_returns, confidence_level=0.95)
    assert var > 0


def test_normal_case_higher_confidence_gives_larger_var(gjr_returns):
    var_95 = gjr_garch_scaled_historical_var(gjr_returns, confidence_level=0.95)
    var_99 = gjr_garch_scaled_historical_var(gjr_returns, confidence_level=0.99)
    assert var_99 >= var_95


def test_normal_case_reacts_to_recent_volatility_spike():
    calm = _simulated_gjr_returns(n=300, seed=1, omega=0.000005, alpha=0.02, gamma=0.06, beta=0.90)
    spike = np.random.normal(0, 0.06, 30)
    combined = pd.Series(
        np.concatenate([calm.values, spike]),
        index=pd.date_range("2023-01-01", periods=330),
    )
    plain_var = calculate_historical_var(combined, confidence_level=0.95)
    gjr_var = gjr_garch_scaled_historical_var(combined, confidence_level=0.95)
    assert gjr_var > plain_var
