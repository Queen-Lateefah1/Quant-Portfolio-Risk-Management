import numpy as np
import pandas as pd
import pytest

from src.volatility.garch import (
    fit_garch_model,
    get_conditional_volatility,
    forecast_volatility,
    get_standardized_residuals,
    garch_scaled_returns,
    garch_scaled_historical_var,
)
from src.var.historical import calculate_historical_var


def _simulated_garch_returns(n=500, seed=42, omega=0.00001, alpha=0.08, beta=0.90):
    """Simulate a genuine GARCH(1,1) process so tests check the fitter
    recovers sensible, well-behaved output — not just that it runs."""
    np.random.seed(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = np.random.normal(0, np.sqrt(sigma2[t]))
    dates = pd.date_range("2023-01-01", periods=n)
    return pd.Series(returns, index=dates)


@pytest.fixture
def garch_returns():
    return _simulated_garch_returns()


# --- fit_garch_model ---

def test_normal_case_fits_without_error(garch_returns):
    fitted = fit_garch_model(garch_returns)
    assert fitted is not None
    assert "omega" in fitted.params.index


def test_edge_case_minimum_sample_size():
    returns = _simulated_garch_returns(n=30)
    fitted = fit_garch_model(returns)
    assert fitted is not None


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        fit_garch_model(pd.Series(dtype=float))


def test_invalid_input_too_short_raises():
    returns = pd.Series(np.random.normal(0, 0.02, 10))
    with pytest.raises(ValueError, match="at least 30 observations"):
        fit_garch_model(returns)


# --- get_conditional_volatility ---

def test_normal_case_conditional_volatility_positive(garch_returns):
    fitted = fit_garch_model(garch_returns)
    vol = get_conditional_volatility(fitted)
    assert (vol > 0).all()
    assert len(vol) == len(garch_returns)


def test_known_answer_volatility_clusters_with_known_regime():
    """A clearly two-regime series (calm then volatile) should show
    conditional volatility rising in the volatile segment."""
    calm = np.random.normal(0, 0.002, 200)
    volatile = np.random.normal(0, 0.04, 200)
    returns = pd.Series(np.concatenate([calm, volatile]),
                         index=pd.date_range("2023-01-01", periods=400))
    fitted = fit_garch_model(returns)
    vol = get_conditional_volatility(fitted)
    assert vol.iloc[-50:].mean() > vol.iloc[:50].mean()


# --- forecast_volatility ---

def test_normal_case_forecast_shape(garch_returns):
    fitted = fit_garch_model(garch_returns)
    forecast = forecast_volatility(fitted, horizon=5)
    assert len(forecast) == 5
    assert (forecast > 0).all()


def test_edge_case_single_day_forecast(garch_returns):
    fitted = fit_garch_model(garch_returns)
    forecast = forecast_volatility(fitted, horizon=1)
    assert len(forecast) == 1


def test_invalid_input_bad_horizon_raises(garch_returns):
    fitted = fit_garch_model(garch_returns)
    with pytest.raises(ValueError, match="horizon must be"):
        forecast_volatility(fitted, horizon=0)


# --- get_standardized_residuals ---

def test_normal_case_standardized_residuals_unit_variance(garch_returns):
    """A well-fit GARCH model's standardized residuals should have
    variance close to 1 (that's the whole point of standardizing)."""
    fitted = fit_garch_model(garch_returns)
    resid = get_standardized_residuals(fitted)
    assert resid.std() == pytest.approx(1.0, abs=0.15)


# --- garch_scaled_returns ---

def test_known_answer_scaling_last_point_unchanged(garch_returns):
    fitted = fit_garch_model(garch_returns)
    scaled = garch_scaled_returns(garch_returns, fitted)
    assert scaled.iloc[-1] == pytest.approx(garch_returns.iloc[-1])


# --- garch_scaled_historical_var ---

def test_normal_case_garch_var_positive(garch_returns):
    var = garch_scaled_historical_var(garch_returns, confidence_level=0.95)
    assert var > 0


def test_normal_case_higher_confidence_gives_larger_var(garch_returns):
    var_95 = garch_scaled_historical_var(garch_returns, confidence_level=0.95)
    var_99 = garch_scaled_historical_var(garch_returns, confidence_level=0.99)
    assert var_99 >= var_95


def test_normal_case_garch_var_reacts_to_recent_volatility_spike():
    """Same validation logic as EWMA: a recent volatility spike should
    push the GARCH-scaled VaR above the plain historical VaR."""
    calm = _simulated_garch_returns(n=300, seed=1, omega=0.000005, alpha=0.05, beta=0.90)
    spike = np.random.normal(0, 0.06, 30)
    combined = pd.Series(
        np.concatenate([calm.values, spike]),
        index=pd.date_range("2023-01-01", periods=330),
    )
    plain_var = calculate_historical_var(combined, confidence_level=0.95)
    garch_var = garch_scaled_historical_var(combined, confidence_level=0.95)
    assert garch_var > plain_var
