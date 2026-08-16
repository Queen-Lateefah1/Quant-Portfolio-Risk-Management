import numpy as np
import pandas as pd
import pytest

from src.volatility.ewma import (
    ewma_variance,
    ewma_volatility,
    ewma_scaled_returns,
    ewma_scaled_historical_var,
)
from src.var.historical import calculate_historical_var


def test_known_answer_ewma_variance_first_value():
    """First EWMA variance value should equal the first squared return
    (no prior history to decay from)."""
    returns = pd.Series([0.02, 0.01, -0.015, 0.005])
    variance = ewma_variance(returns, lambda_=0.94)
    assert variance.iloc[0] == pytest.approx(0.02**2)


def test_known_answer_ewma_variance_recursion():
    """Second value should match the recursive formula exactly."""
    returns = pd.Series([0.02, 0.01, -0.015, 0.005])
    variance = ewma_variance(returns, lambda_=0.94)
    expected_second = 0.94 * (0.02**2) + 0.06 * (0.01**2)
    assert variance.iloc[1] == pytest.approx(expected_second)


def test_normal_case_volatility_is_sqrt_of_variance():
    np.random.seed(5)
    returns = pd.Series(np.random.normal(0, 0.02, 100))
    variance = ewma_variance(returns)
    volatility = ewma_volatility(returns)
    assert (volatility**2).equals(variance) or np.allclose(volatility**2, variance)


def test_edge_case_constant_returns_gives_constant_variance():
    returns = pd.Series([0.01] * 20)
    variance = ewma_variance(returns)
    assert np.allclose(variance, 0.01**2)


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        ewma_variance(pd.Series(dtype=float))


def test_invalid_input_bad_lambda_raises():
    returns = pd.Series([0.01, -0.02])
    with pytest.raises(ValueError, match="lambda_"):
        ewma_variance(returns, lambda_=1.5)


# --- ewma_scaled_returns ---

def test_known_answer_scaling_last_point_unchanged():
    """The most recent return, scaled to its OWN current volatility,
    should be unchanged (ratio = 1)."""
    np.random.seed(6)
    returns = pd.Series(np.random.normal(0, 0.02, 50))
    scaled = ewma_scaled_returns(returns)
    assert scaled.iloc[-1] == pytest.approx(returns.iloc[-1])


def test_normal_case_scaling_amplifies_calm_period_in_volatile_regime():
    """If volatility trends upward, early (calm) returns should be
    amplified when scaled to the (higher) current volatility."""
    calm = [0.001] * 30
    volatile = [0.05, -0.05, 0.04, -0.04] * 5
    returns = pd.Series(calm + volatile)
    scaled = ewma_scaled_returns(returns)
    # an early calm return should be scaled up in magnitude
    assert abs(scaled.iloc[5]) >= abs(returns.iloc[5])


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        ewma_scaled_returns(pd.Series(dtype=float))


# --- ewma_scaled_historical_var ---

def test_normal_case_ewma_var_is_positive():
    np.random.seed(7)
    returns = pd.Series(np.random.normal(0, 0.02, 300))
    var = ewma_scaled_historical_var(returns, confidence_level=0.95)
    assert var > 0


def test_normal_case_ewma_var_reacts_to_recent_volatility_spike():
    """A recent volatility spike should push the EWMA-scaled VaR above
    the plain historical VaR computed on the same data."""
    np.random.seed(8)
    calm = np.random.normal(0, 0.005, 200)
    spike = np.random.normal(0, 0.05, 20)
    returns = pd.Series(np.concatenate([calm, spike]))

    plain_var = calculate_historical_var(returns, confidence_level=0.95)
    ewma_var = ewma_scaled_historical_var(returns, confidence_level=0.95)

    assert ewma_var > plain_var


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        ewma_scaled_historical_var(pd.Series(dtype=float))
