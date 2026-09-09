import numpy as np
import pandas as pd
import pytest

from src.stress_testing.volatility_shocks import (
    apply_volatility_doubling_shock,
    apply_correlation_spike_shock,
)


def _sample_returns(n=500, seed=1):
    np.random.seed(seed)
    return pd.Series(np.random.normal(0, 0.015, n), index=pd.date_range("2023-01-01", periods=n))


# --- apply_volatility_doubling_shock ---

def test_known_answer_var_scales_linearly_with_multiplier():
    returns = _sample_returns()
    result = apply_volatility_doubling_shock(returns, vol_multiplier=2.0)
    assert result.stressed_var == pytest.approx(result.base_var * 2.0)
    assert result.stressed_es == pytest.approx(result.base_es * 2.0)


def test_normal_case_multiplier_one_leaves_var_unchanged():
    returns = _sample_returns()
    result = apply_volatility_doubling_shock(returns, vol_multiplier=1.0)
    assert result.stressed_var == pytest.approx(result.base_var)


def test_edge_case_multiplier_less_than_one_reduces_var():
    returns = _sample_returns()
    result = apply_volatility_doubling_shock(returns, vol_multiplier=0.5)
    assert result.stressed_var < result.base_var


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        apply_volatility_doubling_shock(pd.Series(dtype=float))


def test_invalid_input_negative_multiplier_raises():
    returns = _sample_returns()
    with pytest.raises(ValueError, match="must be positive"):
        apply_volatility_doubling_shock(returns, vol_multiplier=-1.0)


# --- apply_correlation_spike_shock ---

def test_known_answer_perfect_correlation_equals_weighted_vol_sum():
    """With correlation = 1 across the board, portfolio volatility should
    equal the weighted SUM of individual volatilities (no diversification
    benefit at all) — a hand-verifiable closed-form case."""
    np.random.seed(2)
    n = 500
    a = pd.Series(np.random.normal(0, 0.02, n))
    b = pd.Series(np.random.normal(0, 0.03, n))
    returns_matrix = pd.DataFrame({"A": a, "B": b})
    weights = pd.Series({"A": 0.5, "B": 0.5})

    result = apply_correlation_spike_shock(returns_matrix, weights, stressed_correlation=1.0)
    expected_vol = 0.5 * a.std() + 0.5 * b.std()
    assert result.stressed_portfolio_vol == pytest.approx(expected_vol, rel=1e-6)


def test_normal_case_low_base_correlation_shows_diversification_loss():
    """Two genuinely uncorrelated assets stressed to high correlation
    should show INCREASED portfolio volatility (diversification benefit
    disappearing)."""
    np.random.seed(3)
    n = 500
    a = pd.Series(np.random.normal(0, 0.02, n))
    b = pd.Series(np.random.normal(0, 0.02, n))  # independent of a
    returns_matrix = pd.DataFrame({"A": a, "B": b})
    weights = pd.Series({"A": 0.5, "B": 0.5})

    result = apply_correlation_spike_shock(returns_matrix, weights, stressed_correlation=0.95)
    assert result.stressed_portfolio_vol > result.base_portfolio_vol
    assert result.stressed_var > result.base_var


def test_edge_case_already_perfectly_correlated_shows_no_change():
    """If assets already move in lockstep, stressing correlation further
    to the same value should barely change anything."""
    np.random.seed(4)
    n = 300
    a = pd.Series(np.random.normal(0, 0.02, n))
    b = a * 1.0  # perfectly correlated by construction
    returns_matrix = pd.DataFrame({"A": a, "B": b})
    weights = pd.Series({"A": 0.5, "B": 0.5})

    result = apply_correlation_spike_shock(returns_matrix, weights, stressed_correlation=0.999)
    assert result.vol_scale_factor == pytest.approx(1.0, abs=0.05)


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        apply_correlation_spike_shock(pd.DataFrame(), pd.Series({"A": 1.0}))


def test_invalid_input_bad_correlation_raises():
    returns_matrix = pd.DataFrame({"A": [0.01, 0.02], "B": [0.01, -0.01]})
    weights = pd.Series({"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError, match="must be in \\[-1, 1\\]"):
        apply_correlation_spike_shock(returns_matrix, weights, stressed_correlation=1.5)


def test_invalid_input_weight_ticker_not_in_matrix_raises():
    returns_matrix = pd.DataFrame({"A": [0.01, 0.02]})
    weights = pd.Series({"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError, match="not in returns_matrix"):
        apply_correlation_spike_shock(returns_matrix, weights)
