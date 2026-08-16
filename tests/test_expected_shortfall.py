import numpy as np
import pandas as pd
import pytest

from src.var.expected_shortfall import calculate_expected_shortfall, rolling_expected_shortfall
from src.var.historical import calculate_historical_var


def test_known_answer_es_averages_tail():
    """Hand-verifiable: returns where the worst 5 out of 100 are known exactly."""
    good_returns = [0.01] * 95
    bad_returns = [-0.10, -0.09, -0.08, -0.07, -0.06]
    returns = pd.Series(good_returns + bad_returns)
    es = calculate_expected_shortfall(returns, confidence_level=0.95)
    # tail should be roughly the worst ~5 observations
    expected_tail_mean = np.mean([0.10, 0.09, 0.08, 0.07, 0.06])
    assert es == pytest.approx(expected_tail_mean, rel=0.2)


def test_normal_case_es_greater_than_or_equal_var():
    """Validation check from doc: Expected Shortfall > VaR."""
    np.random.seed(3)
    returns = pd.Series(np.random.normal(0, 0.02, 500))
    var = calculate_historical_var(returns, confidence_level=0.95)
    es = calculate_expected_shortfall(returns, confidence_level=0.95)
    assert es >= var


def test_edge_case_small_sample():
    returns = pd.Series([0.01, -0.02, 0.005, -0.03, 0.02, -0.01, 0.015, -0.025, 0.03, -0.04])
    es = calculate_expected_shortfall(returns, confidence_level=0.8)
    assert es > 0


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_expected_shortfall(pd.Series(dtype=float))


def test_invalid_input_bad_confidence_raises():
    returns = pd.Series([0.01, -0.02, 0.03])
    with pytest.raises(ValueError, match="confidence_level"):
        calculate_expected_shortfall(returns, confidence_level=0)


# --- rolling_expected_shortfall ---

def test_normal_case_rolling_es_shape():
    np.random.seed(4)
    returns = pd.Series(np.random.normal(0, 0.02, 200))
    rolling = rolling_expected_shortfall(returns, window=50, confidence_level=0.95)
    assert len(rolling) == len(returns)
    assert rolling.iloc[49:].notna().all()


def test_invalid_input_window_too_small_raises():
    returns = pd.Series([0.01, -0.02])
    with pytest.raises(ValueError, match="window must be"):
        rolling_expected_shortfall(returns, window=1)
