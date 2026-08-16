import numpy as np
import pandas as pd
import pytest

from src.var.historical import (
    calculate_historical_var,
    calculate_historical_var_multi,
    rolling_historical_var,
)


def test_known_answer_simple_var():
    """100 returns, uniformly -0.01 to 0.01 in 1% steps around a known
    distribution -> 5th percentile is hand-verifiable."""
    returns = pd.Series(np.linspace(-0.10, 0.09, 100))  # sorted ascending
    var_95 = calculate_historical_var(returns, confidence_level=0.95)
    # 5th percentile of returns (index ~5) should be around -0.095
    expected = -returns.quantile(0.05)
    assert var_95 == pytest.approx(expected)


def test_normal_case_var_is_positive():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 500))
    var = calculate_historical_var(returns, confidence_level=0.95)
    assert var > 0


def test_normal_case_higher_confidence_gives_larger_var():
    """Validation check from the doc: more confidence level -> larger VaR."""
    np.random.seed(1)
    returns = pd.Series(np.random.normal(0, 0.02, 500))
    var_95 = calculate_historical_var(returns, confidence_level=0.95)
    var_99 = calculate_historical_var(returns, confidence_level=0.99)
    var_995 = calculate_historical_var(returns, confidence_level=0.995)
    assert var_99 >= var_95
    assert var_995 >= var_99


def test_edge_case_all_identical_returns():
    returns = pd.Series([0.01] * 50)
    var = calculate_historical_var(returns, confidence_level=0.95)
    assert var == pytest.approx(-0.01)


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_historical_var(pd.Series(dtype=float))


def test_invalid_input_bad_confidence_raises():
    returns = pd.Series([0.01, -0.02, 0.03])
    with pytest.raises(ValueError, match="confidence_level"):
        calculate_historical_var(returns, confidence_level=1.5)


# --- calculate_historical_var_multi ---

def test_normal_case_multi_table_shape():
    np.random.seed(2)
    returns = pd.Series(np.random.normal(0, 0.02, 800))
    table = calculate_historical_var_multi(returns, windows=[250, 500], confidence_levels=[0.95, 0.99])
    assert table.shape == (2, 2)
    assert set(table.index) == {250, 500}
    assert set(table.columns) == {0.95, 0.99}


def test_edge_case_window_larger_than_history():
    returns = pd.Series(np.random.normal(0, 0.02, 100))
    table = calculate_historical_var_multi(returns, windows=[250], confidence_levels=[0.95])
    # should not raise, just use all available data
    assert not table.isna().any().any()


def test_invalid_input_empty_windows_raises():
    returns = pd.Series([0.01, -0.02])
    with pytest.raises(ValueError, match="windows cannot be empty"):
        calculate_historical_var_multi(returns, windows=[])


# --- rolling_historical_var ---

def test_normal_case_rolling_var_shape():
    returns = pd.Series(np.random.normal(0, 0.02, 300))
    rolling = rolling_historical_var(returns, window=50, confidence_level=0.95)
    assert len(rolling) == len(returns)
    assert rolling.iloc[:49].isna().all()
    assert rolling.iloc[49:].notna().all()


def test_edge_case_minimum_window():
    returns = pd.Series([0.01, -0.02, 0.005, -0.01, 0.02])
    rolling = rolling_historical_var(returns, window=2, confidence_level=0.95)
    assert rolling.iloc[0] != rolling.iloc[0]  # NaN check


def test_invalid_input_window_too_small_raises():
    returns = pd.Series([0.01, -0.02])
    with pytest.raises(ValueError, match="window must be"):
        rolling_historical_var(returns, window=1)
