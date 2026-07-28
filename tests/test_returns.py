import numpy as np
import pandas as pd
import pytest

from src.data_collection.returns import calculate_log_returns


def test_known_answer_single_asset():
    """Hand-verifiable case: prices 100 -> 105 should give log(1.05)."""
    prices = pd.DataFrame({"AAPL": [100.0, 105.0]}, index=pd.date_range("2026-01-01", periods=2))
    result = calculate_log_returns(prices)
    expected = np.log(105.0 / 100.0)
    assert result.iloc[0]["AAPL"] == pytest.approx(expected)


def test_normal_case_multi_asset():
    """Normal case: multiple tickers, multiple days, correct shape."""
    dates = pd.date_range("2026-01-01", periods=5)
    prices = pd.DataFrame(
        {"AAPL": [100, 101, 99, 102, 103], "MSFT": [200, 202, 198, 204, 206]},
        index=dates,
    )
    result = calculate_log_returns(prices)
    assert result.shape == (4, 2)  # one fewer row than input
    assert list(result.columns) == ["AAPL", "MSFT"]


def test_edge_case_two_rows_minimum():
    """Edge case: minimum viable input is two price observations."""
    prices = pd.DataFrame({"AAPL": [100.0, 100.0]}, index=pd.date_range("2026-01-01", periods=2))
    result = calculate_log_returns(prices)
    assert len(result) == 1
    assert result.iloc[0]["AAPL"] == pytest.approx(0.0)


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_log_returns(pd.DataFrame())


def test_invalid_input_non_positive_raises():
    prices = pd.DataFrame({"AAPL": [100.0, -5.0]}, index=pd.date_range("2026-01-01", periods=2))
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_log_returns(prices)
