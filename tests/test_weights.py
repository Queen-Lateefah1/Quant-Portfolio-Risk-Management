from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.portfolio.weights import (
    equal_weights,
    long_short_weights,
    market_cap_weights,
    user_defined_weights,
)


# --- equal_weights ---

def test_known_answer_equal_weights_two_tickers():
    result = equal_weights(["AAPL", "MSFT"])
    assert result["AAPL"] == pytest.approx(0.5)
    assert result["MSFT"] == pytest.approx(0.5)
    assert result.sum() == pytest.approx(1.0)


def test_normal_case_equal_weights_ten_tickers():
    tickers = [f"T{i}" for i in range(10)]
    result = equal_weights(tickers)
    assert len(result) == 10
    assert result.sum() == pytest.approx(1.0)


def test_edge_case_single_ticker():
    result = equal_weights(["AAPL"])
    assert result["AAPL"] == pytest.approx(1.0)


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        equal_weights([])


# --- user_defined_weights ---

def test_known_answer_user_weights_valid():
    result = user_defined_weights({"AAPL": 0.6, "MSFT": 0.4})
    assert result["AAPL"] == pytest.approx(0.6)
    assert result["MSFT"] == pytest.approx(0.4)


def test_normal_case_user_weights_many_tickers():
    weights = {f"T{i}": 0.1 for i in range(10)}
    result = user_defined_weights(weights)
    assert result.sum() == pytest.approx(1.0)


def test_edge_case_long_short_weights_sum_to_one():
    """Long positions and short positions can coexist as long as net = 1.0."""
    result = user_defined_weights({"AAPL": 1.5, "MSFT": -0.5})
    assert result.sum() == pytest.approx(1.0)


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        user_defined_weights({})


def test_invalid_input_weights_dont_sum_to_one_raises():
    with pytest.raises(ValueError, match="sum to 1.0"):
        user_defined_weights({"AAPL": 0.5, "MSFT": 0.3})


# --- market_cap_weights ---

def _mock_ticker(market_cap):
    mock = MagicMock()
    mock.info = {"marketCap": market_cap} if market_cap is not None else {}
    return mock


def test_normal_case_market_cap_weights():
    with patch("src.portfolio.weights.yf.Ticker") as mock_cls:
        mock_cls.side_effect = lambda t: _mock_ticker({"AAPL": 3_000_000_000_000, "JPM": 500_000_000_000}[t])
        result = market_cap_weights(["AAPL", "JPM"])
        assert result.sum() == pytest.approx(1.0)
        assert result["AAPL"] > result["JPM"]


def test_known_answer_market_cap_proportional():
    with patch("src.portfolio.weights.yf.Ticker") as mock_cls:
        mock_cls.side_effect = lambda t: _mock_ticker({"A": 100.0, "B": 300.0}[t])
        result = market_cap_weights(["A", "B"])
        assert result["A"] == pytest.approx(0.25)
        assert result["B"] == pytest.approx(0.75)


def test_edge_case_excludes_ticker_with_no_market_cap():
    with patch("src.portfolio.weights.yf.Ticker") as mock_cls:
        mock_cls.side_effect = lambda t: _mock_ticker({"A": 100.0, "B": None}[t])
        result = market_cap_weights(["A", "B"])
        assert "B" not in result.index
        assert result["A"] == pytest.approx(1.0)


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        market_cap_weights([])


def test_invalid_input_all_lookups_fail_raises():
    with patch("src.portfolio.weights.yf.Ticker") as mock_cls:
        mock_cls.side_effect = Exception("network error")
        with pytest.raises(ValueError, match="Could not determine market cap"):
            market_cap_weights(["AAPL"])


# --- long_short_weights (placeholder) ---

def test_long_short_not_implemented():
    with pytest.raises(NotImplementedError):
        long_short_weights(["AAPL"], ["TSLA"])
