import numpy as np
import pandas as pd
import pytest

from src.data_collection.validator import clean_prices, validate_prices
from src.data_collection.returns import calculate_log_returns


def test_normal_case_no_missing_values():
    dates = pd.date_range("2026-01-01", periods=5)
    prices = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates, dtype=float)
    cleaned, report = clean_prices(prices)
    assert report.missing_values_before == 0
    assert report.tickers_dropped == []
    assert cleaned.equals(prices)


def test_known_answer_fills_isolated_gap():
    """A single NaN between two known prices should be forward-filled,
    and the report must say exactly how many values were filled."""
    dates = pd.date_range("2026-01-01", periods=4)
    prices = pd.DataFrame({"AAPL": [100.0, np.nan, 102.0, 103.0]}, index=dates)
    cleaned, report = clean_prices(prices)
    assert cleaned["AAPL"].iloc[1] == 100.0  # forward-filled
    assert report.missing_values_before == 1
    assert report.missing_values_filled == 1
    assert report.missing_values_remaining == 0


def test_edge_case_drops_fully_empty_ticker():
    dates = pd.date_range("2026-01-01", periods=3)
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0], "BROKEN": [np.nan, np.nan, np.nan]}, index=dates
    )
    cleaned, report = clean_prices(prices)
    assert "BROKEN" in report.tickers_dropped
    assert "BROKEN" not in cleaned.columns


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        clean_prices(pd.DataFrame())


def test_validation_passes_on_clean_data():
    dates = pd.date_range("2026-01-01", periods=5)
    prices = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates, dtype=float)
    returns = calculate_log_returns(prices)
    result = validate_prices(prices, returns)
    assert result.passed is True
    assert result.missing_values == 0


def test_validation_fails_on_duplicate_dates():
    dates = pd.DatetimeIndex(["2026-01-01", "2026-01-01", "2026-01-02"])
    prices = pd.DataFrame({"AAPL": [100.0, 100.5, 101.0]}, index=dates)
    returns = calculate_log_returns(prices)
    result = validate_prices(prices, returns)
    assert result.has_duplicate_dates is True
    assert result.passed is False
