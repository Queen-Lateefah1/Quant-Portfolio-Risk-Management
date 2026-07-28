import numpy as np
import pandas as pd
import pytest

from src.data_cleaning.alignment import align_trading_dates


def test_normal_case_already_aligned():
    dates = pd.date_range("2026-01-01", periods=5)
    prices = pd.DataFrame({"AAPL": range(100, 105), "MSFT": range(200, 205)}, index=dates, dtype=float)
    aligned, report = align_trading_dates(prices)
    assert aligned.equals(prices)
    assert report.dates_dropped == 0


def test_known_answer_trims_to_common_range():
    """MSFT starts 2 days later than AAPL -> common range should exclude those 2 days."""
    dates = pd.date_range("2026-01-01", periods=5)
    aapl = [100.0, 101.0, 102.0, 103.0, 104.0]
    msft = [np.nan, np.nan, 200.0, 201.0, 202.0]
    prices = pd.DataFrame({"AAPL": aapl, "MSFT": msft}, index=dates)
    aligned, report = align_trading_dates(prices)
    assert len(aligned) == 3
    assert report.dates_dropped == 2


def test_edge_case_fills_internal_gap():
    dates = pd.date_range("2026-01-01", periods=4)
    prices = pd.DataFrame({"AAPL": [100.0, np.nan, 102.0, 103.0]}, index=dates)
    aligned, report = align_trading_dates(prices)
    assert aligned["AAPL"].isna().sum() == 0
    assert "AAPL" in report.tickers_with_gaps_filled


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        align_trading_dates(pd.DataFrame())
