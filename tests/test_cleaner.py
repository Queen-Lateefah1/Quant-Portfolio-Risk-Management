import numpy as np
import pandas as pd
import pytest

from src.data_cleaning.cleaner import remove_stale_prices, winsorize_returns


# --- remove_stale_prices ---

def test_normal_case_no_stale_prices():
    dates = pd.date_range("2026-01-01", periods=5)
    prices = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates, dtype=float)
    cleaned, report = remove_stale_prices(prices, min_repeat_run=3)
    assert cleaned.equals(prices)
    assert report.values_replaced == 0


def test_known_answer_detects_stale_run():
    """5 identical values in a row should be flagged and forward-filled from the prior real price."""
    dates = pd.date_range("2026-01-01", periods=7)
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 101.0, 101.0, 101.0, 101.0, 105.0]}, index=dates
    )
    cleaned, report = remove_stale_prices(prices, min_repeat_run=5)
    assert report.stale_runs_found.get("AAPL", 0) > 0
    assert cleaned["AAPL"].iloc[-1] == 105.0  # unaffected real move preserved


def test_edge_case_short_repeat_not_flagged():
    """A repeat run shorter than min_repeat_run should NOT be treated as stale."""
    dates = pd.date_range("2026-01-01", periods=4)
    prices = pd.DataFrame({"AAPL": [100.0, 101.0, 101.0, 102.0]}, index=dates)
    cleaned, report = remove_stale_prices(prices, min_repeat_run=5)
    assert report.values_replaced == 0
    assert cleaned.equals(prices)


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        remove_stale_prices(pd.DataFrame())


def test_invalid_input_bad_min_repeat_run_raises():
    prices = pd.DataFrame({"AAPL": [100.0, 101.0]})
    with pytest.raises(ValueError, match="min_repeat_run"):
        remove_stale_prices(prices, min_repeat_run=1)


# --- winsorize_returns ---

def test_normal_case_no_outliers_unchanged_within_bounds():
    dates = pd.date_range("2026-01-01", periods=10)
    returns = pd.DataFrame({"AAPL": np.linspace(-0.01, 0.01, 10)}, index=dates)
    winsorized, report = winsorize_returns(returns, lower_percentile=0.01, upper_percentile=0.99)
    assert winsorized.shape == returns.shape


def test_known_answer_extreme_value_clipped():
    dates = pd.date_range("2026-01-01", periods=10)
    values = [0.01] * 9 + [5.0]  # one obvious fat-finger outlier
    returns = pd.DataFrame({"AAPL": values}, index=dates)
    winsorized, report = winsorize_returns(returns, lower_percentile=0.01, upper_percentile=0.90)
    assert winsorized["AAPL"].max() < 5.0
    assert report.values_clipped_high.get("AAPL", 0) >= 1


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        winsorize_returns(pd.DataFrame())


def test_invalid_input_bad_percentiles_raises():
    returns = pd.DataFrame({"AAPL": [0.01, 0.02]})
    with pytest.raises(ValueError, match="lower_percentile"):
        winsorize_returns(returns, lower_percentile=0.9, upper_percentile=0.1)
