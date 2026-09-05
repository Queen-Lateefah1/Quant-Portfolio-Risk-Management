import numpy as np
import pandas as pd
import pytest

from src.backtesting.exceptions import identify_exceptions, summarize_exceptions
from src.backtesting.kupiec import kupiec_pof_test
from src.backtesting.christoffersen import (
    christoffersen_independence_test,
    christoffersen_conditional_coverage_test,
)
from src.backtesting.basel import basel_traffic_light
from src.backtesting.report import run_backtest
from src.var.historical import rolling_historical_var


# --- exceptions.py ---

def test_known_answer_exception_identified_correctly():
    dates = pd.date_range("2024-01-01", periods=4)
    returns = pd.Series([0.01, -0.05, 0.02, -0.01], index=dates)
    var = pd.Series([0.02, 0.02, 0.02, 0.02], index=dates)
    exceptions = identify_exceptions(returns, var)
    assert list(exceptions) == [False, True, False, False]


def test_normal_case_static_var_float():
    returns = pd.Series([0.01, -0.05, 0.02, -0.01])
    exceptions = identify_exceptions(returns, 0.02)
    assert list(exceptions) == [False, True, False, False]


def test_edge_case_loss_exactly_equal_to_var_not_exception():
    """A loss exactly at the VaR threshold is NOT an exception (strict inequality)."""
    returns = pd.Series([-0.02])
    exceptions = identify_exceptions(returns, 0.02)
    assert exceptions.iloc[0] == False


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        identify_exceptions(pd.Series(dtype=float), 0.02)


def test_normal_case_exception_summary_matches_expected_rate():
    np.random.seed(1)
    returns = pd.Series(np.random.normal(0, 0.02, 1000))
    var = 0.033  # roughly the 95% VaR for this distribution
    exceptions = identify_exceptions(returns, var)
    summary = summarize_exceptions(exceptions, confidence_level=0.95)
    assert summary.n_observations == 1000
    assert summary.expected_ratio == pytest.approx(0.05)
    assert summary.exception_ratio == pytest.approx(0.05, abs=0.03)


# --- kupiec.py ---

def test_known_answer_kupiec_perfect_calibration_passes():
    """Exactly the expected number of exceptions should give a high p-value."""
    result = kupiec_pof_test(n_observations=1000, n_exceptions=50, confidence_level=0.95)
    assert result.passed is True
    assert result.p_value > 0.5


def test_normal_case_kupiec_gross_miscalibration_fails():
    result = kupiec_pof_test(n_observations=1000, n_exceptions=200, confidence_level=0.95)
    assert result.passed is False
    assert result.p_value < 0.01


def test_edge_case_zero_exceptions():
    result = kupiec_pof_test(n_observations=100, n_exceptions=0, confidence_level=0.95)
    assert result.lr_statistic >= 0
    assert not np.isnan(result.p_value)


def test_edge_case_all_exceptions():
    result = kupiec_pof_test(n_observations=10, n_exceptions=10, confidence_level=0.95)
    assert result.lr_statistic >= 0
    assert not np.isnan(result.p_value)


def test_invalid_input_negative_exceptions_raises():
    with pytest.raises(ValueError, match="cannot be negative"):
        kupiec_pof_test(n_observations=100, n_exceptions=-1)


def test_invalid_input_exceptions_exceed_observations_raises():
    with pytest.raises(ValueError, match="cannot exceed"):
        kupiec_pof_test(n_observations=100, n_exceptions=101)


# --- christoffersen.py ---

def test_known_answer_independence_clustered_exceptions_fails():
    """Exceptions all clustered together (not scattered) should fail independence."""
    exceptions = pd.Series([False] * 40 + [True] * 10 + [False] * 40)
    result = christoffersen_independence_test(exceptions)
    assert result.passed is False


def test_normal_case_independence_scattered_exceptions_passes():
    np.random.seed(3)
    exceptions = pd.Series(np.random.random(500) < 0.05)  # scattered, roughly 5% rate
    result = christoffersen_independence_test(exceptions)
    assert result.passed is True


def test_edge_case_no_exceptions_at_all():
    exceptions = pd.Series([False] * 100)
    result = christoffersen_independence_test(exceptions)
    assert not np.isnan(result.p_value)


def test_invalid_input_too_short_raises():
    with pytest.raises(ValueError, match="at least 2"):
        christoffersen_independence_test(pd.Series([True]))


def test_normal_case_conditional_coverage_combines_both_tests():
    np.random.seed(4)
    exceptions = pd.Series(np.random.random(1000) < 0.05)
    result = christoffersen_conditional_coverage_test(exceptions, confidence_level=0.95)
    assert result.lr_conditional_coverage == pytest.approx(result.lr_pof + result.lr_independence)


# --- basel.py ---

def test_known_answer_basel_official_green_boundary():
    """Basel's published table: 4 exceptions out of 250 at 99% is Green."""
    result = basel_traffic_light(n_observations=250, n_exceptions=4, confidence_level=0.99)
    assert result.zone == "Green"


def test_known_answer_basel_official_yellow_range():
    """Basel's published table: 7 exceptions out of 250 at 99% is Yellow."""
    result = basel_traffic_light(n_observations=250, n_exceptions=7, confidence_level=0.99)
    assert result.zone == "Yellow"


def test_known_answer_basel_official_red_range():
    """Basel's published table: 12 exceptions out of 250 at 99% is Red."""
    result = basel_traffic_light(n_observations=250, n_exceptions=12, confidence_level=0.99)
    assert result.zone == "Red"


def test_invalid_input_exceptions_exceed_observations_raises():
    with pytest.raises(ValueError, match="cannot exceed"):
        basel_traffic_light(n_observations=100, n_exceptions=101)


# --- report.py (integration) ---

def test_normal_case_well_calibrated_model_passes():
    np.random.seed(7)
    n = 1000
    returns = pd.Series(np.random.normal(0, 0.015, n), index=pd.date_range("2020-01-01", periods=n))
    var = rolling_historical_var(returns, window=250, confidence_level=0.95)
    report = run_backtest(returns, var, confidence_level=0.95)
    assert report.overall_conclusion == "Pass"
    assert report.basel.zone in ("Green", "Yellow")


def test_normal_case_miscalibrated_model_fails():
    """A model using a much-too-tight VaR (from an 80% confidence
    calculation, tested as if it were 95%) should be correctly rejected."""
    np.random.seed(7)
    n = 1000
    returns = pd.Series(np.random.normal(0, 0.015, n), index=pd.date_range("2020-01-01", periods=n))
    bad_var = rolling_historical_var(returns, window=250, confidence_level=0.80)
    report = run_backtest(returns, bad_var, confidence_level=0.95)
    assert report.overall_conclusion == "Fail"
    assert report.kupiec.passed is False


def test_invalid_input_empty_returns_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        run_backtest(pd.Series(dtype=float), pd.Series(dtype=float))
