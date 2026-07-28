import pandas as pd
import pytest

from src.portfolio.construction import equal_weighted_portfolio_returns


def test_known_answer_two_equal_assets():
    """Two assets, equal weight -> portfolio return is the simple average."""
    dates = pd.date_range("2026-01-01", periods=3)
    returns = pd.DataFrame({"AAPL": [0.02, -0.01, 0.03], "MSFT": [0.00, 0.01, -0.01]}, index=dates)
    result = equal_weighted_portfolio_returns(returns)
    assert result.iloc[0] == pytest.approx(0.01)   # (0.02 + 0.00) / 2
    assert result.iloc[1] == pytest.approx(0.00)   # (-0.01 + 0.01) / 2
    assert result.iloc[2] == pytest.approx(0.01)   # (0.03 - 0.01) / 2


def test_normal_case_ten_assets_sums_to_one_weight():
    dates = pd.date_range("2026-01-01", periods=3)
    returns = pd.DataFrame(
        {f"T{i}": [0.01, 0.02, -0.01] for i in range(10)}, index=dates
    )
    result = equal_weighted_portfolio_returns(returns)
    # all assets identical -> portfolio return equals the shared asset return
    assert result.iloc[0] == pytest.approx(0.01)


def test_edge_case_single_asset():
    dates = pd.date_range("2026-01-01", periods=2)
    returns = pd.DataFrame({"AAPL": [0.05, -0.02]}, index=dates)
    result = equal_weighted_portfolio_returns(returns)
    assert result.iloc[0] == pytest.approx(0.05)


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        equal_weighted_portfolio_returns(pd.DataFrame())
