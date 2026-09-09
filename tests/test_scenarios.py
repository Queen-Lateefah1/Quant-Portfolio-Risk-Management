import numpy as np
import pandas as pd
import pytest

from src.stress_testing.scenarios import (
    HISTORICAL_SCENARIOS,
    HYPOTHETICAL_MARKET_SHOCKS,
    compute_scenario_return,
    stressed_loss_from_return,
    fetch_scenario_prices,
)


# --- compute_scenario_return (pure calculation, no network) ---

def test_known_answer_single_asset_scenario_return():
    dates = pd.date_range("2020-02-19", periods=2)
    prices = pd.DataFrame({"AAPL": [100.0, 65.0]}, index=dates)  # -35% peak to trough
    weights = pd.Series({"AAPL": 1.0})
    result = compute_scenario_return(prices, weights)
    assert result == pytest.approx(-0.35)


def test_normal_case_multi_asset_weighted_correctly():
    dates = pd.date_range("2020-02-19", periods=2)
    prices = pd.DataFrame({"AAPL": [100.0, 80.0], "JPM": [100.0, 90.0]}, index=dates)  # -20%, -10%
    weights = pd.Series({"AAPL": 0.5, "JPM": 0.5})
    result = compute_scenario_return(prices, weights)
    assert result == pytest.approx(-0.15)  # 0.5*-0.20 + 0.5*-0.10


def test_known_answer_excludes_pre_ipo_ticker_and_renormalizes():
    """A ticker with no price data for the window (e.g. hadn't IPO'd
    yet, like Tesla during the 2008 crisis) should be excluded, with
    the REMAINING tickers' weights renormalized to 100% — not silently
    treated as a 0% contribution, which would understate the loss."""
    dates = pd.date_range("2008-09-01", periods=2)
    prices = pd.DataFrame(
        {"AAPL": [100.0, 70.0], "TSLA": [np.nan, np.nan]},  # TSLA didn't exist in 2008
        index=dates,
    )
    weights = pd.Series({"AAPL": 0.5, "TSLA": 0.5})
    result = compute_scenario_return(prices, weights)
    # AAPL's -30% return should apply at FULL weight (renormalized to 1.0), not half-weight
    assert result == pytest.approx(-0.30)


def test_invalid_input_all_tickers_missing_data_raises():
    dates = pd.date_range("2008-09-01", periods=2)
    prices = pd.DataFrame({"TSLA": [np.nan, np.nan]}, index=dates)
    weights = pd.Series({"TSLA": 1.0})
    with pytest.raises(ValueError, match="no ticker"):
        compute_scenario_return(prices, weights)


def test_edge_case_no_price_change_gives_zero_return():
    prices = pd.DataFrame({"AAPL": [100.0, 100.0]}, index=pd.date_range("2020-01-01", periods=2))
    weights = pd.Series({"AAPL": 1.0})
    result = compute_scenario_return(prices, weights)
    assert result == pytest.approx(0.0)


def test_invalid_input_empty_prices_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_scenario_return(pd.DataFrame(), pd.Series({"AAPL": 1.0}))


def test_invalid_input_weight_ticker_not_in_prices_raises():
    prices = pd.DataFrame({"AAPL": [100.0, 90.0]})
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    with pytest.raises(ValueError, match="not in prices"):
        compute_scenario_return(prices, weights)


# --- stressed_loss_from_return ---

def test_known_answer_negative_return_gives_positive_loss():
    loss = stressed_loss_from_return(portfolio_return=-0.20, portfolio_value=1_000_000.0)
    assert loss == pytest.approx(200_000.0)


def test_known_answer_positive_return_gives_negative_loss():
    """A 'stressed' scenario that's actually a gain should show as a negative loss."""
    loss = stressed_loss_from_return(portfolio_return=0.05, portfolio_value=1_000_000.0)
    assert loss == pytest.approx(-50_000.0)


def test_invalid_input_zero_portfolio_value_raises():
    with pytest.raises(ValueError, match="must be positive"):
        stressed_loss_from_return(portfolio_return=-0.1, portfolio_value=0)


# --- fetch_scenario_prices (network call: input validation only) ---

def test_invalid_input_unknown_scenario_raises():
    with pytest.raises(ValueError, match="Unknown scenario"):
        fetch_scenario_prices(["AAPL"], "Not A Real Scenario")


# --- scenario definitions themselves ---

def test_normal_case_five_historical_scenarios_defined():
    assert len(HISTORICAL_SCENARIOS) == 5
    for name, (start, end) in HISTORICAL_SCENARIOS.items():
        assert start < end


def test_normal_case_three_hypothetical_market_shocks_defined():
    assert len(HYPOTHETICAL_MARKET_SHOCKS) == 3
    for name, shock in HYPOTHETICAL_MARKET_SHOCKS.items():
        assert shock < 0  # all defined as market DOWN scenarios
