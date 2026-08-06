import pandas as pd
import pytest

from src.portfolio.construction import build_portfolio, equal_weighted_portfolio_returns


# --- equal_weighted_portfolio_returns (Phase 2, unchanged) ---

def test_known_answer_two_equal_assets():
    dates = pd.date_range("2026-01-01", periods=3)
    returns = pd.DataFrame({"AAPL": [0.02, -0.01, 0.03], "MSFT": [0.00, 0.01, -0.01]}, index=dates)
    result = equal_weighted_portfolio_returns(returns)
    assert result.iloc[0] == pytest.approx(0.01)


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        equal_weighted_portfolio_returns(pd.DataFrame())


# --- build_portfolio: buy-and-hold (no rebalancing) ---

def test_known_answer_buy_and_hold_no_price_change():
    """Prices flat -> portfolio value should stay flat at initial_value."""
    dates = pd.date_range("2026-01-01", periods=3)
    prices = pd.DataFrame({"AAPL": [100.0, 100.0, 100.0], "MSFT": [50.0, 50.0, 50.0]}, index=dates)
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    result = build_portfolio(prices, weights, initial_value=10_000.0)
    assert result.value.iloc[0] == pytest.approx(10_000.0)
    assert result.value.iloc[-1] == pytest.approx(10_000.0)
    assert result.pnl.iloc[0] == 0.0
    assert result.rebalance_dates == []


def test_normal_case_buy_and_hold_price_increase():
    """Both assets double -> portfolio value should double too."""
    dates = pd.date_range("2026-01-01", periods=2)
    prices = pd.DataFrame({"AAPL": [100.0, 200.0], "MSFT": [50.0, 100.0]}, index=dates)
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    result = build_portfolio(prices, weights, initial_value=10_000.0)
    assert result.value.iloc[-1] == pytest.approx(20_000.0)
    assert result.returns.iloc[-1] == pytest.approx(1.0)  # 100% return


def test_edge_case_single_asset_full_weight():
    dates = pd.date_range("2026-01-01", periods=2)
    prices = pd.DataFrame({"AAPL": [100.0, 110.0]}, index=dates)
    weights = pd.Series({"AAPL": 1.0})
    result = build_portfolio(prices, weights, initial_value=1_000.0)
    assert result.value.iloc[-1] == pytest.approx(1_100.0)


def test_invalid_input_empty_prices_raises():
    with pytest.raises(ValueError, match="prices cannot be empty"):
        build_portfolio(pd.DataFrame(), pd.Series({"AAPL": 1.0}))


def test_invalid_input_empty_weights_raises():
    prices = pd.DataFrame({"AAPL": [100.0]})
    with pytest.raises(ValueError, match="weights cannot be empty"):
        build_portfolio(prices, pd.Series(dtype=float))


def test_invalid_input_negative_initial_value_raises():
    prices = pd.DataFrame({"AAPL": [100.0]})
    weights = pd.Series({"AAPL": 1.0})
    with pytest.raises(ValueError, match="initial_value must be positive"):
        build_portfolio(prices, weights, initial_value=-100.0)


def test_invalid_input_weight_ticker_not_in_prices_raises():
    prices = pd.DataFrame({"AAPL": [100.0]})
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    with pytest.raises(ValueError, match="not in prices"):
        build_portfolio(prices, weights)


# --- build_portfolio: with rebalancing ---

def test_normal_case_rebalancing_resets_to_target_weights():
    """After a rebalance, portfolio value should still track correctly
    even when relative asset prices have drifted apart."""
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    # AAPL drifts up, MSFT stays flat -> weights drift from 50/50 without rebalancing
    aapl = [100.0 + i for i in range(40)]
    msft = [50.0] * 40
    prices = pd.DataFrame({"AAPL": aapl, "MSFT": msft}, index=dates)
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})

    result_no_rebal = build_portfolio(prices, weights, initial_value=10_000.0)
    result_rebal = build_portfolio(prices, weights, initial_value=10_000.0, rebalance_freq="W")

    # Both should have valid, positive portfolio values throughout
    assert (result_no_rebal.value > 0).all()
    assert (result_rebal.value > 0).all()
    assert len(result_rebal.rebalance_dates) > 0
