import numpy as np
import pandas as pd
import pytest

from src.stress_testing.engine import (
    run_hypothetical_market_shocks,
    run_volatility_and_correlation_shocks,
    run_sector_shock,
    run_all_stress_tests,
    rank_scenarios,
)


def _sample_data(n=500, seed=6):
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n)
    a = pd.Series(np.random.normal(0, 0.015, n), index=dates)
    b = pd.Series(np.random.normal(0, 0.02, n), index=dates)
    returns_matrix = pd.DataFrame({"AAPL": a, "JPM": b})
    weights = pd.Series({"AAPL": 0.6, "JPM": 0.4})
    portfolio_returns = returns_matrix.mul(weights, axis=1).sum(axis=1)
    sector_map = {"AAPL": "Technology", "JPM": "Financial Services"}
    return portfolio_returns, returns_matrix, weights, sector_map


# --- run_hypothetical_market_shocks ---

def test_normal_case_three_market_shock_rows():
    rows = run_hypothetical_market_shocks(portfolio_value=1_000_000.0)
    assert len(rows) == 3
    assert all(r.category == "Hypothetical" for r in rows)


def test_known_answer_down_20_percent_loss():
    rows = run_hypothetical_market_shocks(portfolio_value=1_000_000.0)
    down_20 = next(r for r in rows if "20%" in r.scenario)
    assert down_20.stressed_loss == pytest.approx(200_000.0)


# --- run_volatility_and_correlation_shocks ---

def test_normal_case_two_rows_returned():
    portfolio_returns, returns_matrix, weights, _ = _sample_data()
    rows = run_volatility_and_correlation_shocks(portfolio_returns, returns_matrix, weights, portfolio_value=1_000_000.0)
    assert len(rows) == 2
    assert {r.scenario for r in rows} == {"Volatility doubles", "Correlation spikes"}


def test_normal_case_stressed_var_populated():
    portfolio_returns, returns_matrix, weights, _ = _sample_data()
    rows = run_volatility_and_correlation_shocks(portfolio_returns, returns_matrix, weights, portfolio_value=1_000_000.0)
    assert all(r.stressed_var is not None and r.stressed_var > 0 for r in rows)


# --- run_sector_shock ---

def test_normal_case_shocks_largest_sector():
    _, _, weights, sector_map = _sample_data()
    row = run_sector_shock(weights, sector_map, portfolio_value=1_000_000.0)
    assert row is not None
    assert "Technology" in row.scenario  # AAPL has the larger weight (0.6)


def test_edge_case_no_sector_data_returns_none():
    weights = pd.Series({"AAPL": 1.0})
    sector_map = {"AAPL": "Unknown"}
    row = run_sector_shock(weights, sector_map, portfolio_value=1_000_000.0)
    assert row is None


# --- run_all_stress_tests (network disabled for fast/reliable testing) ---

def test_normal_case_full_run_without_historical():
    portfolio_returns, returns_matrix, weights, sector_map = _sample_data()
    table = run_all_stress_tests(
        portfolio_returns, returns_matrix, weights, sector_map,
        portfolio_value=1_000_000.0, include_historical=False,
    )
    # 3 market shocks + 2 parametric + 1 sector = 6 rows
    assert len(table) == 6
    expected_columns = {"category", "stressed_return", "stressed_loss", "stressed_var", "stressed_es"}
    assert expected_columns.issubset(set(table.columns))


def test_invalid_input_empty_returns_raises():
    _, returns_matrix, weights, sector_map = _sample_data()
    with pytest.raises(ValueError, match="cannot be empty"):
        run_all_stress_tests(pd.Series(dtype=float), returns_matrix, weights, sector_map)


# --- rank_scenarios ---

def test_normal_case_ranked_worst_first():
    portfolio_returns, returns_matrix, weights, sector_map = _sample_data()
    table = run_all_stress_tests(
        portfolio_returns, returns_matrix, weights, sector_map,
        portfolio_value=1_000_000.0, include_historical=False,
    )
    ranked = rank_scenarios(table)
    assert ranked["stressed_loss"].is_monotonic_decreasing
    assert list(ranked["rank"]) == list(range(1, len(ranked) + 1))


def test_invalid_input_empty_table_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        rank_scenarios(pd.DataFrame())
