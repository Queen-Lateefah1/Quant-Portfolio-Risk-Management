import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization import stress_testing_charts
from src.stress_testing.engine import run_all_stress_tests, rank_scenarios


@pytest.fixture
def stress_table():
    np.random.seed(8)
    n = 400
    dates = pd.date_range("2023-01-01", periods=n)
    a = pd.Series(np.random.normal(0, 0.015, n), index=dates)
    b = pd.Series(np.random.normal(0, 0.02, n), index=dates)
    returns_matrix = pd.DataFrame({"AAPL": a, "JPM": b})
    weights = pd.Series({"AAPL": 0.6, "JPM": 0.4})
    portfolio_returns = returns_matrix.mul(weights, axis=1).sum(axis=1)
    sector_map = {"AAPL": "Technology", "JPM": "Financial Services"}
    return run_all_stress_tests(portfolio_returns, returns_matrix, weights, sector_map,
                                  portfolio_value=1_000_000.0, include_historical=False)


def test_plot_scenario_loss_bar_returns_figure(stress_table):
    fig = stress_testing_charts.plot_scenario_loss_bar(stress_table)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_stress_var_comparison_returns_figure(stress_table):
    fig = stress_testing_charts.plot_stress_var_comparison(stress_table)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_sector_shock_impact_returns_figure(stress_table):
    fig = stress_testing_charts.plot_sector_shock_impact(stress_table)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_worst_loss_ranking_returns_figure(stress_table):
    ranked = rank_scenarios(stress_table)
    fig = stress_testing_charts.plot_worst_loss_ranking(ranked)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_stress_var_comparison_raises_when_no_var_data():
    empty_var_table = pd.DataFrame({
        "category": ["Hypothetical"], "stressed_return": [-0.1], "stressed_loss": [100000.0],
        "stressed_var": [None], "stressed_es": [None],
    }, index=["Some scenario"])
    with pytest.raises(ValueError, match="no scenarios"):
        stress_testing_charts.plot_stress_var_comparison(empty_var_table)
