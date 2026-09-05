import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization import backtesting_charts
from src.backtesting.exceptions import identify_exceptions, summarize_exceptions
from src.backtesting.report import run_backtest
from src.backtesting.comparison import compare_models, rank_models
from src.var.historical import rolling_historical_var


@pytest.fixture
def sample_data():
    np.random.seed(15)
    n = 1000
    returns = pd.Series(np.random.normal(0, 0.015, n), index=pd.date_range("2020-01-01", periods=n))
    var = rolling_historical_var(returns, window=250, confidence_level=0.95)
    return returns, var


def test_plot_actual_pnl_vs_var_returns_figure(sample_data):
    returns, var = sample_data
    exceptions = identify_exceptions(returns, var)
    fig = backtesting_charts.plot_actual_pnl_vs_var(returns, var, exceptions)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_exception_timeline_returns_figure(sample_data):
    returns, var = sample_data
    exceptions = identify_exceptions(returns, var)
    fig = backtesting_charts.plot_exception_timeline(exceptions)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_basel_traffic_light_returns_figure(sample_data):
    returns, var = sample_data
    report = run_backtest(returns, var, confidence_level=0.95)
    fig = backtesting_charts.plot_basel_traffic_light(report.basel)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_expected_vs_actual_exceptions_returns_figure(sample_data):
    returns, var = sample_data
    exceptions = identify_exceptions(returns, var)
    summary = summarize_exceptions(exceptions, confidence_level=0.95)
    fig = backtesting_charts.plot_expected_vs_actual_exceptions(summary)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_var_model_comparison_returns_figure(sample_data):
    returns, var = sample_data
    var2 = rolling_historical_var(returns, window=500, confidence_level=0.95)
    fig = backtesting_charts.plot_var_model_comparison({"Historical-250": var, "Historical-500": var2})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_exceptions_overlay_returns_figure(sample_data):
    returns, var = sample_data
    var2 = rolling_historical_var(returns, window=500, confidence_level=0.95)
    fig = backtesting_charts.plot_exceptions_overlay(returns, {"A": var, "B": var2})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_backtesting_summary_bar_returns_figure(sample_data):
    returns, var = sample_data
    var2 = rolling_historical_var(returns, window=500, confidence_level=0.95)
    table = compare_models(returns, {"A": var, "B": var2}, confidence_level=0.95)
    fig = backtesting_charts.plot_backtesting_summary_bar(table)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_model_ranking_table_returns_figure(sample_data):
    returns, var = sample_data
    var2 = rolling_historical_var(returns, window=500, confidence_level=0.95)
    table = compare_models(returns, {"A": var, "B": var2}, confidence_level=0.95)
    ranked = rank_models(table)
    fig = backtesting_charts.plot_model_ranking_table(ranked)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
