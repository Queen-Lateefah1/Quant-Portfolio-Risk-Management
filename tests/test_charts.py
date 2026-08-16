import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed for tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization import charts
from src.var.historical import calculate_historical_var, rolling_historical_var
from src.var.expected_shortfall import calculate_expected_shortfall
from src.volatility.ewma import ewma_volatility, ewma_scaled_returns, ewma_scaled_historical_var


@pytest.fixture
def sample_returns():
    np.random.seed(9)
    dates = pd.date_range("2024-01-01", periods=300)
    return pd.Series(np.random.normal(0, 0.02, 300), index=dates)


def test_plot_return_distribution_returns_figure(sample_returns):
    var = calculate_historical_var(sample_returns)
    es = calculate_expected_shortfall(sample_returns)
    fig = charts.plot_return_distribution(sample_returns, var, es)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_rolling_var_returns_figure(sample_returns):
    rolling = rolling_historical_var(sample_returns, window=50)
    fig = charts.plot_rolling_var(rolling, sample_returns)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_tail_loss_returns_figure(sample_returns):
    var = calculate_historical_var(sample_returns)
    fig = charts.plot_tail_loss(sample_returns, var)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_var_vs_es_returns_figure(sample_returns):
    var_dict = {0.95: calculate_historical_var(sample_returns, 0.95),
                0.99: calculate_historical_var(sample_returns, 0.99)}
    es_dict = {0.95: calculate_expected_shortfall(sample_returns, 0.95),
               0.99: calculate_expected_shortfall(sample_returns, 0.99)}
    fig = charts.plot_var_vs_es(var_dict, es_dict)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_ewma_volatility_returns_figure(sample_returns):
    vol = ewma_volatility(sample_returns)
    fig = charts.plot_ewma_volatility(vol)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_raw_vs_scaled_returns_returns_figure(sample_returns):
    scaled = ewma_scaled_returns(sample_returns)
    fig = charts.plot_raw_vs_scaled_returns(sample_returns, scaled)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_var_comparison_returns_figure(sample_returns):
    hist_var = calculate_historical_var(sample_returns)
    ewma_var = ewma_scaled_historical_var(sample_returns)
    fig = charts.plot_var_comparison(hist_var, ewma_var)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_exception_comparison_returns_figure(sample_returns):
    hist_var = calculate_historical_var(sample_returns)
    ewma_var = ewma_scaled_historical_var(sample_returns)
    fig = charts.plot_exception_comparison(sample_returns, hist_var, ewma_var)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
