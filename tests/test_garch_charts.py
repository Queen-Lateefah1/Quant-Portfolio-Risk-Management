import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization import garch_charts
from src.volatility.garch import (
    fit_garch_model,
    get_conditional_volatility,
    forecast_volatility,
    get_standardized_residuals,
)


@pytest.fixture
def garch_returns():
    np.random.seed(11)
    n = 300
    omega, alpha, beta = 0.00001, 0.08, 0.90
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = np.random.normal(0, np.sqrt(sigma2[t]))
    return pd.Series(returns, index=pd.date_range("2023-01-01", periods=n))


def test_plot_garch_conditional_volatility_returns_figure(garch_returns):
    fitted = fit_garch_model(garch_returns)
    vol = get_conditional_volatility(fitted)
    fig = garch_charts.plot_garch_conditional_volatility(vol)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_garch_forecast_returns_figure(garch_returns):
    fitted = fit_garch_model(garch_returns)
    vol = get_conditional_volatility(fitted)
    forecast = forecast_volatility(fitted, horizon=5)
    fig = garch_charts.plot_garch_forecast(vol, forecast)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_standardized_residuals_returns_figure(garch_returns):
    fitted = fit_garch_model(garch_returns)
    resid = get_standardized_residuals(fitted)
    fig = garch_charts.plot_standardized_residuals(resid)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_volatility_clustering_returns_figure(garch_returns):
    fig = garch_charts.plot_volatility_clustering(garch_returns)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
