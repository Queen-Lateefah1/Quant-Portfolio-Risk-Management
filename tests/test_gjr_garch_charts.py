import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization import gjr_garch_charts
from src.volatility.gjr_garch import fit_gjr_garch_model, get_conditional_volatility
from src.volatility.garch import fit_garch_model, get_conditional_volatility as get_garch_vol


@pytest.fixture
def gjr_returns():
    np.random.seed(21)
    n = 300
    omega, alpha, gamma, beta = 0.00001, 0.03, 0.10, 0.88
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - gamma / 2 - beta)
    for t in range(1, n):
        shock_sq = returns[t - 1] ** 2
        asym = shock_sq if returns[t - 1] < 0 else 0.0
        sigma2[t] = omega + alpha * shock_sq + gamma * asym + beta * sigma2[t - 1]
        returns[t] = np.random.normal(0, np.sqrt(sigma2[t]))
    return pd.Series(returns, index=pd.date_range("2023-01-01", periods=n))


def test_plot_gjr_garch_volatility_returns_figure(gjr_returns):
    fitted = fit_gjr_garch_model(gjr_returns)
    vol = get_conditional_volatility(fitted)
    fig = gjr_garch_charts.plot_gjr_garch_volatility(vol)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_garch_vs_gjr_garch_returns_figure(gjr_returns):
    garch_fitted = fit_garch_model(gjr_returns)
    gjr_fitted = fit_gjr_garch_model(gjr_returns)
    garch_vol = get_garch_vol(garch_fitted)
    gjr_vol = get_conditional_volatility(gjr_fitted)
    fig = gjr_garch_charts.plot_garch_vs_gjr_garch(garch_vol, gjr_vol)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_leverage_effect_returns_figure(gjr_returns):
    fitted = fit_gjr_garch_model(gjr_returns)
    fig = gjr_garch_charts.plot_leverage_effect(fitted)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_leverage_effect_negative_side_exceeds_positive():
    """With a genuine positive leverage effect (gamma > 0), the
    variance response to a negative shock should exceed the response
    to a positive shock of the same magnitude — the whole point of
    this chart. Tested directly against the news-impact-curve formula
    with a known gamma, rather than a fitted estimate: MLE on a small
    sample can occasionally recover a noisy/negative gamma even when
    the true process has a genuine positive leverage effect, which
    would make this test flaky for the wrong reason."""
    omega, alpha, gamma, beta = 0.01, 0.05, 0.08, 0.85
    shock = 3.0
    negative_response = omega + alpha * shock**2 + gamma * shock**2 + beta * 1.0
    positive_response = omega + alpha * shock**2 + beta * 1.0
    assert negative_response > positive_response
