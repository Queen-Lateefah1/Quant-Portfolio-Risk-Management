"""
garch_charts.py

Responsibility: visualizations for Phase 6 (GARCH Volatility Scaling).
Kept in a separate file from charts.py (Phase 4/5) purely so each chart
module stays focused on the phase that introduced it — same one-module-
one-job principle as the rest of the project.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_garch_conditional_volatility(conditional_vol: pd.Series):
    """Line chart of GARCH conditional volatility over time."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(conditional_vol.index, conditional_vol, color="#C44E52", linewidth=1.3)
    ax.set_title("GARCH(1,1) Conditional Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (daily)")
    fig.tight_layout()
    return fig


def plot_garch_forecast(conditional_vol: pd.Series, forecast: pd.Series):
    """Line chart showing historical conditional volatility with the
    forward forecast path appended, so the transition is visible."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(conditional_vol.index, conditional_vol, color="#C44E52", linewidth=1.3, label="Historical conditional volatility")

    last_date = conditional_vol.index[-1]
    freq = pd.infer_freq(conditional_vol.index) or "D"
    forecast_dates = pd.date_range(start=last_date, periods=len(forecast) + 1, freq=freq)[1:]
    ax.plot(forecast_dates, forecast, color="#4C72B0", linewidth=1.8, linestyle="--", marker="o", label="Forecast")

    ax.axvline(last_date, color="gray", linestyle=":", linewidth=1)
    ax.set_title("GARCH Volatility Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (daily)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_standardized_residuals(residuals: pd.Series):
    """Two-panel diagnostic: residuals over time, and their distribution
    (should look roughly like standard-normal white noise for a
    well-specified model)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(residuals.index, residuals, color="#55A868", linewidth=0.8)
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_title("GARCH Standardized Residuals")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Standardized residual")

    ax2.hist(residuals, bins=40, color="#55A868", alpha=0.75, density=True, edgecolor="white")
    x = np.linspace(-4, 4, 200)
    ax2.plot(x, (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2), color="#C44E52", linewidth=1.5, label="Standard normal")
    ax2.set_title("Residual Distribution")
    ax2.legend()

    fig.tight_layout()
    return fig


def plot_volatility_clustering(returns: pd.Series):
    """Overlay of raw returns and squared returns to visually illustrate
    volatility clustering — the pattern GARCH is specifically built to
    capture (large moves followed by large moves, calm followed by calm)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(returns.index, returns, color="#4C72B0", linewidth=0.7)
    ax1.axhline(0, color="gray", linewidth=0.6)
    ax1.set_title("Daily Returns")
    ax1.set_ylabel("Return")

    ax2.plot(returns.index, returns**2, color="#C44E52", linewidth=0.7)
    ax2.set_title("Squared Returns (Volatility Clustering)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Squared return")

    fig.tight_layout()
    return fig
