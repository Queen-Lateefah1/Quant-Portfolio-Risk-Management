"""
charts.py

Responsibility: visualizations for Phase 4 (Historical Simulation VaR)
and Phase 5 (EWMA Volatility Scaling). Every function returns a
matplotlib Figure rather than calling plt.show() — the caller (notebook
or dashboard) decides how/where to display or save it.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- Phase 4: Historical Simulation VaR charts ---

def plot_return_distribution(returns: pd.Series, var: float, es: float = None, confidence_level: float = 0.95):
    """Histogram of returns with a VaR threshold line (and optional ES line)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(returns, bins=50, color="#4C72B0", alpha=0.75, edgecolor="white")
    ax.axvline(-var, color="#C44E52", linestyle="--", linewidth=2,
               label=f"VaR ({confidence_level*100:.1f}%): {-var:.4f}")
    if es is not None:
        ax.axvline(-es, color="#8172B2", linestyle=":", linewidth=2,
                   label=f"Expected Shortfall: {-es:.4f}")
    ax.set_title("Historical Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rolling_var(rolling_var: pd.Series, returns: pd.Series = None):
    """Line chart of rolling VaR over time, optionally overlaid with actual returns."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(rolling_var.index, -rolling_var, color="#C44E52", linewidth=1.5, label="VaR threshold")
    if returns is not None:
        aligned = returns.reindex(rolling_var.index)
        ax.plot(aligned.index, aligned, color="#4C72B0", alpha=0.5, linewidth=0.8, label="Actual return")
    ax.set_title("Rolling Historical VaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return / VaR threshold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_tail_loss(returns: pd.Series, var: float):
    """Highlight observations in the loss tail beyond the VaR threshold."""
    losses = -returns
    tail = losses[losses >= var]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(losses.index, losses, s=10, color="#4C72B0", alpha=0.5, label="Daily loss")
    ax.scatter(tail.index, tail, s=25, color="#C44E52", label="Tail loss (beyond VaR)")
    ax.axhline(var, color="#C44E52", linestyle="--", linewidth=1.5, label=f"VaR: {var:.4f}")
    ax.set_title("Tail Loss Chart")
    ax.set_xlabel("Date")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_var_vs_es(var_by_confidence: dict, es_by_confidence: dict):
    """Bar chart comparing VaR vs Expected Shortfall across confidence levels.

    Parameters
    ----------
    var_by_confidence, es_by_confidence:
        Dicts mapping confidence_level -> value, same keys in both.
    """
    levels = sorted(var_by_confidence.keys())
    var_vals = [var_by_confidence[l] for l in levels]
    es_vals = [es_by_confidence[l] for l in levels]

    x = np.arange(len(levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, var_vals, width, label="VaR", color="#4C72B0")
    ax.bar(x + width / 2, es_vals, width, label="Expected Shortfall", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l*100:.1f}%" for l in levels])
    ax.set_title("VaR vs Expected Shortfall by Confidence Level")
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Loss magnitude")
    ax.legend()
    fig.tight_layout()
    return fig


# --- Phase 5: EWMA Volatility Scaling charts ---

def plot_ewma_volatility(ewma_vol: pd.Series):
    """Line chart of EWMA volatility over time."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ewma_vol.index, ewma_vol, color="#55A868", linewidth=1.5)
    ax.set_title("EWMA Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (daily)")
    fig.tight_layout()
    return fig


def plot_raw_vs_scaled_returns(raw_returns: pd.Series, scaled_returns: pd.Series):
    """Overlay raw returns against EWMA volatility-scaled returns."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(raw_returns.index, raw_returns, color="#4C72B0", alpha=0.6, linewidth=0.8, label="Raw return")
    ax.plot(scaled_returns.index, scaled_returns, color="#DD8452", alpha=0.8, linewidth=0.8, label="EWMA-scaled return")
    ax.set_title("Raw Returns vs Volatility-Scaled Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_var_comparison(historical_var: float, ewma_var: float, confidence_level: float = 0.95):
    """Bar chart comparing plain Historical VaR to EWMA-scaled VaR."""
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Historical VaR", "EWMA-scaled VaR"], [historical_var, ewma_var],
                   color=["#4C72B0", "#DD8452"])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center")
    ax.set_title(f"Historical VaR vs EWMA-Scaled VaR ({confidence_level*100:.1f}%)")
    ax.set_ylabel("Loss magnitude")
    fig.tight_layout()
    return fig


def plot_exception_comparison(returns: pd.Series, historical_var: float, ewma_var: float):
    """Bar chart comparing the number of VaR exceptions (actual loss > VaR)
    under the plain historical VaR vs the EWMA-scaled VaR."""
    losses = -returns
    hist_exceptions = int((losses > historical_var).sum())
    ewma_exceptions = int((losses > ewma_var).sum())

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Historical VaR", "EWMA-scaled VaR"], [hist_exceptions, ewma_exceptions],
                   color=["#4C72B0", "#DD8452"])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(str(int(height)), xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center")
    ax.set_title("VaR Exceptions: Historical vs EWMA-Scaled")
    ax.set_ylabel("Number of exceptions")
    fig.tight_layout()
    return fig
