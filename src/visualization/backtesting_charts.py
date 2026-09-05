"""
backtesting_charts.py

Responsibility: visualizations for Phase 8 (VaR Backtesting) and
Phase 9 (Model Comparison).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- Phase 8 charts ---

def plot_actual_pnl_vs_var(returns: pd.Series, var: pd.Series, exceptions: pd.Series = None):
    """Actual daily loss vs the VaR threshold over time, with exceptions marked."""
    losses = -returns
    aligned_losses, aligned_var = losses.align(var, join="inner")
    valid = aligned_var.notna()
    aligned_losses, aligned_var = aligned_losses[valid], aligned_var[valid]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(aligned_losses.index, aligned_losses, color="#4C72B0", linewidth=0.8, label="Actual daily loss")
    ax.plot(aligned_var.index, aligned_var, color="#C44E52", linewidth=1.3, label="VaR threshold")

    if exceptions is not None:
        exc_dates = exceptions[exceptions].index.intersection(aligned_losses.index)
        ax.scatter(exc_dates, aligned_losses.loc[exc_dates], color="#DD8452", s=30, zorder=5, label="Exception")

    ax.set_title("Actual P&L vs VaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_exception_timeline(exceptions: pd.Series):
    """Timeline showing exactly when exceptions occurred — helps spot clustering visually."""
    fig, ax = plt.subplots(figsize=(12, 2.5))
    exc_dates = exceptions[exceptions].index
    ax.eventplot(exc_dates, color="#C44E52", linewidths=1.5)
    ax.set_title("VaR Exception Timeline")
    ax.set_xlabel("Date")
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_basel_traffic_light(basel_result):
    """Simple visual indicator of the Basel zone with the underlying numbers."""
    zone_colors = {"Green": "#55A868", "Yellow": "#DD8452", "Red": "#C44E52"}
    color = zone_colors.get(basel_result.zone, "gray")

    fig, ax = plt.subplots(figsize=(5, 5))
    circle = plt.Circle((0.5, 0.5), 0.4, color=color)
    ax.add_patch(circle)
    ax.text(0.5, 0.5, basel_result.zone, ha="center", va="center", fontsize=18, color="white", fontweight="bold")
    ax.text(0.5, 0.08, f"{basel_result.n_exceptions} exceptions / {basel_result.n_observations} obs",
            ha="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Basel Traffic Light")
    fig.tight_layout()
    return fig


def plot_expected_vs_actual_exceptions(exception_summary):
    """Bar chart: expected vs actual exception count."""
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Expected", "Actual"],
        [exception_summary.expected_exceptions, exception_summary.n_exceptions],
        color=["#4C72B0", "#C44E52"],
    )
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center")
    ax.set_title("Expected vs Actual Exceptions")
    ax.set_ylabel("Number of exceptions")
    fig.tight_layout()
    return fig


# --- Phase 9 charts ---

def plot_var_model_comparison(var_by_model: dict):
    """Line chart overlaying multiple models' VaR series over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#DD8452"]
    for (name, series), color in zip(var_by_model.items(), colors):
        ax.plot(series.index, series, label=name, linewidth=1.2, color=color, alpha=0.85)
    ax.set_title("VaR Model Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("VaR (loss magnitude)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_exceptions_overlay(returns: pd.Series, var_by_model: dict):
    """Actual losses with each model's exceptions marked in a different color."""
    losses = -returns
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(losses.index, losses, color="gray", linewidth=0.6, alpha=0.6, label="Actual loss")

    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
    for (name, var_series), color in zip(var_by_model.items(), colors):
        aligned_losses, aligned_var = losses.align(var_series, join="inner")
        valid = aligned_var.notna()
        aligned_losses, aligned_var = aligned_losses[valid], aligned_var[valid]
        exc_mask = aligned_losses > aligned_var
        ax.scatter(aligned_losses[exc_mask].index, aligned_losses[exc_mask], color=color, s=20, label=f"{name} exception")

    ax.set_title("Exceptions Overlay by Model")
    ax.set_xlabel("Date")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_backtesting_summary_bar(comparison_table: pd.DataFrame):
    """Grouped bar chart: exception ratio (actual vs expected) per model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comparison_table))
    width = 0.35

    ax.bar(x - width / 2, comparison_table["exception_ratio"], width, label="Actual ratio", color="#C44E52")
    ax.bar(x + width / 2, comparison_table["expected_ratio"], width, label="Expected ratio", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_table.index, rotation=20, ha="right")
    ax.set_title("Backtesting Summary: Exception Ratios by Model")
    ax.set_ylabel("Exception ratio")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_model_ranking_table(ranked_table: pd.DataFrame):
    """Render the model ranking as a matplotlib table image (so it can
    be saved/exported alongside the other charts, not just printed)."""
    display_cols = ["rank", "n_exceptions", "exception_ratio", "expected_shortfall", "basel_zone", "overall_conclusion"]
    display_data = ranked_table[display_cols].round(4)

    fig, ax = plt.subplots(figsize=(11, 0.6 + 0.5 * len(display_data)))
    ax.axis("off")
    table = ax.table(
        cellText=display_data.values,
        rowLabels=display_data.index,
        colLabels=display_cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("Model Ranking", pad=20)
    fig.tight_layout()
    return fig
