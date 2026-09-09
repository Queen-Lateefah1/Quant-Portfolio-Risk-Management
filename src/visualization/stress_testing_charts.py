"""
stress_testing_charts.py

Responsibility: visualizations for Phase 10 (Stress Testing).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scenario_loss_bar(stress_test_table: pd.DataFrame):
    """Horizontal bar chart of stressed loss per scenario, color-coded
    by category (Historical vs Hypothetical)."""
    table = stress_test_table.sort_values("stressed_loss")
    colors = ["#C44E52" if cat == "Historical" else "#4C72B0" for cat in table["category"]]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(table))))
    ax.barh(table.index, table["stressed_loss"], color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title("Stressed Loss by Scenario")
    ax.set_xlabel("Stressed loss (currency)")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#C44E52", label="Historical"), Patch(facecolor="#4C72B0", label="Hypothetical")]
    ax.legend(handles=legend_elements)

    fig.tight_layout()
    return fig


def plot_stress_var_comparison(stress_test_table: pd.DataFrame):
    """Bar chart comparing stressed VaR across scenarios that have one
    (the parametric volatility/correlation scenarios)."""
    var_rows = stress_test_table[stress_test_table["stressed_var"].notna()]
    if var_rows.empty:
        raise ValueError("no scenarios in this table have a stressed_var value to chart")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(var_rows.index, var_rows["stressed_var"], color="#8172B2")
    ax.set_title("Stress VaR Comparison")
    ax.set_ylabel("Stressed VaR")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


def plot_sector_shock_impact(stress_test_table: pd.DataFrame):
    """Bar chart isolating the sector-specific shock scenario(s)."""
    sector_rows = stress_test_table[stress_test_table.index.str.contains("Sector", case=False)]
    if sector_rows.empty:
        raise ValueError("no sector-shock scenario found in this table")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sector_rows.index, sector_rows["stressed_loss"], color="#DD8452")
    ax.set_title("Sector Shock Impact")
    ax.set_ylabel("Stressed loss (currency)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    return fig


def plot_worst_loss_ranking(ranked_table: pd.DataFrame, top_n: int = 10):
    """Bar chart of the top N worst scenarios by stressed loss, from
    `engine.rank_scenarios` output."""
    top = ranked_table.head(top_n).sort_values("stressed_loss")

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(top))))
    ax.barh(top.index, top["stressed_loss"], color="#C44E52")
    ax.set_title(f"Worst-Case Loss Ranking (Top {min(top_n, len(top))})")
    ax.set_xlabel("Stressed loss (currency)")
    fig.tight_layout()
    return fig
