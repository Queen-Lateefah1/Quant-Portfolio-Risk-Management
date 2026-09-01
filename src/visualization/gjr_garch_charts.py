"""
gjr_garch_charts.py

Responsibility: visualizations for Phase 7 (GJR-GARCH Volatility Scaling).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.univariate.base import ARCHModelResult


def plot_gjr_garch_volatility(conditional_vol: pd.Series):
    """Line chart of GJR-GARCH conditional volatility over time."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(conditional_vol.index, conditional_vol, color="#8172B2", linewidth=1.3)
    ax.set_title("GJR-GARCH Conditional Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (daily)")
    fig.tight_layout()
    return fig


def plot_garch_vs_gjr_garch(garch_vol: pd.Series, gjr_vol: pd.Series):
    """Overlay plain GARCH vs GJR-GARCH conditional volatility, to see
    where the asymmetric model diverges from the symmetric one — the
    gap widens after negative-return periods, which is the leverage
    effect showing up visually."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(garch_vol.index, garch_vol, color="#C44E52", linewidth=1.2, alpha=0.8, label="GARCH(1,1)")
    ax.plot(gjr_vol.index, gjr_vol, color="#8172B2", linewidth=1.2, alpha=0.8, label="GJR-GARCH")
    ax.set_title("GARCH vs GJR-GARCH Conditional Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (daily)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_leverage_effect(fitted_model: ARCHModelResult, shock_range: float = 0.05):
    """News Impact Curve: shows how next-period conditional variance
    responds to a shock of a given size, holding prior variance fixed
    at its long-run (unconditional) level. This is the standard way to
    visualize the leverage effect — the curve should be visibly steeper
    on the negative-shock side than the positive side when gamma > 0.

    Parameters
    ----------
    fitted_model:
        Result from `fit_gjr_garch_model`.
    shock_range:
        Range of shocks to plot, in decimal return units (e.g. 0.05
        plots shocks from -5% to +5%).
    """
    omega = fitted_model.params.get("omega", 0.0)
    alpha = fitted_model.params.get("alpha[1]", 0.0)
    gamma = fitted_model.params.get("gamma[1]", 0.0)
    beta = fitted_model.params.get("beta[1]", 0.0)

    # Long-run (unconditional) variance, used as the held-fixed prior variance.
    # For zero-mean shocks, P(shock < 0) = 0.5, so E[asymmetry term] = gamma/2.
    denom = 1 - alpha - gamma / 2 - beta
    long_run_var = omega / denom if denom > 0 else omega / max(1e-6, 1 - alpha - beta)

    shocks_pct = np.linspace(-shock_range, shock_range, 200) * 100  # model fit on x100 scale
    variance_response = np.where(
        shocks_pct < 0,
        omega + alpha * shocks_pct**2 + gamma * shocks_pct**2 + beta * long_run_var,
        omega + alpha * shocks_pct**2 + beta * long_run_var,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    shocks_decimal = shocks_pct / 100
    negative_mask = shocks_decimal < 0
    ax.plot(shocks_decimal[negative_mask], variance_response[negative_mask], color="#C44E52", linewidth=2, label="Negative shock")
    ax.plot(shocks_decimal[~negative_mask], variance_response[~negative_mask], color="#4C72B0", linewidth=2, label="Positive shock")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"News Impact Curve (leverage effect, gamma={gamma:.4f})")
    ax.set_xlabel("Shock (return)")
    ax.set_ylabel("Next-period conditional variance")
    ax.legend()
    fig.tight_layout()
    return fig
