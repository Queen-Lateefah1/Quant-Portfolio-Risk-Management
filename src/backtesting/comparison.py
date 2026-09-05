"""
comparison.py

Responsibility: Phase 9 — compare Historical, EWMA-scaled, GARCH-scaled,
and GJR-GARCH-scaled VaR against each other on the same return series,
using a common set of metrics and backtests, and produce a ranking.

This module only aggregates and compares results computed elsewhere —
it does not reimplement VaR calculation, volatility scaling, or
backtesting logic. Callers pass in a dict of pre-computed VaR series
(one per model) alongside the return series.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.backtesting.exceptions import identify_exceptions, summarize_exceptions
from src.backtesting.report import run_backtest
from src.var.expected_shortfall import calculate_expected_shortfall

logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonRow:
    """One model's row in the comparison table (per guide 2.5)."""

    model: str
    n_exceptions: int
    exception_ratio: float
    expected_ratio: float
    average_var: float
    var_volatility: float
    worst_daily_loss: float
    expected_shortfall: float
    kupiec_p_value: float
    christoffersen_p_value: float
    basel_zone: str
    overall_conclusion: str


def compare_models(
    returns: pd.Series,
    var_by_model: Dict[str, pd.Series],
    confidence_level: float = 0.95,
    significance: float = 0.05,
) -> pd.DataFrame:
    """Build the Phase 9 model comparison table across multiple VaR series.

    Parameters
    ----------
    returns:
        Daily portfolio return series.
    var_by_model:
        Dict mapping model name (e.g. "Historical", "EWMA", "GARCH",
        "GJR-GARCH") to its VaR series, aligned by date to `returns`.
        Each series is typically a rolling VaR (e.g. from
        `rolling_historical_var` or a volatility-scaled equivalent) so
        that backtesting has an out-of-sample period to evaluate.
    confidence_level:
        The VaR confidence level all models were built at (must be the
        same across models being compared, or the comparison isn't
        apples-to-apples).
    significance:
        Significance level for the Kupiec/Christoffersen pass-fail
        decisions feeding into `overall_conclusion`.

    Returns
    -------
    pd.DataFrame
        One row per model, indexed by model name, with columns matching
        every metric the doc's Phase 9 "Metrics" list asks for.

    Raises
    ------
    ValueError
        If `returns` is empty or `var_by_model` is empty.
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")
    if not var_by_model:
        raise ValueError("var_by_model cannot be empty")

    rows: List[ModelComparisonRow] = []

    for model_name, var_series in var_by_model.items():
        exceptions = identify_exceptions(returns, var_series)
        summary = summarize_exceptions(exceptions, confidence_level)

        aligned_returns, aligned_var = (-returns).align(var_series, join="inner")
        valid = aligned_var.notna()
        aligned_returns = aligned_returns[valid]
        aligned_var = aligned_var[valid]

        worst_daily_loss = float(aligned_returns.max())  # max loss = max of the loss series
        average_var = float(aligned_var.mean())
        var_volatility = float(aligned_var.std())

        # Expected Shortfall computed on the same out-of-sample window, using
        # the full aligned return series (not the VaR series) as its input.
        matching_returns = returns.loc[aligned_var.index]
        es = calculate_expected_shortfall(matching_returns, confidence_level)

        backtest = run_backtest(returns, var_series, confidence_level, significance)

        rows.append(ModelComparisonRow(
            model=model_name,
            n_exceptions=summary.n_exceptions,
            exception_ratio=summary.exception_ratio,
            expected_ratio=summary.expected_ratio,
            average_var=average_var,
            var_volatility=var_volatility,
            worst_daily_loss=worst_daily_loss,
            expected_shortfall=es,
            kupiec_p_value=backtest.kupiec.p_value,
            christoffersen_p_value=backtest.independence.p_value,
            basel_zone=backtest.basel.zone,
            overall_conclusion=backtest.overall_conclusion,
        ))

    table = pd.DataFrame([vars(row) for row in rows]).set_index("model")
    logger.info("Compared %d models: %s", len(rows), list(table.index))
    return table


def rank_models(comparison_table: pd.DataFrame) -> pd.DataFrame:
    """Rank models from the Phase 9 comparison table.

    Ranking heuristic (in priority order):
    1. Models that PASS backtesting (overall_conclusion == "Pass") rank
       above models that fail, regardless of any other metric — a model
       that isn't statistically well-calibrated shouldn't be preferred
       just because it happens to have a lower ES.
    2. Among passing models (or, if none pass, among all models), rank
       by how close the observed exception ratio is to the expected
       ratio — the closer to expected, the better-calibrated the model.

    This is a reasonable default, not the only valid ranking — a risk
    manager might weight Expected Shortfall or Basel zone more heavily
    depending on their priorities. The full table (returned unranked by
    `compare_models`) has every metric needed to re-rank differently.

    Parameters
    ----------
    comparison_table:
        Output of `compare_models`.

    Returns
    -------
    pd.DataFrame
        Same table, sorted best-to-worst, with an added `rank` column
        (1 = best).

    Raises
    ------
    ValueError
        If `comparison_table` is empty.
    """
    if comparison_table.empty:
        raise ValueError("comparison_table cannot be empty")

    table = comparison_table.copy()
    table["calibration_gap"] = (table["exception_ratio"] - table["expected_ratio"]).abs()
    table["passed_bool"] = table["overall_conclusion"] == "Pass"

    ranked = table.sort_values(by=["passed_bool", "calibration_gap"], ascending=[False, True])
    ranked = ranked.drop(columns=["passed_bool", "calibration_gap"])
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    logger.info("Model ranking: %s", list(ranked.index))
    return ranked
