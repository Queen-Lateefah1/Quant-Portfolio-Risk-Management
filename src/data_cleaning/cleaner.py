"""
cleaner.py

Responsibility: Phase 2 data-quality treatment — stale price detection
and winsorization of extreme return errors.

Missing-value handling for raw downloads lives in
data_collection/validator.py (Phase 1 concern: did the download
succeed?). This module handles Phase 2 concerns: does the cleaned data
look economically sensible?
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StalePriceReport:
    """Structured record of stale-price treatment."""

    stale_runs_found: Dict[str, int] = field(default_factory=dict)
    values_replaced: int = 0


@dataclass
class WinsorizeReport:
    """Structured record of winsorization treatment."""

    lower_percentile: float = 0.0
    upper_percentile: float = 0.0
    values_clipped_low: Dict[str, int] = field(default_factory=dict)
    values_clipped_high: Dict[str, int] = field(default_factory=dict)


def remove_stale_prices(prices: pd.DataFrame, min_repeat_run: int = 5) -> tuple[pd.DataFrame, StalePriceReport]:
    """Flag and treat "stale" prices — a price repeated for an unusually
    long run of consecutive days, which typically signals a trading halt
    or a stale data feed rather than genuine zero price movement.

    Treatment: any run of `min_repeat_run` or more identical consecutive
    values is set to NaN (except the first value in the run) and then
    forward-filled from the last genuinely-traded price. This is
    reported explicitly per ticker, per guide 3.4.

    Parameters
    ----------
    prices:
        Clean, aligned price matrix.
    min_repeat_run:
        Minimum number of consecutive identical values to treat as a
        stale run. Default 5 trading days (~1 week).

    Returns
    -------
    (pd.DataFrame, StalePriceReport)

    Raises
    ------
    ValueError
        If `prices` is empty or `min_repeat_run` < 2.
    """
    if prices.empty:
        raise ValueError("prices cannot be empty")
    if min_repeat_run < 2:
        raise ValueError("min_repeat_run must be >= 2")

    result = prices.copy()
    report = StalePriceReport()

    for col in result.columns:
        series = result[col]
        is_repeat = series.eq(series.shift(1))
        run_id = (~is_repeat).cumsum()
        run_lengths = series.groupby(run_id).transform("size")

        stale_mask = is_repeat & (run_lengths >= min_repeat_run)
        n_stale = int(stale_mask.sum())
        if n_stale > 0:
            result.loc[stale_mask, col] = np.nan
            report.stale_runs_found[col] = n_stale
            report.values_replaced += n_stale

    if report.stale_runs_found:
        result = result.ffill()
        logger.warning("Stale prices treated: %s", report.stale_runs_found)

    return result, report


def winsorize_returns(
    returns: pd.DataFrame,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> tuple[pd.DataFrame, WinsorizeReport]:
    """Winsorize extreme daily returns, per ticker, to reduce the
    influence of likely data errors (fat-finger prints, bad ticks)
    without deleting observations.

    Each ticker's return series is independently clipped at its own
    `lower_percentile`/`upper_percentile` values — a return more extreme
    than that percentile is replaced with the percentile value itself.

    Parameters
    ----------
    returns:
        Daily return matrix (log or simple), one column per ticker.
    lower_percentile:
        Lower clipping bound, e.g. 0.01 for the 1st percentile.
    upper_percentile:
        Upper clipping bound, e.g. 0.99 for the 99th percentile.

    Returns
    -------
    (pd.DataFrame, WinsorizeReport)

    Raises
    ------
    ValueError
        If `returns` is empty, or if the percentile bounds are invalid.
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")
    if not (0 <= lower_percentile < upper_percentile <= 1):
        raise ValueError("require 0 <= lower_percentile < upper_percentile <= 1")

    result = returns.copy()
    report = WinsorizeReport(lower_percentile=lower_percentile, upper_percentile=upper_percentile)

    for col in result.columns:
        series = result[col]
        lower_bound = series.quantile(lower_percentile)
        upper_bound = series.quantile(upper_percentile)

        n_low = int((series < lower_bound).sum())
        n_high = int((series > upper_bound).sum())

        if n_low:
            report.values_clipped_low[col] = n_low
        if n_high:
            report.values_clipped_high[col] = n_high

        result[col] = series.clip(lower=lower_bound, upper=upper_bound)

    logger.info(
        "Winsorization complete at [%s, %s]: clipped low=%s, high=%s",
        lower_percentile, upper_percentile, report.values_clipped_low, report.values_clipped_high,
    )

    return result, report
