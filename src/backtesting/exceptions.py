"""
exceptions.py

Responsibility: identify VaR "exceptions" — days where the actual loss
exceeded the VaR prediction — and summarize them. Every backtesting
test in this package (kupiec.py, christoffersen.py, basel.py) operates
on the exception indicator series this module produces.
"""

import logging
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExceptionSummary:
    """Structured exception count summary (per guide 2.5)."""

    n_observations: int
    n_exceptions: int
    expected_exceptions: float
    exception_ratio: float
    expected_ratio: float


def identify_exceptions(returns: pd.Series, var: Union[float, pd.Series]) -> pd.Series:
    """Identify which days had a VaR exception (actual loss > VaR).

    Parameters
    ----------
    returns:
        Daily portfolio return series (decimal form, e.g. -0.02 = -2%).
    var:
        Either a single static VaR figure (positive loss magnitude)
        applied to every day, or a pd.Series of VaR estimates aligned
        (by index) to `returns` -- e.g. from `rolling_historical_var`.

    Returns
    -------
    pd.Series
        Boolean series. Rows where VaR is NaN (e.g. the initial
        rolling-window warm-up period) are dropped rather than
        silently treated as False.

    Raises
    ------
    ValueError
        If `returns` is empty.
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")

    losses = -returns

    if isinstance(var, pd.Series):
        aligned_losses, aligned_var = losses.align(var, join="inner")
        valid = aligned_var.notna()
        aligned_losses = aligned_losses[valid]
        aligned_var = aligned_var[valid]
        exceptions = aligned_losses > aligned_var
    else:
        exceptions = losses > var

    exceptions.name = "exception"
    return exceptions


def summarize_exceptions(exceptions: pd.Series, confidence_level: float = 0.95) -> ExceptionSummary:
    """Summarize an exception indicator series against its expected rate.

    Parameters
    ----------
    exceptions:
        Boolean series from `identify_exceptions`.
    confidence_level:
        The VaR confidence level the exceptions were measured against.

    Returns
    -------
    ExceptionSummary

    Raises
    ------
    ValueError
        If `exceptions` is empty, or `confidence_level` is not in (0, 1).
    """
    if exceptions.empty:
        raise ValueError("exceptions cannot be empty")
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be strictly between 0 and 1")

    n_obs = len(exceptions)
    n_exceptions = int(exceptions.sum())
    expected_ratio = 1 - confidence_level
    expected_exceptions = n_obs * expected_ratio
    exception_ratio = n_exceptions / n_obs

    summary = ExceptionSummary(
        n_observations=n_obs,
        n_exceptions=n_exceptions,
        expected_exceptions=expected_exceptions,
        exception_ratio=exception_ratio,
        expected_ratio=expected_ratio,
    )
    logger.info(
        "Exceptions: %d/%d observed (%.2f%%), %.2f expected (%.2f%%)",
        n_exceptions, n_obs, exception_ratio * 100, expected_exceptions, expected_ratio * 100,
    )
    return summary
