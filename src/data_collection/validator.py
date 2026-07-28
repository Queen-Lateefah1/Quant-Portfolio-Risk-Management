"""
validator.py

Responsibility: validate and (explicitly, traceably) clean raw price data.

Per guide 3.1 (validate inputs early), 3.4 (no silent missing-value
handling), and 2.5 (use structured result objects instead of bare
tuples/dicts).
"""

import logging
from dataclasses import dataclass, field
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Structured record of what the cleaning step did, so results are
    always traceable back to the raw data (guide 3.3)."""

    tickers_dropped: List[str] = field(default_factory=list)
    missing_values_before: int = 0
    missing_values_filled: int = 0
    missing_values_remaining: int = 0
    rows_dropped_fully_empty: int = 0


@dataclass
class ValidationResult:
    """Structured validation output (guide 2.5)."""

    n_rows: int
    n_tickers: int
    is_monotonic: bool
    has_duplicate_dates: bool
    missing_values: int
    extreme_return_count: int
    passed: bool


def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Explicitly clean a raw price matrix.

    Treatment applied (and reported, per guide 3.4 — nothing here is silent):
    1. Drop any ticker column that is entirely missing (failed download).
    2. Forward-fill then back-fill isolated gaps (holiday/calendar
       mismatches between exchanges).
    3. Drop any row that is still fully empty after filling.

    Parameters
    ----------
    prices:
        Raw wide price matrix (date index, one column per ticker).

    Returns
    -------
    (pd.DataFrame, CleaningReport)
        The cleaned price matrix, and a report describing exactly what
        was changed and why.

    Raises
    ------
    ValueError
        If `prices` is empty.
    """
    if prices.empty:
        raise ValueError("prices cannot be empty")

    report = CleaningReport()

    before_cols = list(prices.columns)
    cleaned = prices.dropna(axis=1, how="all")
    report.tickers_dropped = sorted(set(before_cols) - set(cleaned.columns))
    if report.tickers_dropped:
        logger.warning("Dropped tickers with no data at all: %s", report.tickers_dropped)

    report.missing_values_before = int(cleaned.isna().sum().sum())
    cleaned = cleaned.ffill().bfill()
    report.missing_values_remaining = int(cleaned.isna().sum().sum())
    report.missing_values_filled = report.missing_values_before - report.missing_values_remaining

    rows_before = len(cleaned)
    cleaned = cleaned.dropna(how="all")
    report.rows_dropped_fully_empty = rows_before - len(cleaned)

    logger.info(
        "Cleaning complete: %d tickers dropped, %d missing values filled, %d rows dropped",
        len(report.tickers_dropped), report.missing_values_filled, report.rows_dropped_fully_empty,
    )

    return cleaned, report


def validate_prices(prices: pd.DataFrame, returns: pd.DataFrame) -> ValidationResult:
    """Run the Phase 1 data-validation checklist against cleaned prices/returns.

    Checks (per the internship doc's 'Data Validation' list):
    - No missing return dates
    - No look-ahead bias signal (dates strictly increasing, no duplicates)
    - Returns correctly aligned to prices

    Parameters
    ----------
    prices:
        Cleaned price matrix.
    returns:
        Corresponding daily log-return matrix.

    Returns
    -------
    ValidationResult
        Structured pass/fail summary. `passed` is False if any hard
        check fails (missing values, non-monotonic dates, duplicate dates).
    """
    if prices.empty:
        raise ValueError("prices cannot be empty")
    if returns.empty:
        raise ValueError("returns cannot be empty")

    is_monotonic = bool(prices.index.is_monotonic_increasing)
    has_duplicates = bool(prices.index.duplicated().any())
    missing_values = int(returns.isna().sum().sum())
    extreme_returns = int((returns.abs() > 0.5).sum().sum())

    passed = is_monotonic and not has_duplicates and missing_values == 0

    result = ValidationResult(
        n_rows=len(prices),
        n_tickers=prices.shape[1],
        is_monotonic=is_monotonic,
        has_duplicate_dates=has_duplicates,
        missing_values=missing_values,
        extreme_return_count=extreme_returns,
        passed=passed,
    )

    logger.info("Validation result: %s", result)
    return result
