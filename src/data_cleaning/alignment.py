"""
alignment.py

Responsibility: align trading dates across all tickers in the universe.

Different tickers can have slightly different trading calendars (late
IPOs, temporary halts, exchange-specific holidays). This module makes
sure every column in the price matrix shares the same date index before
returns are computed, so no ticker silently skews the sample.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AlignmentReport:
    """Structured record of what alignment changed."""

    dates_before: int = 0
    dates_after: int = 0
    dates_dropped: int = 0
    tickers_with_gaps_filled: List[str] = field(default_factory=list)


def align_trading_dates(prices: pd.DataFrame) -> tuple[pd.DataFrame, AlignmentReport]:
    """Align all tickers to a common set of trading dates.

    Treatment (explicit, per guide 3.4 — nothing here is silent):
    1. Restrict to the date range every ticker has at least one observation in
       (the intersection of each column's first/last valid date).
    2. Within that shared range, forward-fill any ticker-specific gaps so
       every column has a value on every date in the common index.

    Parameters
    ----------
    prices:
        Wide price matrix (date index, one column per ticker). May contain
        NaNs where a given ticker didn't trade on a shared date.

    Returns
    -------
    (pd.DataFrame, AlignmentReport)
        The aligned price matrix, and a report describing exactly what
        was changed.

    Raises
    ------
    ValueError
        If `prices` is empty.
    """
    if prices.empty:
        raise ValueError("prices cannot be empty")

    report = AlignmentReport(dates_before=len(prices))

    # Common range: latest of the per-ticker first valid dates,
    # to the earliest of the per-ticker last valid dates.
    first_valid = prices.apply(lambda col: col.first_valid_index())
    last_valid = prices.apply(lambda col: col.last_valid_index())
    common_start = first_valid.max()
    common_end = last_valid.min()

    aligned = prices.loc[common_start:common_end].copy()

    gaps_before = aligned.isna().sum()
    aligned = aligned.ffill().bfill()
    gaps_after = aligned.isna().sum()

    report.tickers_with_gaps_filled = sorted(
        gaps_before[gaps_before > gaps_after].index.tolist()
    )
    report.dates_after = len(aligned)
    report.dates_dropped = report.dates_before - report.dates_after

    logger.info(
        "Alignment complete: %d dates dropped, gaps filled for tickers: %s",
        report.dates_dropped, report.tickers_with_gaps_filled,
    )

    return aligned, report
