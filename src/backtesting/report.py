"""
report.py

Responsibility: orchestrate a full VaR backtest — exceptions, Kupiec,
Christoffersen (both tests), and Basel Traffic Light — into the single
combined report Phase 8 asks for. Each individual test still lives in
its own module (exceptions.py, kupiec.py, christoffersen.py, basel.py);
this module just calls them in sequence and packages the results.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from src.backtesting.basel import BaselResult, basel_traffic_light
from src.backtesting.christoffersen import (
    ChristoffersenIndependenceResult,
    ConditionalCoverageResult,
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
)
from src.backtesting.exceptions import ExceptionSummary, summarize_exceptions, identify_exceptions
from src.backtesting.kupiec import KupiecResult, kupiec_pof_test

logger = logging.getLogger(__name__)


@dataclass
class BacktestReport:
    """The full Phase 8 output set, in one place."""

    exception_summary: ExceptionSummary
    kupiec: KupiecResult
    independence: ChristoffersenIndependenceResult
    conditional_coverage: ConditionalCoverageResult
    basel: BaselResult
    overall_conclusion: str  # "Pass" or "Fail"
    exceptions: pd.Series  # the underlying boolean series, for charting


def run_backtest(
    returns: pd.Series,
    var: pd.Series,
    confidence_level: float = 0.95,
    significance: float = 0.05,
) -> BacktestReport:
    """Run the complete Phase 8 backtest: exceptions, Kupiec POF,
    Christoffersen Independence, Christoffersen Conditional Coverage,
    and Basel Traffic Light — all against the same VaR series.

    Parameters
    ----------
    returns:
        Daily return series.
    var:
        VaR forecast series (positive loss magnitude). Typically a
        rolling VaR (e.g. from var.historical.rolling_historical_var,
        or a volatility-scaled equivalent), so out-of-sample days are
        naturally the ones where `var` has a valid (non-NaN) value.
    confidence_level:
        VaR confidence level the model was built at.
    significance:
        Significance level for the Kupiec/Christoffersen pass/fail
        decisions (default 5%).

    Returns
    -------
    BacktestReport
        `overall_conclusion` is "Pass" only if BOTH the Kupiec test and
        the Christoffersen Independence test pass — a model can have the
        right exception COUNT but still fail if exceptions cluster, and
        both matter for a model to be considered well-calibrated.

    Raises
    ------
    ValueError
        Propagated from the underlying test functions (see
        exceptions.identify_exceptions, kupiec.kupiec_pof_test, etc.)
        for cases like empty inputs or no overlapping data.
    """
    exceptions = identify_exceptions(returns, var)
    summary = summarize_exceptions(exceptions, confidence_level)

    kupiec_result = kupiec_pof_test(
        summary.n_observations, summary.n_exceptions, confidence_level, significance
    )
    independence_result = christoffersen_independence_test(exceptions, significance)
    conditional_coverage_result = christoffersen_conditional_coverage_test(exceptions, confidence_level, significance)
    basel_result = basel_traffic_light(
        summary.n_observations, summary.n_exceptions, confidence_level
    )

    overall_conclusion = "Pass" if (kupiec_result.passed and independence_result.passed) else "Fail"

    logger.info(
        "Backtest complete: %d/%d exceptions (expected %.1f), Kupiec=%s, Independence=%s, Basel=%s, Overall=%s",
        summary.n_exceptions, summary.n_observations, summary.expected_exceptions,
        kupiec_result.passed, independence_result.passed, basel_result.zone, overall_conclusion,
    )

    return BacktestReport(
        exception_summary=summary,
        kupiec=kupiec_result,
        independence=independence_result,
        conditional_coverage=conditional_coverage_result,
        basel=basel_result,
        overall_conclusion=overall_conclusion,
        exceptions=exceptions,
    )
