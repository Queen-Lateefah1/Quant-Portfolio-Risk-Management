"""
christoffersen.py

Responsibility: Christoffersen Independence and Conditional Coverage tests.

Kupiec's POF test only checks whether the TOTAL exception count is
right — it would pass a model whose exceptions are correctly counted
but come in obviously non-random clusters (e.g. 5 exceptions in a row
during one crash, then nothing for a year). Clustering like that is
dangerous in practice: it means the model fails to adapt when it
matters most. The Christoffersen Independence test specifically checks
for this, treating the exception sequence as a two-state Markov chain
(exception / no-exception) and testing whether the probability of an
exception tomorrow depends on whether there was one today.

Conditional Coverage combines both checks (correct rate AND
independence) into a single joint test: LR_cc = LR_pof + LR_ind,
chi-squared with 2 degrees of freedom under the null.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.backtesting.kupiec import kupiec_pof_test

logger = logging.getLogger(__name__)


@dataclass
class ChristoffersenIndependenceResult:
    """Structured Christoffersen Independence test output."""

    n00: int
    n01: int
    n10: int
    n11: int
    lr_statistic: float
    p_value: float
    passed: bool  # True = fail to reject null = exceptions are independent


@dataclass
class ConditionalCoverageResult:
    """Structured Christoffersen Conditional Coverage test output
    (joint test of correct unconditional coverage AND independence)."""

    lr_pof: float
    lr_independence: float
    lr_conditional_coverage: float
    p_value: float
    passed: bool


def christoffersen_independence_test(exceptions: pd.Series, significance: float = 0.05) -> ChristoffersenIndependenceResult:
    """Test whether VaR exceptions are independently distributed over
    time (no clustering), via a first-order Markov chain likelihood
    ratio test.

    Parameters
    ----------
    exceptions:
        Boolean exception series (from exceptions.identify_exceptions),
        in chronological order.
    significance:
        Test significance level for the pass/fail decision.

    Returns
    -------
    ChristoffersenIndependenceResult

    Raises
    ------
    ValueError
        If `exceptions` has fewer than 2 observations, or `significance`
        not in (0, 1).
    """
    if len(exceptions) < 2:
        raise ValueError("exceptions must have at least 2 observations")
    if not (0 < significance < 1):
        raise ValueError("significance must be strictly between 0 and 1")

    values = exceptions.astype(int).to_numpy()
    prev = values[:-1]
    curr = values[1:]

    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    n0_total = n00 + n01
    n1_total = n10 + n11
    n_total = n0_total + n1_total

    pi01 = n01 / n0_total if n0_total > 0 else 0.0
    pi11 = n11 / n1_total if n1_total > 0 else 0.0
    pi = (n01 + n11) / n_total if n_total > 0 else 0.0

    def _safe_log_term(count, prob):
        if count == 0:
            return 0.0
        if prob <= 0 or prob >= 1:
            return count * 0.0  # degenerate case, contributes nothing further
        return count * np.log(prob)

    log_null = _safe_log_term(n0_total, 1 - pi) + _safe_log_term(n1_total, pi)
    log_alt = (
        _safe_log_term(n00, 1 - pi01) + _safe_log_term(n01, pi01)
        + _safe_log_term(n10, 1 - pi11) + _safe_log_term(n11, pi11)
    )

    lr_statistic = max(0.0, float(-2 * (log_null - log_alt)))
    p_value = 1 - stats.chi2.cdf(lr_statistic, df=1)
    passed = bool(p_value >= significance)

    logger.info(
        "Christoffersen Independence: n00=%d n01=%d n10=%d n11=%d, LR=%.4f, p=%.4f, passed=%s",
        n00, n01, n10, n11, lr_statistic, p_value, passed,
    )

    return ChristoffersenIndependenceResult(
        n00=n00, n01=n01, n10=n10, n11=n11,
        lr_statistic=lr_statistic, p_value=float(p_value), passed=passed,
    )


def christoffersen_conditional_coverage_test(
    exceptions: pd.Series, confidence_level: float = 0.95, significance: float = 0.05
) -> ConditionalCoverageResult:
    """Joint test of correct unconditional coverage (Kupiec) AND
    independence (Christoffersen), combined into a single chi-squared(2) test.

    Parameters
    ----------
    exceptions:
        Boolean exception series.
    confidence_level:
        VaR confidence level.
    significance:
        Test significance level.

    Returns
    -------
    ConditionalCoverageResult
    """
    n_observations = len(exceptions)
    n_exceptions = int(exceptions.sum())

    kupiec_result = kupiec_pof_test(n_observations, n_exceptions, confidence_level, significance)
    independence_result = christoffersen_independence_test(exceptions, significance)

    lr_cc = kupiec_result.lr_statistic + independence_result.lr_statistic
    p_value = 1 - stats.chi2.cdf(lr_cc, df=2)
    passed = bool(p_value >= significance)

    logger.info("Christoffersen Conditional Coverage: LR=%.4f, p=%.4f, passed=%s", lr_cc, p_value, passed)

    return ConditionalCoverageResult(
        lr_pof=kupiec_result.lr_statistic,
        lr_independence=independence_result.lr_statistic,
        lr_conditional_coverage=lr_cc,
        p_value=float(p_value),
        passed=passed,
    )
