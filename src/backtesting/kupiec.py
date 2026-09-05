"""
kupiec.py

Responsibility: Kupiec Proportion of Failures (POF) test.

Tests whether the OBSERVED exception rate is statistically consistent
with the EXPECTED exception rate implied by the VaR confidence level.
Under the null hypothesis (the model is correctly calibrated),
exceptions occur independently with probability p = 1 - confidence_level,
so the exception count follows a Binomial(n, p) distribution.

The Kupiec likelihood ratio statistic:

    LR_pof = -2 * ln[ (1-p)^(n-x) * p^x ] + 2 * ln[ (1-x/n)^(n-x) * (x/n)^x ]

is asymptotically chi-squared distributed with 1 degree of freedom
under the null. A low p-value means the observed exception rate is
significantly different from what the model predicted — i.e. the model
is miscalibrated (either too many exceptions = model understates risk,
or too few = model is overly conservative).
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class KupiecResult:
    """Structured Kupiec POF test output."""

    n_observations: int
    n_exceptions: int
    expected_ratio: float
    observed_ratio: float
    lr_statistic: float
    p_value: float
    passed: bool  # True = fail to reject null = model calibration is acceptable


def kupiec_pof_test(n_observations: int, n_exceptions: int, confidence_level: float = 0.95, significance: float = 0.05) -> KupiecResult:
    """Run the Kupiec Proportion of Failures test.

    Parameters
    ----------
    n_observations:
        Total number of days evaluated.
    n_exceptions:
        Number of VaR exceptions observed.
    confidence_level:
        The VaR confidence level being tested (e.g. 0.95).
    significance:
        Test significance level for the pass/fail decision (default 5%).
        `passed=True` if p_value >= significance (fail to reject the
        null that the model is correctly calibrated).

    Returns
    -------
    KupiecResult

    Raises
    ------
    ValueError
        If `n_observations` <= 0, `n_exceptions` < 0, `n_exceptions` >
        `n_observations`, or `confidence_level`/`significance` not in (0, 1).
    """
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    if n_exceptions < 0:
        raise ValueError("n_exceptions cannot be negative")
    if n_exceptions > n_observations:
        raise ValueError("n_exceptions cannot exceed n_observations")
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if not (0 < significance < 1):
        raise ValueError("significance must be strictly between 0 and 1")

    p = 1 - confidence_level  # expected exception probability
    n, x = n_observations, n_exceptions
    observed_ratio = x / n

    # Handle the boundary cases (0 or all exceptions) where log(0) would occur.
    if x == 0:
        log_likelihood_ratio = -2 * n * np.log(1 - p)
    elif x == n:
        log_likelihood_ratio = -2 * n * np.log(p)
    else:
        log_null = (n - x) * np.log(1 - p) + x * np.log(p)
        log_alt = (n - x) * np.log(1 - observed_ratio) + x * np.log(observed_ratio)
        log_likelihood_ratio = -2 * (log_null - log_alt)

    lr_statistic = max(0.0, float(log_likelihood_ratio))  # numerical guard against tiny negative noise
    p_value = 1 - stats.chi2.cdf(lr_statistic, df=1)
    passed = bool(p_value >= significance)

    logger.info(
        "Kupiec POF: n=%d, x=%d, expected_ratio=%.4f, observed_ratio=%.4f, LR=%.4f, p=%.4f, passed=%s",
        n, x, p, observed_ratio, lr_statistic, p_value, passed,
    )

    return KupiecResult(
        n_observations=n, n_exceptions=x, expected_ratio=p, observed_ratio=observed_ratio,
        lr_statistic=lr_statistic, p_value=float(p_value), passed=passed,
    )
