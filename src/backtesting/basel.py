"""
basel.py

Responsibility: Basel Traffic Light test — the regulatory standard for
classifying VaR model backtesting results into Green/Yellow/Red zones,
based on how (statistically) surprising the observed exception count is
under the model's own assumptions.

Official methodology: compute the cumulative binomial probability of
observing AT MOST the actual number of exceptions, given n observations
and the model's expected exception probability p = 1 - confidence_level.
- Green zone: cumulative probability < 95% (exception count unremarkable)
- Yellow zone: 95% <= cumulative probability < 99.99% (statistically
  unlikely — increased regulatory scrutiny / capital multiplier)
- Red zone: cumulative probability >= 99.99% (model rejected — the
  observed exception count would essentially never happen if the model
  were correctly calibrated)

This binomial-cumulative-probability formulation generalizes the
Basel Committee's published 250-observation/99%-confidence table (which
lists Green: 0-4, Yellow: 5-9, Red: 10+ exceptions) to any sample size
and confidence level.
"""

import logging
from dataclasses import dataclass

from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class BaselResult:
    """Structured Basel Traffic Light test output."""

    n_observations: int
    n_exceptions: int
    cumulative_probability: float
    zone: str  # "Green", "Yellow", or "Red"


def basel_traffic_light(n_observations: int, n_exceptions: int, confidence_level: float = 0.99) -> BaselResult:
    """Classify a VaR model's backtesting result into a Basel traffic light zone.

    Parameters
    ----------
    n_observations:
        Total number of days evaluated.
    n_exceptions:
        Number of VaR exceptions observed.
    confidence_level:
        VaR confidence level (Basel's official table is defined at 99%,
        but this function works for any confidence level).

    Returns
    -------
    BaselResult

    Raises
    ------
    ValueError
        If `n_observations` <= 0, `n_exceptions` < 0, `n_exceptions` >
        `n_observations`, or `confidence_level` not in (0, 1).
    """
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    if n_exceptions < 0:
        raise ValueError("n_exceptions cannot be negative")
    if n_exceptions > n_observations:
        raise ValueError("n_exceptions cannot exceed n_observations")
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be strictly between 0 and 1")

    p = 1 - confidence_level
    cumulative_probability = float(stats.binom.cdf(n_exceptions, n_observations, p))

    if cumulative_probability < 0.95:
        zone = "Green"
    elif cumulative_probability < 0.9999:
        zone = "Yellow"
    else:
        zone = "Red"

    logger.info(
        "Basel Traffic Light: n=%d, x=%d, p=%.4f, cum_prob=%.6f, zone=%s",
        n_observations, n_exceptions, p, cumulative_probability, zone,
    )

    return BaselResult(
        n_observations=n_observations, n_exceptions=n_exceptions,
        cumulative_probability=cumulative_probability, zone=zone,
    )
