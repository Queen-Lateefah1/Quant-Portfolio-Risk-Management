"""
sector_shock.py

Responsibility: apply an isolated shock to only the tickers in a given
sector, leaving the rest of the portfolio unshocked. This tests
concentration risk — how much a single sector's blowup would actually
hurt, given the portfolio's real weights in that sector (not the whole
portfolio moving together, unlike the market-wide hypothetical shocks).
"""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def apply_sector_shock(
    weights: pd.Series,
    sector_map: Dict[str, str],
    target_sector: str,
    shock_pct: float,
) -> float:
    """Apply a shock to only the tickers belonging to `target_sector`.

    Parameters
    ----------
    weights:
        Portfolio weights, indexed by ticker.
    sector_map:
        Mapping of ticker -> sector name (from data_collection.sectors.get_sector_map).
    target_sector:
        The sector to shock (must match a value in `sector_map` exactly).
    shock_pct:
        Return shock applied to every ticker in `target_sector` (e.g.
        -0.20 for a 20% sector-wide drop). Tickers outside the sector
        are unaffected (implicit 0% shock).

    Returns
    -------
    float
        Portfolio-level return impact (decimal) from this isolated
        sector shock — will be smaller in magnitude than `shock_pct`
        itself unless the whole portfolio is concentrated in that sector.

    Raises
    ------
    ValueError
        If `weights` is empty, or no ticker in `weights` belongs to
        `target_sector` (nothing to shock).
    """
    if weights.empty:
        raise ValueError("weights cannot be empty")

    affected_tickers: List[str] = [t for t in weights.index if sector_map.get(t) == target_sector]

    if not affected_tickers:
        raise ValueError(
            f"No tickers in the portfolio belong to sector '{target_sector}'. "
            f"Sectors present: {sorted(set(sector_map.get(t, 'Unknown') for t in weights.index))}"
        )

    affected_weight = float(weights[affected_tickers].sum())
    portfolio_return_impact = affected_weight * shock_pct

    logger.info(
        "Sector shock: %s (%d tickers, %.1f%% of portfolio) shocked %.1f%% -> portfolio impact %.4f",
        target_sector, len(affected_tickers), affected_weight * 100, shock_pct * 100, portfolio_return_impact,
    )

    return portfolio_return_impact
