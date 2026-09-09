"""
engine.py

Responsibility: orchestrate every Phase 10 scenario (5 historical, 3
market shocks, volatility doubling, correlation spike, sector shock)
into one summary table, and rank them by severity. Individual scenario
math lives in scenarios.py, volatility_shocks.py, and sector_shock.py —
this module only calls them and assembles results.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from src.stress_testing.scenarios import (
    HISTORICAL_SCENARIOS,
    HYPOTHETICAL_MARKET_SHOCKS,
    compute_scenario_return,
    fetch_scenario_prices,
    stressed_loss_from_return,
)
from src.stress_testing.volatility_shocks import (
    apply_correlation_spike_shock,
    apply_volatility_doubling_shock,
)
from src.stress_testing.sector_shock import apply_sector_shock

logger = logging.getLogger(__name__)


@dataclass
class StressTestRow:
    """One scenario's row in the Phase 10 summary table."""

    scenario: str
    category: str
    stressed_return: Optional[float]
    stressed_loss: float
    stressed_var: Optional[float]
    stressed_es: Optional[float]


def run_historical_scenarios(weights: pd.Series, portfolio_value: float) -> List[StressTestRow]:
    """Run every defined historical scenario (real market data, network required).

    Parameters
    ----------
    weights:
        Portfolio weights, indexed by ticker.
    portfolio_value:
        Current portfolio value.

    Returns
    -------
    List[StressTestRow]
        One row per historical scenario. A scenario that fails to
        download (e.g. network issue) is skipped with a warning rather
        than aborting the whole run — partial results are still useful.
    """
    tickers = list(weights.index)
    rows = []
    for scenario_name in HISTORICAL_SCENARIOS:
        try:
            prices = fetch_scenario_prices(tickers, scenario_name)
            scenario_return = compute_scenario_return(prices, weights)
            loss = stressed_loss_from_return(scenario_return, portfolio_value)
            rows.append(StressTestRow(
                scenario=scenario_name, category="Historical",
                stressed_return=scenario_return, stressed_loss=loss,
                stressed_var=None, stressed_es=None,
            ))
        except Exception as exc:
            logger.warning("Skipping historical scenario '%s': %s", scenario_name, exc)
    return rows


def run_hypothetical_market_shocks(portfolio_value: float) -> List[StressTestRow]:
    """Run every defined hypothetical market-wide shock (pure calculation, no network)."""
    rows = []
    for scenario_name, shock_pct in HYPOTHETICAL_MARKET_SHOCKS.items():
        loss = stressed_loss_from_return(shock_pct, portfolio_value)
        rows.append(StressTestRow(
            scenario=scenario_name, category="Hypothetical",
            stressed_return=shock_pct, stressed_loss=loss,
            stressed_var=None, stressed_es=None,
        ))
    return rows


def run_volatility_and_correlation_shocks(
    portfolio_returns: pd.Series,
    returns_matrix: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> List[StressTestRow]:
    """Run the volatility-doubling and correlation-spike scenarios."""
    rows = []

    vol_result = apply_volatility_doubling_shock(portfolio_returns, vol_multiplier=2.0, confidence_level=confidence_level)
    rows.append(StressTestRow(
        scenario="Volatility doubles", category="Hypothetical",
        stressed_return=None, stressed_loss=vol_result.stressed_var * portfolio_value,
        stressed_var=vol_result.stressed_var, stressed_es=vol_result.stressed_es,
    ))

    corr_result = apply_correlation_spike_shock(returns_matrix, weights, stressed_correlation=0.9, confidence_level=confidence_level)
    rows.append(StressTestRow(
        scenario="Correlation spikes", category="Hypothetical",
        stressed_return=None, stressed_loss=corr_result.stressed_var * portfolio_value,
        stressed_var=corr_result.stressed_var, stressed_es=corr_result.stressed_es,
    ))

    return rows


def run_sector_shock(
    weights: pd.Series,
    sector_map: Dict[str, str],
    portfolio_value: float,
    shock_pct: float = -0.20,
) -> Optional[StressTestRow]:
    """Run a sector-specific shock against the portfolio's largest sector by weight.

    Parameters
    ----------
    weights, sector_map, portfolio_value:
        See sector_shock.apply_sector_shock.
    shock_pct:
        Shock magnitude applied to the target sector.

    Returns
    -------
    Optional[StressTestRow]
        None if `sector_map` has no usable sector data (e.g. all "Unknown").
    """
    sector_weights: Dict[str, float] = {}
    for ticker, weight in weights.items():
        sector = sector_map.get(ticker, "Unknown")
        if sector != "Unknown":
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

    if not sector_weights:
        logger.warning("No usable sector data — skipping sector shock scenario")
        return None

    target_sector = max(sector_weights, key=sector_weights.get)
    impact = apply_sector_shock(weights, sector_map, target_sector, shock_pct)
    loss = stressed_loss_from_return(impact, portfolio_value)

    return StressTestRow(
        scenario=f"Sector-specific shock ({target_sector} {shock_pct*100:.0f}%)",
        category="Hypothetical",
        stressed_return=impact, stressed_loss=loss,
        stressed_var=None, stressed_es=None,
    )


def run_all_stress_tests(
    portfolio_returns: pd.Series,
    returns_matrix: pd.DataFrame,
    weights: pd.Series,
    sector_map: Dict[str, str],
    portfolio_value: float = 1_000_000.0,
    confidence_level: float = 0.95,
    include_historical: bool = True,
) -> pd.DataFrame:
    """Run every Phase 10 scenario and assemble the full summary table.

    Parameters
    ----------
    portfolio_returns:
        Daily portfolio-level return series.
    returns_matrix:
        Daily asset-level returns (needed for the correlation shock).
    weights:
        Portfolio weights.
    sector_map:
        Ticker -> sector mapping (from data_collection.sectors).
    portfolio_value:
        Current portfolio value.
    confidence_level:
        VaR/ES confidence level for the parametric scenarios.
    include_historical:
        Set False to skip the network-dependent historical scenarios
        (useful for fast local testing without live market data).

    Returns
    -------
    pd.DataFrame
        One row per scenario, indexed by scenario name, with columns:
        category, stressed_return, stressed_loss, stressed_var, stressed_es.

    Raises
    ------
    ValueError
        If `portfolio_returns` is empty.
    """
    if portfolio_returns.empty:
        raise ValueError("portfolio_returns cannot be empty")

    rows: List[StressTestRow] = []

    if include_historical:
        rows.extend(run_historical_scenarios(weights, portfolio_value))

    rows.extend(run_hypothetical_market_shocks(portfolio_value))
    rows.extend(run_volatility_and_correlation_shocks(
        portfolio_returns, returns_matrix, weights, portfolio_value, confidence_level
    ))

    sector_row = run_sector_shock(weights, sector_map, portfolio_value)
    if sector_row is not None:
        rows.append(sector_row)

    table = pd.DataFrame([vars(r) for r in rows]).set_index("scenario")
    logger.info("Ran %d stress test scenarios", len(table))
    return table


def rank_scenarios(stress_test_table: pd.DataFrame) -> pd.DataFrame:
    """Rank scenarios from worst-case loss to least severe.

    Parameters
    ----------
    stress_test_table:
        Output of `run_all_stress_tests`.

    Returns
    -------
    pd.DataFrame
        Same table, sorted by `stressed_loss` descending (worst first),
        with a `rank` column added (1 = worst).

    Raises
    ------
    ValueError
        If `stress_test_table` is empty.
    """
    if stress_test_table.empty:
        raise ValueError("stress_test_table cannot be empty")

    ranked = stress_test_table.sort_values(by="stressed_loss", ascending=False).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked
