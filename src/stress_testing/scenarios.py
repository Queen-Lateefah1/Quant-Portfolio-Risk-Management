"""
scenarios.py

Responsibility: define and apply historical and hypothetical market
shock scenarios.

Historical scenarios are replayed using the portfolio's ACTUAL current
holdings against ACTUAL market data from real crisis periods — this is
the standard "historical scenario replay" methodology used by risk
desks, not a made-up shock magnitude. It answers: "if this exact crisis
happened again to this exact portfolio, what would happen?"

The fetch step (network-dependent) and the calculation step (pure) are
deliberately split into two functions, so the calculation logic can be
unit tested without needing live market data.
"""

import logging
from typing import Dict, Tuple

import pandas as pd

from src.data_collection.downloader import fetch_adjusted_close

logger = logging.getLogger(__name__)


# Date ranges chosen to capture peak-to-trough of each crisis for the
# assets this project's universe would have been exposed to.
HISTORICAL_SCENARIOS: Dict[str, Tuple[str, str]] = {
    "2008 Financial Crisis": ("2008-09-01", "2009-03-09"),   # Lehman collapse to market bottom
    "COVID Crash 2020": ("2020-02-19", "2020-03-23"),         # pre-crash peak to bottom
    "Inflation Shock 2022": ("2022-01-03", "2022-10-13"),     # 2022 drawdown, peak to trough
    "Banking Stress 2023": ("2023-03-08", "2023-03-31"),      # SVB collapse window
    "Tech Selloff Scenario": ("2022-01-03", "2022-12-28"),    # full-year 2022 tech-heavy drawdown
}

HYPOTHETICAL_MARKET_SHOCKS: Dict[str, float] = {
    "Equity market down 5%": -0.05,
    "Equity market down 10%": -0.10,
    "Equity market down 20%": -0.20,
}


def fetch_scenario_prices(tickers, scenario_name: str) -> pd.DataFrame:
    """Download actual historical prices for the given scenario's date window.

    Parameters
    ----------
    tickers:
        List of ticker symbols (the portfolio's current universe).
    scenario_name:
        Must be a key in HISTORICAL_SCENARIOS.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices for `tickers` across the scenario window.

    Raises
    ------
    ValueError
        If `scenario_name` is not a recognized historical scenario.
    """
    if scenario_name not in HISTORICAL_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. Available: {list(HISTORICAL_SCENARIOS.keys())}"
        )
    start, end = HISTORICAL_SCENARIOS[scenario_name]
    logger.info("Fetching scenario prices for '%s' (%s to %s)", scenario_name, start, end)
    return fetch_adjusted_close(tickers, start_date=start, end_date=end)


def compute_scenario_return(prices: pd.DataFrame, weights: pd.Series) -> float:
    """Compute the portfolio-level return over a historical scenario window.

    Each ticker's total (peak-to-trough or start-to-end) return over the
    window is weighted by its current portfolio weight — this answers
    "what would TODAY's portfolio have experienced during THAT crisis,"
    which is a different (and more decision-relevant) question than
    "what did the market do during that crisis."

    A ticker with no data for the scenario window (most commonly: it
    hadn't IPO'd yet — e.g. Tesla in 2008, Meta before 2012) is excluded
    from the calculation, and the REMAINING tickers' weights are
    renormalized to sum back to 1.0, so the result answers "what would
    the portion of the portfolio that existed back then have done,"
    rather than silently understating the loss by treating the missing
    weight as a 0% contribution. Excluded tickers are logged explicitly
    (never silent) — check logs/warnings when replaying scenarios far
    enough in the past that young companies may not have existed yet.

    Parameters
    ----------
    prices:
        Adjusted close price matrix over the scenario window (from
        `fetch_scenario_prices`), one column per ticker.
    weights:
        Current portfolio weights, indexed by ticker.

    Returns
    -------
    float
        Portfolio return over the scenario window (decimal, can be
        negative for a loss or positive for a gain), computed over
        whichever tickers had usable data, reweighted to sum to 1.0.

    Raises
    ------
    ValueError
        If `prices` is empty, if `weights` references tickers not
        present in `prices` at all (a genuine data problem, not a
        pre-IPO gap), or if EVERY ticker is missing data for this
        window (nothing usable to compute).
    """
    if prices.empty:
        raise ValueError("prices cannot be empty")

    missing = set(weights.index) - set(prices.columns)
    if missing:
        raise ValueError(f"weights reference tickers not in prices: {sorted(missing)}")

    ticker_returns = prices.iloc[-1] / prices.iloc[0] - 1
    ticker_returns = ticker_returns[weights.index]

    valid_mask = ticker_returns.notna()
    if not valid_mask.all():
        excluded = ticker_returns.index[~valid_mask].tolist()
        excluded_weight = float(weights[~valid_mask].sum())
        logger.warning(
            "Excluded %d ticker(s) with no data for this scenario window (likely pre-IPO): "
            "%s (%.1f%% of portfolio weight) — remaining weights renormalized to 100%%",
            len(excluded), excluded, excluded_weight * 100,
        )

    if not valid_mask.any():
        raise ValueError("no ticker in this portfolio has usable data for this scenario window")

    valid_returns = ticker_returns[valid_mask]
    valid_weights = weights[valid_mask]
    valid_weights = valid_weights / valid_weights.sum()  # renormalize to 1.0

    portfolio_return = float((valid_returns * valid_weights).sum())

    logger.info("Scenario portfolio return: %.4f", portfolio_return)
    return portfolio_return


def stressed_loss_from_return(portfolio_return: float, portfolio_value: float) -> float:
    """Convert a stressed portfolio return into a currency loss figure.

    Shared by every scenario type in this package (historical,
    hypothetical, volatility, correlation, sector) so the sign
    convention (positive = loss) stays consistent everywhere.

    Parameters
    ----------
    portfolio_return:
        Stressed return (decimal). Negative = loss, positive = gain.
    portfolio_value:
        Current portfolio value (currency units).

    Returns
    -------
    float
        Loss in currency units. Positive means a loss; negative means
        the scenario would actually be a gain.

    Raises
    ------
    ValueError
        If `portfolio_value` <= 0.
    """
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    return -portfolio_return * portfolio_value
