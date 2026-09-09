import pandas as pd
import pytest

from src.stress_testing.sector_shock import apply_sector_shock


def test_known_answer_weighted_sector_impact():
    weights = pd.Series({"AAPL": 0.3, "MSFT": 0.2, "JPM": 0.5})
    sector_map = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financial Services"}
    impact = apply_sector_shock(weights, sector_map, "Technology", shock_pct=-0.20)
    assert impact == pytest.approx(-0.10)  # (0.3+0.2) * -0.20


def test_normal_case_unaffected_sector_zero_impact():
    weights = pd.Series({"AAPL": 0.5, "JPM": 0.5})
    sector_map = {"AAPL": "Technology", "JPM": "Financial Services"}
    impact = apply_sector_shock(weights, sector_map, "Financial Services", shock_pct=-0.30)
    assert impact == pytest.approx(-0.15)


def test_edge_case_entire_portfolio_in_shocked_sector():
    weights = pd.Series({"AAPL": 0.6, "MSFT": 0.4})
    sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
    impact = apply_sector_shock(weights, sector_map, "Technology", shock_pct=-0.25)
    assert impact == pytest.approx(-0.25)  # 100% weight -> full shock passes through


def test_invalid_input_empty_weights_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        apply_sector_shock(pd.Series(dtype=float), {}, "Technology", -0.1)


def test_invalid_input_sector_not_present_raises():
    weights = pd.Series({"AAPL": 1.0})
    sector_map = {"AAPL": "Technology"}
    with pytest.raises(ValueError, match="No tickers"):
        apply_sector_shock(weights, sector_map, "Energy", -0.1)
