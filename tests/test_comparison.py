import numpy as np
import pandas as pd
import pytest

from src.backtesting.comparison import compare_models, rank_models
from src.var.historical import rolling_historical_var


def _sample_returns(n=1000, seed=5):
    np.random.seed(seed)
    return pd.Series(
        np.random.normal(0, 0.015, n),
        index=pd.date_range("2020-01-01", periods=n),
    )


# --- compare_models ---

def test_normal_case_compares_multiple_models():
    returns = _sample_returns()
    var_a = rolling_historical_var(returns, window=250, confidence_level=0.95)
    var_b = rolling_historical_var(returns, window=500, confidence_level=0.95)
    table = compare_models(returns, {"Historical-250": var_a, "Historical-500": var_b}, confidence_level=0.95)

    assert list(table.index) == ["Historical-250", "Historical-500"]
    expected_columns = {
        "n_exceptions", "exception_ratio", "expected_ratio", "average_var",
        "var_volatility", "worst_daily_loss", "expected_shortfall",
        "kupiec_p_value", "christoffersen_p_value", "basel_zone", "overall_conclusion",
    }
    assert expected_columns.issubset(set(table.columns))


def test_known_answer_well_calibrated_model_has_ratio_near_expected():
    returns = _sample_returns()
    var = rolling_historical_var(returns, window=250, confidence_level=0.95)
    table = compare_models(returns, {"Historical": var}, confidence_level=0.95)
    row = table.loc["Historical"]
    assert row["exception_ratio"] == pytest.approx(row["expected_ratio"], abs=0.03)


def test_normal_case_expected_shortfall_exceeds_average_var():
    """Validation check from the doc: Expected Shortfall > VaR, which
    should hold here too since average_var and expected_shortfall are
    both reported per model."""
    returns = _sample_returns()
    var = rolling_historical_var(returns, window=250, confidence_level=0.95)
    table = compare_models(returns, {"Historical": var}, confidence_level=0.95)
    row = table.loc["Historical"]
    assert row["expected_shortfall"] >= row["average_var"]


def test_invalid_input_empty_returns_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        compare_models(pd.Series(dtype=float), {"A": pd.Series([0.02])})


def test_invalid_input_empty_var_dict_raises():
    returns = _sample_returns()
    with pytest.raises(ValueError, match="cannot be empty"):
        compare_models(returns, {})


# --- rank_models ---

def test_normal_case_ranking_orders_by_pass_then_calibration():
    returns = _sample_returns()
    good_var = rolling_historical_var(returns, window=250, confidence_level=0.95)
    bad_var = rolling_historical_var(returns, window=250, confidence_level=0.60)  # deliberately miscalibrated

    table = compare_models(returns, {"Good": good_var, "Bad": bad_var}, confidence_level=0.95)
    ranked = rank_models(table)

    assert ranked.iloc[0].name == "Good"
    assert list(ranked["rank"]) == [1, 2]


def test_known_answer_two_passing_models_ranked_by_calibration_gap():
    returns = _sample_returns()
    var_a = rolling_historical_var(returns, window=250, confidence_level=0.95)
    var_b = rolling_historical_var(returns, window=750, confidence_level=0.95)
    table = compare_models(returns, {"A": var_a, "B": var_b}, confidence_level=0.95)
    ranked = rank_models(table)
    # Whichever model's exception_ratio is closer to expected_ratio should rank first
    gaps = (table["exception_ratio"] - table["expected_ratio"]).abs()
    assert ranked.iloc[0].name == gaps.idxmin()


def test_invalid_input_empty_table_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        rank_models(pd.DataFrame())
