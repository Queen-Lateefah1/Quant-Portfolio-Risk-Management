from unittest.mock import MagicMock, patch

import pytest

from src.data_collection.sectors import get_sector_map


def _mock_ticker(sector):
    mock = MagicMock()
    mock.info = {"sector": sector} if sector is not None else {}
    return mock


def test_normal_case_returns_sector_for_each_ticker():
    with patch("src.data_collection.sectors.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.side_effect = lambda t: _mock_ticker(
            {"AAPL": "Technology", "JPM": "Financial Services"}[t]
        )
        result = get_sector_map(["AAPL", "JPM"])
        assert result == {"AAPL": "Technology", "JPM": "Financial Services"}


def test_known_answer_single_ticker():
    with patch("src.data_collection.sectors.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value = _mock_ticker("Healthcare")
        result = get_sector_map(["UNH"])
        assert result["UNH"] == "Healthcare"


def test_edge_case_missing_sector_maps_to_unknown():
    with patch("src.data_collection.sectors.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value = _mock_ticker(None)
        result = get_sector_map(["WEIRD"])
        assert result["WEIRD"] == "Unknown"


def test_edge_case_lookup_failure_maps_to_unknown():
    with patch("src.data_collection.sectors.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.side_effect = Exception("network error")
        result = get_sector_map(["BROKEN"])
        assert result["BROKEN"] == "Unknown"


def test_invalid_input_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        get_sector_map([])
