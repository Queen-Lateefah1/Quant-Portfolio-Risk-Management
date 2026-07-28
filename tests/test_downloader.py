import pytest

from src.data_collection.downloader import fetch_adjusted_close, fetch_index_prices

# Note: these tests cover input validation only. Live-network tests
# (actual Yahoo Finance calls) are intentionally excluded from the
# automated suite so tests run without an internet connection, per
# guide 4.3 (tests should run fast and not depend on external APIs).


def test_invalid_input_empty_tickers_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        fetch_adjusted_close([], start_date="2024-01-01")


def test_invalid_input_empty_index_ticker_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        fetch_index_prices("", start_date="2024-01-01")
