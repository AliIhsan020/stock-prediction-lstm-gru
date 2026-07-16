"""Tests for normalizing single-ticker Yahoo Finance responses."""

import pandas as pd
import pytest

from bist100_forecasting.data import (
    DEFAULT_SYMBOL,
    REQUIRED_COLUMNS,
    normalize_yahoo_history,
)


def yahoo_history() -> pd.DataFrame:
    """Create an unsorted response with an extra Yahoo Finance column."""
    return pd.DataFrame(
        {
            "Open": [10_100.0, 10_000.0, 10_050.0],
            "High": [10_200.0, 10_150.0, 10_125.0],
            "Low": [10_000.0, 9_950.0, 9_990.0],
            "Close": [10_050.0, 10_100.0, 10_075.0],
            "Adj Close": [10_050.0, 10_100.0, 10_075.0],
            "Volume": [1_100_000, 1_000_000, 950_000],
        },
        index=["2025-01-03", "2025-01-02", "2025-01-06"],
    )


def test_normalize_yahoo_history_selects_schema_and_sorts_dates() -> None:
    raw = yahoo_history()

    history = normalize_yahoo_history(raw)

    assert list(history.columns) == list(REQUIRED_COLUMNS)
    assert isinstance(history.index, pd.DatetimeIndex)
    assert history.index.is_monotonic_increasing
    assert history.index.name == "Date"
    assert "Adj Close" in raw.columns


def test_normalize_yahoo_history_flattens_single_ticker_columns() -> None:
    raw = yahoo_history()
    raw.columns = pd.MultiIndex.from_product(
        [raw.columns, [DEFAULT_SYMBOL]], names=["Price", "Ticker"]
    )

    history = normalize_yahoo_history(raw)

    assert list(history.columns) == list(REQUIRED_COLUMNS)


def test_normalize_yahoo_history_rejects_multiple_tickers() -> None:
    raw = yahoo_history()
    multiple_tickers = pd.concat(
        {DEFAULT_SYMBOL: raw, "GARAN.IS": raw}, axis="columns", names=["Ticker"]
    )

    with pytest.raises(ValueError, match="exactly one ticker"):
        normalize_yahoo_history(multiple_tickers)


def test_normalize_yahoo_history_removes_timezone() -> None:
    raw = yahoo_history()
    raw.index = pd.to_datetime(raw.index).tz_localize("Europe/Istanbul")

    history = normalize_yahoo_history(raw)

    assert history.index.tz is None


def test_normalize_yahoo_history_converts_numeric_strings() -> None:
    raw = yahoo_history().astype(str)

    history = normalize_yahoo_history(raw)

    assert all(pd.api.types.is_numeric_dtype(history[column]) for column in history)


def test_normalize_yahoo_history_rejects_unparseable_dates() -> None:
    raw = yahoo_history()
    raw.index = ["2025-01-02", "not-a-date", "2025-01-06"]

    with pytest.raises(ValueError, match="dates could not be parsed"):
        normalize_yahoo_history(raw)


def test_normalize_yahoo_history_reports_missing_columns() -> None:
    raw = yahoo_history().drop(columns=["High", "Volume"])

    with pytest.raises(ValueError, match=r"High, Volume"):
        normalize_yahoo_history(raw)


def test_normalize_yahoo_history_does_not_modify_input() -> None:
    raw = yahoo_history()
    original = raw.copy(deep=True)

    normalize_yahoo_history(raw)

    pd.testing.assert_frame_equal(raw, original)
