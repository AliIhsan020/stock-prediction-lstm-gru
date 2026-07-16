"""Tests for daily BIST 100 market data validation."""

import numpy as np
import pandas as pd
import pytest

from bist100_forecasting.data import REQUIRED_COLUMNS, validate_history


def valid_history() -> pd.DataFrame:
    """Create a small, deterministic OHLCV data frame."""
    return pd.DataFrame(
        {
            "Open": [10_000.0, 10_100.0, 10_050.0],
            "High": [10_150.0, 10_200.0, 10_125.0],
            "Low": [9_950.0, 10_000.0, 9_990.0],
            "Close": [10_100.0, 10_050.0, 10_075.0],
            "Volume": [1_000_000, 1_100_000, 950_000],
        },
        index=pd.date_range("2025-01-02", periods=3, freq="B", name="Date"),
    )


def test_validate_history_accepts_valid_ohlcv_data() -> None:
    history = valid_history()

    validate_history(history)

    assert list(history.columns) == list(REQUIRED_COLUMNS)


def test_validate_history_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        validate_history(pd.DataFrame())


def test_validate_history_requires_datetime_index() -> None:
    history = valid_history().reset_index(drop=True)

    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_history(history)


def test_validate_history_rejects_duplicate_dates() -> None:
    history = valid_history()
    history.index = pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-06"])

    with pytest.raises(ValueError, match="duplicate dates"):
        validate_history(history)


def test_validate_history_rejects_unsorted_dates() -> None:
    history = valid_history().iloc[::-1]

    with pytest.raises(ValueError, match="ascending order"):
        validate_history(history)


def test_validate_history_reports_all_missing_columns() -> None:
    history = valid_history().drop(columns=["High", "Volume"])

    with pytest.raises(ValueError, match=r"High, Volume"):
        validate_history(history)


def test_validate_history_rejects_missing_values() -> None:
    history = valid_history()
    history.loc[history.index[0], "Close"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        validate_history(history)


def test_validate_history_rejects_non_numeric_values() -> None:
    history = valid_history().astype({"Close": object})
    history.loc[history.index[0], "Close"] = "not-a-price"

    with pytest.raises(ValueError, match="must be numeric"):
        validate_history(history)


def test_validate_history_rejects_infinite_values() -> None:
    history = valid_history()
    history.loc[history.index[0], "Close"] = np.inf

    with pytest.raises(ValueError, match="infinite values"):
        validate_history(history)


@pytest.mark.parametrize("column", ["Open", "High", "Low", "Close"])
def test_validate_history_requires_positive_prices(column: str) -> None:
    history = valid_history()
    history.loc[history.index[0], column] = 0

    with pytest.raises(ValueError, match="greater than zero"):
        validate_history(history)


def test_validate_history_rejects_negative_volume() -> None:
    history = valid_history()
    history.loc[history.index[0], "Volume"] = -1

    with pytest.raises(ValueError, match="cannot be negative"):
        validate_history(history)


def test_validate_history_checks_daily_high() -> None:
    history = valid_history()
    history.loc[history.index[0], "High"] = history.loc[history.index[0], "Close"] - 1

    with pytest.raises(ValueError, match="High must be the greatest"):
        validate_history(history)


def test_validate_history_checks_daily_low() -> None:
    history = valid_history()
    history.loc[history.index[0], "Low"] = history.loc[history.index[0], "Close"] + 1

    with pytest.raises(ValueError, match="Low must be the smallest"):
        validate_history(history)
