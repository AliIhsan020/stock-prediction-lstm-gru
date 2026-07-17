"""Tests for storing and loading validated market history."""

from pathlib import Path

import pandas as pd
import pytest

from bist100_forecasting.data import load_history, save_history


def valid_history() -> pd.DataFrame:
    """Create valid OHLCV data for CSV round-trip tests."""
    return pd.DataFrame(
        {
            "Open": [10_000.0, 10_100.0],
            "High": [10_150.0, 10_200.0],
            "Low": [9_950.0, 10_000.0],
            "Close": [10_100.0, 10_050.0],
            "Volume": [1_000_000, 1_100_000],
        },
        index=pd.date_range("2025-01-02", periods=2, freq="B", name="Date"),
    )


def test_save_and_load_history_round_trip(tmp_path: Path) -> None:
    history = valid_history()
    output_path = tmp_path / "raw" / "bist100.csv"

    result = save_history(history, output_path)
    loaded = load_history(output_path)

    assert result == output_path
    pd.testing.assert_frame_equal(loaded, history, check_freq=False)


def test_save_history_creates_parent_directories(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "raw" / "bist100.csv"

    save_history(valid_history(), output_path)

    assert output_path.is_file()


def test_save_history_does_not_write_invalid_data(tmp_path: Path) -> None:
    history = valid_history()
    history.loc[history.index[0], "Close"] = -1
    output_path = tmp_path / "bist100.csv"

    with pytest.raises(ValueError, match="greater than zero"):
        save_history(history, output_path)

    assert not output_path.exists()


def test_save_history_removes_temporary_file(tmp_path: Path) -> None:
    output_path = tmp_path / "bist100.csv"

    save_history(valid_history(), output_path)

    assert not output_path.with_suffix(".csv.tmp").exists()


def test_load_history_reports_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Market history file not found"):
        load_history(missing_path)


def test_load_history_requires_date_column(tmp_path: Path) -> None:
    input_path = tmp_path / "missing-date.csv"
    valid_history().reset_index(drop=True).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="must contain a Date column"):
        load_history(input_path)


def test_load_history_rejects_invalid_dates(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid-date.csv"
    history = valid_history().reset_index()
    history["Date"] = history["Date"].astype(object)
    history.loc[0, "Date"] = "not-a-date"
    history.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="contains invalid dates"):
        load_history(input_path)


def test_load_history_revalidates_saved_values(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid-price.csv"
    history = valid_history()
    history.loc[history.index[0], "Close"] = -1
    history.to_csv(input_path, index_label="Date")

    with pytest.raises(ValueError, match="greater than zero"):
        load_history(input_path)
