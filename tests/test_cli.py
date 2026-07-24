"""Tests for the BIST 100 download command-line interface."""

from pathlib import Path
from typing import Any

import pandas as pd

from bist100_forecasting.cli import build_parser, main
from bist100_forecasting.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
)


def sample_history() -> pd.DataFrame:
    """Create a small normalized history returned by a fake download."""
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


def test_parser_uses_project_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.symbol == DEFAULT_SYMBOL
    assert args.start == DEFAULT_START_DATE
    assert args.end == DEFAULT_END_DATE
    assert args.output == DEFAULT_DATA_PATH


def test_main_downloads_and_saves_default_snapshot(capsys: Any, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}
    history = sample_history()
    saved_path = tmp_path / "bist100.csv"

    def fake_download(**kwargs: Any) -> pd.DataFrame:
        calls["download"] = kwargs
        return history

    def fake_save(frame: pd.DataFrame, output_path: Path) -> Path:
        calls["save"] = (frame, output_path)
        return saved_path

    exit_code = main([], download=fake_download, save=fake_save)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert calls["download"] == {
        "symbol": DEFAULT_SYMBOL,
        "start": DEFAULT_START_DATE,
        "end": DEFAULT_END_DATE,
    }
    assert calls["save"] == (history, DEFAULT_DATA_PATH)
    assert "Saved 2 rows for XU100.IS" in output
    assert "2025-01-02 to 2025-01-03" in output
    assert str(saved_path) in output


def test_main_forwards_custom_arguments(tmp_path: Path) -> None:
    calls: dict[str, Any] = {}
    output_path = tmp_path / "custom.csv"

    def fake_download(**kwargs: Any) -> pd.DataFrame:
        calls["download"] = kwargs
        return sample_history()

    def fake_save(frame: pd.DataFrame, path: Path) -> Path:
        calls["saved_frame"] = frame
        calls["saved_path"] = path
        return path

    exit_code = main(
        [
            "--symbol",
            "GARAN.IS",
            "--start",
            "2024-01-01",
            "--end",
            "2025-01-01",
            "--output",
            str(output_path),
        ],
        download=fake_download,
        save=fake_save,
    )

    assert exit_code == 0
    assert calls["download"] == {
        "symbol": "GARAN.IS",
        "start": "2024-01-01",
        "end": "2025-01-01",
    }
    assert calls["saved_path"] == output_path
