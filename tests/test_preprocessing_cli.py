"""Tests for the BIST 100 preprocessing command."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bist100_forecasting.data import DEFAULT_DATA_PATH
from bist100_forecasting.preprocessing import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_LOOKBACK,
    DEFAULT_PROCESSED_DATA_PATH,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VALIDATION_RATIO,
)
from bist100_forecasting.preprocessing_cli import build_parser, main


def market_history(rows: int = 30) -> pd.DataFrame:
    """Create valid OHLCV history for end-to-end CLI testing."""
    close = pd.Series(range(100, 100 + rows), dtype=float)
    return pd.DataFrame(
        {
            "Open": close.to_numpy(),
            "High": close.add(2).to_numpy(),
            "Low": close.sub(2).to_numpy(),
            "Close": close.add(1).to_numpy(),
            "Volume": range(1_000, 1_000 + rows),
        },
        index=pd.date_range("2025-01-02", periods=rows, freq="B", name="Date"),
    )


def test_parser_uses_preprocessing_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.input == DEFAULT_DATA_PATH
    assert args.output == DEFAULT_PROCESSED_DATA_PATH
    assert args.train_ratio == DEFAULT_TRAIN_RATIO
    assert args.validation_ratio == DEFAULT_VALIDATION_RATIO
    assert args.lookback == DEFAULT_LOOKBACK
    assert tuple(args.features) == DEFAULT_FEATURE_COLUMNS
    assert args.target == DEFAULT_TARGET_COLUMN


def test_main_creates_model_ready_archive(capsys: Any, tmp_path: Path) -> None:
    history = market_history()
    output_path = tmp_path / "prepared.npz"
    input_path = tmp_path / "history.csv"

    def fake_load(path: Path) -> pd.DataFrame:
        assert path == input_path
        return history

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--train-ratio",
            "0.6",
            "--validation-ratio",
            "0.2",
            "--lookback",
            "3",
            "--features",
            "Open",
            "Close",
            "Volume",
        ],
        load=fake_load,
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert output_path.is_file()
    assert "Training samples: 15" in output
    assert "Validation samples: 6" in output
    assert "Test samples: 6" in output

    with np.load(output_path, allow_pickle=False) as prepared:
        assert prepared["train_features"].shape == (15, 3, 3)
        assert prepared["validation_features"].shape == (6, 3, 3)
        assert prepared["test_features"].shape == (6, 3, 3)
        assert prepared["train_targets"].shape == (15,)
        assert prepared["feature_columns"].tolist() == ["Open", "Close", "Volume"]
        assert prepared["target_column"].item() == "Close"
        assert prepared["lookback"].item() == 3
        assert prepared["feature_scale"].shape == (3,)
        assert prepared["feature_offset"].shape == (3,)
        assert prepared["target_scale"].shape == (1,)
        assert prepared["target_offset"].shape == (1,)
        np.testing.assert_array_equal(
            prepared["validation_target_dates"],
            split_validation_dates(history),
        )


def split_validation_dates(history: pd.DataFrame) -> np.ndarray:
    """Return expected validation dates for the custom CLI ratios."""
    train_end = int(len(history) * 0.6)
    validation_end = train_end + int(len(history) * 0.2)
    return history.index[train_end:validation_end].to_numpy(dtype="datetime64[ns]")
