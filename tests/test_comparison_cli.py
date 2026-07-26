"""Tests for command-line model comparison and CSV reporting."""

from pathlib import Path

import pandas as pd
import pytest
from torch.optim import Adam

from bist100_forecasting.checkpoints import save_checkpoint
from bist100_forecasting.comparison import DEFAULT_COMPARISON_PATH
from bist100_forecasting.comparison_cli import build_parser, main
from bist100_forecasting.data import save_history
from bist100_forecasting.models import GRUForecaster, LSTMForecaster
from bist100_forecasting.preprocessing import (
    create_windowed_split,
    fit_scalers,
    save_windowed_split,
    split_history,
)
from bist100_forecasting.training import EpochResult


def market_history(rows: int = 36) -> pd.DataFrame:
    """Create valid deterministic OHLCV history for comparison tests."""
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


def comparison_inputs(tmp_path: Path) -> dict[str, Path]:
    """Save history, prepared arrays, and both recurrent checkpoints."""
    history = market_history()
    history_path = save_history(history, tmp_path / "history.csv")
    split = split_history(
        history,
        train_ratio=0.60,
        validation_ratio=0.20,
    )
    scalers = fit_scalers(split.train)
    windows = create_windowed_split(split, scalers, lookback=3)
    archive_path = save_windowed_split(
        windows,
        scalers,
        tmp_path / "prepared.npz",
    )
    checkpoints = {}
    for name, model_class in (
        ("lstm", LSTMForecaster),
        ("gru", GRUForecaster),
    ):
        model = model_class(
            input_size=len(scalers.feature_columns),
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
        )
        optimizer = Adam(model.parameters())
        checkpoints[name] = save_checkpoint(
            tmp_path / f"{name}.pt",
            model,
            optimizer,
            epoch=1,
            validation_loss=0.5,
            history=(
                EpochResult(
                    epoch=1,
                    train_loss=0.6,
                    validation_loss=0.5,
                ),
            ),
        )
    return {
        "history": history_path,
        "archive": archive_path,
        **checkpoints,
    }


def test_parser_uses_project_artifact_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.lstm_checkpoint == Path("models/checkpoints/lstm.pt")
    assert args.gru_checkpoint == Path("models/checkpoints/gru.pt")
    assert args.output == DEFAULT_COMPARISON_PATH
    assert args.moving_average_window == 20


def test_main_compares_checkpoints_and_saves_csv(
    tmp_path: Path,
    capsys,
) -> None:
    paths = comparison_inputs(tmp_path)
    output_path = tmp_path / "comparison.csv"

    exit_code = main(
        [
            "--history",
            str(paths["history"]),
            "--input",
            str(paths["archive"]),
            "--lstm-checkpoint",
            str(paths["lstm"]),
            "--gru-checkpoint",
            str(paths["gru"]),
            "--output",
            str(output_path),
            "--batch-size",
            "4",
            "--moving-average-window",
            "3",
            "--device",
            "cpu",
        ]
    )

    terminal_output = capsys.readouterr().out
    stored = pd.read_csv(output_path, index_col="Method")
    assert exit_code == 0
    assert set(stored.index) == {
        "LSTM",
        "GRU",
        "Persistence",
        "3-day moving average",
    }
    assert list(stored["Rank"].sort_values()) == [1, 2, 3, 4]
    assert "Best method by RMSE:" in terminal_output
    assert "Comparison report saved to:" in terminal_output
    assert not output_path.with_suffix(".csv.tmp").exists()


def test_main_rejects_checkpoint_with_wrong_model_type(tmp_path: Path) -> None:
    paths = comparison_inputs(tmp_path)

    with pytest.raises(ValueError, match="LSTM checkpoint contains GRUForecaster"):
        main(
            [
                "--history",
                str(paths["history"]),
                "--input",
                str(paths["archive"]),
                "--lstm-checkpoint",
                str(paths["gru"]),
                "--gru-checkpoint",
                str(paths["gru"]),
                "--output",
                str(tmp_path / "comparison.csv"),
                "--device",
                "cpu",
            ]
        )


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("--batch-size", "Batch size must be positive"),
        ("--moving-average-window", "Moving-average window must be positive"),
    ],
)
def test_main_rejects_non_positive_integer_options(
    option: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        main([option, "0"])
