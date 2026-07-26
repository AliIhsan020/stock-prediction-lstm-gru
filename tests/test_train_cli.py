"""Tests for command-line recurrent-model training."""

from pathlib import Path

import pandas as pd
import pytest

from bist100_forecasting.checkpoints import load_checkpoint
from bist100_forecasting.models import GRUForecaster, LSTMForecaster
from bist100_forecasting.preprocessing import (
    create_windowed_split,
    fit_scalers,
    save_windowed_split,
    split_history,
)
from bist100_forecasting.train_cli import (
    build_parser,
    create_model,
    default_checkpoint_path,
    main,
    resolve_device,
)


def market_history(rows: int = 36) -> pd.DataFrame:
    """Create deterministic OHLCV history for a small training run."""
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


def prepared_archive_path(tmp_path: Path) -> Path:
    """Save a small but complete sequence archive for CLI tests."""
    split = split_history(
        market_history(),
        train_ratio=0.60,
        validation_ratio=0.20,
    )
    scalers = fit_scalers(split.train)
    windows = create_windowed_split(split, scalers, lookback=3)
    return save_windowed_split(windows, scalers, tmp_path / "prepared.npz")


def test_parser_requires_a_supported_model() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "transformer"])

    args = parser.parse_args(["--model", "gru"])
    assert args.model == "gru"
    assert args.output is None
    assert args.epochs == 100


@pytest.mark.parametrize(
    ("model_name", "model_class"),
    [("lstm", LSTMForecaster), ("gru", GRUForecaster)],
)
def test_create_model_builds_requested_architecture(
    model_name: str,
    model_class,
) -> None:
    model = create_model(
        model_name,
        input_size=5,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )

    assert isinstance(model, model_class)
    assert model.config.input_size == 5
    assert model.config.hidden_size == 8


def test_default_checkpoint_path_is_model_specific() -> None:
    assert default_checkpoint_path("lstm") == Path("models/checkpoints/lstm.pt")
    assert default_checkpoint_path("gru") == Path("models/checkpoints/gru.pt")


def test_resolve_device_rejects_unavailable_cuda(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(ValueError, match="not available"):
        resolve_device("cuda")


@pytest.mark.parametrize(
    ("model_name", "model_class"),
    [("lstm", LSTMForecaster), ("gru", GRUForecaster)],
)
def test_main_trains_saves_and_evaluates_selected_model(
    model_name: str,
    model_class,
    tmp_path: Path,
    capsys,
) -> None:
    input_path = prepared_archive_path(tmp_path)
    output_path = tmp_path / f"{model_name}.pt"

    exit_code = main(
        [
            "--model",
            model_name,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--hidden-size",
            "4",
            "--num-layers",
            "1",
            "--dropout",
            "0",
            "--patience",
            "1",
            "--device",
            "cpu",
        ]
    )

    output = capsys.readouterr().out
    restored = load_checkpoint(output_path)
    assert exit_code == 0
    assert output_path.is_file()
    assert isinstance(restored.model, model_class)
    assert restored.model.config.hidden_size == 4
    assert f"Model: {model_name.upper()}" in output
    assert "MAE:" in output
    assert "RMSE:" in output
    assert "MAPE:" in output
    assert "R2:" in output
    assert "Best checkpoint saved to:" in output


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--epochs", "0", "Epochs must be positive"),
        ("--batch-size", "0", "Batch size must be positive"),
        ("--learning-rate", "0", "Learning rate must be finite and positive"),
        ("--patience", "0", "Patience must be positive"),
        ("--min-delta", "-1", "Minimum delta must be finite and non-negative"),
        ("--gradient-clip", "0", "Gradient clip must be finite and positive"),
    ],
)
def test_main_rejects_invalid_training_options(
    option: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        main(["--model", "gru", option, value])
