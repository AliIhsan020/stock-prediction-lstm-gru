"""Reusable loading and evaluation of saved recurrent-model results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from bist100_forecasting.baselines import DEFAULT_MOVING_AVERAGE_WINDOW
from bist100_forecasting.checkpoints import LoadedCheckpoint, load_checkpoint
from bist100_forecasting.comparison import best_method, compare_forecasts
from bist100_forecasting.inference import ForecastEvaluation, evaluate_model_forecast
from bist100_forecasting.models import GRUForecaster, LSTMForecaster
from bist100_forecasting.preprocessing import (
    DEFAULT_PROCESSED_DATA_PATH,
    load_prepared_archive,
)
from bist100_forecasting.training import DEFAULT_BATCH_SIZE

DEFAULT_LSTM_CHECKPOINT_PATH = Path("models/checkpoints/lstm.pt")
DEFAULT_GRU_CHECKPOINT_PATH = Path("models/checkpoints/gru.pt")


@dataclass(frozen=True, slots=True)
class SavedModelResult:
    """Checkpoint metadata and original-scale test forecast for one model."""

    checkpoint_path: Path
    best_epoch: int
    validation_loss: float
    evaluation: ForecastEvaluation


@dataclass(frozen=True, slots=True)
class SavedModelResults:
    """Aligned LSTM, GRU, and baseline comparison ready for presentation."""

    lstm: SavedModelResult
    gru: SavedModelResult
    comparison: pd.DataFrame
    winner: str


def evaluate_saved_models(
    history: pd.DataFrame,
    *,
    archive_path: Path = DEFAULT_PROCESSED_DATA_PATH,
    lstm_checkpoint_path: Path = DEFAULT_LSTM_CHECKPOINT_PATH,
    gru_checkpoint_path: Path = DEFAULT_GRU_CHECKPOINT_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    moving_average_window: int = DEFAULT_MOVING_AVERAGE_WINDOW,
    device: str | torch.device = "cpu",
) -> SavedModelResults:
    """Evaluate saved LSTM and GRU checkpoints over one shared test period."""
    _validate_positive_integer(batch_size, name="Batch size")
    _validate_positive_integer(
        moving_average_window,
        name="Moving-average window",
    )
    resolved_device = torch.device(device)
    archive = load_prepared_archive(archive_path)
    loaded_lstm = _load_expected_checkpoint(
        lstm_checkpoint_path,
        expected_type=LSTMForecaster,
        model_name="LSTM",
        device=resolved_device,
    )
    loaded_gru = _load_expected_checkpoint(
        gru_checkpoint_path,
        expected_type=GRUForecaster,
        model_name="GRU",
        device=resolved_device,
    )
    lstm_result = _evaluate_loaded_checkpoint(
        loaded_lstm,
        checkpoint_path=lstm_checkpoint_path,
        archive=archive,
        batch_size=batch_size,
        device=resolved_device,
    )
    gru_result = _evaluate_loaded_checkpoint(
        loaded_gru,
        checkpoint_path=gru_checkpoint_path,
        archive=archive,
        batch_size=batch_size,
        device=resolved_device,
    )
    comparison = compare_forecasts(
        history,
        {
            "LSTM": lstm_result.evaluation,
            "GRU": gru_result.evaluation,
        },
        moving_average_window=moving_average_window,
    )
    return SavedModelResults(
        lstm=lstm_result,
        gru=gru_result,
        comparison=comparison,
        winner=best_method(comparison),
    )


def missing_model_artifacts(
    *,
    archive_path: Path = DEFAULT_PROCESSED_DATA_PATH,
    lstm_checkpoint_path: Path = DEFAULT_LSTM_CHECKPOINT_PATH,
    gru_checkpoint_path: Path = DEFAULT_GRU_CHECKPOINT_PATH,
) -> tuple[Path, ...]:
    """Return model artifacts that do not currently exist on disk."""
    paths = (
        Path(archive_path),
        Path(lstm_checkpoint_path),
        Path(gru_checkpoint_path),
    )
    return tuple(path for path in paths if not path.is_file())


def _load_expected_checkpoint(
    checkpoint_path: Path,
    *,
    expected_type: type[LSTMForecaster] | type[GRUForecaster],
    model_name: str,
    device: torch.device,
) -> LoadedCheckpoint:
    """Load a checkpoint and verify that it contains the expected model."""
    loaded = load_checkpoint(checkpoint_path, device=device)
    if not isinstance(loaded.model, expected_type):
        raise ValueError(
            f"{model_name} checkpoint contains {type(loaded.model).__name__}."
        )
    return loaded


def _evaluate_loaded_checkpoint(
    loaded: LoadedCheckpoint,
    *,
    checkpoint_path: Path,
    archive,
    batch_size: int,
    device: torch.device,
) -> SavedModelResult:
    """Evaluate one loaded recurrent checkpoint and retain its metadata."""
    evaluation = evaluate_model_forecast(
        loaded.model,
        archive.windows.test,
        archive,
        batch_size=batch_size,
        device=device,
    )
    return SavedModelResult(
        checkpoint_path=Path(checkpoint_path),
        best_epoch=loaded.epoch,
        validation_loss=loaded.validation_loss,
        evaluation=evaluation,
    )


def _validate_positive_integer(value: int, *, name: str) -> None:
    """Validate a positive, non-boolean integer option."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
