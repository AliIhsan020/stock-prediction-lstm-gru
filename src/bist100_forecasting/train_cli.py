"""Command-line training for BIST 100 recurrent forecasting models."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from bist100_forecasting.checkpoints import BestCheckpoint, load_checkpoint
from bist100_forecasting.inference import ForecastEvaluation, evaluate_model_forecast
from bist100_forecasting.models import GRUForecaster, LSTMForecaster
from bist100_forecasting.preprocessing import (
    DEFAULT_PROCESSED_DATA_PATH,
    load_prepared_archive,
)
from bist100_forecasting.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRADIENT_CLIP,
    DEFAULT_RANDOM_SEED,
    EarlyStopping,
    create_data_loaders,
    fit_model,
)

DEFAULT_CHECKPOINT_DIRECTORY = Path("models/checkpoints")
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.20
DEFAULT_PATIENCE = 10
DEFAULT_MIN_DELTA = 0.0
MODEL_NAMES = ("lstm", "gru")

RecurrentModel = LSTMForecaster | GRUForecaster


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Summary of one completed training and test-evaluation run."""

    model_name: str
    device: torch.device
    checkpoint_path: Path
    epochs_run: int
    best_epoch: int
    evaluation: ForecastEvaluation


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for a recurrent-model training run."""
    parser = argparse.ArgumentParser(
        description="Train an LSTM or GRU on prepared BIST 100 sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_PROCESSED_DATA_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        help="Checkpoint path; defaults to models/checkpoints/<model>.pt",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--gradient-clip", type=float, default=DEFAULT_GRADIENT_CLIP)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device; automatically selects CUDA when available",
    )
    return parser


def create_model(
    model_name: str,
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> RecurrentModel:
    """Create the requested recurrent model from validated hyperparameters."""
    model_classes = {
        "lstm": LSTMForecaster,
        "gru": GRUForecaster,
    }
    try:
        model_class = model_classes[model_name]
    except KeyError as error:
        supported = ", ".join(MODEL_NAMES)
        raise ValueError(f"Model must be one of: {supported}.") from error
    return model_class(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


def run_training(args: argparse.Namespace) -> TrainingResult:
    """Train, restore the best checkpoint, and evaluate the test period."""
    _validate_run_options(args)
    device = resolve_device(args.device)
    set_random_seed(args.seed)
    archive = load_prepared_archive(args.input)
    model = create_model(
        args.model,
        input_size=len(archive.feature_columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    data_loaders = create_data_loaders(
        archive.windows,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    checkpoint_path = args.output or default_checkpoint_path(args.model)
    checkpoint = BestCheckpoint(checkpoint_path)
    history = fit_model(
        model,
        data_loaders,
        optimizer,
        epochs=args.epochs,
        device=device,
        gradient_clip=args.gradient_clip,
        early_stopping=EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
        ),
        checkpoint=checkpoint,
    )
    restored = load_checkpoint(checkpoint.path, device=device)
    evaluation = evaluate_model_forecast(
        restored.model,
        archive.windows.test,
        archive,
        batch_size=args.batch_size,
        device=device,
    )
    return TrainingResult(
        model_name=args.model,
        device=device,
        checkpoint_path=checkpoint.path,
        epochs_run=len(history),
        best_epoch=restored.epoch,
        evaluation=evaluation,
    )


def default_checkpoint_path(model_name: str) -> Path:
    """Return the stable default checkpoint path for a supported model."""
    if model_name not in MODEL_NAMES:
        supported = ", ".join(MODEL_NAMES)
        raise ValueError(f"Model must be one of: {supported}.")
    return DEFAULT_CHECKPOINT_DIRECTORY / f"{model_name}.pt"


def resolve_device(requested: str | None) -> torch.device:
    """Resolve automatic or explicitly requested PyTorch execution device."""
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def set_random_seed(seed: int) -> None:
    """Seed NumPy and PyTorch for reproducible parameter initialization."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")
    if seed < 0:
        raise ValueError("Seed must be non-negative.")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: Sequence[str] | None = None) -> int:
    """Run model training from command-line arguments."""
    args = build_parser().parse_args(argv)
    result = run_training(args)
    metrics = result.evaluation.metrics

    print(f"Model: {result.model_name.upper()}")
    print(f"Device: {result.device}")
    print(f"Epochs completed: {result.epochs_run}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Test samples: {len(result.evaluation.actual):,}")
    print(f"MAE: {metrics.mae:.4f}")
    print(f"RMSE: {metrics.rmse:.4f}")
    print(f"MAPE: {metrics.mape_pct:.2f}%")
    print(f"R2: {metrics.r2:.4f}")
    print(f"Best checkpoint saved to: {result.checkpoint_path}")
    return 0


def _validate_run_options(args: argparse.Namespace) -> None:
    """Reject invalid scalar options before data loading or model creation."""
    _validate_positive_integer(args.epochs, name="Epochs")
    _validate_positive_integer(args.batch_size, name="Batch size")
    _validate_positive_number(args.learning_rate, name="Learning rate")
    _validate_positive_integer(args.patience, name="Patience")
    _validate_non_negative_number(args.min_delta, name="Minimum delta")
    _validate_positive_number(args.gradient_clip, name="Gradient clip")


def _validate_positive_integer(value: int, *, name: str) -> None:
    """Validate a positive, non-boolean integer option."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_positive_number(value: float, *, name: str) -> None:
    """Validate a finite positive numeric option."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _validate_non_negative_number(value: float, *, name: str) -> None:
    """Validate a finite non-negative numeric option."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
