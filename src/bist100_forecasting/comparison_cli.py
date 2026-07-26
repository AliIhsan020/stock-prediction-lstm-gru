"""Command-line comparison of trained BIST 100 forecasting models."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from bist100_forecasting.baselines import DEFAULT_MOVING_AVERAGE_WINDOW
from bist100_forecasting.comparison import (
    DEFAULT_COMPARISON_PATH,
    save_comparison_table,
)
from bist100_forecasting.data import DEFAULT_DATA_PATH, load_history
from bist100_forecasting.model_results import (
    DEFAULT_GRU_CHECKPOINT_PATH,
    DEFAULT_LSTM_CHECKPOINT_PATH,
    evaluate_saved_models,
)
from bist100_forecasting.preprocessing import (
    DEFAULT_PROCESSED_DATA_PATH,
)
from bist100_forecasting.train_cli import resolve_device
from bist100_forecasting.training import DEFAULT_BATCH_SIZE


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Saved table and execution details for one comparison run."""

    table: pd.DataFrame
    winner: str
    output_path: Path
    test_samples: int
    device: torch.device


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for model comparison."""
    parser = argparse.ArgumentParser(
        description="Compare trained LSTM and GRU models with simple baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_PROCESSED_DATA_PATH)
    parser.add_argument(
        "--lstm-checkpoint",
        type=Path,
        default=DEFAULT_LSTM_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--gru-checkpoint",
        type=Path,
        default=DEFAULT_GRU_CHECKPOINT_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=DEFAULT_MOVING_AVERAGE_WINDOW,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device; automatically selects CUDA when available",
    )
    return parser


def run_comparison(args: argparse.Namespace) -> ComparisonResult:
    """Load both checkpoints, compare forecasts, and save the result."""
    _validate_positive_integer(args.batch_size, name="Batch size")
    _validate_positive_integer(
        args.moving_average_window,
        name="Moving-average window",
    )
    device = resolve_device(args.device)
    history = load_history(args.history)
    saved_results = evaluate_saved_models(
        history,
        archive_path=args.input,
        lstm_checkpoint_path=args.lstm_checkpoint,
        gru_checkpoint_path=args.gru_checkpoint,
        batch_size=args.batch_size,
        moving_average_window=args.moving_average_window,
        device=device,
    )
    output_path = save_comparison_table(saved_results.comparison, args.output)
    return ComparisonResult(
        table=saved_results.comparison,
        winner=saved_results.winner,
        output_path=output_path,
        test_samples=len(saved_results.lstm.evaluation.actual),
        device=device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run checkpoint comparison from command-line arguments."""
    args = build_parser().parse_args(argv)
    result = run_comparison(args)

    print(f"Device: {result.device}")
    print(f"Test samples: {result.test_samples:,}")
    print(result.table.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Best method by RMSE: {result.winner}")
    print(f"Comparison report saved to: {result.output_path}")
    return 0


def _validate_positive_integer(value: int, *, name: str) -> None:
    """Validate a positive, non-boolean integer option."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
