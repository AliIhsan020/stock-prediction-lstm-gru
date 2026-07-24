"""Command-line interface for preparing BIST 100 model arrays."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from bist100_forecasting.data import DEFAULT_DATA_PATH, load_history
from bist100_forecasting.preprocessing import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_LOOKBACK,
    DEFAULT_PROCESSED_DATA_PATH,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VALIDATION_RATIO,
    FittedScalers,
    WindowedSplit,
    create_windowed_split,
    fit_scalers,
    save_windowed_split,
    split_history,
)

LoadFunction = Callable[[Path], pd.DataFrame]
SaveFunction = Callable[[WindowedSplit, FittedScalers, Path], Path]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for model-data preparation."""
    parser = argparse.ArgumentParser(
        description="Create leakage-safe BIST 100 sequence arrays for modeling."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROCESSED_DATA_PATH)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
    )
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURE_COLUMNS),
    )
    parser.add_argument("--target", default=DEFAULT_TARGET_COLUMN)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    load: LoadFunction = load_history,
    save: SaveFunction = save_windowed_split,
) -> int:
    """Prepare and save chronological, scaled sequence arrays."""
    args = build_parser().parse_args(argv)
    history = load(args.input)
    split = split_history(
        history,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    scalers = fit_scalers(
        split.train,
        feature_columns=args.features,
        target_column=args.target,
    )
    windows = create_windowed_split(split, scalers, lookback=args.lookback)
    output_path = save(windows, scalers, args.output)

    print(f"Features: {', '.join(scalers.feature_columns)}")
    print(f"Target: {scalers.target_column}")
    print(f"Lookback: {args.lookback} trading days")
    print(f"Training samples: {len(windows.train.targets):,}")
    print(f"Validation samples: {len(windows.validation.targets):,}")
    print(f"Test samples: {len(windows.test.targets):,}")
    print(f"Prepared arrays saved to: {output_path}")
    return 0
