"""Command-line interface for downloading BIST 100 market history."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from bist100_forecasting.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    download_history,
    save_history,
)

DownloadFunction = Callable[..., pd.DataFrame]
SaveFunction = Callable[[pd.DataFrame, Path], Path]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for BIST 100 data collection."""
    parser = argparse.ArgumentParser(
        description="Download and validate daily BIST 100 market history."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument(
        "--end",
        default=DEFAULT_END_DATE,
        help="Exclusive end date (default: %(default)s).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    download: DownloadFunction = download_history,
    save: SaveFunction = save_history,
) -> int:
    """Download, validate, and save market history from command-line arguments."""
    args = build_parser().parse_args(argv)
    history = download(symbol=args.symbol, start=args.start, end=args.end)
    output_path = save(history, args.output)

    first_date = history.index.min().date()
    last_date = history.index.max().date()
    print(
        f"Saved {len(history):,} rows for {args.symbol} "
        f"({first_date} to {last_date}) to {output_path}"
    )
    return 0
