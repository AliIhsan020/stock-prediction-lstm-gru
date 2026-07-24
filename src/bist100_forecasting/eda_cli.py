"""Command-line interface for BIST 100 exploratory data analysis."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from bist100_forecasting.data import DEFAULT_DATA_PATH, load_history
from bist100_forecasting.eda import (
    DEFAULT_EDA_FIGURE,
    HistorySummary,
    save_eda_figure,
    summarize_history,
)

LoadFunction = Callable[[Path], pd.DataFrame]
SummaryFunction = Callable[[pd.DataFrame], HistorySummary]
SaveFigureFunction = Callable[[pd.DataFrame, Path], Path]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for exploratory analysis."""
    parser = argparse.ArgumentParser(
        description="Summarize saved BIST 100 history and generate an EDA figure."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Validated market-history CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EDA_FIGURE,
        help="Output PNG path (default: %(default)s).",
    )
    return parser


def format_summary(summary: HistorySummary) -> str:
    """Format an exploratory summary for terminal output."""
    return "\n".join(
        [
            "BIST 100 exploratory summary",
            f"Observations: {summary.observations:,}",
            f"Date range: {summary.start_date} to {summary.end_date}",
            f"First close: {summary.first_close:,.2f}",
            f"Latest close: {summary.latest_close:,.2f}",
            f"Minimum close: {summary.minimum_close:,.2f}",
            f"Maximum close: {summary.maximum_close:,.2f}",
            f"Total return: {summary.total_return_pct:,.2f}%",
            f"Mean daily return: {summary.mean_daily_return_pct:,.4f}%",
            f"Daily volatility: {summary.daily_volatility_pct:,.4f}%",
            f"Maximum drawdown: {summary.maximum_drawdown_pct:,.2f}%",
        ]
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    load: LoadFunction = load_history,
    summarize: SummaryFunction = summarize_history,
    save_figure: SaveFigureFunction = save_eda_figure,
) -> int:
    """Load saved history, print its summary, and save its EDA figure."""
    args = build_parser().parse_args(argv)
    history = load(args.input)
    summary = summarize(history)
    output_path = save_figure(history, args.output)

    print(format_summary(summary))
    print(f"Figure saved to: {output_path}")
    return 0
