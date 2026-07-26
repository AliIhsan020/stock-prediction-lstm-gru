"""Descriptive statistics for BIST 100 exploratory analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from bist100_forecasting.data import validate_history

DEFAULT_EDA_FIGURE = Path("reports/figures/bist100_eda.png")


@dataclass(frozen=True, slots=True)
class HistorySummary:
    """Key descriptive statistics for a validated market history."""

    observations: int
    start_date: date
    end_date: date
    first_close: float
    latest_close: float
    minimum_close: float
    maximum_close: float
    total_return_pct: float
    mean_daily_return_pct: float
    daily_volatility_pct: float
    maximum_drawdown_pct: float


def calculate_daily_returns(history: pd.DataFrame) -> pd.Series:
    """Calculate close-to-close returns without filling missing observations."""
    validate_history(history)
    if len(history) < 2:
        raise ValueError("Daily returns require at least two observations.")

    returns = history["Close"].pct_change(fill_method=None).dropna()
    return returns.rename("Daily Return")


def summarize_history(history: pd.DataFrame) -> HistorySummary:
    """Summarize levels, returns, volatility, and maximum drawdown."""
    validate_history(history)
    if len(history) < 3:
        raise ValueError("History summary requires at least three observations.")

    close = history["Close"]
    daily_returns = calculate_daily_returns(history)
    drawdown = close.div(close.cummax()).sub(1)

    return HistorySummary(
        observations=len(history),
        start_date=history.index[0].date(),
        end_date=history.index[-1].date(),
        first_close=float(close.iloc[0]),
        latest_close=float(close.iloc[-1]),
        minimum_close=float(close.min()),
        maximum_close=float(close.max()),
        total_return_pct=float((close.iloc[-1] / close.iloc[0] - 1) * 100),
        mean_daily_return_pct=float(daily_returns.mean() * 100),
        daily_volatility_pct=float(daily_returns.std() * 100),
        maximum_drawdown_pct=float(drawdown.min() * 100),
    )


def build_eda_figure(
    history: pd.DataFrame, instrument_label: str = "BIST 100"
) -> Figure:
    """Build closing-value and daily-return panels for validated history."""
    daily_returns_pct = calculate_daily_returns(history).mul(100)
    figure = Figure(figsize=(12, 8))
    axes = figure.subplots(nrows=2, sharex=True)

    axes[0].plot(
        history.index,
        history["Close"],
        color="tab:blue",
        linewidth=1.2,
    )
    axes[0].set_title(f"{instrument_label} Daily Closing Value")
    axes[0].set_ylabel("Closing value")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        daily_returns_pct.index,
        daily_returns_pct,
        color="tab:orange",
        linewidth=0.7,
    )
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].set_title(f"{instrument_label} Daily Return")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Return (%)")
    axes[1].grid(alpha=0.25)

    figure.suptitle(f"{instrument_label} Exploratory Data Analysis", fontsize=14)
    figure.tight_layout()
    return figure


def save_eda_figure(
    history: pd.DataFrame,
    output_path: Path = DEFAULT_EDA_FIGURE,
) -> Path:
    """Save the EDA figure as a PNG without requiring a GUI backend."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = build_eda_figure(history)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path
