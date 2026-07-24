"""Descriptive statistics for BIST 100 exploratory analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from bist100_forecasting.data import validate_history


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
