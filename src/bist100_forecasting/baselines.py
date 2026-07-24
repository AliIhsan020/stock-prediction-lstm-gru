"""Simple reference forecasts for BIST 100 model comparison."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bist100_forecasting.data import validate_history

DEFAULT_MOVING_AVERAGE_WINDOW = 20


@dataclass(frozen=True, slots=True)
class BaselineForecast:
    """Aligned actual and predicted closing values for a reference method."""

    name: str
    target_dates: pd.DatetimeIndex
    actual: np.ndarray
    predicted: np.ndarray


def persistence_forecast(
    history: pd.DataFrame,
    target_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
) -> BaselineForecast:
    """Predict each target close as the immediately preceding closing value."""
    dates, positions = _validate_targets(history, target_dates)
    if np.any(positions < 1):
        raise ValueError("Persistence targets require a prior observation.")

    close = history["Close"].to_numpy(dtype=np.float64)
    return BaselineForecast(
        name="Persistence",
        target_dates=dates,
        actual=close[positions].copy(),
        predicted=close[positions - 1].copy(),
    )


def moving_average_forecast(
    history: pd.DataFrame,
    target_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    window: int = DEFAULT_MOVING_AVERAGE_WINDOW,
) -> BaselineForecast:
    """Predict each close from the mean of closing values before its target date."""
    _validate_window(window)
    dates, positions = _validate_targets(history, target_dates)
    if np.any(positions < window):
        raise ValueError(
            "Moving-average targets require at least window prior observations."
        )

    close = history["Close"].to_numpy(dtype=np.float64)
    predictions = np.asarray(
        [close[position - window : position].mean() for position in positions],
        dtype=np.float64,
    )
    return BaselineForecast(
        name=f"{window}-day moving average",
        target_dates=dates,
        actual=close[positions].copy(),
        predicted=predictions,
    )


def _validate_targets(
    history: pd.DataFrame,
    target_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Validate ordered target dates and return their history positions."""
    validate_history(history)
    try:
        dates = pd.DatetimeIndex(target_dates, name=history.index.name)
    except (TypeError, ValueError) as error:
        raise ValueError("Target dates must be valid dates.") from error
    if dates.empty:
        raise ValueError("Target dates must not be empty.")
    if dates.has_duplicates:
        raise ValueError("Target dates must not contain duplicates.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Target dates must be in ascending order.")

    positions = history.index.get_indexer(dates)
    if np.any(positions < 0):
        raise ValueError("Every target date must exist in market history.")
    return dates, positions


def _validate_window(window: int) -> None:
    """Validate a moving-average lookback length."""
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("Moving-average window must be an integer.")
    if window <= 0:
        raise ValueError("Moving-average window must be positive.")
