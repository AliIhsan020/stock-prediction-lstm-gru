"""Aligned model and baseline comparison for BIST 100 forecasts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from bist100_forecasting.baselines import (
    DEFAULT_MOVING_AVERAGE_WINDOW,
    moving_average_forecast,
    persistence_forecast,
)
from bist100_forecasting.data import validate_history
from bist100_forecasting.evaluation import ForecastMetrics, evaluate_forecast
from bist100_forecasting.inference import ForecastEvaluation

COMPARISON_COLUMNS = ("Rank", "MAE", "RMSE", "MAPE (%)", "R2")


def compare_forecasts(
    history: pd.DataFrame,
    model_forecasts: Mapping[str, ForecastEvaluation],
    *,
    moving_average_window: int = DEFAULT_MOVING_AVERAGE_WINDOW,
) -> pd.DataFrame:
    """Compare aligned model forecasts with persistence and moving average."""
    validate_history(history)
    names, target_dates, actual = _validate_model_forecasts(model_forecasts)
    persistence = persistence_forecast(history, target_dates)
    moving_average = moving_average_forecast(
        history,
        target_dates,
        window=moving_average_window,
    )
    _validate_history_alignment(
        actual,
        persistence.actual,
        moving_average.actual,
    )

    metrics_by_method = {
        persistence.name: evaluate_forecast(
            persistence.actual,
            persistence.predicted,
        ),
        moving_average.name: evaluate_forecast(
            moving_average.actual,
            moving_average.predicted,
        ),
    }
    metrics_by_method.update({name: model_forecasts[name].metrics for name in names})
    return _build_ranked_table(metrics_by_method)


def best_method(comparison: pd.DataFrame) -> str:
    """Return the method ranked first by RMSE."""
    _validate_comparison_table(comparison)
    ranked_first = comparison.index[comparison["Rank"] == 1]
    if len(ranked_first) != 1:
        raise ValueError(
            "Comparison table must contain exactly one first-place method."
        )
    return str(ranked_first[0])


def _validate_model_forecasts(
    model_forecasts: Mapping[str, ForecastEvaluation],
) -> tuple[tuple[str, ...], pd.DatetimeIndex, np.ndarray]:
    """Validate names, dates, and actual values shared by model forecasts."""
    if not model_forecasts:
        raise ValueError("At least one model forecast is required.")

    names = tuple(model_forecasts)
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model names must be non-empty strings.")
        if name in {"Persistence"} or name.endswith("-day moving average"):
            raise ValueError(f"Model name conflicts with a baseline: {name}.")

    first = model_forecasts[names[0]]
    target_dates = first.target_dates
    actual = np.asarray(first.actual, dtype=np.float64)
    for name in names[1:]:
        forecast = model_forecasts[name]
        if not forecast.target_dates.equals(target_dates):
            raise ValueError("All model forecasts must use identical target dates.")
        if not np.allclose(forecast.actual, actual):
            raise ValueError("All model forecasts must use identical actual values.")
    return names, target_dates, actual


def _validate_history_alignment(
    model_actual: np.ndarray,
    persistence_actual: np.ndarray,
    moving_average_actual: np.ndarray,
) -> None:
    """Ensure models and baselines evaluate exactly the same observed values."""
    if not np.allclose(model_actual, persistence_actual):
        raise ValueError("Model targets do not match market history closing values.")
    if not np.allclose(model_actual, moving_average_actual):
        raise ValueError("Baseline targets do not match market history closing values.")


def _build_ranked_table(
    metrics_by_method: Mapping[str, ForecastMetrics],
) -> pd.DataFrame:
    """Create a stable metric table ranked by ascending RMSE."""
    table = pd.DataFrame.from_dict(
        {
            method: {
                "MAE": metrics.mae,
                "RMSE": metrics.rmse,
                "MAPE (%)": metrics.mape_pct,
                "R2": metrics.r2,
            }
            for method, metrics in metrics_by_method.items()
        },
        orient="index",
    )
    table.index.name = "Method"
    table["Rank"] = table["RMSE"].rank(method="first").astype(int)
    table = table.sort_values(["Rank", "RMSE"], kind="stable")
    return table.loc[:, COMPARISON_COLUMNS]


def _validate_comparison_table(comparison: pd.DataFrame) -> None:
    """Reject incomplete comparison frames before selecting a winner."""
    if comparison.empty:
        raise ValueError("Comparison table must not be empty.")
    missing_columns = set(COMPARISON_COLUMNS).difference(comparison.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Comparison table is missing columns: {names}.")
