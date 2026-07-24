"""Forecast evaluation metrics on the original BIST 100 value scale."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

NumericVector = Sequence[float] | np.ndarray


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    """Common regression metrics for one-step-ahead forecasts."""

    mae: float
    rmse: float
    mape_pct: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        """Return display-ready metric names and values."""
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE (%)": self.mape_pct,
            "R²": self.r2,
        }


def evaluate_forecast(
    actual: NumericVector,
    predicted: NumericVector,
) -> ForecastMetrics:
    """Calculate forecast errors after validating aligned one-dimensional values."""
    actual_values = _as_numeric_vector(actual, name="Actual")
    predicted_values = _as_numeric_vector(predicted, name="Predicted")
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted values must have the same length.")
    if np.any(actual_values == 0):
        raise ValueError("MAPE requires non-zero actual values.")

    errors = predicted_values - actual_values
    absolute_errors = np.abs(errors)
    squared_errors = np.square(errors)
    residual_sum = float(squared_errors.sum())
    total_sum = float(np.square(actual_values - actual_values.mean()).sum())
    if np.isclose(total_sum, 0.0):
        r2 = 1.0 if np.isclose(residual_sum, 0.0) else 0.0
    else:
        r2 = 1.0 - residual_sum / total_sum

    return ForecastMetrics(
        mae=float(absolute_errors.mean()),
        rmse=float(np.sqrt(squared_errors.mean())),
        mape_pct=float(np.mean(absolute_errors / np.abs(actual_values)) * 100),
        r2=r2,
    )


def _as_numeric_vector(values: NumericVector, *, name: str) -> np.ndarray:
    """Convert metric input to a finite one-dimensional float array."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} values must be numeric.") from error
    if array.ndim != 1:
        raise ValueError(f"{name} values must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} values must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} values must be finite.")
    return array
