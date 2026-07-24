"""Ordered prediction and original-scale evaluation for recurrent models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from bist100_forecasting.evaluation import ForecastMetrics, evaluate_forecast
from bist100_forecasting.preprocessing import (
    PreparedArchive,
    SequenceWindows,
)
from bist100_forecasting.training import DEFAULT_BATCH_SIZE, SequenceDataset


@dataclass(frozen=True, slots=True)
class ScaledForecast:
    """Ordered scaled targets and model predictions."""

    target_dates: pd.DatetimeIndex
    actual: np.ndarray
    predicted: np.ndarray


@dataclass(frozen=True, slots=True)
class ForecastEvaluation:
    """Original-scale predictions, dates, and regression metrics."""

    target_dates: pd.DatetimeIndex
    actual: np.ndarray
    predicted: np.ndarray
    metrics: ForecastMetrics

    def as_frame(self) -> pd.DataFrame:
        """Return dated actual, predicted, and signed-error columns."""
        return pd.DataFrame(
            {
                "Actual": self.actual,
                "Predicted": self.predicted,
                "Error": self.predicted - self.actual,
            },
            index=self.target_dates.copy(),
        )


@torch.no_grad()
def predict_windows(
    model: nn.Module,
    windows: SequenceWindows,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | torch.device = "cpu",
) -> ScaledForecast:
    """Predict ordered sequence windows without tracking gradients."""
    _validate_batch_size(batch_size)
    dataset = SequenceDataset(windows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()

    actual_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    for features, targets in loader:
        predictions = model(features.to(resolved_device))
        if not isinstance(predictions, torch.Tensor):
            raise TypeError("Model predictions must be a torch.Tensor.")
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Prediction shape {tuple(predictions.shape)} does not match "
                f"target shape {tuple(targets.shape)}."
            )
        if not torch.isfinite(predictions).all():
            raise FloatingPointError("Model predictions must be finite.")
        actual_batches.append(targets)
        prediction_batches.append(predictions.detach().cpu())

    return ScaledForecast(
        target_dates=dataset.target_dates,
        actual=torch.cat(actual_batches).numpy().astype(np.float64),
        predicted=torch.cat(prediction_batches).numpy().astype(np.float64),
    )


def evaluate_model_forecast(
    model: nn.Module,
    windows: SequenceWindows,
    archive: PreparedArchive,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | torch.device = "cpu",
) -> ForecastEvaluation:
    """Evaluate model predictions after restoring the original target scale."""
    expected_shape = (archive.lookback, len(archive.feature_columns))
    if windows.features.shape[1:] != expected_shape:
        raise ValueError("Sequence feature shape does not match archive metadata.")

    scaled = predict_windows(
        model,
        windows,
        batch_size=batch_size,
        device=device,
    )
    actual = archive.inverse_target(scaled.actual)
    predicted = archive.inverse_target(scaled.predicted)
    return ForecastEvaluation(
        target_dates=scaled.target_dates,
        actual=actual,
        predicted=predicted,
        metrics=evaluate_forecast(actual, predicted),
    )


def _validate_batch_size(batch_size: int) -> None:
    """Validate prediction batch size."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("Batch size must be an integer.")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
