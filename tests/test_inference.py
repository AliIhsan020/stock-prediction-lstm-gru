"""Tests for ordered recurrent-model prediction and evaluation."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from bist100_forecasting.inference import (
    evaluate_model_forecast,
    predict_windows,
)
from bist100_forecasting.preprocessing import (
    PreparedArchive,
    SequenceWindows,
    WindowedSplit,
)


class LastFeatureForecaster(nn.Module):
    """Return the final value of the first input feature."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, -1, 0]


def sequence_windows(samples: int = 7) -> SequenceWindows:
    """Create sequences whose final feature equals their target."""
    targets = np.linspace(0.1, 0.7, num=samples, dtype=np.float32)
    features = np.repeat(targets[:, None, None], repeats=3, axis=1)
    return SequenceWindows(
        features=features,
        targets=targets,
        target_dates=pd.date_range(
            "2025-01-02",
            periods=samples,
            freq="B",
            name="Date",
        ),
    )


def prepared_archive() -> PreparedArchive:
    """Create scaling metadata for deterministic original-scale assertions."""
    windows = sequence_windows()
    split = WindowedSplit(train=windows, validation=windows, test=windows)
    return PreparedArchive(
        windows=split,
        feature_columns=("Close",),
        target_column="Close",
        feature_scale=np.asarray([0.01]),
        feature_offset=np.asarray([-1.0]),
        target_scale=np.asarray([0.01]),
        target_offset=np.asarray([-1.0]),
        lookback=3,
    )


def test_predict_windows_preserves_dates_and_sample_order() -> None:
    windows = sequence_windows()

    forecast = predict_windows(
        LastFeatureForecaster(),
        windows,
        batch_size=3,
    )

    assert forecast.target_dates.equals(windows.target_dates)
    np.testing.assert_allclose(forecast.actual, windows.targets)
    np.testing.assert_allclose(forecast.predicted, windows.targets)


def test_evaluate_model_forecast_restores_original_scale() -> None:
    archive = prepared_archive()

    evaluation = evaluate_model_forecast(
        LastFeatureForecaster(),
        archive.windows.test,
        archive,
    )

    expected = archive.inverse_target(archive.windows.test.targets)
    np.testing.assert_allclose(evaluation.actual, expected)
    np.testing.assert_allclose(evaluation.predicted, expected)
    assert evaluation.metrics.mae == pytest.approx(0.0)
    assert evaluation.metrics.rmse == pytest.approx(0.0)
    assert evaluation.metrics.mape_pct == pytest.approx(0.0)
    assert evaluation.metrics.r2 == pytest.approx(1.0)


def test_forecast_evaluation_frame_contains_dated_errors() -> None:
    archive = prepared_archive()

    evaluation = evaluate_model_forecast(
        LastFeatureForecaster(),
        archive.windows.test,
        archive,
    )
    frame = evaluation.as_frame()

    assert frame.index.equals(archive.windows.test.target_dates)
    assert list(frame.columns) == ["Actual", "Predicted", "Error"]
    np.testing.assert_allclose(frame["Error"], 0.0)


@pytest.mark.parametrize(
    ("batch_size", "error", "message"),
    [
        (0, ValueError, "positive"),
        (-1, ValueError, "positive"),
        (True, TypeError, "integer"),
        (1.5, TypeError, "integer"),
    ],
)
def test_predict_windows_rejects_invalid_batch_size(
    batch_size,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        predict_windows(
            LastFeatureForecaster(),
            sequence_windows(),
            batch_size=batch_size,
        )


def test_predict_windows_rejects_prediction_shape_mismatch() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3, 1))

    with pytest.raises(ValueError, match="Prediction shape"):
        predict_windows(model, sequence_windows())


def test_predict_windows_rejects_non_finite_predictions() -> None:
    class NonFiniteModel(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.full(
                (len(inputs),),
                float("nan"),
                device=inputs.device,
            )

    with pytest.raises(FloatingPointError, match="finite"):
        predict_windows(NonFiniteModel(), sequence_windows())


def test_evaluate_model_forecast_requires_matching_archive_metadata() -> None:
    archive = prepared_archive()
    mismatched = SequenceWindows(
        features=np.ones((4, 2, 1), dtype=np.float32),
        targets=np.ones(4, dtype=np.float32),
        target_dates=pd.date_range("2025-01-01", periods=4, name="Date"),
    )

    with pytest.raises(ValueError, match="archive metadata"):
        evaluate_model_forecast(
            LastFeatureForecaster(),
            mismatched,
            archive,
        )
