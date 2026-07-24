"""Tests for forecast evaluation metrics."""

import numpy as np
import pytest

from bist100_forecasting.evaluation import evaluate_forecast


def test_evaluate_forecast_calculates_regression_metrics() -> None:
    actual = [100.0, 110.0, 120.0]
    predicted = [90.0, 115.0, 125.0]

    metrics = evaluate_forecast(actual, predicted)

    assert metrics.mae == pytest.approx(20 / 3)
    assert metrics.rmse == pytest.approx(np.sqrt(50))
    assert metrics.mape_pct == pytest.approx(
        np.mean([10 / 100, 5 / 110, 5 / 120]) * 100
    )
    assert metrics.r2 == pytest.approx(0.25)


def test_evaluate_forecast_reports_perfect_predictions() -> None:
    metrics = evaluate_forecast(
        np.asarray([8_000.0, 8_100.0, 8_200.0]),
        np.asarray([8_000.0, 8_100.0, 8_200.0]),
    )

    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.mape_pct == 0.0
    assert metrics.r2 == 1.0
    assert metrics.as_dict() == {
        "MAE": 0.0,
        "RMSE": 0.0,
        "MAPE (%)": 0.0,
        "R²": 1.0,
    }


def test_evaluate_forecast_handles_nonperfect_constant_actual_values() -> None:
    metrics = evaluate_forecast([100.0, 100.0], [99.0, 101.0])

    assert metrics.r2 == 0.0


@pytest.mark.parametrize(
    ("actual", "predicted", "message"),
    [
        ([], [], "must not be empty"),
        ([1.0, 2.0], [1.0], "same length"),
        ([[1.0], [2.0]], [[1.0], [2.0]], "one-dimensional"),
        ([1.0, np.nan], [1.0, 2.0], "finite"),
        ([1.0, 2.0], [1.0, np.inf], "finite"),
        ([1.0, "invalid"], [1.0, 2.0], "numeric"),
        ([0.0, 1.0], [0.0, 1.0], "non-zero"),
    ],
)
def test_evaluate_forecast_rejects_invalid_inputs(
    actual,
    predicted,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_forecast(actual, predicted)
