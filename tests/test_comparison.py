"""Tests for aligned model and baseline comparison."""

import numpy as np
import pandas as pd
import pytest

from bist100_forecasting.comparison import best_method, compare_forecasts
from bist100_forecasting.evaluation import evaluate_forecast
from bist100_forecasting.inference import ForecastEvaluation


def market_history(rows: int = 40) -> pd.DataFrame:
    """Create deterministic OHLCV values with non-linear close changes."""
    positions = np.arange(rows, dtype=np.float64)
    close = 100.0 + positions + np.sin(positions) * 2.0
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": 1_000.0 + positions,
        },
        index=pd.date_range("2025-01-02", periods=rows, freq="B", name="Date"),
    )


def model_forecast(
    history: pd.DataFrame,
    *,
    offset: float,
    target_dates: pd.DatetimeIndex | None = None,
) -> ForecastEvaluation:
    """Create an evaluated model forecast with a fixed signed error."""
    dates = target_dates if target_dates is not None else history.index[-10:]
    actual = history.loc[dates, "Close"].to_numpy(dtype=np.float64)
    predicted = actual + offset
    return ForecastEvaluation(
        target_dates=dates,
        actual=actual,
        predicted=predicted,
        metrics=evaluate_forecast(actual, predicted),
    )


def test_compare_forecasts_ranks_models_and_baselines_by_rmse() -> None:
    history = market_history()
    forecasts = {
        "LSTM": model_forecast(history, offset=0.0),
        "GRU": model_forecast(history, offset=1.0),
    }

    comparison = compare_forecasts(history, forecasts, moving_average_window=5)

    assert list(comparison.columns) == ["Rank", "MAE", "RMSE", "MAPE (%)", "R2"]
    assert set(comparison.index) == {
        "LSTM",
        "GRU",
        "Persistence",
        "5-day moving average",
    }
    assert comparison.index[0] == "LSTM"
    assert comparison.loc["LSTM", "Rank"] == 1
    assert comparison.loc["LSTM", "RMSE"] == pytest.approx(0.0)
    assert best_method(comparison) == "LSTM"


def test_compare_forecasts_requires_identical_model_dates() -> None:
    history = market_history()
    forecasts = {
        "LSTM": model_forecast(history, offset=0.0),
        "GRU": model_forecast(
            history,
            offset=0.0,
            target_dates=history.index[-11:-1],
        ),
    }

    with pytest.raises(ValueError, match="identical target dates"):
        compare_forecasts(history, forecasts)


def test_compare_forecasts_requires_identical_actual_values() -> None:
    history = market_history()
    lstm = model_forecast(history, offset=0.0)
    gru = model_forecast(history, offset=0.0)
    mismatched_actual = gru.actual.copy()
    mismatched_actual[0] += 1.0
    gru = ForecastEvaluation(
        target_dates=gru.target_dates,
        actual=mismatched_actual,
        predicted=gru.predicted,
        metrics=gru.metrics,
    )

    with pytest.raises(ValueError, match="identical actual values"):
        compare_forecasts(history, {"LSTM": lstm, "GRU": gru})


def test_compare_forecasts_requires_targets_to_match_history() -> None:
    history = market_history()
    forecast = model_forecast(history, offset=0.0)
    mismatched_actual = forecast.actual.copy()
    mismatched_actual[0] += 1.0
    forecast = ForecastEvaluation(
        target_dates=forecast.target_dates,
        actual=mismatched_actual,
        predicted=mismatched_actual.copy(),
        metrics=evaluate_forecast(mismatched_actual, mismatched_actual),
    )

    with pytest.raises(ValueError, match="do not match market history"):
        compare_forecasts(history, {"LSTM": forecast})


@pytest.mark.parametrize(
    "forecasts",
    [
        {},
        {"": None},
        {"Persistence": None},
        {"20-day moving average": None},
    ],
)
def test_compare_forecasts_rejects_missing_or_conflicting_models(
    forecasts,
) -> None:
    with pytest.raises(ValueError, match="required|non-empty|conflicts"):
        compare_forecasts(market_history(), forecasts)


def test_best_method_rejects_incomplete_table() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        best_method(pd.DataFrame({"RMSE": [1.0]}, index=["GRU"]))
