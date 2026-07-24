"""Tests for simple BIST 100 reference forecasts."""

import numpy as np
import pandas as pd
import pytest

from bist100_forecasting.baselines import (
    moving_average_forecast,
    persistence_forecast,
)


def market_history(rows: int = 10) -> pd.DataFrame:
    """Create valid OHLCV history with predictable closing values."""
    close = pd.Series(range(100, 100 + rows), dtype=float)
    return pd.DataFrame(
        {
            "Open": close.to_numpy(),
            "High": close.add(2).to_numpy(),
            "Low": close.sub(2).to_numpy(),
            "Close": close.add(1).to_numpy(),
            "Volume": range(1_000, 1_000 + rows),
        },
        index=pd.date_range("2025-01-02", periods=rows, freq="B", name="Date"),
    )


def test_persistence_forecast_uses_previous_close() -> None:
    history = market_history()
    target_dates = history.index[3:6]

    forecast = persistence_forecast(history, target_dates)

    assert forecast.name == "Persistence"
    assert forecast.target_dates.equals(target_dates)
    np.testing.assert_array_equal(forecast.actual, [104.0, 105.0, 106.0])
    np.testing.assert_array_equal(forecast.predicted, [103.0, 104.0, 105.0])


def test_moving_average_forecast_uses_only_prior_closes() -> None:
    history = market_history()
    target_dates = history.index[3:6]

    forecast = moving_average_forecast(history, target_dates, window=3)

    assert forecast.name == "3-day moving average"
    assert forecast.target_dates.equals(target_dates)
    np.testing.assert_array_equal(forecast.actual, [104.0, 105.0, 106.0])
    np.testing.assert_allclose(forecast.predicted, [102.0, 103.0, 104.0])


def test_future_values_do_not_change_earlier_moving_average_prediction() -> None:
    history = market_history()
    target_date = history.index[5:6]
    original = moving_average_forecast(history, target_date, window=3)
    changed_history = history.copy()
    changed_history.loc[history.index[6] :, "Close"] *= 10
    changed_history.loc[history.index[6] :, "High"] = changed_history.loc[
        history.index[6] :, "Close"
    ]

    changed = moving_average_forecast(changed_history, target_date, window=3)

    np.testing.assert_array_equal(changed.predicted, original.predicted)


@pytest.mark.parametrize(
    "target_dates",
    [
        pd.DatetimeIndex([]),
        pd.DatetimeIndex(["2025-01-03", "2025-01-03"]),
        pd.DatetimeIndex(["2025-01-06", "2025-01-03"]),
        pd.DatetimeIndex(["2030-01-01"]),
    ],
)
def test_baselines_reject_invalid_target_dates(target_dates) -> None:
    with pytest.raises(ValueError):
        persistence_forecast(market_history(), target_dates)


def test_persistence_forecast_requires_previous_observation() -> None:
    history = market_history()

    with pytest.raises(ValueError, match="prior observation"):
        persistence_forecast(history, history.index[:1])


def test_moving_average_forecast_requires_enough_history() -> None:
    history = market_history()

    with pytest.raises(ValueError, match="window prior observations"):
        moving_average_forecast(history, history.index[2:3], window=3)


@pytest.mark.parametrize("window", [0, -1])
def test_moving_average_forecast_requires_positive_window(window: int) -> None:
    history = market_history()

    with pytest.raises(ValueError, match="positive"):
        moving_average_forecast(history, history.index[3:4], window=window)


@pytest.mark.parametrize("window", [True, 2.5])
def test_moving_average_forecast_requires_integer_window(window) -> None:
    history = market_history()

    with pytest.raises(TypeError, match="integer"):
        moving_average_forecast(history, history.index[3:4], window=window)
