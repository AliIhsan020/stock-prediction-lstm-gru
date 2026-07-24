"""Tests for one-step-ahead sequence window creation."""

import numpy as np
import pandas as pd
import pytest

from bist100_forecasting.preprocessing import (
    create_sequence_windows,
    create_windowed_split,
    fit_scalers,
    split_history,
)


def market_history(rows: int = 30) -> pd.DataFrame:
    """Create valid OHLCV history for deterministic sequence tests."""
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


def test_create_sequence_windows_returns_model_ready_shapes() -> None:
    history = market_history(rows=10)
    scalers = fit_scalers(history.iloc[:7])

    windows = create_sequence_windows(history, scalers, lookback=3)

    assert windows.features.shape == (7, 3, 5)
    assert windows.targets.shape == (7,)
    assert len(windows.target_dates) == 7
    assert windows.features.dtype == np.float32
    assert windows.targets.dtype == np.float32


def test_each_window_uses_only_observations_before_its_target() -> None:
    history = market_history(rows=10)
    scalers = fit_scalers(history.iloc[:7])
    scaled_features = scalers.transform_features(history)
    scaled_targets = scalers.transform_target(history)

    windows = create_sequence_windows(history, scalers, lookback=3)

    np.testing.assert_allclose(windows.features[0], scaled_features[:3])
    assert windows.targets[0] == pytest.approx(scaled_targets[3])
    assert windows.target_dates[0] == history.index[3]
    np.testing.assert_allclose(windows.features[-1], scaled_features[-4:-1])
    assert windows.targets[-1] == pytest.approx(scaled_targets[-1])
    assert windows.target_dates[-1] == history.index[-1]


def test_create_windowed_split_keeps_targets_inside_their_periods() -> None:
    split = split_history(
        market_history(),
        train_ratio=0.60,
        validation_ratio=0.20,
    )
    scalers = fit_scalers(split.train)

    windows = create_windowed_split(split, scalers, lookback=3)

    assert windows.train.features.shape[0] == len(split.train) - 3
    assert windows.validation.features.shape[0] == len(split.validation)
    assert windows.test.features.shape[0] == len(split.test)
    np.testing.assert_allclose(
        windows.validation.features[0],
        scalers.transform_features(split.train.tail(3)),
    )
    np.testing.assert_allclose(
        windows.test.features[0],
        scalers.transform_features(split.validation.tail(3)),
    )
    assert windows.validation.target_dates.equals(split.validation.index)
    assert windows.test.target_dates.equals(split.test.index)


def test_window_targets_can_be_restored_to_closing_values() -> None:
    history = market_history(rows=10)
    scalers = fit_scalers(history.iloc[:7])

    windows = create_sequence_windows(history, scalers, lookback=3)
    restored_targets = scalers.inverse_target(windows.targets)

    np.testing.assert_allclose(
        restored_targets,
        history["Close"].iloc[3:].to_numpy(),
        rtol=1e-6,
    )


@pytest.mark.parametrize("lookback", [0, -1])
def test_create_sequence_windows_requires_positive_lookback(lookback: int) -> None:
    history = market_history(rows=10)
    scalers = fit_scalers(history)

    with pytest.raises(ValueError, match="positive integer"):
        create_sequence_windows(history, scalers, lookback=lookback)


@pytest.mark.parametrize("lookback", [True, 2.5])
def test_create_sequence_windows_requires_integer_lookback(lookback) -> None:
    history = market_history(rows=10)
    scalers = fit_scalers(history)

    with pytest.raises(TypeError, match="integer"):
        create_sequence_windows(history, scalers, lookback=lookback)


def test_create_sequence_windows_requires_a_future_target() -> None:
    history = market_history(rows=10)
    scalers = fit_scalers(history)

    with pytest.raises(ValueError, match="more observations"):
        create_sequence_windows(history, scalers, lookback=len(history))
