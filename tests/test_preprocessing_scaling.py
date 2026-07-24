"""Tests for leakage-safe market-history scaling."""

import numpy as np
import pandas as pd
import pytest

from bist100_forecasting.preprocessing import fit_scalers, split_history


def market_history(rows: int = 20) -> pd.DataFrame:
    """Create valid OHLCV history whose values increase after training."""
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


def test_fit_scalers_maps_training_features_to_unit_range() -> None:
    split = split_history(
        market_history(),
        train_ratio=0.60,
        validation_ratio=0.20,
    )

    scalers = fit_scalers(split.train)
    scaled_train = scalers.transform_features(split.train)

    np.testing.assert_allclose(scaled_train.min(axis=0), 0.0)
    np.testing.assert_allclose(scaled_train.max(axis=0), 1.0)
    assert scaled_train.shape == split.train.shape


def test_validation_and_test_use_training_feature_ranges() -> None:
    split = split_history(
        market_history(),
        train_ratio=0.60,
        validation_ratio=0.20,
    )

    scalers = fit_scalers(split.train)
    scaled_validation = scalers.transform_features(split.validation)
    scaled_test = scalers.transform_features(split.test)

    assert scaled_validation.min() > 1.0
    assert scaled_test.min() > scaled_validation.min()


def test_target_scaler_uses_training_close_range() -> None:
    split = split_history(
        market_history(),
        train_ratio=0.60,
        validation_ratio=0.20,
    )

    scalers = fit_scalers(split.train)

    np.testing.assert_allclose(
        scalers.target_scaler.data_min_,
        [split.train["Close"].min()],
    )
    np.testing.assert_allclose(
        scalers.target_scaler.data_max_,
        [split.train["Close"].max()],
    )


def test_inverse_target_restores_original_close_values() -> None:
    history = market_history()
    split = split_history(history)
    scalers = fit_scalers(split.train)

    scaled_target = scalers.transform_target(split.test)
    restored_target = scalers.inverse_target(scaled_target)

    np.testing.assert_allclose(restored_target, split.test["Close"].to_numpy())
    assert restored_target.shape == scaled_target.shape


def test_fit_scalers_supports_selected_features() -> None:
    split = split_history(market_history())

    scalers = fit_scalers(
        split.train,
        feature_columns=("Open", "Close", "Volume"),
    )

    assert scalers.feature_columns == ("Open", "Close", "Volume")
    assert scalers.transform_features(split.validation).shape[1] == 3


@pytest.mark.parametrize(
    ("feature_columns", "target_column", "error", "message"),
    [
        ((), "Close", ValueError, "At least one feature"),
        (("Close", "Close"), "Close", ValueError, "duplicates"),
        ("Close", "Close", TypeError, "sequence"),
        (("Open", "Missing"), "Close", ValueError, "Missing"),
        (("Open", "Close"), "Target", ValueError, "Target"),
    ],
)
def test_fit_scalers_rejects_invalid_column_configuration(
    feature_columns,
    target_column: str,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        fit_scalers(
            market_history(),
            feature_columns=feature_columns,
            target_column=target_column,
        )
