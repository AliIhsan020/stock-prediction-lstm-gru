"""Chronological data preparation for BIST 100 forecasting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from bist100_forecasting.data import REQUIRED_COLUMNS, validate_history

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VALIDATION_RATIO = 0.15
DEFAULT_FEATURE_COLUMNS = REQUIRED_COLUMNS
DEFAULT_TARGET_COLUMN = "Close"
DEFAULT_LOOKBACK = 60


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Train, validation, and test periods kept in chronological order."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True, slots=True)
class FittedScalers:
    """Feature and target scalers fitted only on the training period."""

    feature_columns: tuple[str, ...]
    target_column: str
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler

    def transform_features(self, history: pd.DataFrame) -> np.ndarray:
        """Scale model inputs with training-period feature ranges."""
        validate_history(history)
        return self.feature_scaler.transform(history.loc[:, self.feature_columns])

    def transform_target(self, history: pd.DataFrame) -> np.ndarray:
        """Scale the prediction target with its training-period range."""
        validate_history(history)
        values = self.target_scaler.transform(history.loc[:, [self.target_column]])
        return values.reshape(-1)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        """Restore scaled target values to their original index-value scale."""
        scaled_values = np.asarray(values, dtype=np.float64)
        original_shape = scaled_values.shape
        restored = self.target_scaler.inverse_transform(scaled_values.reshape(-1, 1))
        return restored.reshape(original_shape)


@dataclass(frozen=True, slots=True)
class SequenceWindows:
    """Scaled input sequences and their one-step-ahead targets."""

    features: np.ndarray
    targets: np.ndarray
    target_dates: pd.DatetimeIndex


@dataclass(frozen=True, slots=True)
class WindowedSplit:
    """Sequence windows whose targets remain inside each chronological period."""

    train: SequenceWindows
    validation: SequenceWindows
    test: SequenceWindows


def split_history(
    history: pd.DataFrame,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
) -> ChronologicalSplit:
    """Split validated history by position without shuffling observations."""
    validate_history(history)
    _validate_split_ratios(train_ratio, validation_ratio)

    train_end = int(len(history) * train_ratio)
    validation_end = train_end + int(len(history) * validation_ratio)
    if train_end == 0 or validation_end == train_end or validation_end >= len(history):
        raise ValueError(
            "Dataset is too small to create non-empty train, validation, "
            "and test splits with the requested ratios."
        )

    return ChronologicalSplit(
        train=history.iloc[:train_end].copy(),
        validation=history.iloc[train_end:validation_end].copy(),
        test=history.iloc[validation_end:].copy(),
    )


def fit_scalers(
    train_history: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> FittedScalers:
    """Fit feature and target Min-Max scalers using training history only."""
    validate_history(train_history)
    columns = _validate_scaling_columns(
        train_history,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_history.loc[:, columns])
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_history.loc[:, [target_column]])

    return FittedScalers(
        feature_columns=columns,
        target_column=target_column,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )


def create_sequence_windows(
    history: pd.DataFrame,
    scalers: FittedScalers,
    *,
    lookback: int = DEFAULT_LOOKBACK,
) -> SequenceWindows:
    """Create past-to-future windows for one-step-ahead close prediction."""
    _validate_lookback(lookback, observations=len(history))
    scaled_features = scalers.transform_features(history).astype(np.float32)
    scaled_targets = scalers.transform_target(history).astype(np.float32)

    sample_count = len(history) - lookback
    windows = np.empty(
        (sample_count, lookback, scaled_features.shape[1]),
        dtype=np.float32,
    )
    for sample_index, target_index in enumerate(range(lookback, len(history))):
        windows[sample_index] = scaled_features[target_index - lookback : target_index]

    return SequenceWindows(
        features=windows,
        targets=scaled_targets[lookback:].copy(),
        target_dates=history.index[lookback:].copy(),
    )


def create_windowed_split(
    split: ChronologicalSplit,
    scalers: FittedScalers,
    *,
    lookback: int = DEFAULT_LOOKBACK,
) -> WindowedSplit:
    """Build split targets with only the historical context available before them."""
    train_windows = create_sequence_windows(split.train, scalers, lookback=lookback)
    validation_history = pd.concat(
        [split.train.tail(lookback), split.validation],
    )
    history_before_test = pd.concat([split.train, split.validation])
    test_history = pd.concat(
        [history_before_test.tail(lookback), split.test],
    )

    return WindowedSplit(
        train=train_windows,
        validation=create_sequence_windows(
            validation_history,
            scalers,
            lookback=lookback,
        ),
        test=create_sequence_windows(test_history, scalers, lookback=lookback),
    )


def _validate_split_ratios(train_ratio: float, validation_ratio: float) -> None:
    """Validate that all three chronological split proportions are positive."""
    if not 0 < train_ratio < 1:
        raise ValueError("Train ratio must be between 0 and 1.")
    if not 0 < validation_ratio < 1:
        raise ValueError("Validation ratio must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than 1.")


def _validate_lookback(lookback: int, *, observations: int) -> None:
    """Validate the number of past observations used by each sequence."""
    if isinstance(lookback, bool) or not isinstance(lookback, int):
        raise TypeError("Lookback must be an integer.")
    if lookback <= 0:
        raise ValueError("Lookback must be a positive integer.")
    if observations <= lookback:
        raise ValueError(
            "History must contain more observations than the requested lookback."
        )


def _validate_scaling_columns(
    history: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[str, ...]:
    """Return normalized feature names after validating the scaling schema."""
    if isinstance(feature_columns, str):
        raise TypeError("Feature columns must be a sequence of column names.")

    columns = tuple(feature_columns)
    if not columns:
        raise ValueError("At least one feature column is required.")
    if len(columns) != len(set(columns)):
        raise ValueError("Feature columns must not contain duplicates.")

    missing_columns = sorted(set((*columns, target_column)).difference(history.columns))
    if missing_columns:
        names = ", ".join(missing_columns)
        raise ValueError(f"Scaling columns are missing from history: {names}.")
    return columns
