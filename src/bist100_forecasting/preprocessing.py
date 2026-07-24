"""Chronological data preparation for BIST 100 forecasting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from bist100_forecasting.data import REQUIRED_COLUMNS, validate_history

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VALIDATION_RATIO = 0.15
DEFAULT_FEATURE_COLUMNS = REQUIRED_COLUMNS
DEFAULT_TARGET_COLUMN = "Close"
DEFAULT_LOOKBACK = 60
DEFAULT_PROCESSED_DATA_PATH = Path("data/processed/bist100_sequences.npz")


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


@dataclass(frozen=True, slots=True)
class PreparedArchive:
    """Loaded model arrays and the metadata needed for evaluation."""

    windows: WindowedSplit
    feature_columns: tuple[str, ...]
    target_column: str
    feature_scale: np.ndarray
    feature_offset: np.ndarray
    target_scale: np.ndarray
    target_offset: np.ndarray
    lookback: int

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        """Restore scaled target values without refitting a scaler."""
        scaled_values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(scaled_values).all():
            raise ValueError("Scaled target values must be finite.")
        return (scaled_values - self.target_offset.item()) / self.target_scale.item()


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


def save_windowed_split(
    windows: WindowedSplit,
    scalers: FittedScalers,
    output_path: Path = DEFAULT_PROCESSED_DATA_PATH,
) -> Path:
    """Atomically save model-ready arrays and scaling metadata as compressed NPZ."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    try:
        with temporary_path.open("wb") as temporary_file:
            np.savez_compressed(
                temporary_file,
                train_features=windows.train.features,
                train_targets=windows.train.targets,
                train_target_dates=windows.train.target_dates.to_numpy(
                    dtype="datetime64[ns]"
                ),
                validation_features=windows.validation.features,
                validation_targets=windows.validation.targets,
                validation_target_dates=windows.validation.target_dates.to_numpy(
                    dtype="datetime64[ns]"
                ),
                test_features=windows.test.features,
                test_targets=windows.test.targets,
                test_target_dates=windows.test.target_dates.to_numpy(
                    dtype="datetime64[ns]"
                ),
                feature_columns=np.asarray(scalers.feature_columns),
                target_column=np.asarray(scalers.target_column),
                feature_scale=scalers.feature_scaler.scale_,
                feature_offset=scalers.feature_scaler.min_,
                target_scale=scalers.target_scaler.scale_,
                target_offset=scalers.target_scaler.min_,
                lookback=np.asarray(windows.train.features.shape[1], dtype=np.int64),
            )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def load_prepared_archive(
    input_path: Path = DEFAULT_PROCESSED_DATA_PATH,
) -> PreparedArchive:
    """Load and validate a saved model-ready NPZ archive."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Prepared data archive not found: {input_path}")

    with np.load(input_path, allow_pickle=False) as stored:
        required_keys = {
            "train_features",
            "train_targets",
            "train_target_dates",
            "validation_features",
            "validation_targets",
            "validation_target_dates",
            "test_features",
            "test_targets",
            "test_target_dates",
            "feature_columns",
            "target_column",
            "feature_scale",
            "feature_offset",
            "target_scale",
            "target_offset",
            "lookback",
        }
        missing_keys = sorted(required_keys.difference(stored.files))
        if missing_keys:
            names = ", ".join(missing_keys)
            raise ValueError(f"Prepared archive is missing fields: {names}.")

        feature_columns = tuple(
            str(column) for column in stored["feature_columns"].tolist()
        )
        target_column = str(stored["target_column"].item())
        lookback = int(stored["lookback"].item())
        feature_scale = stored["feature_scale"].astype(np.float64, copy=True)
        feature_offset = stored["feature_offset"].astype(np.float64, copy=True)
        target_scale = stored["target_scale"].astype(np.float64, copy=True)
        target_offset = stored["target_offset"].astype(np.float64, copy=True)
        windows = WindowedSplit(
            train=_load_stored_windows(stored, "train"),
            validation=_load_stored_windows(stored, "validation"),
            test=_load_stored_windows(stored, "test"),
        )

    archive = PreparedArchive(
        windows=windows,
        feature_columns=feature_columns,
        target_column=target_column,
        feature_scale=feature_scale,
        feature_offset=feature_offset,
        target_scale=target_scale,
        target_offset=target_offset,
        lookback=lookback,
    )
    _validate_prepared_archive(archive)
    return archive


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


def _load_stored_windows(stored, split_name: str) -> SequenceWindows:
    """Copy one split from an open NumPy archive."""
    return SequenceWindows(
        features=stored[f"{split_name}_features"].astype(np.float32, copy=True),
        targets=stored[f"{split_name}_targets"].astype(np.float32, copy=True),
        target_dates=pd.DatetimeIndex(
            stored[f"{split_name}_target_dates"].copy(),
            name="Date",
        ),
    )


def _validate_prepared_archive(archive: PreparedArchive) -> None:
    """Validate model dimensions, dates, and scaling metadata."""
    if not archive.feature_columns:
        raise ValueError("Prepared archive must contain feature columns.")
    if len(archive.feature_columns) != len(set(archive.feature_columns)):
        raise ValueError("Prepared archive feature columns must be unique.")
    if not archive.target_column:
        raise ValueError("Prepared archive target column must not be empty.")
    if archive.lookback <= 0:
        raise ValueError("Prepared archive lookback must be positive.")

    feature_count = len(archive.feature_columns)
    for name, window in (
        ("Training", archive.windows.train),
        ("Validation", archive.windows.validation),
        ("Test", archive.windows.test),
    ):
        if window.features.ndim != 3:
            raise ValueError(f"{name} features must be three-dimensional.")
        if window.targets.ndim != 1:
            raise ValueError(f"{name} targets must be one-dimensional.")
        if len(window.features) == 0:
            raise ValueError(f"{name} split must not be empty.")
        if len(window.features) != len(window.targets):
            raise ValueError(f"{name} features and targets must have equal lengths.")
        if len(window.target_dates) != len(window.targets):
            raise ValueError(f"{name} target dates must align with targets.")
        if window.features.shape[1:] != (archive.lookback, feature_count):
            raise ValueError(f"{name} feature shape does not match archive metadata.")
        if (
            not np.isfinite(window.features).all()
            or not np.isfinite(window.targets).all()
        ):
            raise ValueError(f"{name} arrays must contain only finite values.")
        if window.target_dates.has_duplicates:
            raise ValueError(f"{name} target dates must be unique.")
        if not window.target_dates.is_monotonic_increasing:
            raise ValueError(f"{name} target dates must be in ascending order.")

    if archive.feature_scale.shape != (feature_count,):
        raise ValueError("Feature scale shape does not match feature columns.")
    if archive.feature_offset.shape != (feature_count,):
        raise ValueError("Feature offset shape does not match feature columns.")
    if archive.target_scale.shape != (1,) or archive.target_offset.shape != (1,):
        raise ValueError("Target scaling metadata must contain one value.")
    scaling_values = np.concatenate(
        [
            archive.feature_scale,
            archive.feature_offset,
            archive.target_scale,
            archive.target_offset,
        ]
    )
    if not np.isfinite(scaling_values).all():
        raise ValueError("Prepared archive scaling metadata must be finite.")
    if np.any(archive.feature_scale == 0) or archive.target_scale.item() == 0:
        raise ValueError("Prepared archive scaling factors must be non-zero.")
