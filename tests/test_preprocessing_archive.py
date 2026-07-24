"""Tests for loading saved model-ready sequence archives."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bist100_forecasting.preprocessing import (
    create_windowed_split,
    fit_scalers,
    load_prepared_archive,
    save_windowed_split,
    split_history,
)


def market_history(rows: int = 30) -> pd.DataFrame:
    """Create valid OHLCV history for archive round-trip tests."""
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


def saved_archive(tmp_path: Path) -> tuple[Path, object, object]:
    """Save deterministic windows and return their source objects."""
    split = split_history(
        market_history(),
        train_ratio=0.60,
        validation_ratio=0.20,
    )
    scalers = fit_scalers(split.train)
    windows = create_windowed_split(split, scalers, lookback=3)
    path = save_windowed_split(windows, scalers, tmp_path / "prepared.npz")
    return path, windows, scalers


def test_load_prepared_archive_restores_arrays_and_metadata(tmp_path: Path) -> None:
    path, windows, scalers = saved_archive(tmp_path)

    archive = load_prepared_archive(path)

    assert archive.feature_columns == scalers.feature_columns
    assert archive.target_column == scalers.target_column
    assert archive.lookback == 3
    np.testing.assert_array_equal(
        archive.windows.train.features,
        windows.train.features,
    )
    np.testing.assert_array_equal(
        archive.windows.validation.targets,
        windows.validation.targets,
    )
    assert archive.windows.test.target_dates.equals(windows.test.target_dates)
    np.testing.assert_allclose(archive.feature_scale, scalers.feature_scaler.scale_)
    np.testing.assert_allclose(archive.target_offset, scalers.target_scaler.min_)


def test_loaded_archive_restores_targets_to_original_scale(tmp_path: Path) -> None:
    path, windows, scalers = saved_archive(tmp_path)
    archive = load_prepared_archive(path)

    expected = scalers.inverse_target(windows.test.targets)
    restored = archive.inverse_target(archive.windows.test.targets)

    np.testing.assert_allclose(restored, expected)


def test_load_prepared_archive_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.npz"
    np.savez(path, train_features=np.ones((2, 3, 1)))

    with pytest.raises(ValueError, match="missing fields"):
        load_prepared_archive(path)


def test_load_prepared_archive_rejects_metadata_shape_mismatch(
    tmp_path: Path,
) -> None:
    path, _, _ = saved_archive(tmp_path)
    with np.load(path, allow_pickle=False) as stored:
        payload = {name: stored[name].copy() for name in stored.files}
    payload["lookback"] = np.asarray(4)
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="feature shape"):
        load_prepared_archive(path)


def test_load_prepared_archive_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_prepared_archive(tmp_path / "missing.npz")


def test_inverse_target_rejects_non_finite_values(tmp_path: Path) -> None:
    path, _, _ = saved_archive(tmp_path)
    archive = load_prepared_archive(path)

    with pytest.raises(ValueError, match="finite"):
        archive.inverse_target(np.asarray([np.nan]))
