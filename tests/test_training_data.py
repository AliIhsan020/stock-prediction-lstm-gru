"""Tests for PyTorch sequence datasets and data loaders."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from bist100_forecasting.preprocessing import SequenceWindows, WindowedSplit
from bist100_forecasting.training import SequenceDataset, create_data_loaders


def sequence_windows(samples: int = 10, *, offset: int = 0) -> SequenceWindows:
    """Create sequences whose target values expose their sample order."""
    sample_values = np.arange(offset, offset + samples, dtype=np.float64)
    features = np.repeat(sample_values[:, None, None], repeats=3, axis=1)
    features = np.repeat(features, repeats=2, axis=2)
    return SequenceWindows(
        features=features,
        targets=sample_values,
        target_dates=pd.date_range(
            "2025-01-02",
            periods=samples,
            freq="B",
            name="Date",
        ),
    )


def windowed_split() -> WindowedSplit:
    """Create three non-overlapping synthetic sequence sets."""
    return WindowedSplit(
        train=sequence_windows(10),
        validation=sequence_windows(6, offset=100),
        test=sequence_windows(5, offset=200),
    )


def test_sequence_dataset_creates_float32_tensors() -> None:
    windows = sequence_windows()

    dataset = SequenceDataset(windows)
    features, target = dataset[3]

    assert len(dataset) == 10
    assert dataset.features.shape == (10, 3, 2)
    assert dataset.features.dtype == torch.float32
    assert dataset.targets.dtype == torch.float32
    assert features.shape == (3, 2)
    assert target.item() == pytest.approx(3.0)
    assert dataset.target_dates.equals(windows.target_dates)


def test_data_loaders_use_expected_batch_shapes_and_samplers() -> None:
    loaders = create_data_loaders(windowed_split(), batch_size=4)

    train_features, train_targets = next(iter(loaders.train))
    validation_features, validation_targets = next(iter(loaders.validation))

    assert train_features.shape == (4, 3, 2)
    assert train_targets.shape == (4,)
    assert validation_features.shape == (4, 3, 2)
    assert validation_targets.shape == (4,)
    assert isinstance(loaders.train.sampler, RandomSampler)
    assert isinstance(loaders.validation.sampler, SequentialSampler)
    assert isinstance(loaders.test.sampler, SequentialSampler)


def test_training_shuffle_is_reproducible_with_the_same_seed() -> None:
    first = create_data_loaders(windowed_split(), batch_size=3, seed=7)
    second = create_data_loaders(windowed_split(), batch_size=3, seed=7)

    first_order = torch.cat([targets for _, targets in first.train])
    second_order = torch.cat([targets for _, targets in second.train])

    assert torch.equal(first_order, second_order)
    assert not torch.equal(first_order, torch.arange(10, dtype=torch.float32))


def test_evaluation_loaders_preserve_chronological_order() -> None:
    loaders = create_data_loaders(windowed_split(), batch_size=2)

    validation_order = torch.cat([targets for _, targets in loaders.validation])
    test_order = torch.cat([targets for _, targets in loaders.test])

    assert torch.equal(validation_order, torch.arange(100, 106, dtype=torch.float32))
    assert torch.equal(test_order, torch.arange(200, 205, dtype=torch.float32))


@pytest.mark.parametrize(
    ("features", "targets", "dates", "message"),
    [
        (np.ones((3, 2)), np.ones(3), pd.date_range("2025-01-01", periods=3), "three"),
        (
            np.ones((3, 2, 1)),
            np.ones((3, 1)),
            pd.date_range("2025-01-01", periods=3),
            "one-dimensional",
        ),
        (
            np.ones((0, 2, 1)),
            np.ones(0),
            pd.DatetimeIndex([]),
            "at least one",
        ),
        (
            np.ones((3, 2, 1)),
            np.ones(2),
            pd.date_range("2025-01-01", periods=2),
            "same length",
        ),
        (
            np.ones((3, 2, 1)),
            np.ones(3),
            pd.date_range("2025-01-01", periods=2),
            "Target dates",
        ),
        (
            np.full((3, 2, 1), np.nan),
            np.ones(3),
            pd.date_range("2025-01-01", periods=3),
            "finite",
        ),
    ],
)
def test_sequence_dataset_rejects_invalid_arrays(
    features,
    targets,
    dates,
    message: str,
) -> None:
    windows = SequenceWindows(
        features=features,
        targets=targets,
        target_dates=pd.DatetimeIndex(dates),
    )

    with pytest.raises(ValueError, match=message):
        SequenceDataset(windows)


@pytest.mark.parametrize(
    ("keyword", "value", "error", "message"),
    [
        ("batch_size", 0, ValueError, "at least 1"),
        ("batch_size", True, TypeError, "integer"),
        ("seed", -1, ValueError, "at least 0"),
        ("seed", 1.5, TypeError, "integer"),
        ("num_workers", -1, ValueError, "at least 0"),
        ("num_workers", False, TypeError, "integer"),
    ],
)
def test_data_loaders_reject_invalid_options(
    keyword: str,
    value,
    error: type[Exception],
    message: str,
) -> None:
    arguments = {keyword: value}

    with pytest.raises(error, match=message):
        create_data_loaders(windowed_split(), **arguments)
