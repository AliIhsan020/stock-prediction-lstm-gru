"""PyTorch datasets and data loaders for recurrent model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bist100_forecasting.preprocessing import SequenceWindows, WindowedSplit

DEFAULT_BATCH_SIZE = 32
DEFAULT_RANDOM_SEED = 42


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Validated float32 tensors for one-step-ahead sequence learning."""

    def __init__(self, windows: SequenceWindows) -> None:
        features = np.asarray(windows.features)
        targets = np.asarray(windows.targets)
        if features.ndim != 3:
            raise ValueError("Sequence features must be a three-dimensional array.")
        if targets.ndim != 1:
            raise ValueError("Sequence targets must be a one-dimensional array.")
        if len(features) == 0:
            raise ValueError("Sequence dataset must contain at least one sample.")
        if len(features) != len(targets):
            raise ValueError("Sequence features and targets must have the same length.")
        if len(windows.target_dates) != len(targets):
            raise ValueError(
                "Target dates and target values must have the same length."
            )
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("Sequence features and targets must be finite.")

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.target_dates = windows.target_dates.copy()

    def __len__(self) -> int:
        """Return the number of available forecast samples."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one input sequence and its scaled target."""
        return self.features[index], self.targets[index]


@dataclass(frozen=True, slots=True)
class SequenceDataLoaders:
    """Training, validation, and test data loaders."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader


def create_data_loaders(
    windows: WindowedSplit,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
    num_workers: int = 0,
) -> SequenceDataLoaders:
    """Create reproducible training and ordered evaluation data loaders."""
    _validate_loader_options(
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    return SequenceDataLoaders(
        train=DataLoader(
            SequenceDataset(windows.train),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            drop_last=False,
        ),
        validation=DataLoader(
            SequenceDataset(windows.validation),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        ),
        test=DataLoader(
            SequenceDataset(windows.test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        ),
    )


def _validate_loader_options(
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> None:
    """Validate deterministic DataLoader configuration values."""
    _validate_integer(batch_size, name="Batch size", minimum=1)
    _validate_integer(seed, name="Seed", minimum=0)
    _validate_integer(num_workers, name="Number of workers", minimum=0)


def _validate_integer(value: int, *, name: str, minimum: int) -> None:
    """Validate an integer option with an inclusive lower bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
