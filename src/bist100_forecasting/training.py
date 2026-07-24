"""PyTorch datasets and data loaders for recurrent model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from bist100_forecasting.preprocessing import SequenceWindows, WindowedSplit

DEFAULT_BATCH_SIZE = 32
DEFAULT_RANDOM_SEED = 42
DEFAULT_GRADIENT_CLIP = 1.0


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


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Training and validation loss recorded for one epoch."""

    epoch: int
    train_loss: float
    validation_loss: float


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


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    *,
    device: str | torch.device = "cpu",
    loss_function: nn.Module | None = None,
    gradient_clip: float | None = DEFAULT_GRADIENT_CLIP,
) -> float:
    """Update model parameters for one epoch and return sample-weighted loss."""
    _validate_gradient_clip(gradient_clip)
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.train()
    criterion = loss_function or nn.MSELoss()
    criterion.to(resolved_device)

    total_loss = 0.0
    sample_count = 0
    for features, targets in data_loader:
        features = features.to(resolved_device)
        targets = targets.to(resolved_device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        _validate_prediction_shape(predictions, targets)
        loss = criterion(predictions, targets)
        _validate_finite_loss(loss)
        loss.backward()
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        batch_size = len(targets)
        total_loss += loss.detach().item() * batch_size
        sample_count += batch_size

    return _average_loss(total_loss, sample_count)


@torch.no_grad()
def evaluate_model_loss(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: str | torch.device = "cpu",
    loss_function: nn.Module | None = None,
) -> float:
    """Evaluate a model without gradient tracking and return sample-weighted loss."""
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    criterion = loss_function or nn.MSELoss()
    criterion.to(resolved_device)

    total_loss = 0.0
    sample_count = 0
    for features, targets in data_loader:
        features = features.to(resolved_device)
        targets = targets.to(resolved_device)
        predictions = model(features)
        _validate_prediction_shape(predictions, targets)
        loss = criterion(predictions, targets)
        _validate_finite_loss(loss)

        batch_size = len(targets)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return _average_loss(total_loss, sample_count)


def fit_model(
    model: nn.Module,
    data_loaders: SequenceDataLoaders,
    optimizer: Optimizer,
    *,
    epochs: int,
    device: str | torch.device = "cpu",
    loss_function: nn.Module | None = None,
    gradient_clip: float | None = DEFAULT_GRADIENT_CLIP,
) -> tuple[EpochResult, ...]:
    """Train for a fixed number of epochs and record validation loss."""
    _validate_integer(epochs, name="Epochs", minimum=1)
    _validate_gradient_clip(gradient_clip)
    history: list[EpochResult] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            data_loaders.train,
            optimizer,
            device=device,
            loss_function=loss_function,
            gradient_clip=gradient_clip,
        )
        validation_loss = evaluate_model_loss(
            model,
            data_loaders.validation,
            device=device,
            loss_function=loss_function,
        )
        history.append(
            EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
            )
        )

    return tuple(history)


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


def _validate_gradient_clip(gradient_clip: float | None) -> None:
    """Validate the optional maximum gradient norm."""
    if gradient_clip is None:
        return
    if isinstance(gradient_clip, bool) or not isinstance(gradient_clip, (int, float)):
        raise TypeError("Gradient clip must be a number or None.")
    if gradient_clip <= 0:
        raise ValueError("Gradient clip must be positive.")


def _validate_prediction_shape(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Prevent broadcasting from hiding a model-output shape error."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {tuple(predictions.shape)} does not match "
            f"target shape {tuple(targets.shape)}."
        )


def _validate_finite_loss(loss: torch.Tensor) -> None:
    """Stop training when the loss becomes NaN or infinite."""
    if not torch.isfinite(loss):
        raise FloatingPointError("Model loss became non-finite.")


def _average_loss(total_loss: float, sample_count: int) -> float:
    """Return sample-weighted loss after rejecting an empty data loader."""
    if sample_count == 0:
        raise ValueError("Data loader must contain at least one sample.")
    return total_loss / sample_count


def _validate_integer(value: int, *, name: str, minimum: int) -> None:
    """Validate an integer option with an inclusive lower bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
