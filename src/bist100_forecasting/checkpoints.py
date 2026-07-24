"""Atomic checkpoint storage for BIST 100 recurrent models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.optim import Optimizer

from bist100_forecasting.models import GRUForecaster, LSTMForecaster

if TYPE_CHECKING:
    from bist100_forecasting.training import EpochResult

CHECKPOINT_FORMAT_VERSION = 1
RecurrentModel = LSTMForecaster | GRUForecaster


@dataclass(frozen=True, slots=True)
class CheckpointEpoch:
    """One saved training-history observation."""

    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Restored model and resumable optimizer state."""

    model: RecurrentModel
    optimizer_state: dict
    epoch: int
    validation_loss: float
    history: tuple[CheckpointEpoch, ...]

    def restore_optimizer(self, optimizer: Optimizer) -> None:
        """Load the saved optimizer tensors into a compatible optimizer."""
        optimizer.load_state_dict(self.optimizer_state)


@dataclass(slots=True)
class BestCheckpoint:
    """Save a checkpoint whenever validation loss reaches a new minimum."""

    path: Path
    best_loss: float = field(init=False, default=math.inf)
    best_epoch: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def update(
        self,
        model: RecurrentModel,
        optimizer: Optimizer,
        *,
        epoch: int,
        validation_loss: float,
        history: Sequence[EpochResult],
    ) -> bool:
        """Save improved state and return whether a checkpoint was written."""
        _validate_checkpoint_observation(epoch, validation_loss)
        if validation_loss >= self.best_loss:
            return False

        save_checkpoint(
            self.path,
            model,
            optimizer,
            epoch=epoch,
            validation_loss=validation_loss,
            history=history,
        )
        self.best_loss = float(validation_loss)
        self.best_epoch = epoch
        return True


def save_checkpoint(
    output_path: Path,
    model: RecurrentModel,
    optimizer: Optimizer,
    *,
    epoch: int,
    validation_loss: float,
    history: Sequence[EpochResult],
) -> Path:
    """Atomically save model, optimizer, configuration, and loss history."""
    _validate_checkpoint_observation(epoch, validation_loss)
    model_name = _model_name(model)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": model_name,
        "model_config": model.config.as_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": float(validation_loss),
        "history": [
            {
                "epoch": result.epoch,
                "train_loss": result.train_loss,
                "validation_loss": result.validation_loss,
            }
            for result in history
        ],
    }

    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def load_checkpoint(
    input_path: Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Restore a supported recurrent model and its saved training state."""
    input_path = Path(input_path)
    payload = torch.load(
        input_path,
        map_location=torch.device(device),
        weights_only=True,
    )
    _validate_payload(payload)

    model_class = {
        "lstm": LSTMForecaster,
        "gru": GRUForecaster,
    }[payload["model_name"]]
    model = model_class(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.to(torch.device(device))
    model.eval()
    history = tuple(
        CheckpointEpoch(
            epoch=item["epoch"],
            train_loss=item["train_loss"],
            validation_loss=item["validation_loss"],
        )
        for item in payload["history"]
    )
    return LoadedCheckpoint(
        model=model,
        optimizer_state=payload["optimizer_state"],
        epoch=payload["epoch"],
        validation_loss=payload["validation_loss"],
        history=history,
    )


def _model_name(model: RecurrentModel) -> str:
    """Return the stable checkpoint name for a supported model."""
    if isinstance(model, LSTMForecaster):
        return "lstm"
    if isinstance(model, GRUForecaster):
        return "gru"
    raise TypeError("Checkpoint model must be an LSTMForecaster or GRUForecaster.")


def _validate_checkpoint_observation(epoch: int, validation_loss: float) -> None:
    """Validate checkpoint selection values."""
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError("Checkpoint epoch must be an integer.")
    if epoch <= 0:
        raise ValueError("Checkpoint epoch must be positive.")
    if isinstance(validation_loss, bool) or not isinstance(
        validation_loss, (int, float)
    ):
        raise TypeError("Checkpoint validation loss must be a number.")
    if not math.isfinite(validation_loss):
        raise ValueError("Checkpoint validation loss must be finite.")


def _validate_payload(payload: object) -> None:
    """Reject unsupported or incomplete checkpoint payloads."""
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    required_keys = {
        "format_version",
        "model_name",
        "model_config",
        "model_state",
        "optimizer_state",
        "epoch",
        "validation_loss",
        "history",
    }
    missing_keys = sorted(required_keys.difference(payload))
    if missing_keys:
        names = ", ".join(missing_keys)
        raise ValueError(f"Checkpoint is missing required fields: {names}.")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Checkpoint format version is not supported.")
    if payload["model_name"] not in {"lstm", "gru"}:
        raise ValueError("Checkpoint model type is not supported.")
