"""Tests for validation-loss early stopping."""

import math

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from bist100_forecasting.training import (
    EarlyStopping,
    SequenceDataLoaders,
    fit_model,
)


class ConstantForecaster(nn.Module):
    """Regressor kept fixed to produce a constant validation loss."""

    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Linear(2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(inputs.flatten(start_dim=1)).squeeze(-1)


def constant_loaders() -> SequenceDataLoaders:
    """Create identical loaders with non-zero targets."""
    features = torch.ones(6, 2, 1)
    targets = torch.ones(6)
    loader = DataLoader(TensorDataset(features, targets), batch_size=3)
    return SequenceDataLoaders(train=loader, validation=loader, test=loader)


def test_early_stopping_tracks_improvements_and_resets_patience() -> None:
    stopping = EarlyStopping(patience=2, min_delta=0.01)

    assert not stopping.update(1.0, epoch=1)
    assert stopping.best_loss == 1.0
    assert stopping.best_epoch == 1
    assert not stopping.update(0.995, epoch=2)
    assert stopping.epochs_without_improvement == 1
    assert not stopping.update(0.95, epoch=3)
    assert stopping.best_loss == 0.95
    assert stopping.best_epoch == 3
    assert stopping.epochs_without_improvement == 0


def test_early_stopping_stops_after_configured_patience() -> None:
    stopping = EarlyStopping(patience=2)

    assert not stopping.update(1.0, epoch=1)
    assert not stopping.update(1.1, epoch=2)
    assert stopping.update(1.2, epoch=3)
    assert stopping.should_stop
    assert stopping.best_epoch == 1
    assert stopping.update(0.5, epoch=4)
    assert stopping.best_epoch == 1


def test_early_stopping_can_be_reset() -> None:
    stopping = EarlyStopping(patience=1)
    stopping.update(1.0, epoch=1)
    stopping.update(1.1, epoch=2)

    stopping.reset()

    assert math.isinf(stopping.best_loss)
    assert stopping.best_epoch == 0
    assert stopping.epochs_without_improvement == 0
    assert not stopping.should_stop


def test_fit_model_stops_when_validation_loss_does_not_improve() -> None:
    model = ConstantForecaster()
    optimizer = SGD(model.parameters(), lr=0.0)
    stopping = EarlyStopping(patience=2)

    history = fit_model(
        model,
        constant_loaders(),
        optimizer,
        epochs=10,
        gradient_clip=None,
        early_stopping=stopping,
    )

    assert len(history) == 3
    assert stopping.should_stop
    assert stopping.best_epoch == 1
    assert stopping.best_loss == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("keyword", "value", "error", "message"),
    [
        ("patience", 0, ValueError, "at least 1"),
        ("patience", True, TypeError, "integer"),
        ("min_delta", -0.1, ValueError, "non-negative"),
        ("min_delta", float("inf"), ValueError, "finite"),
        ("min_delta", True, TypeError, "number"),
    ],
)
def test_early_stopping_rejects_invalid_configuration(
    keyword: str,
    value,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        EarlyStopping(**{keyword: value})


@pytest.mark.parametrize(
    ("loss", "epoch", "error", "message"),
    [
        (float("nan"), 1, ValueError, "finite"),
        (float("inf"), 1, ValueError, "finite"),
        (True, 1, TypeError, "number"),
        (1.0, 0, ValueError, "at least 1"),
        (1.0, True, TypeError, "integer"),
    ],
)
def test_early_stopping_rejects_invalid_observations(
    loss,
    epoch,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        EarlyStopping().update(loss, epoch=epoch)
