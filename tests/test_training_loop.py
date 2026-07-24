"""Tests for shared PyTorch model training and validation loops."""

import math

import pytest
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from bist100_forecasting.training import (
    SequenceDataLoaders,
    evaluate_model_loss,
    fit_model,
    train_one_epoch,
)


class TinyForecaster(nn.Module):
    """Small deterministic regressor used to test training behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(inputs.flatten(start_dim=1)).squeeze(-1)


def regression_loader(*, samples: int = 12, batch_size: int = 4) -> DataLoader:
    """Create a simple learnable sequence-regression dataset."""
    features = torch.linspace(-1.0, 1.0, steps=samples * 2).reshape(samples, 2, 1)
    targets = features.flatten(start_dim=1).sum(dim=1)
    return DataLoader(TensorDataset(features, targets), batch_size=batch_size)


def data_loaders() -> SequenceDataLoaders:
    """Create loaders for fixed-epoch fitting tests."""
    return SequenceDataLoaders(
        train=regression_loader(),
        validation=regression_loader(),
        test=regression_loader(),
    )


def test_train_one_epoch_updates_parameters() -> None:
    torch.manual_seed(7)
    model = TinyForecaster()
    optimizer = Adam(model.parameters(), lr=0.05)
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_one_epoch(model, regression_loader(), optimizer)

    assert math.isfinite(loss)
    assert loss >= 0
    assert model.training
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters(), strict=True)
    )


def test_evaluate_model_loss_matches_manual_mean_squared_error() -> None:
    model = TinyForecaster()
    nn.init.zeros_(model.output.weight)
    nn.init.zeros_(model.output.bias)
    loader = regression_loader(samples=5, batch_size=2)
    expected_targets = torch.cat([targets for _, targets in loader])
    expected_loss = torch.square(expected_targets).mean().item()

    loss = evaluate_model_loss(model, loader)

    assert loss == pytest.approx(expected_loss)
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fit_model_records_training_and_validation_history() -> None:
    torch.manual_seed(7)
    model = TinyForecaster()
    optimizer = Adam(model.parameters(), lr=0.05)

    history = fit_model(model, data_loaders(), optimizer, epochs=6)

    assert [result.epoch for result in history] == [1, 2, 3, 4, 5, 6]
    assert all(math.isfinite(result.train_loss) for result in history)
    assert all(math.isfinite(result.validation_loss) for result in history)
    assert history[-1].train_loss < history[0].train_loss
    assert not model.training


def test_train_one_epoch_applies_gradient_clipping(monkeypatch) -> None:
    model = TinyForecaster()
    optimizer = Adam(model.parameters())
    clipping_calls: list[float] = []

    def record_clipping(parameters, max_norm: float) -> None:
        list(parameters)
        clipping_calls.append(max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clipping)

    train_one_epoch(
        model,
        regression_loader(samples=6, batch_size=2),
        optimizer,
        gradient_clip=0.75,
    )

    assert clipping_calls == [0.75, 0.75, 0.75]


def test_evaluate_model_loss_rejects_empty_loader() -> None:
    empty_loader = DataLoader(
        TensorDataset(torch.empty(0, 2, 1), torch.empty(0)),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="at least one sample"):
        evaluate_model_loss(TinyForecaster(), empty_loader)


def test_training_rejects_prediction_shape_mismatch() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(2, 1))
    optimizer = Adam(model.parameters())

    with pytest.raises(ValueError, match="Prediction shape"):
        train_one_epoch(model, regression_loader(), optimizer)


def test_evaluation_rejects_non_finite_loss() -> None:
    features = torch.ones(2, 2, 1)
    targets = torch.tensor([float("inf"), 1.0])
    loader = DataLoader(TensorDataset(features, targets), batch_size=2)

    with pytest.raises(FloatingPointError, match="non-finite"):
        evaluate_model_loss(TinyForecaster(), loader)


@pytest.mark.parametrize(
    ("keyword", "value", "error", "message"),
    [
        ("epochs", 0, ValueError, "at least 1"),
        ("epochs", True, TypeError, "integer"),
        ("gradient_clip", 0.0, ValueError, "positive"),
        ("gradient_clip", -1.0, ValueError, "positive"),
        ("gradient_clip", True, TypeError, "number or None"),
    ],
)
def test_fit_model_rejects_invalid_options(
    keyword: str,
    value,
    error: type[Exception],
    message: str,
) -> None:
    model = TinyForecaster()
    optimizer = Adam(model.parameters())
    arguments = {"epochs": 1, keyword: value}

    with pytest.raises(error, match=message):
        fit_model(model, data_loaders(), optimizer, **arguments)
