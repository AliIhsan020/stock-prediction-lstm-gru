"""Tests for recurrent-model checkpoint persistence."""

from pathlib import Path

import pytest
import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset

from bist100_forecasting.checkpoints import (
    BestCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from bist100_forecasting.models import GRUForecaster, LSTMForecaster
from bist100_forecasting.training import (
    EarlyStopping,
    EpochResult,
    SequenceDataLoaders,
    fit_model,
)


def one_optimizer_step(model) -> Adam:
    """Populate model and Adam optimizer state before checkpointing."""
    optimizer = Adam(model.parameters(), lr=0.01)
    inputs = torch.randn(4, 5, model.config.input_size)
    targets = torch.randn(4)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return optimizer


def sample_history() -> tuple[EpochResult, ...]:
    """Create checkpoint-friendly epoch records."""
    return (
        EpochResult(epoch=1, train_loss=0.5, validation_loss=0.6),
        EpochResult(epoch=2, train_loss=0.4, validation_loss=0.45),
    )


@pytest.mark.parametrize(
    ("model_class", "model_name"),
    [(LSTMForecaster, "LSTMForecaster"), (GRUForecaster, "GRUForecaster")],
)
def test_checkpoint_round_trip_restores_model_and_training_state(
    model_class,
    model_name: str,
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    model = model_class(input_size=3, hidden_size=4, num_layers=1)
    optimizer = one_optimizer_step(model)
    inputs = torch.randn(3, 5, 3)
    model.eval()
    expected_predictions = model(inputs).detach()
    checkpoint_path = tmp_path / "model.pt"

    result_path = save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=2,
        validation_loss=0.45,
        history=sample_history(),
    )
    restored = load_checkpoint(checkpoint_path)

    assert result_path == checkpoint_path
    assert restored.model.__class__.__name__ == model_name
    assert restored.model.config == model.config
    assert restored.epoch == 2
    assert restored.validation_loss == pytest.approx(0.45)
    assert restored.history[1].train_loss == pytest.approx(0.4)
    assert not restored.model.training
    torch.testing.assert_close(restored.model(inputs), expected_predictions)

    restored_optimizer = Adam(restored.model.parameters(), lr=0.01)
    restored.restore_optimizer(restored_optimizer)
    assert restored_optimizer.state_dict()["state"]
    assert not checkpoint_path.with_suffix(".pt.tmp").exists()


def test_best_checkpoint_does_not_overwrite_with_worse_model(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    model = GRUForecaster(input_size=2, hidden_size=3, num_layers=1)
    optimizer = Adam(model.parameters())
    manager = BestCheckpoint(tmp_path / "best.pt")
    inputs = torch.randn(2, 4, 2)

    assert manager.update(
        model,
        optimizer,
        epoch=1,
        validation_loss=0.5,
        history=sample_history()[:1],
    )
    saved_predictions = load_checkpoint(manager.path).model(inputs).detach()
    with torch.no_grad():
        model.output.bias.add_(10)

    assert not manager.update(
        model,
        optimizer,
        epoch=2,
        validation_loss=0.6,
        history=sample_history(),
    )
    restored_predictions = load_checkpoint(manager.path).model(inputs).detach()

    torch.testing.assert_close(restored_predictions, saved_predictions)
    assert manager.best_epoch == 1
    assert manager.best_loss == pytest.approx(0.5)


def test_fit_model_saves_best_checkpoint_before_early_stopping(
    tmp_path: Path,
) -> None:
    features = torch.ones(6, 3, 1)
    targets = torch.ones(6)
    loader = DataLoader(TensorDataset(features, targets), batch_size=3)
    data_loaders = SequenceDataLoaders(train=loader, validation=loader, test=loader)
    model = GRUForecaster(input_size=1, hidden_size=2, num_layers=1)
    optimizer = SGD(model.parameters(), lr=0.0)
    checkpoint = BestCheckpoint(tmp_path / "best.pt")

    history = fit_model(
        model,
        data_loaders,
        optimizer,
        epochs=5,
        early_stopping=EarlyStopping(patience=1),
        checkpoint=checkpoint,
    )

    assert len(history) == 2
    assert checkpoint.path.is_file()
    assert checkpoint.best_epoch == 1
    restored = load_checkpoint(checkpoint.path)
    assert restored.epoch == 1
    assert len(restored.history) == 1


def test_load_checkpoint_rejects_unsupported_format(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "invalid.pt"
    torch.save(
        {
            "format_version": 999,
            "model_name": "gru",
            "model_config": {},
            "model_state": {},
            "optimizer_state": {},
            "epoch": 1,
            "validation_loss": 1.0,
            "history": [],
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="format version"):
        load_checkpoint(checkpoint_path)
