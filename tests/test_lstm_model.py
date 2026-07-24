"""Tests for the PyTorch LSTM forecasting architecture."""

import pytest
import torch

from bist100_forecasting.models import LSTMForecaster, RecurrentConfig


def test_lstm_returns_one_prediction_per_sequence() -> None:
    model = LSTMForecaster(
        input_size=5,
        hidden_size=16,
        num_layers=2,
        dropout=0.10,
    )
    inputs = torch.randn(8, 60, 5)

    predictions = model(inputs)

    assert predictions.shape == (8,)
    assert predictions.dtype == inputs.dtype
    assert model.recurrent.batch_first


def test_lstm_supports_variable_sequence_lengths() -> None:
    model = LSTMForecaster(input_size=3, hidden_size=8)

    short_predictions = model(torch.randn(2, 10, 3))
    long_predictions = model(torch.randn(2, 90, 3))

    assert short_predictions.shape == (2,)
    assert long_predictions.shape == (2,)


def test_lstm_parameters_receive_gradients() -> None:
    model = LSTMForecaster(input_size=5, hidden_size=8, num_layers=1)
    inputs = torch.randn(4, 12, 5)
    targets = torch.randn(4)

    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_single_layer_lstm_disables_internal_dropout() -> None:
    model = LSTMForecaster(input_size=5, num_layers=1, dropout=0.50)

    assert model.config.dropout == 0.50
    assert model.recurrent.dropout == 0.0


def test_recurrent_config_is_checkpoint_friendly() -> None:
    config = RecurrentConfig(
        input_size=5,
        hidden_size=32,
        num_layers=3,
        dropout=0.25,
    )

    assert config.as_dict() == {
        "input_size": 5,
        "hidden_size": 32,
        "num_layers": 3,
        "dropout": 0.25,
    }


@pytest.mark.parametrize(
    ("keyword", "value", "error", "message"),
    [
        ("input_size", 0, ValueError, "Input size must be positive"),
        ("input_size", True, TypeError, "Input size must be an integer"),
        ("hidden_size", 0, ValueError, "Hidden size must be positive"),
        ("num_layers", 0, ValueError, "Number of layers must be positive"),
        ("dropout", -0.1, ValueError, "Dropout must be between"),
        ("dropout", 1.0, ValueError, "Dropout must be between"),
        ("dropout", True, TypeError, "Dropout must be a number"),
    ],
)
def test_lstm_rejects_invalid_configuration(
    keyword: str,
    value,
    error: type[Exception],
    message: str,
) -> None:
    arguments = {"input_size": 5, keyword: value}

    with pytest.raises(error, match=message):
        LSTMForecaster(**arguments)


@pytest.mark.parametrize(
    ("inputs", "error", "message"),
    [
        ([[[1.0]]], TypeError, "torch.Tensor"),
        (torch.randn(3, 5), ValueError, "shape"),
        (torch.empty(0, 10, 5), ValueError, "non-empty"),
        (torch.empty(2, 0, 5), ValueError, "non-empty"),
        (torch.randn(2, 10, 4), ValueError, "Expected 5"),
        (torch.ones(2, 10, 5, dtype=torch.int64), TypeError, "floating-point"),
    ],
)
def test_lstm_rejects_invalid_inputs(
    inputs,
    error: type[Exception],
    message: str,
) -> None:
    model = LSTMForecaster(input_size=5)

    with pytest.raises(error, match=message):
        model(inputs)
