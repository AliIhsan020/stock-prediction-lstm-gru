"""Tests for the PyTorch GRU forecasting architecture."""

import pytest
import torch

from bist100_forecasting.models import GRUForecaster, LSTMForecaster


def test_gru_returns_one_prediction_per_sequence() -> None:
    model = GRUForecaster(
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


def test_gru_supports_variable_sequence_lengths() -> None:
    model = GRUForecaster(input_size=3, hidden_size=8)

    short_predictions = model(torch.randn(2, 10, 3))
    long_predictions = model(torch.randn(2, 90, 3))

    assert short_predictions.shape == (2,)
    assert long_predictions.shape == (2,)


def test_gru_parameters_receive_gradients() -> None:
    model = GRUForecaster(input_size=5, hidden_size=8, num_layers=1)
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


def test_single_layer_gru_disables_internal_dropout() -> None:
    model = GRUForecaster(input_size=5, num_layers=1, dropout=0.50)

    assert model.config.dropout == 0.50
    assert model.recurrent.dropout == 0.0


def test_gru_has_fewer_parameters_than_equivalent_lstm() -> None:
    gru = GRUForecaster(input_size=5, hidden_size=32, num_layers=2)
    lstm = LSTMForecaster(input_size=5, hidden_size=32, num_layers=2)

    gru_parameters = sum(parameter.numel() for parameter in gru.parameters())
    lstm_parameters = sum(parameter.numel() for parameter in lstm.parameters())

    assert gru_parameters < lstm_parameters


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("input_size", 0, ValueError),
        ("hidden_size", 0, ValueError),
        ("num_layers", 0, ValueError),
        ("dropout", -0.1, ValueError),
        ("dropout", 1.0, ValueError),
        ("dropout", True, TypeError),
    ],
)
def test_gru_rejects_invalid_configuration(
    keyword: str,
    value,
    error: type[Exception],
) -> None:
    arguments = {"input_size": 5, keyword: value}

    with pytest.raises(error):
        GRUForecaster(**arguments)


@pytest.mark.parametrize(
    ("inputs", "error"),
    [
        ([[[1.0]]], TypeError),
        (torch.randn(3, 5), ValueError),
        (torch.empty(0, 10, 5), ValueError),
        (torch.empty(2, 0, 5), ValueError),
        (torch.randn(2, 10, 4), ValueError),
        (torch.ones(2, 10, 5, dtype=torch.int64), TypeError),
    ],
)
def test_gru_rejects_invalid_inputs(inputs, error: type[Exception]) -> None:
    model = GRUForecaster(input_size=5)

    with pytest.raises(error):
        model(inputs)
