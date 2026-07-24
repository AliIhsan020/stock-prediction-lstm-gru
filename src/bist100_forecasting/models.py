"""PyTorch sequence models for one-step-ahead BIST 100 forecasting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class RecurrentConfig:
    """Serializable hyperparameters shared by recurrent forecasting models."""

    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.20

    def __post_init__(self) -> None:
        """Validate dimensions before a PyTorch module is constructed."""
        _validate_positive_integer(self.input_size, name="Input size")
        _validate_positive_integer(self.hidden_size, name="Hidden size")
        _validate_positive_integer(self.num_layers, name="Number of layers")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("Dropout must be a number.")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be between 0 inclusive and 1 exclusive.")

    def as_dict(self) -> dict[str, int | float]:
        """Return checkpoint-friendly configuration values."""
        return asdict(self)


class LSTMForecaster(nn.Module):
    """Many-to-one LSTM regressor using the final sequence representation."""

    def __init__(
        self,
        input_size: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.config = RecurrentConfig(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one scaled closing-value prediction per input sequence."""
        _validate_model_inputs(inputs, self.config)
        recurrent_output, _ = self.recurrent(inputs)
        final_hidden = recurrent_output[:, -1, :]
        return self.output(final_hidden).squeeze(-1)


class GRUForecaster(nn.Module):
    """Many-to-one GRU regressor using the final sequence representation."""

    def __init__(
        self,
        input_size: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.config = RecurrentConfig(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one scaled closing-value prediction per input sequence."""
        _validate_model_inputs(inputs, self.config)
        recurrent_output, _ = self.recurrent(inputs)
        final_hidden = recurrent_output[:, -1, :]
        return self.output(final_hidden).squeeze(-1)


def _validate_model_inputs(
    inputs: torch.Tensor,
    config: RecurrentConfig,
) -> None:
    """Validate shared batch, sequence, and feature dimensions."""
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("Model inputs must be a torch.Tensor.")
    if inputs.ndim != 3:
        raise ValueError("Model inputs must have shape (batch, sequence, features).")
    if inputs.shape[0] == 0 or inputs.shape[1] == 0:
        raise ValueError("Model inputs must contain a non-empty batch and sequence.")
    if inputs.shape[2] != config.input_size:
        raise ValueError(
            f"Expected {config.input_size} input features, received {inputs.shape[2]}."
        )
    if not inputs.is_floating_point():
        raise TypeError("Model inputs must use a floating-point dtype.")


def _validate_positive_integer(value: int, *, name: str) -> None:
    """Validate a positive, non-boolean integer hyperparameter."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
