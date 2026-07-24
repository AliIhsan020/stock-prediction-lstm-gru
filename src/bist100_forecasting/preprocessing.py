"""Chronological data preparation for BIST 100 forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bist100_forecasting.data import validate_history

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VALIDATION_RATIO = 0.15


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Train, validation, and test periods kept in chronological order."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_history(
    history: pd.DataFrame,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
) -> ChronologicalSplit:
    """Split validated history by position without shuffling observations."""
    validate_history(history)
    _validate_split_ratios(train_ratio, validation_ratio)

    train_end = int(len(history) * train_ratio)
    validation_end = train_end + int(len(history) * validation_ratio)
    if train_end == 0 or validation_end == train_end or validation_end >= len(history):
        raise ValueError(
            "Dataset is too small to create non-empty train, validation, "
            "and test splits with the requested ratios."
        )

    return ChronologicalSplit(
        train=history.iloc[:train_end].copy(),
        validation=history.iloc[train_end:validation_end].copy(),
        test=history.iloc[validation_end:].copy(),
    )


def _validate_split_ratios(train_ratio: float, validation_ratio: float) -> None:
    """Validate that all three chronological split proportions are positive."""
    if not 0 < train_ratio < 1:
        raise ValueError("Train ratio must be between 0 and 1.")
    if not 0 < validation_ratio < 1:
        raise ValueError("Validation ratio must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than 1.")
