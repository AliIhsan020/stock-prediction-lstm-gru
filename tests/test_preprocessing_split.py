"""Tests for chronological market-history splitting."""

import pandas as pd
import pytest

from bist100_forecasting.preprocessing import split_history


def market_history(rows: int = 20) -> pd.DataFrame:
    """Create valid, increasing OHLCV history with the requested length."""
    close = pd.Series(range(100, 100 + rows), dtype=float)
    return pd.DataFrame(
        {
            "Open": close.to_numpy(),
            "High": close.add(2).to_numpy(),
            "Low": close.sub(2).to_numpy(),
            "Close": close.add(1).to_numpy(),
            "Volume": range(1_000, 1_000 + rows),
        },
        index=pd.date_range("2025-01-02", periods=rows, freq="B", name="Date"),
    )


def test_split_history_uses_default_chronological_proportions() -> None:
    history = market_history()

    split = split_history(history)

    assert len(split.train) == 14
    assert len(split.validation) == 3
    assert len(split.test) == 3
    assert split.train.index.max() < split.validation.index.min()
    assert split.validation.index.max() < split.test.index.min()


def test_split_history_preserves_every_observation_exactly_once() -> None:
    history = market_history()

    split = split_history(history)
    combined = pd.concat([split.train, split.validation, split.test])

    pd.testing.assert_frame_equal(combined, history)


def test_split_history_supports_custom_ratios() -> None:
    split = split_history(
        market_history(),
        train_ratio=0.60,
        validation_ratio=0.20,
    )

    assert len(split.train) == 12
    assert len(split.validation) == 4
    assert len(split.test) == 4


@pytest.mark.parametrize(
    ("train_ratio", "validation_ratio", "message"),
    [
        (0.0, 0.2, "Train ratio"),
        (1.0, 0.2, "Train ratio"),
        (0.7, 0.0, "Validation ratio"),
        (0.7, 1.0, "Validation ratio"),
        (0.8, 0.2, "sum to less than 1"),
    ],
)
def test_split_history_rejects_invalid_ratios(
    train_ratio: float,
    validation_ratio: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        split_history(
            market_history(),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )


def test_split_history_rejects_splits_with_empty_periods() -> None:
    with pytest.raises(ValueError, match="too small"):
        split_history(market_history(rows=3))


def test_split_history_reuses_market_data_validation() -> None:
    history = market_history().iloc[::-1]

    with pytest.raises(ValueError, match="ascending order"):
        split_history(history)
