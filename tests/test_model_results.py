"""Tests for reusable saved-model result loading."""

from pathlib import Path

import pandas as pd
import pytest
from torch.optim import Adam

from bist100_forecasting.checkpoints import save_checkpoint
from bist100_forecasting.data import save_history
from bist100_forecasting.model_results import (
    evaluate_saved_models,
    missing_model_artifacts,
)
from bist100_forecasting.models import GRUForecaster, LSTMForecaster
from bist100_forecasting.preprocessing import (
    create_windowed_split,
    fit_scalers,
    save_windowed_split,
    split_history,
)
from bist100_forecasting.training import EpochResult


def market_history(rows: int = 36) -> pd.DataFrame:
    """Create valid deterministic OHLCV history."""
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


def saved_inputs(tmp_path: Path) -> dict[str, Path | pd.DataFrame]:
    """Save prepared arrays and deterministic recurrent checkpoints."""
    history = market_history()
    save_history(history, tmp_path / "history.csv")
    split = split_history(
        history,
        train_ratio=0.60,
        validation_ratio=0.20,
    )
    scalers = fit_scalers(split.train)
    windows = create_windowed_split(split, scalers, lookback=3)
    archive_path = save_windowed_split(
        windows,
        scalers,
        tmp_path / "prepared.npz",
    )
    checkpoints = {}
    for name, model_class in (
        ("lstm", LSTMForecaster),
        ("gru", GRUForecaster),
    ):
        model = model_class(
            input_size=len(scalers.feature_columns),
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
        )
        checkpoints[name] = save_checkpoint(
            tmp_path / f"{name}.pt",
            model,
            Adam(model.parameters()),
            epoch=2,
            validation_loss=0.25,
            history=(
                EpochResult(
                    epoch=1,
                    train_loss=0.5,
                    validation_loss=0.4,
                ),
                EpochResult(
                    epoch=2,
                    train_loss=0.3,
                    validation_loss=0.25,
                ),
            ),
        )
    return {
        "history": history,
        "archive": archive_path,
        **checkpoints,
    }


def test_evaluate_saved_models_returns_aligned_results(tmp_path: Path) -> None:
    inputs = saved_inputs(tmp_path)

    results = evaluate_saved_models(
        inputs["history"],
        archive_path=inputs["archive"],
        lstm_checkpoint_path=inputs["lstm"],
        gru_checkpoint_path=inputs["gru"],
        batch_size=4,
        moving_average_window=3,
    )

    assert results.lstm.best_epoch == 2
    assert results.gru.validation_loss == pytest.approx(0.25)
    assert results.lstm.evaluation.target_dates.equals(
        results.gru.evaluation.target_dates
    )
    assert set(results.comparison.index) == {
        "LSTM",
        "GRU",
        "Persistence",
        "3-day moving average",
    }
    assert results.winner == results.comparison.index[0]


def test_evaluate_saved_models_rejects_swapped_checkpoint(tmp_path: Path) -> None:
    inputs = saved_inputs(tmp_path)

    with pytest.raises(ValueError, match="LSTM checkpoint contains GRUForecaster"):
        evaluate_saved_models(
            inputs["history"],
            archive_path=inputs["archive"],
            lstm_checkpoint_path=inputs["gru"],
            gru_checkpoint_path=inputs["gru"],
        )


def test_missing_model_artifacts_returns_only_absent_paths(tmp_path: Path) -> None:
    archive_path = tmp_path / "prepared.npz"
    lstm_path = tmp_path / "lstm.pt"
    gru_path = tmp_path / "gru.pt"
    archive_path.touch()

    missing = missing_model_artifacts(
        archive_path=archive_path,
        lstm_checkpoint_path=lstm_path,
        gru_checkpoint_path=gru_path,
    )

    assert missing == (lstm_path, gru_path)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("batch_size", 0, "Batch size must be positive"),
        ("moving_average_window", 0, "Moving-average window must be positive"),
    ],
)
def test_evaluate_saved_models_rejects_invalid_options(
    option: str,
    value: int,
    message: str,
) -> None:
    arguments = {option: value}

    with pytest.raises(ValueError, match=message):
        evaluate_saved_models(market_history(), **arguments)
