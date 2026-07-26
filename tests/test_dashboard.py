"""Tests for the Streamlit data exploration dashboard."""

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import bist100_forecasting.dashboard as dashboard
from bist100_forecasting.evaluation import evaluate_forecast
from bist100_forecasting.inference import ForecastEvaluation
from bist100_forecasting.instruments import BIST100_INDEX, get_bist100_instrument
from bist100_forecasting.model_results import SavedModelResult, SavedModelResults

APP_PATH = Path(__file__).parents[1] / "app.py"


def market_history() -> pd.DataFrame:
    """Create deterministic OHLCV data for dashboard tests."""
    return pd.DataFrame(
        {
            "Open": [99.0, 105.0, 105.0, 101.0],
            "High": [101.0, 112.0, 106.0, 110.0],
            "Low": [98.0, 104.0, 98.0, 100.0],
            "Close": [100.0, 110.0, 99.0, 108.0],
            "Volume": [1_000, 1_100, 1_050, 1_200],
        },
        index=pd.date_range("2025-01-02", periods=4, freq="B", name="Date"),
    )


def saved_model_results() -> SavedModelResults:
    """Create aligned model results for Streamlit rendering tests."""
    dates = pd.date_range("2025-01-02", periods=4, freq="B", name="Date")
    actual = np.asarray([100.0, 110.0, 99.0, 108.0])
    lstm_predicted = np.asarray([101.0, 109.0, 100.0, 107.0])
    gru_predicted = np.asarray([102.0, 108.0, 101.0, 106.0])
    lstm_evaluation = ForecastEvaluation(
        target_dates=dates,
        actual=actual,
        predicted=lstm_predicted,
        metrics=evaluate_forecast(actual, lstm_predicted),
    )
    gru_evaluation = ForecastEvaluation(
        target_dates=dates,
        actual=actual.copy(),
        predicted=gru_predicted,
        metrics=evaluate_forecast(actual, gru_predicted),
    )
    comparison = pd.DataFrame(
        {
            "Rank": [1, 2, 3, 4],
            "MAE": [1.0, 2.0, 3.0, 4.0],
            "RMSE": [1.0, 2.0, 3.0, 4.0],
            "MAPE (%)": [1.0, 2.0, 3.0, 4.0],
            "R2": [0.9, 0.8, 0.7, 0.6],
        },
        index=pd.Index(
            ["LSTM", "GRU", "Persistence", "20-day moving average"],
            name="Method",
        ),
    )
    return SavedModelResults(
        lstm=SavedModelResult(
            checkpoint_path=Path("models/checkpoints/lstm.pt"),
            best_epoch=8,
            validation_loss=0.01,
            evaluation=lstm_evaluation,
        ),
        gru=SavedModelResult(
            checkpoint_path=Path("models/checkpoints/gru.pt"),
            best_epoch=6,
            validation_loss=0.02,
            evaluation=gru_evaluation,
        ),
        comparison=comparison,
        winner="LSTM",
    )


def test_inclusive_end_date_becomes_exclusive() -> None:
    assert dashboard.inclusive_to_exclusive_end(date(2025, 1, 31)) == "2025-02-01"


def test_download_selected_history_downloads_and_saves(monkeypatch) -> None:
    history = market_history()
    download = Mock(return_value=history)
    save = Mock()
    monkeypatch.setattr(dashboard, "download_history", download)
    monkeypatch.setattr(dashboard, "save_history", save)

    result = dashboard.download_selected_history(
        get_bist100_instrument("THYAO"),
        date(2025, 1, 2),
        date(2025, 1, 6),
    )

    assert result is history
    download.assert_called_once_with(
        symbol="THYAO.IS",
        start="2025-01-02",
        end="2025-01-07",
    )
    save.assert_called_once_with(
        history,
        Path("data/raw/stocks/thyao.csv"),
    )


def test_download_selected_history_rejects_reversed_dates(monkeypatch) -> None:
    download = Mock()
    monkeypatch.setattr(dashboard, "download_history", download)

    with pytest.raises(ValueError, match="Start date must not be later"):
        dashboard.download_selected_history(
            BIST100_INDEX,
            date(2025, 1, 7),
            date(2025, 1, 6),
        )

    download.assert_not_called()


def test_dashboard_renders_saved_history(monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "load_history", lambda _path: market_history())
    monkeypatch.setattr(
        dashboard,
        "missing_model_artifacts",
        lambda: (Path("models/checkpoints/lstm.pt"),),
    )

    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == dashboard.APP_TITLE
    assert [metric.label for metric in app.metric] == [
        "Observations",
        "Latest close",
        "Total return",
        "Maximum drawdown",
    ]
    assert app.metric[0].value == "4"
    assert [tab.label for tab in app.tabs] == [
        "Charts",
        "Recent observations",
        "Model results",
    ]
    assert app.button[0].label == "Download selected data"
    assert len(app.selectbox[0].options) == 101
    assert app.selectbox[0].value == BIST100_INDEX
    assert "both trained checkpoints are required" in app.info[0].value


def test_dashboard_renders_saved_model_comparison(monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "load_history", lambda _path: market_history())
    monkeypatch.setattr(dashboard, "missing_model_artifacts", lambda: ())
    monkeypatch.setattr(
        dashboard,
        "evaluate_saved_models",
        Mock(return_value=saved_model_results()),
    )

    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert "Best method by test RMSE: LSTM" in app.success[0].value
    assert [metric.label for metric in app.metric[-4:]] == [
        "LSTM RMSE",
        "LSTM MAE",
        "GRU RMSE",
        "GRU MAE",
    ]
    assert any(subheader.value == "Saved model results" for subheader in app.subheader)


def test_build_prediction_frame_aligns_model_series() -> None:
    results = saved_model_results()

    frame = dashboard.build_prediction_frame(results)

    assert list(frame.columns) == [
        "Actual",
        "LSTM prediction",
        "GRU prediction",
    ]
    assert frame.index.equals(results.lstm.evaluation.target_dates)
    np.testing.assert_array_equal(
        frame["LSTM prediction"],
        results.lstm.evaluation.predicted,
    )


def test_dashboard_explains_how_to_create_missing_data(monkeypatch) -> None:
    def missing_history(_path: Path) -> pd.DataFrame:
        raise FileNotFoundError

    monkeypatch.setattr(dashboard, "load_history", missing_history)

    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert "No local dataset was found" in app.info[0].value
    assert BIST100_INDEX.label in app.info[0].value
