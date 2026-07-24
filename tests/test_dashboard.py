"""Tests for the Streamlit data exploration dashboard."""

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import bist100_forecasting.dashboard as dashboard

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


def test_inclusive_end_date_becomes_exclusive() -> None:
    assert dashboard.inclusive_to_exclusive_end(date(2025, 1, 31)) == "2025-02-01"


def test_download_selected_history_downloads_and_saves(monkeypatch) -> None:
    history = market_history()
    download = Mock(return_value=history)
    save = Mock()
    monkeypatch.setattr(dashboard, "download_history", download)
    monkeypatch.setattr(dashboard, "save_history", save)

    result = dashboard.download_selected_history(
        date(2025, 1, 2),
        date(2025, 1, 6),
    )

    assert result is history
    download.assert_called_once_with(start="2025-01-02", end="2025-01-07")
    save.assert_called_once_with(history)


def test_download_selected_history_rejects_reversed_dates(monkeypatch) -> None:
    download = Mock()
    monkeypatch.setattr(dashboard, "download_history", download)

    with pytest.raises(ValueError, match="Start date must not be later"):
        dashboard.download_selected_history(
            date(2025, 1, 7),
            date(2025, 1, 6),
        )

    download.assert_not_called()


def test_dashboard_renders_saved_history(monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "load_history", market_history)

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
    assert [tab.label for tab in app.tabs] == ["Charts", "Recent observations"]
    assert app.button[0].label == "Download / refresh data"


def test_dashboard_explains_how_to_create_missing_data(monkeypatch) -> None:
    def missing_history() -> pd.DataFrame:
        raise FileNotFoundError

    monkeypatch.setattr(dashboard, "load_history", missing_history)

    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert "No local BIST 100 dataset was found" in app.info[0].value
    assert app.code[0].value == "uv run bist100-download"
