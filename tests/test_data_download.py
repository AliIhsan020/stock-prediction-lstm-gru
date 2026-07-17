"""Tests for the network-independent BIST 100 download workflow."""

from typing import Any

import pandas as pd
import pytest

from bist100_forecasting.data import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    REQUIRED_COLUMNS,
    download_history,
)


def downloaded_history() -> pd.DataFrame:
    """Create a small valid response returned by a fake downloader."""
    return pd.DataFrame(
        {
            "Open": [10_000.0, 10_100.0],
            "High": [10_150.0, 10_200.0],
            "Low": [9_950.0, 10_000.0],
            "Close": [10_100.0, 10_050.0],
            "Volume": [1_000_000, 1_100_000],
        },
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )


def test_download_history_uses_reproducible_defaults() -> None:
    calls: dict[str, Any] = {}

    def fake_downloader(symbol: str, **kwargs: Any) -> pd.DataFrame:
        calls["symbol"] = symbol
        calls.update(kwargs)
        return downloaded_history()

    history = download_history(downloader=fake_downloader)

    assert list(history.columns) == list(REQUIRED_COLUMNS)
    assert calls == {
        "symbol": DEFAULT_SYMBOL,
        "start": DEFAULT_START_DATE,
        "end": DEFAULT_END_DATE,
        "interval": "1d",
        "actions": False,
        "auto_adjust": False,
        "progress": False,
        "threads": False,
        "multi_level_index": True,
    }


def test_download_history_forwards_custom_symbol_and_dates() -> None:
    calls: dict[str, Any] = {}

    def fake_downloader(symbol: str, **kwargs: Any) -> pd.DataFrame:
        calls["symbol"] = symbol
        calls.update(kwargs)
        return downloaded_history()

    download_history(
        symbol="GARAN.IS",
        start="2024-01-01",
        end="2025-01-01",
        downloader=fake_downloader,
    )

    assert calls["symbol"] == "GARAN.IS"
    assert calls["start"] == "2024-01-01"
    assert calls["end"] == "2025-01-01"


@pytest.mark.parametrize(
    ("start", "end"),
    [("2025-01-01", "2025-01-01"), ("2025-02-01", "2025-01-01")],
)
def test_download_history_rejects_invalid_date_order(start: str, end: str) -> None:
    downloader_called = False

    def fake_downloader(*args: Any, **kwargs: Any) -> pd.DataFrame:
        nonlocal downloader_called
        downloader_called = True
        return downloaded_history()

    with pytest.raises(ValueError, match="earlier than end"):
        download_history(start=start, end=end, downloader=fake_downloader)

    assert downloader_called is False


def test_download_history_rejects_unparseable_dates() -> None:
    with pytest.raises(ValueError, match="valid dates"):
        download_history(start="not-a-date", downloader=lambda *args, **kwargs: None)


def test_download_history_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol cannot be empty"):
        download_history(symbol="  ", downloader=lambda *args, **kwargs: None)


def test_download_history_rejects_empty_response() -> None:
    with pytest.raises(ValueError, match="returned no market history"):
        download_history(downloader=lambda *args, **kwargs: pd.DataFrame())


def test_download_history_rejects_none_response() -> None:
    with pytest.raises(ValueError, match="returned no market history"):
        download_history(downloader=lambda *args, **kwargs: None)
