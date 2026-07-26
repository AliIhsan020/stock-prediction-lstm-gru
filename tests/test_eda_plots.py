"""Tests for BIST 100 exploratory analysis plots."""

from pathlib import Path

import pandas as pd

from bist100_forecasting.eda import build_eda_figure, save_eda_figure


def market_history() -> pd.DataFrame:
    """Create valid OHLCV data for plotting tests."""
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


def test_build_eda_figure_creates_close_and_return_panels() -> None:
    history = market_history()

    figure = build_eda_figure(history)

    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "BIST 100 Daily Closing Value"
    assert figure.axes[0].get_ylabel() == "Closing value"
    assert len(figure.axes[0].lines[0].get_xdata()) == len(history)
    assert figure.axes[1].get_title() == "BIST 100 Daily Return"
    assert figure.axes[1].get_ylabel() == "Return (%)"
    assert len(figure.axes[1].lines[0].get_xdata()) == len(history) - 1
    assert figure.canvas.manager is None


def test_save_eda_figure_creates_nonempty_png(tmp_path: Path) -> None:
    output_path = tmp_path / "figures" / "bist100.png"

    result = save_eda_figure(market_history(), output_path)

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_build_eda_figure_does_not_require_pyplot_manager() -> None:
    figure = build_eda_figure(market_history())

    assert figure.canvas.manager is None


def test_build_eda_figure_uses_selected_instrument_label() -> None:
    figure = build_eda_figure(market_history(), "THYAO — Türk Hava Yolları")

    assert figure.axes[0].get_title() == "THYAO — Türk Hava Yolları Daily Closing Value"
    assert figure.axes[0].get_ylabel() == "Closing value"
    assert figure.axes[1].get_title() == "THYAO — Türk Hava Yolları Daily Return"
