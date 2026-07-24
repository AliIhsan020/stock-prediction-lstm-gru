"""Tests for the BIST 100 exploratory-analysis command."""

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from bist100_forecasting.data import DEFAULT_DATA_PATH
from bist100_forecasting.eda import DEFAULT_EDA_FIGURE, HistorySummary
from bist100_forecasting.eda_cli import build_parser, format_summary, main


def sample_history() -> pd.DataFrame:
    """Create a small history passed between fake CLI dependencies."""
    return pd.DataFrame(
        {"Close": [100.0, 105.0, 102.0]},
        index=pd.date_range("2025-01-02", periods=3, freq="B", name="Date"),
    )


def sample_summary() -> HistorySummary:
    """Create deterministic summary values for output assertions."""
    return HistorySummary(
        observations=3,
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 6),
        first_close=100.0,
        latest_close=102.0,
        minimum_close=100.0,
        maximum_close=105.0,
        total_return_pct=2.0,
        mean_daily_return_pct=1.1,
        daily_volatility_pct=2.25,
        maximum_drawdown_pct=-2.8571,
    )


def test_parser_uses_project_paths_by_default() -> None:
    args = build_parser().parse_args([])

    assert args.input == DEFAULT_DATA_PATH
    assert args.output == DEFAULT_EDA_FIGURE


def test_format_summary_includes_return_and_risk_statistics() -> None:
    output = format_summary(sample_summary())

    assert "Observations: 3" in output
    assert "Date range: 2025-01-02 to 2025-01-06" in output
    assert "Total return: 2.00%" in output
    assert "Daily volatility: 2.2500%" in output
    assert "Maximum drawdown: -2.86%" in output


def test_main_loads_summarizes_and_saves_custom_paths(
    capsys: Any,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}
    history = sample_history()
    input_path = tmp_path / "history.csv"
    output_path = tmp_path / "eda.png"

    def fake_load(path: Path) -> pd.DataFrame:
        calls["load"] = path
        return history

    def fake_summarize(frame: pd.DataFrame) -> HistorySummary:
        calls["summarize"] = frame
        return sample_summary()

    def fake_save_figure(frame: pd.DataFrame, path: Path) -> Path:
        calls["save_figure"] = (frame, path)
        return path

    exit_code = main(
        ["--input", str(input_path), "--output", str(output_path)],
        load=fake_load,
        summarize=fake_summarize,
        save_figure=fake_save_figure,
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert calls["load"] == input_path
    assert calls["summarize"] is history
    assert calls["save_figure"] == (history, output_path)
    assert "BIST 100 exploratory summary" in output
    assert f"Figure saved to: {output_path}" in output
