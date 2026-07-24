"""Tests for BIST 100 exploratory summary statistics."""

from datetime import date

import pandas as pd
import pytest

from bist100_forecasting.eda import calculate_daily_returns, summarize_history


def market_history() -> pd.DataFrame:
    """Create deterministic OHLCV data with known daily returns."""
    return pd.DataFrame(
        {
            "Open": [99.0, 105.0, 105.0],
            "High": [101.0, 112.0, 106.0],
            "Low": [98.0, 104.0, 98.0],
            "Close": [100.0, 110.0, 99.0],
            "Volume": [1_000, 1_100, 1_050],
        },
        index=pd.date_range("2025-01-02", periods=3, freq="B", name="Date"),
    )


def test_calculate_daily_returns_uses_previous_close() -> None:
    history = market_history()

    returns = calculate_daily_returns(history)

    assert returns.name == "Daily Return"
    assert returns.index.equals(history.index[1:])
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)


def test_summarize_history_reports_dates_and_levels() -> None:
    summary = summarize_history(market_history())

    assert summary.observations == 3
    assert summary.start_date == date(2025, 1, 2)
    assert summary.end_date == date(2025, 1, 6)
    assert summary.first_close == pytest.approx(100.0)
    assert summary.latest_close == pytest.approx(99.0)
    assert summary.minimum_close == pytest.approx(99.0)
    assert summary.maximum_close == pytest.approx(110.0)


def test_summarize_history_reports_return_and_risk_statistics() -> None:
    summary = summarize_history(market_history())

    assert summary.total_return_pct == pytest.approx(-1.0)
    assert summary.mean_daily_return_pct == pytest.approx(0.0)
    assert summary.daily_volatility_pct == pytest.approx(14.1421356)
    assert summary.maximum_drawdown_pct == pytest.approx(-10.0)


def test_calculate_daily_returns_requires_two_observations() -> None:
    with pytest.raises(ValueError, match="at least two observations"):
        calculate_daily_returns(market_history().iloc[:1])


def test_summarize_history_requires_three_observations() -> None:
    with pytest.raises(ValueError, match="at least three observations"):
        summarize_history(market_history().iloc[:2])


def test_eda_statistics_reuse_market_data_validation() -> None:
    history = market_history()
    history.index = pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-06"])

    with pytest.raises(ValueError, match="duplicate dates"):
        calculate_daily_returns(history)
