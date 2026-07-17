"""Schema and validation rules for daily BIST 100 market data."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_SYMBOL = "XU100.IS"
DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2026-07-01"
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
PRICE_COLUMNS = ("Open", "High", "Low", "Close")

Downloader = Callable[..., pd.DataFrame | None]


def download_history(
    *,
    symbol: str = DEFAULT_SYMBOL,
    start: str = DEFAULT_START_DATE,
    end: str = DEFAULT_END_DATE,
    downloader: Downloader = yf.download,
) -> pd.DataFrame:
    """Download and normalize daily history; ``end`` is an exclusive date."""
    if not symbol.strip():
        raise ValueError("Market symbol cannot be empty.")

    try:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
    except (TypeError, ValueError) as error:
        raise ValueError("Start and end must be valid dates.") from error
    if pd.isna(start_date) or pd.isna(end_date):
        raise ValueError("Start and end must be valid dates.")
    if start_date >= end_date:
        raise ValueError("Start date must be earlier than end date.")

    raw_history = downloader(
        symbol,
        start=start,
        end=end,
        interval="1d",
        actions=False,
        auto_adjust=False,
        progress=False,
        threads=False,
        multi_level_index=True,
    )
    if raw_history is None:
        raise ValueError("Yahoo Finance returned no market history.")
    return normalize_yahoo_history(raw_history)


def normalize_yahoo_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a single-ticker Yahoo Finance response to validated OHLCV data."""
    if frame.empty:
        raise ValueError("Yahoo Finance returned no market history.")

    history = frame.copy()
    if isinstance(history.columns, pd.MultiIndex):
        price_levels = [
            level
            for level in range(history.columns.nlevels)
            if set(REQUIRED_COLUMNS).issubset(
                set(history.columns.get_level_values(level))
            )
        ]
        if len(price_levels) != 1:
            raise ValueError("Could not identify Yahoo Finance price columns.")

        price_level = price_levels[0]
        ticker_values = history.columns.droplevel(price_level).unique()
        if len(ticker_values) != 1:
            raise ValueError("Expected Yahoo Finance data for exactly one ticker.")
        history.columns = history.columns.get_level_values(price_level)

    if history.columns.has_duplicates:
        raise ValueError("Yahoo Finance response contains duplicate columns.")

    missing_columns = set(REQUIRED_COLUMNS).difference(history.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Yahoo Finance response is missing columns: {names}.")

    history = history.loc[:, REQUIRED_COLUMNS].copy()
    try:
        history.index = pd.to_datetime(history.index, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Yahoo Finance dates could not be parsed.") from error
    if history.index.tz is not None:
        history.index = history.index.tz_localize(None)
    history.index.name = "Date"
    history = history.sort_index()

    for column in REQUIRED_COLUMNS:
        try:
            history[column] = pd.to_numeric(history[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError("Yahoo Finance values must be numeric.") from error

    validate_history(history)
    return history


def validate_history(history: pd.DataFrame) -> None:
    """Raise ``ValueError`` when daily market history is unsafe to process."""
    if history.empty:
        raise ValueError("Market history must contain at least one observation.")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("Market history must use a DatetimeIndex.")
    if history.index.has_duplicates:
        raise ValueError("Market history contains duplicate dates.")
    if not history.index.is_monotonic_increasing:
        raise ValueError("Market history dates must be in ascending order.")

    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(history.columns))
    if missing_columns:
        names = ", ".join(missing_columns)
        raise ValueError(f"Market history is missing required columns: {names}.")

    required_data = history.loc[:, REQUIRED_COLUMNS]
    if required_data.isna().any().any():
        raise ValueError("Market history contains missing values.")

    try:
        numeric_values = required_data.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Market history values must be numeric.") from error

    if not np.isfinite(numeric_values).all():
        raise ValueError("Market history contains infinite values.")
    if (history.loc[:, PRICE_COLUMNS].to_numpy(dtype=np.float64) <= 0).any():
        raise ValueError("Market prices must be greater than zero.")
    if (history["Volume"].to_numpy(dtype=np.float64) < 0).any():
        raise ValueError("Market volume cannot be negative.")

    high = history["High"].to_numpy(dtype=np.float64)
    low = history["Low"].to_numpy(dtype=np.float64)
    open_price = history["Open"].to_numpy(dtype=np.float64)
    close = history["Close"].to_numpy(dtype=np.float64)
    if (high < np.maximum.reduce([open_price, low, close])).any():
        raise ValueError("High must be the greatest price in each observation.")
    if (low > np.minimum.reduce([open_price, high, close])).any():
        raise ValueError("Low must be the smallest price in each observation.")
