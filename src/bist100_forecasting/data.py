"""Schema and validation rules for daily BIST 100 market data."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SYMBOL = "XU100.IS"
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
PRICE_COLUMNS = ("Open", "High", "Low", "Close")


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
