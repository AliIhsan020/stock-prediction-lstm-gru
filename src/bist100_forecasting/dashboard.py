"""Streamlit dashboard for BIST 100 data collection and exploration."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from bist100_forecasting.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    download_history,
    load_history,
    save_history,
)
from bist100_forecasting.eda import build_eda_figure, summarize_history

APP_TITLE = "BIST 100 Forecasting"


def inclusive_to_exclusive_end(end_date: date) -> str:
    """Convert a user-facing inclusive date to the data source's exclusive date."""
    return (end_date + timedelta(days=1)).isoformat()


def download_selected_history(start_date: date, end_date: date) -> pd.DataFrame:
    """Download the selected date range and persist the validated snapshot."""
    if start_date > end_date:
        raise ValueError("Start date must not be later than end date.")

    history = download_history(
        start=start_date.isoformat(),
        end=inclusive_to_exclusive_end(end_date),
    )
    save_history(history)
    return history


def _load_or_stop() -> pd.DataFrame:
    """Load local history or stop the page with an actionable message."""
    try:
        return load_history()
    except FileNotFoundError:
        st.info(
            "No local BIST 100 dataset was found. Use **Download / refresh data** "
            "in the sidebar to create it."
        )
        st.code("uv run bist100-download", language="bash")
        st.stop()


def _render_sidebar() -> tuple[date, date, bool]:
    """Render data controls and return the selected dates and refresh action."""
    default_start = date.fromisoformat(DEFAULT_START_DATE)
    default_end = date.fromisoformat(DEFAULT_END_DATE) - timedelta(days=1)

    with st.sidebar:
        st.header("Data controls")
        st.text_input("Symbol", value=DEFAULT_SYMBOL, disabled=True)
        start_date = st.date_input("Start date", value=default_start)
        end_date = st.date_input("End date", value=default_end)
        refresh = st.button("Download / refresh data", type="primary")
        st.caption(f"Local file: `{DEFAULT_DATA_PATH.as_posix()}`")

    return start_date, end_date, refresh


def _render_summary(history: pd.DataFrame) -> None:
    """Render high-level market statistics."""
    summary = summarize_history(history)
    st.subheader("Dataset overview")

    first_row = st.columns(2)
    first_row[0].metric("Observations", f"{summary.observations:,}")
    first_row[1].metric("Latest close", f"{summary.latest_close:,.2f}")

    second_row = st.columns(2)
    second_row[0].metric("Total return", f"{summary.total_return_pct:,.2f}%")
    second_row[1].metric("Maximum drawdown", f"{summary.maximum_drawdown_pct:,.2f}%")

    st.caption(
        f"Data range: {summary.start_date} to {summary.end_date} · "
        f"Daily volatility: {summary.daily_volatility_pct:.4f}%"
    )


def _render_analysis_tabs(history: pd.DataFrame) -> None:
    """Render the EDA figure and recent observations."""
    chart_tab, data_tab = st.tabs(["Charts", "Recent observations"])

    with chart_tab:
        figure = build_eda_figure(history)
        st.pyplot(figure, clear_figure=True, width="stretch")

    with data_tab:
        recent_history = history.tail(20).sort_index(ascending=False)
        st.dataframe(recent_history, width="stretch")
        csv_data = history.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download current CSV",
            data=csv_data,
            file_name="bist100.csv",
            mime="text/csv",
        )


def render_dashboard() -> None:
    """Render the complete Streamlit data and EDA dashboard."""
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    st.title(APP_TITLE)
    st.write(
        "Explore validated BIST 100 history before the LSTM and GRU modeling stages."
    )

    start_date, end_date, refresh = _render_sidebar()
    if start_date > end_date:
        st.error("Start date must not be later than end date.")
        st.stop()

    if refresh:
        with st.spinner("Downloading and validating BIST 100 history..."):
            history = download_selected_history(start_date, end_date)
        st.success(f"Saved {len(history):,} validated observations.")
    else:
        history = _load_or_stop()

    _render_summary(history)
    _render_analysis_tabs(history)
    st.warning(
        "Educational project only. Historical model accuracy does not guarantee "
        "future performance and is not financial advice."
    )
