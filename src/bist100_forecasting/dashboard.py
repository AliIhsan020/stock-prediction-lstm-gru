"""Streamlit dashboard for BIST 100 data collection and exploration."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from bist100_forecasting.data import (
    DEFAULT_START_DATE,
    download_history,
    history_path_for_symbol,
    load_history,
    save_history,
)
from bist100_forecasting.eda import build_eda_figure, summarize_history
from bist100_forecasting.instruments import (
    BIST100_INDEX,
    SELECTABLE_INSTRUMENTS,
    Instrument,
)
from bist100_forecasting.model_results import (
    SavedModelResult,
    SavedModelResults,
    evaluate_saved_models,
    missing_model_artifacts,
)

APP_TITLE = "BIST 100 Forecasting"


def inclusive_to_exclusive_end(end_date: date) -> str:
    """Convert a user-facing inclusive date to the data source's exclusive date."""
    return (end_date + timedelta(days=1)).isoformat()


def download_selected_history(
    instrument: Instrument,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Download the selected date range and persist the validated snapshot."""
    if start_date > end_date:
        raise ValueError("Start date must not be later than end date.")

    history = download_history(
        symbol=instrument.yahoo_symbol,
        start=start_date.isoformat(),
        end=inclusive_to_exclusive_end(end_date),
    )
    save_history(history, history_path_for_symbol(instrument.code))
    return history


def _load_or_stop(instrument: Instrument) -> pd.DataFrame:
    """Load local history or stop the page with an actionable message."""
    data_path = history_path_for_symbol(instrument.code)
    try:
        return load_history(data_path)
    except FileNotFoundError:
        st.info(
            f"No local dataset was found for **{instrument.label}**. Use "
            "**Download selected data** in the sidebar to create it."
        )
        st.stop()


def _render_sidebar() -> tuple[Instrument, date, date, bool]:
    """Render controls and return the selected instrument, dates, and action."""
    default_start = date.fromisoformat(DEFAULT_START_DATE)
    default_end = date.today()

    with st.sidebar:
        st.header("Data controls")
        instrument = st.selectbox(
            "Instrument",
            options=SELECTABLE_INSTRUMENTS,
            format_func=lambda item: item.label,
        )
        start_date = st.date_input("Start date", value=default_start)
        end_date = st.date_input("End date", value=default_end)
        refresh = st.button("Download selected data", type="primary")
        data_path = history_path_for_symbol(instrument.code)
        st.caption(f"Yahoo symbol: `{instrument.yahoo_symbol}`")
        st.caption(f"Local file: `{data_path.as_posix()}`")

    return instrument, start_date, end_date, refresh


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


def _render_analysis_tabs(history: pd.DataFrame, instrument: Instrument) -> None:
    """Render exploratory data and saved-model results."""
    chart_tab, data_tab, model_tab = st.tabs(
        ["Charts", "Recent observations", "Model results"]
    )

    with chart_tab:
        figure = build_eda_figure(history, instrument.label)
        st.pyplot(figure, clear_figure=True, width="stretch")

    with data_tab:
        recent_history = history.tail(20).sort_index(ascending=False)
        st.dataframe(recent_history, width="stretch")
        csv_data = history.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download current CSV",
            data=csv_data,
            file_name=f"{instrument.code.lower()}.csv",
            mime="text/csv",
        )

    with model_tab:
        if instrument != BIST100_INDEX:
            st.info(
                "Saved LSTM and GRU checkpoints currently belong to the BIST 100 "
                "index. A model for the selected stock has not been trained yet."
            )
        else:
            _render_model_results(history)


def build_prediction_frame(results: SavedModelResults) -> pd.DataFrame:
    """Build aligned actual, LSTM, and GRU values for charting."""
    lstm = results.lstm.evaluation
    gru = results.gru.evaluation
    if not lstm.target_dates.equals(gru.target_dates):
        raise ValueError("LSTM and GRU target dates must match.")
    if not pd.Series(lstm.actual).equals(pd.Series(gru.actual)):
        raise ValueError("LSTM and GRU actual values must match.")

    return pd.DataFrame(
        {
            "Actual": lstm.actual,
            "LSTM prediction": lstm.predicted,
            "GRU prediction": gru.predicted,
        },
        index=lstm.target_dates.copy(),
    )


def _render_model_results(history: pd.DataFrame) -> None:
    """Render model metrics, baseline ranking, and dated predictions."""
    st.subheader("Saved model results")
    missing = missing_model_artifacts()
    if missing:
        st.info(
            "Prepared sequences and both trained checkpoints are required before "
            "model results can be displayed."
        )
        st.write("Missing files:")
        for path in missing:
            st.write(f"- `{path.as_posix()}`")
        st.code(
            "\n".join(
                [
                    "uv run bist100-prepare",
                    "uv run bist100-train --model lstm",
                    "uv run bist100-train --model gru",
                ]
            ),
            language="bash",
        )
        return

    try:
        with st.spinner("Loading checkpoints and evaluating the test period..."):
            results = evaluate_saved_models(history)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        st.error(f"Model results could not be loaded: {error}")
        return

    st.success(f"Best method by test RMSE: {results.winner}")
    model_columns = st.columns(2)
    _render_model_metrics(model_columns[0], "LSTM", results.lstm)
    _render_model_metrics(model_columns[1], "GRU", results.gru)

    st.markdown("#### Test-period comparison")
    st.caption(
        "All methods use the same dates. Rank 1 has the lowest test-period RMSE."
    )
    st.dataframe(results.comparison.round(4), width="stretch")

    st.markdown("#### Actual and predicted close")
    predictions = build_prediction_frame(results)
    st.line_chart(predictions, width="stretch")
    st.caption(
        f"Test range: {predictions.index.min().date()} to "
        f"{predictions.index.max().date()} · {len(predictions):,} observations"
    )


def _render_model_metrics(column, name: str, result: SavedModelResult) -> None:
    """Render one model's primary test metrics and checkpoint metadata."""
    metrics = result.evaluation.metrics
    with column:
        st.markdown(f"#### {name}")
        metric_columns = st.columns(2)
        metric_columns[0].metric(f"{name} RMSE", f"{metrics.rmse:,.2f}")
        metric_columns[1].metric(f"{name} MAE", f"{metrics.mae:,.2f}")
        st.caption(
            f"MAPE: {metrics.mape_pct:.2f}% · R²: {metrics.r2:.4f} · "
            f"best epoch: {result.best_epoch} · "
            f"validation loss: {result.validation_loss:.6f}"
        )


def render_dashboard() -> None:
    """Render the complete Streamlit data and EDA dashboard."""
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    st.title(APP_TITLE)
    st.write(
        "Explore validated BIST 100 history and compare saved LSTM and GRU results."
    )

    instrument, start_date, end_date, refresh = _render_sidebar()
    if start_date > end_date:
        st.error("Start date must not be later than end date.")
        st.stop()

    if refresh:
        with st.spinner(f"Downloading and validating {instrument.label} history..."):
            history = download_selected_history(instrument, start_date, end_date)
        st.success(
            f"Saved {len(history):,} validated observations for {instrument.code}."
        )
    else:
        history = _load_or_stop(instrument)

    st.markdown(f"### {instrument.label}")
    _render_summary(history)
    _render_analysis_tabs(history, instrument)
    st.warning(
        "Educational project only. Historical model accuracy does not guarantee "
        "future performance and is not financial advice."
    )
