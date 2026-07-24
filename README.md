# BIST 100 Forecasting with LSTM and GRU

This educational project will implement and compare LSTM and GRU models in
PyTorch for one-step-ahead forecasting of the BIST 100 index.

## Initial scope

- Use the historical BIST 100 index series represented by `XU100.IS`.
- Predict the next trading day's closing index value from past observations.
- Preserve chronological order throughout training and evaluation.
- Compare LSTM and GRU against simple forecasting baselines.
- Report prediction error and training cost under the same conditions.

The first version models the BIST 100 index itself. It does not train a separate
model for each company included in the index.

> [!IMPORTANT]
> This repository is a machine learning study, not financial advice or a trading
> system. Historical prediction accuracy does not guarantee future performance.

## Setup

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/) for dependency
management.

```bash
uv python install 3.12
uv sync --all-groups
```

Commands can then be run through `uv run` without manually activating the virtual
environment.

## Download BIST 100 data

Download the default daily `XU100.IS` snapshot:

```bash
uv run bist100-download
```

The command downloads observations from `2010-01-01` through `2026-06-30`,
validates the OHLCV data, and writes it to:

```text
data/raw/bist100.csv
```

Yahoo Finance treats the `--end` value as exclusive. The default end value is
therefore `2026-07-01`. A custom date range or output path can be supplied:

```bash
uv run bist100-download --start 2020-01-01 --end 2026-07-01
uv run bist100-download --output data/raw/custom_bist100.csv
```

Display every available option:

```bash
uv run bist100-download --help
```

The same command can also be run through the Python package:

```bash
uv run python -m bist100_forecasting
```

Downloaded CSV files are generated locally and are not committed to Git.

## Explore the data in Streamlit

Start the local dashboard from the project root:

```bash
uv run streamlit run app.py
```

Streamlit opens the dashboard at `http://localhost:8501`. The sidebar can
download or refresh the BIST 100 history for a selected date range. The main
page displays:

- the number of observations and latest closing value;
- total return, maximum drawdown, and daily volatility;
- closing-value and daily-return charts;
- the 20 most recent OHLCV observations;
- a button for downloading the current dataset as CSV.

The dashboard currently supports data collection and exploratory analysis.
LSTM and GRU training and prediction views will be added in later stages.

Press `Ctrl+C` in the terminal to stop the dashboard.
