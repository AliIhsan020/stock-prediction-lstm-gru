# BIST 100 Forecasting with LSTM and GRU

An educational time-series project that trains and compares PyTorch LSTM and
GRU models for one-step-ahead forecasting of the BIST 100 closing index.

The project covers the complete workflow:

- downloading and validating daily `XU100.IS` OHLCV data;
- exploratory data analysis;
- chronological train, validation, and test splitting;
- training-only Min-Max scaling and sequence generation;
- LSTM and GRU training with early stopping;
- best-validation-checkpoint storage;
- evaluation against simple forecasting baselines;
- notebook analysis and a local Streamlit dashboard.

The first version models the BIST 100 index itself. It does not train a separate
model for every company included in the index.

> [!IMPORTANT]
> This repository is a machine learning study, not financial advice or a trading
> system. Historical prediction accuracy does not guarantee future performance.

## Methodology

Each model receives the previous 60 trading days of:

- Open
- High
- Low
- Close
- Volume

and predicts the next trading day's closing index value.

The data is kept in chronological order:

| Split | Share | Purpose |
|---|---:|---|
| Training | 70% | Fit scalers and model parameters |
| Validation | 15% | Early stopping and best-checkpoint selection |
| Test | 15% | Final comparison only |

Min-Max scalers are fitted only on the training period. Validation and test
values are transformed with those fixed training-period parameters, preventing
future information from leaking into model fitting.

The default recurrent-model configuration is:

| Setting | Value |
|---|---:|
| Hidden size | 64 |
| Recurrent layers | 2 |
| Dropout | 0.20 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Maximum epochs | 100 |
| Early-stopping patience | 10 |
| Gradient clipping | 1.0 |
| Random seed | 42 |

## Setup

The project requires Python 3.12 and uses
[uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv python install 3.12
uv sync --all-groups
```

Run commands from the repository root. `uv run` uses the project environment
without requiring manual virtual-environment activation.

## Complete workflow

### 1. Download BIST 100 history

```bash
uv run bist100-download
```

The command requests daily `XU100.IS` observations, validates the OHLCV schema,
and saves the available history to:

```text
data/raw/bist100.csv
```

The default requested range begins on `2010-01-01` and ends before
`2026-07-01`. The first available observation can be later than the requested
start date because data availability is controlled by Yahoo Finance.

Custom ranges and output paths are supported:

```bash
uv run bist100-download --start 2020-01-01 --end 2026-07-01
uv run bist100-download --output data/raw/custom_bist100.csv
```

Yahoo Finance treats `--end` as an exclusive date.

### 2. Run exploratory analysis

```bash
uv run bist100-eda
```

This prints descriptive statistics and stores generated analysis figures under
`reports/figures/`.

The same analysis can be explored interactively in:

```text
notebooks/01_exploratory_data_analysis.ipynb
```

### 3. Prepare model sequences

```bash
uv run bist100-prepare
```

This creates the chronological splits, fits the scalers on training data only,
builds 60-day windows, and saves the model-ready archive to:

```text
data/processed/bist100_sequences.npz
```

The preprocessing decisions can be inspected in:

```text
notebooks/02_preprocessing.ipynb
```

### 4. Train the LSTM

```bash
uv run bist100-train --model lstm
```

The best validation checkpoint is saved to:

```text
models/checkpoints/lstm.pt
```

### 5. Train the GRU

```bash
uv run bist100-train --model gru
```

The best validation checkpoint is saved to:

```text
models/checkpoints/gru.pt
```

Both training commands restore the best checkpoint and print original-scale test
MAE, RMSE, MAPE, and R². Hyperparameters can be changed from the command line:

```bash
uv run bist100-train --model gru --epochs 60 --batch-size 64
uv run bist100-train --help
```

CUDA is selected automatically when available. Use `--device cpu` to explicitly
run on the CPU.

### 6. Compare models and baselines

```bash
uv run bist100-compare
```

The command evaluates both checkpoints and two reference methods over identical
test dates:

- persistence: predict tomorrow's close as today's close;
- 20-day moving average: predict from the preceding 20 closes.

Results are ranked by test RMSE and saved to:

```text
reports/model_comparison.csv
```

The full comparison and dated prediction errors are available in:

```text
notebooks/03_model_comparison.ipynb
```

### 7. Open the Streamlit dashboard

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` if Streamlit does not open it automatically.

The dashboard provides:

- validated dataset statistics;
- closing-value, return, and drawdown charts;
- recent OHLCV observations and CSV download;
- LSTM and GRU metric cards;
- baseline and recurrent-model ranking;
- actual, LSTM, and GRU test-period prediction chart;
- clear setup instructions when generated artifacts are missing.

Press `Ctrl+C` in the terminal to stop the dashboard.

## Reference run

The following values came from the local data snapshot ending on `2026-06-10`.
The test period contained 242 observations from `2025-06-23` through
`2026-06-10`.

| Method | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| Persistence | 134.44 | 185.74 | 1.11% | 0.9847 |
| 20-day moving average | 447.16 | 541.78 | 3.64% | 0.8695 |
| LSTM | 431.18 | 547.24 | 3.37% | 0.8668 |
| GRU | 435.43 | 547.70 | 3.43% | 0.8666 |

Persistence produced the lowest RMSE in this run. This is an important baseline
result: the recurrent models followed the index but did not outperform the
simple previous-close forecast on unseen dates.

Exact results can change when the downloaded data, date range, hyperparameters,
software versions, hardware, or random seed changes.

## Command reference

| Command | Purpose |
|---|---|
| `uv run bist100-download` | Download and validate BIST 100 history |
| `uv run bist100-eda` | Print EDA statistics and generate figures |
| `uv run bist100-prepare` | Create scaled chronological sequences |
| `uv run bist100-train --model lstm` | Train and evaluate the LSTM |
| `uv run bist100-train --model gru` | Train and evaluate the GRU |
| `uv run bist100-compare` | Compare models and baselines |
| `uv run streamlit run app.py` | Start the local dashboard |
| `uv run pytest` | Run the automated test suite |
| `uv run ruff check .` | Run static code-quality checks |

Use `--help` with any project command to list all available options.

## Project structure

```text
.
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── checkpoints/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_comparison.ipynb
├── reports/
│   └── figures/
├── src/
│   └── bist100_forecasting/
├── tests/
├── pyproject.toml
└── uv.lock
```

Downloaded data, prepared arrays, generated figures, comparison reports, and
trained checkpoints are local artifacts excluded from version control.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check src tests app.py
uv lock --check
```

The tests cover data validation and storage, preprocessing leakage safeguards,
sequence creation, recurrent-model dimensions, training behavior, early
stopping, checkpoints, inference ordering, metrics, baselines, CLI workflows,
comparison logic, and Streamlit rendering.

## Limitations

- The project forecasts the BIST 100 index, not its individual constituents.
- Yahoo Finance data quality and availability can change.
- Results come from one chronological holdout rather than walk-forward
  validation across multiple market regimes.
- The current features are limited to daily OHLCV values.
- No transaction costs, slippage, position sizing, or trading rules are modeled.
- Hyperparameter search is intentionally limited, so the results are not proof
  of either model architecture's general superiority.
