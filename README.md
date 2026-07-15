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
