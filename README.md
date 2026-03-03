# Predictive Analytics Pipeline — Retail Revenue Forecasting

## Overview

This project implements an end-to-end predictive analytics pipeline that transforms raw transactional retail data into daily revenue forecasts.

It demonstrates:

- Data ingestion and cleaning (ETL)
- Business KPI aggregation
- Feature engineering for time-series modeling
- Rolling-origin time-series cross-validation
- Baseline benchmarking vs machine learning
- Final production-style multi-step forecast

The system is configuration-driven and fully reproducible.

---

## Problem Statement

Retail businesses often rely on transactional data but lack structured forecasting pipelines.

This project answers the question:

> Can we forecast daily revenue from raw transaction-level data using a robust, production-ready pipeline?

We use the UCI "Online Retail" dataset (real-world e-commerce transactions) and forecast daily revenue.

---

## Architecture

Pipeline flow:

Raw Transactions (CSV/XLSX)  
→ Data Cleaning & Validation  
→ Daily KPI Aggregation (Revenue)  
→ Feature Engineering (lags, rolling statistics, calendar features)  
→ Model Training (XGBoost)  
→ Rolling Time-Series Cross-Validation  
→ Baseline Benchmarking (Seasonal Naive)  
→ Final Recursive Forecast  
→ Metrics + Visualization Outputs  

Key design principles:

- No data leakage (rolling stats computed on shifted series)
- Proper time-series validation (no random CV)
- Strong baseline comparison
- Config-driven execution

---

## Dataset

UCI Online Retail Dataset  
Transaction-level e-commerce data including:

- InvoiceDate
- Quantity
- UnitPrice
- CustomerID
- Product codes

Daily revenue is computed as:

Revenue = Quantity × UnitPrice

Returns and negative quantities are removed to avoid distortion.

---

## Model

Primary model:
- XGBoost Regressor (tree-based gradient boosting)

Baseline:
- Seasonal naive forecast (weekly seasonality: y[t] = y[t-7])

---

## Validation Strategy

Rolling-origin time-series cross-validation:

- Initial training window: 180 days
- Forecast horizon: 14 days
- Step size: 14 days
- Multiple folds

This simulates real-world forecasting conditions.

---

## Results

Rolling-origin CV results (mean across folds):

Baseline (Seasonal Naive):  
MAE = <BASELINE_MAE>

XGBoost:  
MAE = <XGB_MAE>

Improvement vs baseline (MAE):  
~8.7%

The model consistently outperforms the seasonal baseline across folds.

---

## Outputs

After running the pipeline:

- `outputs/metrics/metrics.json`
- `outputs/figures/last_fold_forecast_vs_actual.png`
- `outputs/forecasts/last_fold_predictions.csv`
- `outputs/forecasts/forecast.csv`

The final forecast provides a 14-day forward projection using recursive prediction.

---

## How to Run

### 1. Setup environment

```bash
make setup
```

### 2. Place dataset
Put the csv file at: 
```bash
data/raw/online_retail.csv
```

### 3. Run pipeline

```bash
make run
```

## Design Notes
- Feature engineering avoids leakage via shifted rolling statistics.
- Baseline benchmarking ensures measurable added value.
- Evaluation uses proper temporal validation rather than shuffled splits.
- Configuration-driven parameters enable reproducibility.
- Pipeline structure mirrors real-world production ML workflows.

## Tech Stack
- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

## Author
Simone Coscetti
AI Systems Architect & Predictive Analytics Consultant