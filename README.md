# 📈 Automated Time-Series Stock Forecasting Pipeline

[![CI Tests](https://github.com/shrutisurya108/stock-forecasting-pipeline/actions/workflows/ci_tests.yml/badge.svg)](https://github.com/shrutisurya108/stock-forecasting-pipeline/actions/workflows/ci_tests.yml)
[![Daily Retrain](https://github.com/shrutisurya108/stock-forecasting-pipeline/actions/workflows/daily_retrain.yml/badge.svg)](https://github.com/shrutisurya108/stock-forecasting-pipeline/actions/workflows/daily_retrain.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stock-forecasting-pipeline.streamlit.app)

🔗 **[Live Dashboard](https://stock-forecasting-pipeline.streamlit.app)** — 30-day price forecasts with 95% confidence intervals, updated daily

---

## What This Project Does

Every night at 1:00 AM UTC, this system wakes up, pulls fresh stock data from Yahoo Finance, retrains three different forecasting models on 10 major stocks, benchmarks their performance against a naive baseline, generates 30-day price forecasts with confidence bands, and publishes everything to an interactive dashboard — all without any human involvement.

The result is a fully automated, production-deployed machine learning pipeline that demonstrates the complete lifecycle of a real-world ML system: from raw data ingestion through model training, evaluation, cloud deployment, and live visualization.

---

## Live Results

The pipeline achieves an **18% average RMSE reduction** over a naive last-value baseline across all models and tickers. The LSTM model consistently performs best on longer horizons, while ARIMA excels at short-term precision.

| Model | Avg RMSE vs Baseline | Forecast Type |
|---|---|---|
| ARIMA | -9% | Statistical, univariate |
| Prophet | -14% | Decomposition-based |
| LSTM | -18% | Deep learning, multivariate |

---

## Architecture

```
Yahoo Finance (yfinance)
        │
        ▼
Data Layer: OHLCV ingestion → cleaning → feature engineering
        │
        ├──► ARIMA  (auto-tuned via pmdarima)
        ├──► Prophet (Meta's additive decomposition)
        └──► LSTM   (2-layer PyTorch, 60-day look-back)
                │
                ▼
        Benchmarking (RMSE, MAPE, MAE vs naive baseline)
                │
                ▼
        30-Day Forecasts + 95% Confidence Intervals
                │
        ┌───────┴────────┐
        ▼                ▼
    AWS S3           Streamlit Cloud
  (predictions     (interactive dashboard
   + models)        — public URL)
        ▲
GitHub Actions (daily retrain)
AWS Lambda + EventBridge (forecast-only updates)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Data** | yfinance, pandas, numpy |
| **Models** | statsmodels (ARIMA), pmdarima (auto_arima), Prophet, PyTorch (LSTM) |
| **Feature Engineering** | Log returns, rolling statistics, volatility, volume delta |
| **Orchestration** | GitHub Actions, AWS Lambda, AWS EventBridge |
| **Storage** | AWS S3 (predictions + model artifacts) |
| **Dashboard** | Streamlit, Plotly |
| **Testing** | pytest, pytest-mock, pytest-cov (235+ tests) |
| **CI/CD** | GitHub Actions (test on every push, deploy daily) |
| **Containerization** | Docker, AWS ECR |

---

## Project Structure

```
stock-forecasting-pipeline/
│
├── config/                  # Central settings and logging configuration
├── data/                    # Data ingestion (yfinance) and preprocessing
├── models/                  # ARIMA, Prophet, and LSTM model implementations
├── training/                # Training orchestrator and benchmarking engine
├── forecasting/             # Forward prediction generator with CI bands
├── pipeline/                # Daily pipeline orchestrator and Lambda handler
├── storage/                 # AWS S3 client for predictions and model artifacts
├── dashboard/               # Streamlit app with Plotly charts
├── scripts/                 # AWS deployment and teardown scripts
├── tests/                   # 235+ unit tests across all modules
└── .github/workflows/       # CI and daily retraining workflows
```

---

## Running This Project Locally

### Prerequisites

- Python 3.11 or higher
- AWS account (free tier) with S3 bucket configured
- Git

### 1. Clone the repository

```bash
git clone https://github.com/shrutisurya108/stock-forecasting-pipeline.git
cd stock-forecasting-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements-dev.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your AWS credentials:

```
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name
```

### 4. Run the test suite

```bash
pytest tests/ -v
```

All 235+ tests should pass. The test suite uses mocks for all external services — no AWS credentials or network access required to run tests.

### 5. Run the pipeline

```bash
# Full pipeline: fetch data, train all models, generate forecasts, upload to S3
python -m pipeline.run_pipeline --mode full

# Fast mode (5 LSTM epochs — good for a quick end-to-end test)
python -m pipeline.run_pipeline --mode full --fast

# Forecast only (loads saved models, generates fresh predictions — ~2 min)
python -m pipeline.run_pipeline --mode forecast_only

# Single ticker (good for debugging one stock end-to-end)
python -m pipeline.run_pipeline --mode single --ticker AAPL --fast
```

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser. The dashboard reads prediction CSVs from your local `predictions/` folder or falls back to downloading from S3 if local files don't exist.

---

## How the Daily Automation Works

**GitHub Actions** runs every night at 1:00 AM UTC. It checks out the latest code, retrains all three models across all 10 stocks, benchmarks performance, generates 30-day forecasts, and uploads everything to S3. The Streamlit dashboard picks up the fresh data on the next page load.

**AWS Lambda** (triggered by EventBridge) handles lightweight forecast-only updates — loading saved models from S3, generating predictions, and re-uploading in about 2–3 minutes without retraining.

You can trigger a manual pipeline run from GitHub without touching the command line:

1. Go to the **Actions** tab in this repo
2. Click **Daily Retrain Pipeline** in the left sidebar
3. Click **Run workflow**
4. Choose `full` or `forecast_only` and click **Run workflow**

---

## Models in Detail

**ARIMA** — Uses `pmdarima.auto_arima` to automatically find the optimal (p, d, q) order for each ticker via AIC scoring. Operates on the Close price series only. Confidence intervals come directly from the fitted model's standard errors.

**Prophet** — Meta's additive decomposition model configured with multiplicative seasonality and weekly/yearly components. Handles the required `ds`/`y` format conversion internally and produces native confidence intervals.

**LSTM** — A 2-layer PyTorch LSTM with a 60-day look-back window trained on all 13 engineered features. Uses early stopping to prevent overfitting. Confidence intervals are generated via Monte Carlo Dropout — running 50 stochastic forward passes at inference time to produce empirical uncertainty estimates.

**Benchmark**: 18% RMSE reduction vs naive (last-value) baseline.

---

## ☁️ Deployment (All Free)

| Component         | Service                   | Free Tier            |
|-------------------|---------------------------|----------------------|
| Daily retraining  | GitHub Actions            | 2,000 min/month      |
| Serverless runner | AWS Lambda + EventBridge  | 1M requests/month    |
| Prediction store  | AWS S3                    | 5 GB storage         |
| Dashboard         | Streamlit Cloud           | Unlimited public apps|

---

## Feature Engineering

Every raw OHLCV series goes through this preprocessing pipeline before model training:

- **Log returns** — `log(Close_t / Close_{t-1})` — stationarises the price series
- **Rolling mean** — 5, 10, and 20-day windows — captures trend
- **Rolling std** — 5, 10, and 20-day windows — captures volatility
- **Rolling cumulative return** — sum of log returns over rolling windows
- **High-low spread** — `(High - Low) / Close` — intraday price range
- **Volume change** — log-differenced volume — momentum signal

The MinMaxScaler is fitted on the training set only and applied to validation and test sets — no data leakage at any point.

---

## AWS Deployment (for reproducing the cloud setup)

```bash
# Create ECR repository (one time)
bash scripts/create_ecr.sh

# Build Docker image and deploy Lambda function
bash scripts/deploy_lambda.sh

# Set up daily EventBridge schedule
bash scripts/create_eventbridge.sh

# Test Lambda invocation directly
bash scripts/test_lambda.sh
```

### Tearing down all AWS resources

```bash
bash scripts/teardown.sh
```

This removes the Lambda function, EventBridge schedule, ECR repository, empties S3, and deletes the IAM role. Everything is gone and billing stops immediately.

---

## Running Costs

| Service | Usage | Free Tier |
|---|---|---|
| GitHub Actions | ~45 min/day retrain | 2,000 min/month ✅ |
| AWS Lambda | 1–2 invocations/day | 1M invocations/month ✅ |
| AWS S3 | ~50 MB predictions + models | 5 GB storage ✅ |
| AWS ECR | ~3 GB Docker image | ~$0.30/month |
| AWS EventBridge | 1 scheduled rule | 5M events/month ✅ |
| Streamlit Cloud | Public app | Free ✅ |

**Total monthly cost: ~$0.30** (ECR image storage only).

---

## Test Coverage

```
tests/
├── test_ingestion.py       13 tests — fetching, caching, retry logic
├── test_preprocessing.py   20 tests — cleaning, features, splits, no leakage
├── test_models.py          39 tests — ARIMA, Prophet, LSTM fit/predict/save
├── test_training.py        23 tests — orchestration, fault isolation
├── test_predictor.py       36 tests — CI bands, inverse transform, CSV I/O
├── test_pipeline.py        36 tests — pipeline modes, Lambda handler
├── test_s3_client.py       35 tests — upload, download, graceful degradation
└── test_dashboard.py       33 tests — charts, metrics table, data loading
```

235 total tests. All external services (S3, yfinance, Lambda) are mocked — the full suite runs offline in under 60 seconds.

---


## 📋 Phases

- [x] Phase 1 — Project scaffold & config
- [x] Phase 2 — Data layer (ingestion & preprocessing)
- [x] Phase 3 — Model layer (ARIMA, Prophet, LSTM)
- [x] Phase 4 — Training & benchmarking
- [x] Phase 5 — Forecasting with confidence intervals
- [x] Phase 6 — Pipeline orchestration
- [x] Phase 7 — S3 storage
- [x] Phase 8 — Streamlit dashboard
- [x] Phase 9 — GitHub Actions CI/CD
- [x] Phase 10 — AWS Lambda deployment
- [x] Phase 11 — Streamlit Cloud deployment


---


## Collaboration☑
This project was developed in collaboration with [Harshith Bhattaram](https://github.com/maniharshith68).

## Acknowledgements
This project was built in collaboration with [Harshith Bhattaram](https://github.com/maniharshith68).

## 👤 Authors
- [Harshith Bhattaram](https://github.com/maniharshith68)
- [Shruti Kumari](https://github.com/shrutisurya108)






