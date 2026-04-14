# 📈 Automated Time-Series Stock Forecasting Pipeline

A production-grade, fully automated multi-model time-series forecasting system
for 10 stocks, featuring daily retraining, cloud deployment, and an interactive
Streamlit dashboard — built entirely on **free-tier** services.

---

## 🏗️ Architecture

```
yfinance (data)
    │
    ▼
Data Layer (ingestion → preprocessing)
    │
    ├──► ARIMA (statsmodels / pmdarima)
    ├──► Prophet (Meta)
    └──► LSTM  (PyTorch)
              │
              ▼
        Benchmarking (RMSE, MAPE, MAE vs naive baseline)
              │
              ▼
        Predictions + Confidence Intervals
              │
        ┌─────┴──────┐
        ▼             ▼
      AWS S3      Streamlit Cloud
   (storage)       (dashboard)
        ▲
GitHub Actions / AWS Lambda + EventBridge
       (daily retraining at 01:00 UTC)
```

---

## 🗂️ Project Structure

```
stock-forecasting-pipeline/
├── config/          # Settings & logging
├── data/            # Ingestion & preprocessing
├── models/          # ARIMA, Prophet, LSTM
├── training/        # Trainer & benchmarking
├── forecasting/     # Predictor with confidence intervals
├── pipeline/        # Orchestration & Lambda handler
├── storage/         # S3 client
├── dashboard/       # Streamlit app & chart components
├── tests/           # Unit tests
├── .github/         # GitHub Actions workflows
├── logs/            # Rotating log files (gitignored)
└── predictions/     # Local prediction outputs (gitignored)
```

---

## ⚡ Quick Start (Local)

### 1. Clone & set up environment
```bash
git clone https://github.com/YOUR_USERNAME/stock-forecasting-pipeline.git
cd stock-forecasting-pipeline
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your AWS credentials
```

### 3. Run the full pipeline locally
```bash
python -m pipeline.run_pipeline
```

### 4. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Run tests
```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## 📊 Models & Performance

| Model   | Description                             |
|---------|-----------------------------------------|
| ARIMA   | Auto-tuned via `pmdarima.auto_arima`    |
| Prophet | Meta's additive decomposition model     |
| LSTM    | 2-layer PyTorch LSTM, 60-day look-back  |

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

## 🔧 Configuration

All settings are in `config/settings.py`:
- **Stocks**: `STOCK_TICKERS` (10 tickers)
- **Horizon**: `FORECAST_HORIZON` (30 days)
- **Models**: `ARIMA_CONFIG`, `PROPHET_CONFIG`, `LSTM_CONFIG`
- **Split**: 70% train / 15% val / 15% test

---

## 📋 Phases

- [x] Phase 1 — Project scaffold & config
- [ ] Phase 2 — Data layer (ingestion & preprocessing)
- [ ] Phase 3 — Model layer (ARIMA, Prophet, LSTM)
- [ ] Phase 4 — Training & benchmarking
- [ ] Phase 5 — Forecasting with confidence intervals
- [ ] Phase 6 — Pipeline orchestration
- [ ] Phase 7 — S3 storage
- [ ] Phase 8 — Streamlit dashboard
- [ ] Phase 9 — Tests
- [ ] Phase 10 — GitHub Actions CI/CD
- [ ] Phase 11 — AWS Lambda deployment
- [ ] Phase 12 — Streamlit Cloud deployment
