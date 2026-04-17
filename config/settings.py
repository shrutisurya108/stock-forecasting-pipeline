"""
config/settings.py
==================
Central configuration hub for the entire stock forecasting pipeline.
All hyperparameters, stock tickers, paths, and feature flags live here.
Import this module anywhere in the project — never hardcode values elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file (if present) ───────────────────────────────────────────────
load_dotenv()

# ── Root Paths ────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent

# ── Lambda detection ──────────────────────────────────────────────────────────
# Lambda containers have a read-only filesystem except for /tmp (512 MB).
# When running in Lambda, redirect all write paths to /tmp so directory
# creation and file writes succeed. Local runs use the project root as normal.
import os as _os
_IN_LAMBDA = bool(_os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
_WRITE_ROOT = Path("/tmp") if _IN_LAMBDA else ROOT_DIR

DATA_DIR         = _WRITE_ROOT / "data" / "raw"
PREDICTIONS_DIR  = _WRITE_ROOT / "predictions"
LOGS_DIR         = _WRITE_ROOT / "logs"
MODELS_DIR       = ROOT_DIR / "models"   # source code dir, read-only is fine

# Saved model artifacts live inside predictions/models/{ticker}/
SAVED_MODELS_DIR = PREDICTIONS_DIR / "models"

# Ensure directories always exist at import time
for _dir in [DATA_DIR, PREDICTIONS_DIR, LOGS_DIR, SAVED_MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ── Stock Universe ─────────────────────────────────────────────────────────────
STOCK_TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "TSLA",   # Tesla
    "META",   # Meta
    "NVDA",   # NVIDIA
    "JPM",    # JPMorgan Chase
    "V",      # Visa
    "JNJ",    # Johnson & Johnson
]

# Target column to forecast
TARGET_COL = "Close"

# ── Data Settings ──────────────────────────────────────────────────────────────
# How many calendar years of historical data to fetch
DATA_YEARS          = 5
# yfinance interval: "1d" = daily OHLCV bars
DATA_INTERVAL       = "1d"
# Train / validation / test split ratios (must sum to 1.0)
TRAIN_RATIO         = 0.70
VAL_RATIO           = 0.15
TEST_RATIO          = 0.15

# ── Forecasting Horizon ────────────────────────────────────────────────────────
FORECAST_HORIZON    = 30   # days ahead to forecast
CONFIDENCE_INTERVAL = 0.95  # 95 % CI for prediction bands

# ── ARIMA Hyperparameters ──────────────────────────────────────────────────────
ARIMA_CONFIG = {
    "p": 5,          # autoregressive order
    "d": 1,          # differencing order
    "q": 0,          # moving average order
    "seasonal": False,
    "auto_order": True,   # if True, use auto_arima to find best (p,d,q)
    "max_p": 5,
    "max_q": 5,
    "information_criterion": "aic",
}

# ── Prophet Hyperparameters ────────────────────────────────────────────────────
PROPHET_CONFIG = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "seasonality_mode": "multiplicative",   # or "additive"
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "interval_width": CONFIDENCE_INTERVAL,
}

# ── LSTM Hyperparameters ───────────────────────────────────────────────────────
LSTM_CONFIG = {
    "sequence_length": 60,      # look-back window (trading days)
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-3,
    "early_stopping_patience": 10,
    "clip_grad_norm": 1.0,
    "train_on_gpu": True,       # falls back to CPU if unavailable
}

# ── Benchmarking ───────────────────────────────────────────────────────────────
METRICS = ["RMSE", "MAPE", "MAE"]
NAIVE_STRATEGY = "last_value"   # baseline: predict yesterday's close

# ── AWS / S3 Settings (populated from .env) ────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME        = os.getenv("S3_BUCKET_NAME", "stock-forecasting-pipeline")
S3_PREDICTIONS_PREFIX = "predictions/"
S3_MODELS_PREFIX      = "models/"

# ── Pipeline Settings ──────────────────────────────────────────────────────────
PIPELINE_RUN_HOUR_UTC = 1      # run daily at 01:00 UTC (after market close)
RETRAIN_FREQUENCY     = "daily" # "daily" | "weekly"

# ── Dashboard Settings ─────────────────────────────────────────────────────────
DASHBOARD_TITLE       = "📈 Stock Forecasting Pipeline"
DASHBOARD_THEME       = "dark"
DEFAULT_TICKER        = "AAPL"
CHART_HEIGHT          = 500

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL             = os.getenv("LOG_LEVEL", "INFO")   # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT            = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT       = "%Y-%m-%d %H:%M:%S"
LOG_ROTATION_BYTES    = 10 * 1024 * 1024   # 10 MB per log file
LOG_BACKUP_COUNT      = 5                  # keep 5 rotated files

# ── Model Registry ─────────────────────────────────────────────────────────────
# Canonical names used throughout the project
MODEL_NAMES = ["arima", "prophet", "lstm"]

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
