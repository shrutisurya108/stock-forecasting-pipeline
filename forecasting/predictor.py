"""
forecasting/predictor.py
========================
Generates forward-looking predictions from already-fitted models.

Responsibilities
----------------
1. Load fitted models from disk (no retraining).
2. Call model.predict_with_ci(n_steps) → scaled predictions.
3. Inverse-transform forecasts back to real dollar prices.
4. Attach the correct future business-day dates.
5. Combine all three models into one DataFrame per ticker.
6. Save per-ticker CSVs + one combined CSV the dashboard reads.

Key design
----------
- Works from disk — does NOT re-fit any model.
- Inverse-transform uses the scaler from ProcessedData, which must be
  provided alongside the fitted models (it was fitted on training data).
- All three model forecasts share the same date index so the dashboard
  can overlay them on one chart effortlessly.
- PredictionResult is a typed dataclass — easy to inspect and log.

Output files
------------
predictions/{TICKER}_forecast.csv   ← one per ticker, 30 rows
predictions/all_forecasts.csv       ← wide table, all tickers + models

Usage (standalone):
    python -m forecasting.predictor
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.logging_config import get_logger
from config.settings import (
    CONFIDENCE_INTERVAL,
    FORECAST_HORIZON,
    MODEL_NAMES,
    PREDICTIONS_DIR,
    STOCK_TICKERS,
    TARGET_COL,
    LSTM_CONFIG,
)
from data.preprocessing import ProcessedData
from models.base_model import BaseModel
from training.trainer import load_model

logger = get_logger(__name__)


# ── Output paths ──────────────────────────────────────────────────────────────

def _ticker_forecast_path(ticker: str) -> Path:
    """Return path for a ticker's individual forecast CSV."""
    return PREDICTIONS_DIR / f"{ticker}_forecast.csv"


ALL_FORECASTS_PATH = PREDICTIONS_DIR / "all_forecasts.csv"


# ── PredictionResult container ────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Holds the complete 30-day forward forecast for one ticker.

    Attributes:
        ticker:       Stock symbol.
        generated_at: UTC timestamp when predictions were generated.
        horizon:      Number of forecast steps.
        forecast_df:  DataFrame indexed by future business dates with columns:
                        {model}_forecast, {model}_lower, {model}_upper
                      Values are in real dollar prices (inverse-transformed).
        models_used:  List of model names that contributed forecasts.
        last_known_price: The last actual Close price in the training window
                          (used as chart anchor by the dashboard).
        last_known_date:  Date of that last known price.
    """
    ticker:            str
    generated_at:      str
    horizon:           int
    forecast_df:       pd.DataFrame
    models_used:       list[str]  = field(default_factory=list)
    last_known_price:  float      = 0.0
    last_known_date:   str        = ""

    def summary(self) -> str:
        models = ", ".join(self.models_used)
        return (
            f"PredictionResult({self.ticker}) | "
            f"horizon={self.horizon}d | "
            f"models=[{models}] | "
            f"from={self.forecast_df.index[0].date()} "
            f"to={self.forecast_df.index[-1].date()}"
        )


# ── Date helpers ──────────────────────────────────────────────────────────────

def _future_business_dates(
    last_date: pd.Timestamp,
    n_steps:   int,
) -> pd.DatetimeIndex:
    """
    Generate n_steps business days starting the day AFTER last_date.

    Args:
        last_date: The last date in the training/validation data.
        n_steps:   Number of future trading days to generate.

    Returns:
        DatetimeIndex of n_steps business days (Mon–Fri).
    """
    start = last_date + pd.offsets.BDay(1)
    return pd.date_range(start=start, periods=n_steps, freq="B")


# ── Inverse-transform helpers ─────────────────────────────────────────────────

def _make_inverter(
    scaler:       MinMaxScaler,
    feature_cols: list[str],
    target_col:   str = TARGET_COL,
):
    """
    Return a callable that inverse-transforms a 1-D scaled array
    back to real price space using the fitted MinMaxScaler.

    The scaler was fitted on feature_cols during preprocessing.
    We must embed the value in the correct column position of a
    dummy full-width array before calling inverse_transform.

    Args:
        scaler:       MinMaxScaler fitted on the training set.
        feature_cols: Ordered list of columns the scaler was fitted on.
        target_col:   The column to invert (default "Close").

    Returns:
        A function  inverse(arr: np.ndarray) -> np.ndarray
        that maps scaled values → real prices.
    """
    n_cols     = len(scaler.scale_)   # authoritative width from the scaler
    target_idx = feature_cols.index(target_col)

    def _inverse(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).ravel()
        dummy = np.zeros((len(arr), n_cols))
        dummy[:, target_idx] = arr
        return scaler.inverse_transform(dummy)[:, target_idx]

    return _inverse


# ── Per-model forecasting ─────────────────────────────────────────────────────

def _forecast_one_model(
    model:      BaseModel,
    inverter,            # callable from _make_inverter
    n_steps:    int,
    confidence: float,
    ticker:     str,
    model_name: str,
) -> Optional[dict[str, np.ndarray]]:
    """
    Run predict_with_ci on one model and inverse-transform the output.

    Returns a dict with keys:
        f"{model_name}_forecast", f"{model_name}_lower", f"{model_name}_upper"
    all in real price space, or None if the model fails.
    """
    try:
        raw = model.predict_with_ci(n_steps=n_steps, confidence=confidence)

        fc    = inverter(raw["forecast"])
        lower = inverter(raw["lower"])
        upper = inverter(raw["upper"])

        # Sanity: lower ≤ forecast ≤ upper  (clamp if inversion flips order)
        lower = np.minimum(lower, fc)
        upper = np.maximum(upper, fc)

        logger.info(
            "[%s | %s] Forecast range: $%.2f – $%.2f (step-1 CI: $%.2f–$%.2f)",
            ticker, model_name,
            fc.min(), fc.max(), lower[0], upper[0],
        )
        return {
            f"{model_name}_forecast": fc,
            f"{model_name}_lower":    lower,
            f"{model_name}_upper":    upper,
        }

    except Exception as exc:
        logger.error(
            "[%s | %s] Forecast failed: %s", ticker, model_name, exc, exc_info=True
        )
        return None


# ── Core public function ──────────────────────────────────────────────────────

def generate_predictions(
    ticker:         str,
    processed:      ProcessedData,
    fitted_models:  dict[str, BaseModel] | None = None,
    model_names:    list[str] | None            = None,
    horizon:        int                          = FORECAST_HORIZON,
    confidence:     float                        = CONFIDENCE_INTERVAL,
) -> Optional[PredictionResult]:
    """
    Generate forward predictions for one ticker using all available models.

    Args:
        ticker:        Stock symbol.
        processed:     ProcessedData for this ticker (provides scaler + dates).
        fitted_models: Pre-loaded {model_name: BaseModel} dict.
                       If None, models are loaded from disk automatically.
        model_names:   Subset of models to use. Defaults to all three.
        horizon:       Number of business days to forecast. Default 30.
        confidence:    CI confidence level. Default 0.95.

    Returns:
        PredictionResult with forecast_df in real dollar prices,
        or None if every model failed.
    """
    model_names = model_names or MODEL_NAMES
    now_utc     = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(
        "[%s] Generating %d-day forecast (confidence=%.0f%%)",
        ticker, horizon, confidence * 100,
    )

    # ── Load models from disk if not provided ─────────────────────────────────
    if fitted_models is None:
        fitted_models = {}
        for name in model_names:
            m = load_model(ticker, name)
            if m is not None:
                fitted_models[name] = m
            else:
                logger.warning(
                    "[%s | %s] Could not load from disk — skipping", ticker, name
                )

    if not fitted_models:
        logger.error("[%s] No fitted models available — cannot forecast", ticker)
        return None

    # ── Build the inverse-transform callable ──────────────────────────────────
    inverter = _make_inverter(processed.scaler, processed.feature_cols)

    # ── Determine the "last known" anchor point ───────────────────────────────
    # Use the last date from the combined train+val data (what models were
    # trained on) as the anchor.  raw_test gives us unscaled Close prices.
    last_date_train = processed.val.index.max()
    # Last known Close price — use raw_test's first value minus one step,
    # or the last value of the raw training close if available.
    raw_close_all = processed.raw_test["Close"]
    # Fallback: inverse-transform the last val Close
    last_scaled_val = float(processed.val[TARGET_COL].iloc[-1])
    last_price_arr  = inverter(np.array([last_scaled_val]))
    last_known_price = float(last_price_arr[0])
    last_known_date  = str(last_date_train.date())

    # ── Future date index ─────────────────────────────────────────────────────
    future_dates = _future_business_dates(last_date_train, horizon)

    # ── Run each model ────────────────────────────────────────────────────────
    all_cols:      dict[str, np.ndarray] = {}
    models_used:   list[str]             = []

    for model_name in model_names:
        model = fitted_models.get(model_name)
        if model is None:
            logger.warning(
                "[%s | %s] Not in fitted_models — skipping", ticker, model_name
            )
            continue

        result = _forecast_one_model(
            model=model,
            inverter=inverter,
            n_steps=horizon,
            confidence=confidence,
            ticker=ticker,
            model_name=model_name,
        )
        if result is not None:
            all_cols.update(result)
            models_used.append(model_name)

    if not all_cols:
        logger.error("[%s] All model forecasts failed", ticker)
        return None

    # ── Assemble forecast DataFrame ───────────────────────────────────────────
    forecast_df = pd.DataFrame(all_cols, index=future_dates)
    forecast_df.index.name = "date"

    # Round to 2 decimal places (cents)
    forecast_df = forecast_df.round(2)

    prediction = PredictionResult(
        ticker=ticker,
        generated_at=now_utc,
        horizon=horizon,
        forecast_df=forecast_df,
        models_used=models_used,
        last_known_price=round(last_known_price, 2),
        last_known_date=last_known_date,
    )

    logger.info("[%s] %s", ticker, prediction.summary())
    return prediction


def generate_all_predictions(
    processed_data: dict[str, ProcessedData],
    fitted_models:  dict[str, dict[str, BaseModel]] | None = None,
    model_names:    list[str] | None                       = None,
    horizon:        int                                    = FORECAST_HORIZON,
    confidence:     float                                  = CONFIDENCE_INTERVAL,
    save:           bool                                   = True,
) -> dict[str, PredictionResult]:
    """
    Generate forward predictions for all tickers and optionally save to disk.

    Args:
        processed_data: {ticker: ProcessedData} from preprocessing.
        fitted_models:  {ticker: {model_name: BaseModel}} from trainer.
                        If None, models are auto-loaded from disk.
        model_names:    Subset of models to use.
        horizon:        Days to forecast.
        confidence:     CI confidence level.
        save:           Whether to write CSVs.

    Returns:
        {ticker: PredictionResult} for every ticker that succeeded.
    """
    results:  dict[str, PredictionResult] = {}
    tickers = list(processed_data.keys())

    logger.info(
        "Generating predictions for %d tickers, horizon=%d days",
        len(tickers), horizon,
    )

    for ticker in tickers:
        ticker_models = (
            (fitted_models or {}).get(ticker)   # may be None → auto-load
        )
        result = generate_predictions(
            ticker=ticker,
            processed=processed_data[ticker],
            fitted_models=ticker_models,
            model_names=model_names,
            horizon=horizon,
            confidence=confidence,
        )
        if result is not None:
            results[ticker] = result
        else:
            logger.warning("[%s] No prediction generated", ticker)

    logger.info(
        "Predictions complete: %d/%d tickers succeeded",
        len(results), len(tickers),
    )

    if save and results:
        save_all_predictions(results)

    return results


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_prediction(result: PredictionResult) -> Path:
    """
    Save one ticker's forecast DataFrame to CSV.

    Format: index=date, columns={model}_forecast/lower/upper.
    Includes metadata columns (ticker, generated_at, last_known_price).

    Returns the path written.
    """
    path = _ticker_forecast_path(result.ticker)

    df = result.forecast_df.copy()
    df.insert(0, "ticker",            result.ticker)
    df.insert(1, "generated_at",      result.generated_at)
    df.insert(2, "last_known_price",  result.last_known_price)
    df.insert(3, "last_known_date",   result.last_known_date)

    df.to_csv(path, index=True)
    logger.info("[%s] Forecast saved → %s (%d rows)", result.ticker, path, len(df))
    return path


def save_all_predictions(results: dict[str, PredictionResult]) -> Path:
    """
    Save individual per-ticker CSVs AND a combined all_forecasts.csv.

    Returns path to the combined file.
    """
    frames = []
    for ticker, result in results.items():
        # Per-ticker file
        save_prediction(result)

        # Accumulate for combined file
        df = result.forecast_df.copy()
        df.insert(0, "ticker",           ticker)
        df.insert(1, "last_known_price", result.last_known_price)
        df.insert(2, "last_known_date",  result.last_known_date)
        df.insert(3, "generated_at",     result.generated_at)
        frames.append(df)

    combined = pd.concat(frames).reset_index()
    combined.to_csv(ALL_FORECASTS_PATH, index=False)
    logger.info(
        "Combined forecast saved → %s (%d rows, %d tickers)",
        ALL_FORECASTS_PATH, len(combined), len(results),
    )
    return ALL_FORECASTS_PATH


def load_prediction(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load a saved forecast CSV for one ticker.

    Returns None if file does not exist.
    """
    path = _ticker_forecast_path(ticker)
    if not path.exists():
        logger.warning("[%s] No saved forecast at %s", ticker, path)
        return None
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    logger.debug("[%s] Loaded forecast ← %s (%d rows)", ticker, path, len(df))
    return df


def load_all_forecasts() -> Optional[pd.DataFrame]:
    """Load the combined all_forecasts.csv. Returns None if missing."""
    if not ALL_FORECASTS_PATH.exists():
        logger.warning("No combined forecast file at %s", ALL_FORECASTS_PATH)
        return None
    return pd.read_csv(ALL_FORECASTS_PATH, parse_dates=["date"])


# ── forecasting/__init__.py update helper  ────────────────────────────────────
# (called from __init__ — keeps imports clean for dashboard)

def get_available_tickers() -> list[str]:
    """Return tickers that have a saved forecast CSV on disk."""
    return [
        t for t in STOCK_TICKERS
        if _ticker_forecast_path(t).exists()
    ]


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from data.ingestion import fetch_all_tickers
    from data.preprocessing import preprocess_all
    from training.trainer import train_all

    print("\n" + "═" * 65)
    print("  forecasting/predictor.py — smoke test")
    print("═" * 65 + "\n")

    SMOKE_TICKERS = ["AAPL", "MSFT"]

    # Reduce LSTM epochs for speed
    orig_epochs   = LSTM_CONFIG["epochs"]
    orig_patience = LSTM_CONFIG["early_stopping_patience"]
    LSTM_CONFIG["epochs"]                  = 5
    LSTM_CONFIG["early_stopping_patience"] = 3

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("Step 1: Fetch + preprocess…")
    raw       = fetch_all_tickers(tickers=SMOKE_TICKERS)
    processed = preprocess_all(raw)
    print(f"        → {len(processed)} tickers ready\n")

    # ── 2. Train (or reuse from Phase 4 if already on disk) ───────────────────
    print("Step 2: Train models…")
    fitted, _ = train_all(processed, overwrite=False)   # reuse if exists
    n_fitted  = sum(len(v) for v in fitted.values())
    print(f"        → {n_fitted} models ready\n")

    # Restore LSTM config
    LSTM_CONFIG["epochs"]                  = orig_epochs
    LSTM_CONFIG["early_stopping_patience"] = orig_patience

    # ── 3. Generate predictions ───────────────────────────────────────────────
    print("Step 3: Generating 30-day forecasts…")
    predictions = generate_all_predictions(
        processed_data=processed,
        fitted_models=fitted,
        horizon=FORECAST_HORIZON,
        confidence=CONFIDENCE_INTERVAL,
        save=True,
    )
    print(f"        → {len(predictions)} tickers predicted\n")

    # ── 4. Print results table ────────────────────────────────────────────────
    print("── Forecast Preview (first 5 days) ─────────────────────────────────")
    for ticker, result in predictions.items():
        print(f"\n  {ticker} | last known: ${result.last_known_price:.2f} "
              f"on {result.last_known_date}")
        print(f"  Models: {result.models_used}")
        print(f"  {result.forecast_df.head(5).to_string()}")

    # ── 5. Check saved files ──────────────────────────────────────────────────
    print("\n── Saved Files ──────────────────────────────────────────────────────")
    all_ok = True
    for ticker in SMOKE_TICKERS:
        p = _ticker_forecast_path(ticker)
        if p.exists():
            df = pd.read_csv(p)
            print(f"  ✅ {p.name} ({len(df)} rows, {len(df.columns)} cols)")
        else:
            print(f"  ❌ {p.name} MISSING")
            all_ok = False

    if ALL_FORECASTS_PATH.exists():
        combined = pd.read_csv(ALL_FORECASTS_PATH)
        print(f"  ✅ {ALL_FORECASTS_PATH.name} "
              f"({len(combined)} rows, {combined['ticker'].nunique()} tickers)")
    else:
        print(f"  ❌ all_forecasts.csv MISSING")
        all_ok = False

    # ── 6. Verify load round-trip ─────────────────────────────────────────────
    print("\n── Load Round-trip ──────────────────────────────────────────────────")
    loaded = load_prediction("AAPL")
    if loaded is not None:
        print(f"  ✅ load_prediction('AAPL') → {len(loaded)} rows")
    else:
        print("  ❌ load_prediction failed")
        all_ok = False

    combined_loaded = load_all_forecasts()
    if combined_loaded is not None:
        print(f"  ✅ load_all_forecasts() → {len(combined_loaded)} rows, "
              f"tickers={list(combined_loaded['ticker'].unique())}")
    else:
        print("  ❌ load_all_forecasts failed")
        all_ok = False

    # ── 7. Verify CI ordering ─────────────────────────────────────────────────
    print("\n── CI Ordering Check (lower ≤ forecast ≤ upper) ────────────────────")
    for ticker, result in predictions.items():
        df = result.forecast_df
        for model_name in result.models_used:
            fc    = df[f"{model_name}_forecast"].values
            lower = df[f"{model_name}_lower"].values
            upper = df[f"{model_name}_upper"].values
            ok    = np.all(lower <= fc + 0.01) and np.all(fc <= upper + 0.01)
            mark  = "✅" if ok else "❌"
            print(f"  {mark} {ticker} | {model_name}: "
                  f"lower≤fc≤upper across all {len(fc)} steps")

    print()
    if all_ok and len(predictions) == len(SMOKE_TICKERS):
        print(f"✅ predictor.py smoke test PASSED — "
              f"{len(predictions)} tickers, {FORECAST_HORIZON}-day forecasts\n")
    else:
        print("⚠️  predictor.py smoke test had issues — check logs above\n")
