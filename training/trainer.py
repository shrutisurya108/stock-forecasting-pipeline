"""
training/trainer.py
===================
Orchestrates the full training run: fits all three models (ARIMA, Prophet,
LSTM) across every ticker, with per-(ticker, model) fault isolation, timing,
and model persistence to disk.

Key behaviours
--------------
- Each (ticker, model) pair is wrapped in its own try/except so a single
  failure never aborts the full run.
- Fitted models are saved to  predictions/models/{ticker}/{model}/
  so the pipeline can load them without retraining on every run.
- Returns a nested dict  {ticker: {model_name: BaseModel}}  consumed
  by benchmarking.py and forecasting/predictor.py.
- A TrainingResult dataclass captures success/failure metadata for every
  (ticker, model) pair — written to predictions/training_report.json.

Usage (standalone smoke-test):
    python -m training.trainer
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from config.logging_config import get_logger
from config.settings import (
    MODEL_NAMES,
    PREDICTIONS_DIR,
    SAVED_MODELS_DIR,
    STOCK_TICKERS,
    LSTM_CONFIG,
)
from data.ingestion import fetch_all_tickers
from data.preprocessing import ProcessedData, preprocess_all
from models.arima_model import ARIMAModel
from models.base_model import BaseModel
from models.lstm_model import LSTMModel
from models.prophet_model import ProphetModel

logger = get_logger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────
# {ticker: {model_name: fitted_model}}
FittedModels = dict[str, dict[str, BaseModel]]


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """Captures outcome metadata for one (ticker, model) training run."""
    ticker:      str
    model_name:  str
    success:     bool
    duration_s:  float       = 0.0
    error:       str         = ""
    saved_path:  str         = ""


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_model(model_name: str, ticker: str) -> BaseModel:
    """
    Instantiate a fresh (unfitted) model by name.

    Args:
        model_name: One of "arima", "prophet", "lstm".
        ticker:     Stock symbol the model is for.

    Returns:
        A new unfitted model instance.
    """
    if model_name == "arima":
        return ARIMAModel(ticker)
    elif model_name == "prophet":
        return ProphetModel(ticker)
    elif model_name == "lstm":
        return LSTMModel(ticker)
    else:
        raise ValueError(f"Unknown model name: '{model_name}'. "
                         f"Valid names: {MODEL_NAMES}")


# ── Save / Load helpers ───────────────────────────────────────────────────────

def model_save_path(ticker: str, model_name: str) -> Path:
    """Return the directory where a (ticker, model) pair is persisted."""
    return SAVED_MODELS_DIR / ticker / model_name


def save_model(model: BaseModel, ticker: str, model_name: str) -> Path:
    """Persist a fitted model to disk. Returns the save directory."""
    path = model_save_path(ticker, model_name)
    path.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def load_model(ticker: str, model_name: str) -> Optional[BaseModel]:
    """
    Load a previously saved model from disk.

    Returns None if the save directory or files don't exist.
    """
    path = model_save_path(ticker, model_name)
    if not path.exists():
        logger.warning(
            "No saved model found for [%s | %s] at %s",
            ticker, model_name, path,
        )
        return None
    try:
        model = _make_model(model_name, ticker)
        model.load(path)
        logger.info("Loaded [%s | %s] from %s", ticker, model_name, path)
        return model
    except Exception as exc:
        logger.error(
            "Failed to load [%s | %s]: %s", ticker, model_name, exc
        )
        return None


# ── Core training logic ───────────────────────────────────────────────────────

def _train_single(
    ticker:     str,
    model_name: str,
    processed:  ProcessedData,
    overwrite:  bool = True,
) -> tuple[Optional[BaseModel], TrainingResult]:
    """
    Fit one model on one ticker's preprocessed data.

    Args:
        ticker:     Stock symbol.
        model_name: "arima", "prophet", or "lstm".
        processed:  ProcessedData object from preprocessing.
        overwrite:  If False and a saved model exists, load it instead of
                    retraining (useful for incremental daily runs).

    Returns:
        (fitted_model | None, TrainingResult)
    """
    save_path = model_save_path(ticker, model_name)

    # ── Fast path: load existing model ────────────────────────────────────────
    if not overwrite and save_path.exists():
        model = load_model(ticker, model_name)
        if model is not None:
            return model, TrainingResult(
                ticker=ticker,
                model_name=model_name,
                success=True,
                duration_s=0.0,
                saved_path=str(save_path),
            )

    # ── Full training path ────────────────────────────────────────────────────
    logger.info(
        "━━ Training [%s | %s] — train=%d | val=%d",
        ticker, model_name, len(processed.train), len(processed.val),
    )
    t0 = time.time()

    try:
        model = _make_model(model_name, ticker)
        model.fit(processed.train, processed.val)
        duration = time.time() - t0

        # Persist to disk
        saved = save_model(model, ticker, model_name)

        result = TrainingResult(
            ticker=ticker,
            model_name=model_name,
            success=True,
            duration_s=round(duration, 2),
            saved_path=str(saved),
        )
        logger.info(
            "✓ [%s | %s] trained in %.1fs — saved → %s",
            ticker, model_name, duration, saved,
        )
        return model, result

    except Exception as exc:
        duration = time.time() - t0
        logger.error(
            "✗ [%s | %s] FAILED after %.1fs: %s",
            ticker, model_name, duration, exc, exc_info=True,
        )
        return None, TrainingResult(
            ticker=ticker,
            model_name=model_name,
            success=False,
            duration_s=round(duration, 2),
            error=str(exc),
        )


# ── Public API ────────────────────────────────────────────────────────────────

def train_all(
    processed_data: dict[str, ProcessedData],
    model_names:    list[str] | None = None,
    overwrite:      bool = True,
) -> tuple[FittedModels, list[TrainingResult]]:
    """
    Fit all models across all tickers.

    Args:
        processed_data: Output of preprocessing.preprocess_all().
        model_names:    Subset of models to train. Defaults to all three.
        overwrite:      Retrain even if a saved model already exists.

    Returns:
        (fitted_models, training_results)
        fitted_models   → {ticker: {model_name: BaseModel}}
        training_results → list of TrainingResult (one per ticker×model)
    """
    model_names = model_names or MODEL_NAMES
    tickers     = list(processed_data.keys())

    total = len(tickers) * len(model_names)
    logger.info(
        "Starting training run: %d tickers × %d models = %d total fits",
        len(tickers), len(model_names), total,
    )

    fitted:  FittedModels        = {}
    results: list[TrainingResult] = []
    run_t0 = time.time()
    done   = 0

    for ticker in tickers:
        processed = processed_data[ticker]
        fitted[ticker] = {}

        for model_name in model_names:
            done += 1
            logger.info(
                "── [%d/%d] %s | %s", done, total, ticker, model_name
            )
            model, result = _train_single(
                ticker, model_name, processed, overwrite=overwrite
            )
            results.append(result)
            if model is not None:
                fitted[ticker][model_name] = model

        # Log per-ticker summary
        n_ok = sum(1 for r in results
                   if r.ticker == ticker and r.success)
        logger.info(
            "── Ticker %s complete: %d/%d models OK",
            ticker, n_ok, len(model_names),
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - run_t0
    n_success  = sum(1 for r in results if r.success)
    n_failed   = total - n_success

    logger.info(
        "Training complete — %d/%d succeeded, %d failed, total_time=%.1fs",
        n_success, total, n_failed, total_time,
    )

    if n_failed > 0:
        failed = [(r.ticker, r.model_name) for r in results if not r.success]
        logger.warning("Failed pairs: %s", failed)

    # Write training report
    _save_training_report(results)

    return fitted, results


def _save_training_report(results: list[TrainingResult]) -> None:
    """Write training outcomes to predictions/training_report.json."""
    report_path = PREDICTIONS_DIR / "training_report.json"
    report = {
        "total":   len(results),
        "success": sum(1 for r in results if r.success),
        "failed":  sum(1 for r in results if not r.success),
        "results": [asdict(r) for r in results],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Training report saved → %s", report_path)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    print("\n" + "═" * 65)
    print("  training/trainer.py — smoke test (2 tickers, fast LSTM)")
    print("═" * 65 + "\n")

    # Use only 2 tickers and 5 LSTM epochs for a fast smoke test
    SMOKE_TICKERS = ["AAPL", "MSFT"]

    # Temporarily reduce LSTM epochs so the smoke test stays fast
    orig_epochs   = LSTM_CONFIG["epochs"]
    orig_patience = LSTM_CONFIG["early_stopping_patience"]
    LSTM_CONFIG["epochs"]                  = 5
    LSTM_CONFIG["early_stopping_patience"] = 3

    print("Step 1: Fetching data…")
    raw = fetch_all_tickers(tickers=SMOKE_TICKERS)
    print(f"        → {len(raw)} tickers fetched\n")

    print("Step 2: Preprocessing…")
    processed = preprocess_all(raw)
    print(f"        → {len(processed)} tickers preprocessed\n")

    print("Step 3: Training all 3 models on each ticker…")
    fitted, results = train_all(processed, overwrite=True)

    # Restore LSTM config
    LSTM_CONFIG["epochs"]                  = orig_epochs
    LSTM_CONFIG["early_stopping_patience"] = orig_patience

    # Results table
    print("\n── Training Results ─────────────────────────────────────────────")
    print(f"{'Ticker':<7} {'Model':<10} {'Status':<8} {'Time(s)':>8}  {'Error'}")
    print("─" * 60)
    for r in results:
        status = "✅ OK" if r.success else "❌ FAIL"
        err    = r.error[:35] if r.error else ""
        print(f"{r.ticker:<7} {r.model_name:<10} {status:<8} {r.duration_s:>8.1f}  {err}")

    n_ok = sum(1 for r in results if r.success)
    print(f"\n{n_ok}/{len(results)} training runs succeeded")

    # Check saved model dirs exist
    print("\n── Saved Model Dirs ─────────────────────────────────────────────")
    for ticker in SMOKE_TICKERS:
        for model_name in MODEL_NAMES:
            p = model_save_path(ticker, model_name)
            exists = "✅" if p.exists() else "❌"
            print(f"  {exists}  {p}")

    # Verify load round-trip for one model
    print("\n── Load Round-trip Check ────────────────────────────────────────")
    loaded = load_model("AAPL", "arima")
    if loaded and loaded.is_fitted:
        fc = loaded.predict(5)
        print(f"  ✅ Loaded AAPL|arima — predict(5) = {fc.round(4)}")
    else:
        print("  ❌ Load failed")

    if n_ok == len(results):
        print(f"\n✅ trainer.py smoke test PASSED — {n_ok}/{len(results)} models trained\n")
    else:
        print(f"\n⚠️  trainer.py smoke test partial — {n_ok}/{len(results)} succeeded\n")
