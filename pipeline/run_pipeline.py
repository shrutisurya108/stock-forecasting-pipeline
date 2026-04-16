"""
pipeline/run_pipeline.py
========================
Single entry point that chains every phase of the pipeline together.

Three operating modes
---------------------
full          — fetch + preprocess + train + benchmark + forecast
                Used for the daily production run (~30–45 min on CPU).

forecast_only — skip training, load existing models, generate new forecasts.
                Used when models are already trained and only fresh predictions
                are needed (~1 min).

single        — full pipeline for one ticker only (--ticker AAPL).
                Used for debugging / smoke-testing a single stock.

CLI usage
---------
    python -m pipeline.run_pipeline --mode full
    python -m pipeline.run_pipeline --mode forecast_only
    python -m pipeline.run_pipeline --mode single --ticker AAPL
    python -m pipeline.run_pipeline --mode full --no-benchmark

Programmatic usage (called by lambda_handler.py)
-------------------------------------------------
    from pipeline.run_pipeline import main
    result = main(mode="forecast_only", tickers=["AAPL", "MSFT"])

Output
------
- predictions/{TICKER}_forecast.csv   — 30-day forecasts per ticker
- predictions/all_forecasts.csv       — combined forecast table
- predictions/benchmark_results.csv   — RMSE/MAPE comparison table
- predictions/training_report.json    — per-model training metadata
- logs/run_{YYYY-MM-DD}.log           — per-run log file
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from config.logging_config import get_logger, get_run_logger
from config.settings import (
    CONFIDENCE_INTERVAL,
    FORECAST_HORIZON,
    LSTM_CONFIG,
    MODEL_NAMES,
    PREDICTIONS_DIR,
    STOCK_TICKERS,
)
from data.ingestion import fetch_all_tickers
from data.preprocessing import preprocess_all
from forecasting.predictor import generate_all_predictions, save_all_predictions
from training.benchmarking import run_benchmark
from training.trainer import train_all, load_model
from storage.s3_client import S3Client

logger = get_logger(__name__)

# ── Valid pipeline modes ───────────────────────────────────────────────────────
VALID_MODES = ("full", "forecast_only", "single")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Captures the outcome of a full pipeline run.

    Attributes:
        run_id:          Date string used as unique run identifier.
        mode:            Pipeline mode used ("full", "forecast_only", "single").
        tickers_fetched: Number of tickers successfully fetched.
        tickers_trained: Number of tickers with at least one trained model.
        tickers_forecast:Number of tickers with a saved forecast.
        models_trained:  Total (ticker × model) pairs trained this run.
        duration_s:      Wall-clock seconds for the entire run.
        success:         True if the pipeline completed without fatal errors.
        errors:          List of non-fatal error messages encountered.
        benchmark_avg_rmse_reduction: Aggregate RMSE improvement vs naive (%).
    """
    run_id:                      str
    mode:                        str
    tickers_fetched:             int   = 0
    tickers_trained:             int   = 0
    tickers_forecast:            int   = 0
    models_trained:              int   = 0
    duration_s:                  float = 0.0
    success:                     bool  = False
    errors:                      list  = field(default_factory=list)
    benchmark_avg_rmse_reduction: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        lines = [
            f"  Run ID   : {self.run_id}",
            f"  Mode     : {self.mode}",
            f"  Fetched  : {self.tickers_fetched} tickers",
        ]
        if self.mode != "forecast_only":
            lines.append(
                f"  Trained  : {self.tickers_trained} tickers "
                f"({self.models_trained} model fits)"
            )
        if self.benchmark_avg_rmse_reduction:
            lines.append(
                f"  RMSE ↓   : {self.benchmark_avg_rmse_reduction:+.1f}% vs naive"
            )
        lines += [
            f"  Forecast : {self.tickers_forecast} tickers",
            f"  Duration : {self.duration_s:.1f}s",
            f"  Status   : {'✅ SUCCESS' if self.success else '❌ FAILED'}",
        ]
        if self.errors:
            lines.append(f"  Errors   : {len(self.errors)} non-fatal")
            for e in self.errors[:3]:
                lines.append(f"    • {e}")
        return lines


# ── Step helpers ──────────────────────────────────────────────────────────────

def _step(n: int, total: int, label: str) -> None:
    """Print a formatted step header to stdout."""
    print(f"\n[{n}/{total}] {label}...")
    logger.info("── Step %d/%d: %s", n, total, label)


def _ok(elapsed: float, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"        ✅  {elapsed:.1f}s{suffix}")


def _skip(reason: str) -> None:
    print(f"        ⏭  skipped ({reason})")


def _fail(msg: str) -> None:
    print(f"        ❌  {msg}")


# ── Individual pipeline stages ────────────────────────────────────────────────

def stage_fetch(
    tickers: list[str],
    force_download: bool = False,
) -> tuple[dict, float]:
    """
    Stage 1 — Fetch raw OHLCV data from Yahoo Finance.

    Returns (raw_data_dict, elapsed_seconds).
    """
    t0  = time.time()
    raw = fetch_all_tickers(tickers=tickers, force_download=force_download)
    return raw, time.time() - t0


def stage_preprocess(raw: dict) -> tuple[dict, float]:
    """
    Stage 2 — Clean, engineer features, scale, split.

    Returns (processed_dict, elapsed_seconds).
    """
    t0        = time.time()
    processed = preprocess_all(raw)
    return processed, time.time() - t0


def stage_train(
    processed:   dict,
    model_names: list[str],
    overwrite:   bool = True,
) -> tuple[dict, list, float]:
    """
    Stage 3 — Fit all models on all tickers.

    Returns (fitted_models, training_results, elapsed_seconds).
    """
    t0             = time.time()
    fitted, results = train_all(
        processed_data=processed,
        model_names=model_names,
        overwrite=overwrite,
    )
    return fitted, results, time.time() - t0


def stage_benchmark(
    fitted:    dict,
    processed: dict,
    model_names: list[str],
) -> tuple[object, dict, float]:
    """
    Stage 4 — Evaluate all models vs naive baseline.

    Returns (benchmark_df, summary, elapsed_seconds).
    """
    t0         = time.time()
    df, summary = run_benchmark(
        fitted_models=fitted,
        processed_data=processed,
        model_names=model_names,
        save=True,
    )
    return df, summary, time.time() - t0


def stage_forecast(
    processed:    dict,
    fitted:       dict,
    model_names:  list[str],
    horizon:      int   = FORECAST_HORIZON,
    confidence:   float = CONFIDENCE_INTERVAL,
) -> tuple[dict, float]:
    """
    Stage 5 — Generate forward predictions with CI bands.

    Returns (predictions_dict, elapsed_seconds).
    """
    t0          = time.time()
    predictions = generate_all_predictions(
        processed_data=processed,
        fitted_models=fitted,
        model_names=model_names,
        horizon=horizon,
        confidence=confidence,
        save=True,
    )
    return predictions, time.time() - t0


def _load_existing_models(
    tickers:     list[str],
    model_names: list[str],
) -> dict:
    """
    Load all already-saved models from disk (used in forecast_only mode).

    Returns {ticker: {model_name: fitted_model}}.
    Missing models are logged but don't abort the load.
    """
    fitted: dict = {}
    for ticker in tickers:
        fitted[ticker] = {}
        for name in model_names:
            m = load_model(ticker, name)
            if m is not None:
                fitted[ticker][name] = m
            else:
                logger.warning(
                    "No saved model for [%s | %s] — will skip in forecast",
                    ticker, name,
                )
    return fitted


def stage_upload(client) -> tuple[object, float]:
    """
    Stage 6 — Upload predictions, benchmarks, AND model artifacts to S3.

    Uploading model artifacts means future CI runs can restore them and
    skip retraining, running in forecast_only mode instead of full (~40 min saved).

    Args:
        client: An S3Client instance.

    Returns:
        (UploadResult, elapsed_seconds)
    """
    t0 = time.time()

    # Upload prediction CSVs and benchmark results
    result = client.upload_all_predictions()

    # Upload trained model artifacts so CI can restore them on next run
    model_result = client.upload_models()
    result.files_uploaded += model_result.files_uploaded
    result.bytes_uploaded += model_result.bytes_uploaded
    result.files_failed   += model_result.files_failed
    if model_result.files_failed > 0:
        result.errors.extend(model_result.errors)

    return result, time.time() - t0


# ── Main pipeline entry point ─────────────────────────────────────────────────

def main(
    mode:         str        = "full",
    tickers:      list[str] | None = None,
    model_names:  list[str] | None = None,
    horizon:      int        = FORECAST_HORIZON,
    confidence:   float      = CONFIDENCE_INTERVAL,
    run_benchmark_flag: bool = True,
    force_download: bool     = False,
) -> PipelineResult:
    """
    Run the full forecasting pipeline.

    Args:
        mode:          "full" | "forecast_only" | "single".
        tickers:       Stock symbols to process. Defaults to all 10.
        model_names:   Models to use. Defaults to all three.
        horizon:       Forecast horizon in business days.
        confidence:    CI confidence level (0–1).
        run_benchmark_flag: Whether to run benchmarking (skip in forecast_only).
        force_download: Force re-download even if cache is fresh.

    Returns:
        PipelineResult with run metadata and outcome.
    """
    # ── Validate mode ─────────────────────────────────────────────────────────
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Valid modes: {VALID_MODES}"
        )

    tickers     = tickers     or STOCK_TICKERS
    model_names = model_names or MODEL_NAMES

    run_id  = datetime.date.today().isoformat()
    run_log = get_run_logger(run_id)
    t_start = time.time()

    result    = PipelineResult(run_id=run_id, mode=mode)
    s3_client = S3Client()

    # ── Header ────────────────────────────────────────────────────────────────
    divider = "═" * 55
    print(f"\n{divider}")
    print(f"  Pipeline Run: {run_id} | mode={mode}")
    print(f"  Tickers: {tickers}")
    print(f"  Models : {model_names}")
    print(divider)

    run_log.info(
        "Pipeline started — run_id=%s mode=%s tickers=%s models=%s",
        run_id, mode, tickers, model_names,
    )

    try:
        total_steps = 5 if mode == "forecast_only" else 6

        # ── Stage 1: Fetch ────────────────────────────────────────────────────
        _step(1, total_steps, "Fetching stock data")
        raw, t = stage_fetch(tickers, force_download=force_download)
        result.tickers_fetched = len(raw)

        if not raw:
            _fail("No data fetched — aborting pipeline")
            run_log.error("No data fetched — pipeline aborted")
            result.errors.append("No data fetched")
            return result

        _ok(t, f"{len(raw)} tickers")
        run_log.info("Fetch complete: %d tickers in %.1fs", len(raw), t)

        # ── Stage 2: Preprocess ───────────────────────────────────────────────
        _step(2, total_steps, "Preprocessing")
        processed, t = stage_preprocess(raw)

        if not processed:
            _fail("Preprocessing produced no data — aborting")
            result.errors.append("Preprocessing failed for all tickers")
            return result

        _ok(t, f"{len(processed)} tickers ready")
        run_log.info("Preprocessing complete: %d tickers in %.1fs", len(processed), t)

        # ── Stage 3: Train (skipped in forecast_only) ─────────────────────────
        if mode == "forecast_only":
            _step(3, total_steps, "Loading saved models")
            fitted = _load_existing_models(list(processed.keys()), model_names)
            n_loaded = sum(len(v) for v in fitted.values())
            _ok(0, f"{n_loaded} models loaded from disk")
            run_log.info("Loaded %d models from disk (forecast_only)", n_loaded)
            training_results = []

        else:
            _step(3, total_steps, "Training models")
            overwrite = (mode == "full")
            fitted, training_results, t = stage_train(
                processed, model_names, overwrite=overwrite
            )
            n_ok = sum(1 for r in training_results if r.success)
            result.tickers_trained = len(
                [tk for tk in fitted if fitted[tk]]
            )
            result.models_trained = n_ok
            _ok(t, f"{n_ok}/{len(training_results)} model fits succeeded")
            run_log.info(
                "Training complete: %d/%d succeeded in %.1fs",
                n_ok, len(training_results), t,
            )

            # ── Stage 4: Benchmark ────────────────────────────────────────────
            if run_benchmark_flag and fitted:
                _step(4, total_steps, "Benchmarking")
                try:
                    _, summary, t = stage_benchmark(
                        fitted, processed, model_names
                    )
                    avg_r = summary.get("avg_rmse_reduction_pct", 0.0)
                    result.benchmark_avg_rmse_reduction = avg_r
                    _ok(t, f"avg RMSE reduction: {avg_r:+.1f}% vs naive")
                    run_log.info(
                        "Benchmark complete: avg_rmse_reduction=%.1f%% in %.1fs",
                        avg_r, t,
                    )
                except Exception as exc:
                    _fail(f"Benchmark failed (non-fatal): {exc}")
                    result.errors.append(f"Benchmark: {exc}")
                    run_log.warning("Benchmark failed (non-fatal): %s", exc)
            else:
                _skip("benchmark disabled")

        # ── Stage 5 (or 4): Forecast ──────────────────────────────────────────
        forecast_step = 4 if mode == "forecast_only" else 5
        _step(forecast_step, total_steps, "Generating 30-day forecasts")
        predictions, t = stage_forecast(
            processed, fitted, model_names, horizon, confidence
        )
        result.tickers_forecast = len(predictions)

        if not predictions:
            _fail("No forecasts generated")
            result.errors.append("Forecast stage produced no output")
        else:
            _ok(t, f"{len(predictions)} tickers, {horizon}-day horizon")
            run_log.info(
                "Forecasts generated: %d tickers in %.1fs",
                len(predictions), t,
            )

        # ── Stage 6: Upload to S3 ─────────────────────────────────────────────
        s3_step = forecast_step + 1
        _step(s3_step, s3_step, "Uploading to S3")
        if s3_client.is_available():
            upload_result, t = stage_upload(s3_client)
            if upload_result.success:
                _ok(
                    t,
                    f"{upload_result.files_uploaded} files "
                    f"({upload_result.bytes_uploaded:,} bytes)",
                )
                run_log.info(
                    "S3 upload complete: %d files in %.1fs",
                    upload_result.files_uploaded, t,
                )
            else:
                _fail(
                    f"S3 upload partial: "
                    f"{upload_result.files_uploaded} ok, "
                    f"{upload_result.files_failed} failed"
                )
                result.errors.extend(upload_result.errors)
        else:
            _skip("AWS credentials not configured — set in .env to enable")
            run_log.info("S3 upload skipped (credentials not set)")

        # ── Done ──────────────────────────────────────────────────────────────
        result.duration_s = round(time.time() - t_start, 2)
        result.success    = (
            result.tickers_fetched > 0
            and result.tickers_forecast > 0
            and len(result.errors) == 0
        )

    except Exception as exc:
        result.duration_s = round(time.time() - t_start, 2)
        result.success    = False
        err_msg = f"{type(exc).__name__}: {exc}"
        result.errors.append(err_msg)
        run_log.error("Pipeline FATAL error: %s\n%s", err_msg, traceback.format_exc())
        logger.error("Pipeline fatal error: %s", err_msg, exc_info=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{divider}")
    print("  Pipeline Summary")
    print(divider)
    for line in result.summary_lines():
        print(line)
    print(divider + "\n")

    run_log.info(
        "Pipeline finished — success=%s duration=%.1fs",
        result.success, result.duration_s,
    )

    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stock Forecasting Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="forecast_only",
        help=(
            "full          → fetch + preprocess + train + forecast\n"
            "forecast_only → load saved models, generate new forecasts only\n"
            "single        → full pipeline for one ticker (use with --ticker)"
        ),
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker symbol for --mode single (e.g. AAPL)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=MODEL_NAMES,
        help="Which models to use (default: all three)",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip the benchmarking stage (faster, no CSV written)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download data even if cache is fresh",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use 5 LSTM epochs (smoke-test speed — do NOT use in production)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ── --fast flag: override LSTM epochs for quick testing ───────────────────
    if args.fast:
        LSTM_CONFIG["epochs"]                 = 5
        LSTM_CONFIG["early_stopping_patience"] = 3
        print("⚠️  --fast flag active: LSTM epochs = 5 (smoke-test mode)")

    # ── --mode single: validate --ticker was provided ─────────────────────────
    if args.mode == "single":
        if not args.ticker:
            print("❌  --mode single requires --ticker (e.g. --ticker AAPL)")
            sys.exit(1)
        tickers = [args.ticker.upper()]
    else:
        tickers = None   # use all 10

    result = main(
        mode=args.mode,
        tickers=tickers,
        model_names=args.models,
        run_benchmark_flag=not args.no_benchmark,
        force_download=args.force_download,
    )

    sys.exit(0 if result.success else 1)