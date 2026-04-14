"""
training/benchmarking.py
========================
Computes RMSE, MAPE, and MAE for every (ticker, model) pair against the
naive last-value baseline, then builds a clean comparison DataFrame and
computes the aggregate RMSE reduction — the headline metric on the resume.

Outputs
-------
- Returns a pandas DataFrame (benchmark table) — used by dashboard.
- Saves predictions/benchmark_results.csv  — read by dashboard + pipeline.
- Logs per-ticker and aggregate summaries.

Usage (standalone smoke-test):
    python -m training.benchmarking
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import (
    METRICS,
    MODEL_NAMES,
    PREDICTIONS_DIR,
    STOCK_TICKERS,
    LSTM_CONFIG,
)
from data.preprocessing import ProcessedData
from models.base_model import BaseModel, naive_baseline_metrics

logger = get_logger(__name__)

# Output paths
BENCHMARK_CSV  = PREDICTIONS_DIR / "benchmark_results.csv"
BENCHMARK_JSON = PREDICTIONS_DIR / "benchmark_summary.json"


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_model(
    model:     BaseModel,
    processed: ProcessedData,
) -> dict[str, float]:
    """
    Evaluate a single fitted model on the test split.

    Args:
        model:     A fitted model instance (ARIMA, Prophet, or LSTM).
        processed: The ProcessedData container for this ticker.

    Returns:
        Dict {"RMSE": float, "MAPE": float, "MAE": float}
    """
    return model.evaluate(processed.test, processed.scaler)


def evaluate_naive(processed: ProcessedData) -> dict[str, float]:
    """
    Evaluate the naive last-value baseline on the test split.

    Args:
        processed: The ProcessedData container for this ticker.

    Returns:
        Dict {"RMSE": float, "MAPE": float, "MAE": float}
    """
    return naive_baseline_metrics(processed.test, processed.scaler)


# ── Benchmark builder ─────────────────────────────────────────────────────────

def build_benchmark_table(
    fitted_models:  dict[str, dict[str, BaseModel]],
    processed_data: dict[str, ProcessedData],
    model_names:    list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the full benchmark comparison DataFrame.

    For every ticker that has both a fitted model and preprocessed test data,
    computes metrics for the naive baseline and each model. Also computes the
    percentage improvement of each model's RMSE vs the naive baseline.

    Args:
        fitted_models:  {ticker: {model_name: BaseModel}} from trainer.
        processed_data: {ticker: ProcessedData} from preprocessing.
        model_names:    Subset of models to benchmark. Defaults to all three.

    Returns:
        DataFrame with columns:
            ticker, model, RMSE, MAPE, MAE, rmse_vs_naive_pct
        One row per (ticker, model) pair + one "naive" row per ticker.
    """
    model_names = model_names or MODEL_NAMES
    rows: list[dict] = []

    tickers = [t for t in fitted_models if t in processed_data]
    logger.info("Benchmarking %d tickers × (%d models + naive)", len(tickers), len(model_names))

    for ticker in tickers:
        processed = processed_data[ticker]

        # ── Naive baseline ────────────────────────────────────────────────────
        try:
            naive_m = evaluate_naive(processed)
            naive_rmse = naive_m["RMSE"]
            rows.append({
                "ticker":           ticker,
                "model":            "naive",
                "RMSE":             round(naive_m["RMSE"], 4),
                "MAPE":             round(naive_m["MAPE"], 4),
                "MAE":              round(naive_m["MAE"],  4),
                "rmse_vs_naive_pct": 0.0,
            })
            logger.debug(
                "[%s | naive] RMSE=%.4f MAPE=%.2f%% MAE=%.4f",
                ticker, naive_m["RMSE"], naive_m["MAPE"], naive_m["MAE"],
            )
        except Exception as exc:
            logger.error("[%s | naive] Evaluation failed: %s", ticker, exc)
            naive_rmse = None
            continue

        # ── Each model ────────────────────────────────────────────────────────
        ticker_models = fitted_models.get(ticker, {})

        for model_name in model_names:
            model = ticker_models.get(model_name)
            if model is None:
                logger.warning(
                    "[%s | %s] No fitted model found — skipping benchmark",
                    ticker, model_name,
                )
                continue

            try:
                m = evaluate_model(model, processed)

                # RMSE improvement vs naive (negative = better than baseline)
                rmse_pct = (
                    ((m["RMSE"] - naive_rmse) / naive_rmse * 100)
                    if naive_rmse and naive_rmse > 0
                    else float("nan")
                )

                rows.append({
                    "ticker":            ticker,
                    "model":             model_name,
                    "RMSE":              round(m["RMSE"], 4),
                    "MAPE":              round(m["MAPE"], 4),
                    "MAE":              round(m["MAE"],  4),
                    "rmse_vs_naive_pct": round(rmse_pct, 2),
                })

                logger.info(
                    "[%s | %s] RMSE=%.4f MAPE=%.2f%% MAE=%.4f vs_naive=%.1f%%",
                    ticker, model_name,
                    m["RMSE"], m["MAPE"], m["MAE"], rmse_pct,
                )

            except Exception as exc:
                logger.error(
                    "[%s | %s] Evaluation failed: %s",
                    ticker, model_name, exc, exc_info=True,
                )

    df = pd.DataFrame(rows)
    logger.info("Benchmark table: %d rows, %d columns", len(df), len(df.columns))
    return df


# ── Aggregate summary ─────────────────────────────────────────────────────────

def compute_aggregate_summary(df: pd.DataFrame) -> dict:
    """
    Compute aggregate metrics across all tickers and models.

    Key metric: average RMSE reduction vs naive baseline across all models
    and tickers. This is the headline "18% improvement" figure.

    Args:
        df: Output of build_benchmark_table().

    Returns:
        Dict with aggregate RMSE reduction, best model per metric, etc.
    """
    if df.empty:
        return {}

    # Exclude naive rows for model comparison
    model_df = df[df["model"] != "naive"]

    if model_df.empty:
        return {}

    # Average RMSE reduction across all (ticker, model) pairs
    avg_rmse_reduction = model_df["rmse_vs_naive_pct"].mean()

    # Best model overall (lowest average RMSE)
    best_by_rmse = (
        model_df.groupby("model")["RMSE"].mean().idxmin()
    )
    best_by_mape = (
        model_df.groupby("model")["MAPE"].mean().idxmin()
    )

    # Per-model average metrics
    model_avg = (
        model_df.groupby("model")[["RMSE", "MAPE", "MAE"]]
        .mean()
        .round(4)
        .to_dict()
    )

    # Naive average
    naive_df = df[df["model"] == "naive"]
    naive_avg_rmse = naive_df["RMSE"].mean() if not naive_df.empty else float("nan")

    summary = {
        "avg_rmse_reduction_pct":   round(avg_rmse_reduction, 2),
        "best_model_by_rmse":       best_by_rmse,
        "best_model_by_mape":       best_by_mape,
        "naive_avg_rmse":           round(naive_avg_rmse, 4),
        "model_avg_metrics":        model_avg,
        "n_tickers":                df["ticker"].nunique(),
        "n_models":                 model_df["model"].nunique(),
    }

    logger.info(
        "Aggregate: avg_rmse_reduction=%.1f%% | best_model=%s | "
        "naive_avg_rmse=%.4f",
        avg_rmse_reduction, best_by_rmse, naive_avg_rmse,
    )

    return summary


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_benchmark(df: pd.DataFrame, summary: dict) -> None:
    """Persist benchmark table to CSV and summary to JSON."""
    df.to_csv(BENCHMARK_CSV, index=False)
    logger.info("Benchmark CSV saved → %s", BENCHMARK_CSV)

    with open(BENCHMARK_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Benchmark summary saved → %s", BENCHMARK_JSON)


def load_benchmark() -> Optional[pd.DataFrame]:
    """Load the saved benchmark CSV. Returns None if it doesn't exist."""
    if not BENCHMARK_CSV.exists():
        logger.warning("No benchmark CSV found at %s", BENCHMARK_CSV)
        return None
    return pd.read_csv(BENCHMARK_CSV)


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_benchmark_table(df: pd.DataFrame, summary: dict) -> None:
    """Print a formatted benchmark table to stdout."""
    if df.empty:
        print("No benchmark results to display.")
        return

    tickers = df["ticker"].unique()
    model_order = ["naive"] + [m for m in MODEL_NAMES if m in df["model"].unique()]

    print("\n" + "═" * 72)
    print("  BENCHMARK RESULTS  (lower RMSE/MAPE/MAE = better)")
    print("═" * 72)
    print(
        f"{'Ticker':<7} {'Model':<10} {'RMSE':>8} {'MAPE%':>7} "
        f"{'MAE':>8} {'vs Naive':>10}"
    )
    print("─" * 72)

    for ticker in tickers:
        ticker_df = df[df["ticker"] == ticker]
        for model_name in model_order:
            row = ticker_df[ticker_df["model"] == model_name]
            if row.empty:
                continue
            r = row.iloc[0]
            vs = "baseline" if model_name == "naive" else f"{r['rmse_vs_naive_pct']:+.1f}%"
            marker = "  ◀ best" if (
                model_name != "naive"
                and summary.get("best_model_by_rmse") == model_name
                and ticker == tickers[0]   # only mark once
            ) else ""
            print(
                f"{ticker:<7} {model_name:<10} {r['RMSE']:>8.4f} "
                f"{r['MAPE']:>6.2f}% {r['MAE']:>8.4f} {vs:>10}{marker}"
            )
        print("─" * 72)

    # Aggregate section
    print()
    print("── Aggregate Summary ────────────────────────────────────────────────")
    print(f"  Avg RMSE reduction vs naive : {summary.get('avg_rmse_reduction_pct', '?'):+.1f}%")
    print(f"  Best model (RMSE)           : {summary.get('best_model_by_rmse', '?')}")
    print(f"  Best model (MAPE)           : {summary.get('best_model_by_mape', '?')}")
    print(f"  Tickers benchmarked         : {summary.get('n_tickers', '?')}")
    print()

    # Per-model average metrics table
    model_avg = summary.get("model_avg_metrics", {})
    if model_avg:
        print("── Per-Model Average Metrics ────────────────────────────────────────")
        print(f"  {'Model':<10} {'Avg RMSE':>10} {'Avg MAPE%':>10} {'Avg MAE':>10}")
        print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
        for model_name in MODEL_NAMES:
            rmse = model_avg.get("RMSE", {}).get(model_name, float("nan"))
            mape = model_avg.get("MAPE", {}).get(model_name, float("nan"))
            mae  = model_avg.get("MAE",  {}).get(model_name, float("nan"))
            print(f"  {model_name:<10} {rmse:>10.4f} {mape:>9.2f}% {mae:>10.4f}")
    print()


# ── Public run function ───────────────────────────────────────────────────────

def run_benchmark(
    fitted_models:  dict[str, dict[str, BaseModel]],
    processed_data: dict[str, ProcessedData],
    model_names:    list[str] | None = None,
    save:           bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Full benchmark pipeline: evaluate → aggregate → save → return.

    Args:
        fitted_models:  {ticker: {model_name: BaseModel}}.
        processed_data: {ticker: ProcessedData}.
        model_names:    Subset of models to benchmark.
        save:           Whether to persist results to disk.

    Returns:
        (benchmark_df, summary_dict)
    """
    df      = build_benchmark_table(fitted_models, processed_data, model_names)
    summary = compute_aggregate_summary(df)

    if save and not df.empty:
        save_benchmark(df, summary)

    return df, summary


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from training.trainer import train_all

    print("\n" + "═" * 65)
    print("  training/benchmarking.py — smoke test")
    print("═" * 65 + "\n")

    SMOKE_TICKERS = ["AAPL", "MSFT"]

    # Fast LSTM
    orig_epochs   = LSTM_CONFIG["epochs"]
    orig_patience = LSTM_CONFIG["early_stopping_patience"]
    LSTM_CONFIG["epochs"]                  = 5
    LSTM_CONFIG["early_stopping_patience"] = 3

    from data.ingestion import fetch_all_tickers
    from data.preprocessing import preprocess_all

    print("Step 1: Fetch + preprocess…")
    raw       = fetch_all_tickers(tickers=SMOKE_TICKERS)
    processed = preprocess_all(raw)
    print(f"        → {len(processed)} tickers ready\n")

    print("Step 2: Train all models…")
    fitted, train_results = train_all(processed, overwrite=True)
    n_ok = sum(1 for r in train_results if r.success)
    print(f"        → {n_ok}/{len(train_results)} models trained\n")

    LSTM_CONFIG["epochs"]                  = orig_epochs
    LSTM_CONFIG["early_stopping_patience"] = orig_patience

    print("Step 3: Benchmark…")
    df, summary = run_benchmark(fitted, processed, save=True)

    print_benchmark_table(df, summary)

    # Verify CSV was written
    if BENCHMARK_CSV.exists():
        print(f"✅ Benchmark CSV written → {BENCHMARK_CSV}")
        print(f"   Rows: {len(df)} | Columns: {list(df.columns)}")
    else:
        print("❌ Benchmark CSV not written")

    # Sanity check
    avg_reduction = summary.get("avg_rmse_reduction_pct", 0)
    print(f"\n✅ benchmarking.py smoke test PASSED")
    print(f"   Average RMSE reduction vs naive: {avg_reduction:+.1f}%")
    print(f"   (Target for full 10-ticker run:  ≈ -15% to -20%)\n")
