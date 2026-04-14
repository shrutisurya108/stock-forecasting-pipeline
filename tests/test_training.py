"""
tests/test_training.py
======================
Unit tests for training/trainer.py and training/benchmarking.py.

Strategy:
- Use a MockModel that fits and predicts instantly (no real ML).
- Use the same synthetic ProcessedData fixture from test_models.py.
- Test fault isolation, result structure, CSV output, and aggregate summary.
- No real network calls, no real model training — suite stays under 5s.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from data.preprocessing import preprocess_ticker, ProcessedData
from models.base_model import BaseModel, _compute_metrics
from training.trainer import (
    _make_model,
    _train_single,
    train_all,
    model_save_path,
)
from training.benchmarking import (
    build_benchmark_table,
    compute_aggregate_summary,
    run_benchmark,
    evaluate_naive,
)
from config.settings import LSTM_CONFIG, MODEL_NAMES


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def processed() -> ProcessedData:
    """Shared synthetic ProcessedData — built once for the whole module."""
    np.random.seed(0)
    n = 600
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    raw = pd.DataFrame(
        {
            "Open":   close * 0.99,
            "High":   close * 1.02,
            "Low":    close * 0.97,
            "Close":  close,
            "Volume": np.random.randint(500_000, 2_000_000, n).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    result = preprocess_ticker(raw, "TEST")
    assert result is not None
    return result


class MockModel(BaseModel):
    """
    Instant-fitting mock model for testing the training orchestration.
    Always returns a constant forecast of 0.5 (scaled space).
    """

    def _fit(self, train, val):
        self._n_features = len(train.columns)

    def _predict(self, n_steps):
        return np.full(n_steps, 0.5)

    def _predict_with_ci(self, n_steps, confidence=0.95):
        fc = self._predict(n_steps)
        return {"forecast": fc, "lower": fc - 0.05, "upper": fc + 0.05}

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "mock.txt").write_text("saved")

    def load(self, path):
        self.is_fitted = True
        return self


def _make_mock(model_name: str, ticker: str) -> MockModel:
    return MockModel(ticker=ticker, model_name=model_name)


# ══════════════════════════════════════════════════════════════════════════════
# _make_model factory
# ══════════════════════════════════════════════════════════════════════════════

def test_make_model_arima():
    from models.arima_model import ARIMAModel
    m = _make_model("arima", "AAPL")
    assert isinstance(m, ARIMAModel)


def test_make_model_prophet():
    from models.prophet_model import ProphetModel
    m = _make_model("prophet", "AAPL")
    assert isinstance(m, ProphetModel)


def test_make_model_lstm():
    from models.lstm_model import LSTMModel
    m = _make_model("lstm", "AAPL")
    assert isinstance(m, LSTMModel)


def test_make_model_invalid():
    with pytest.raises(ValueError, match="Unknown model name"):
        _make_model("xgboost", "AAPL")


# ══════════════════════════════════════════════════════════════════════════════
# _train_single
# ══════════════════════════════════════════════════════════════════════════════

def test_train_single_success(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("training.trainer.SAVED_MODELS_DIR", tmp_path)
    monkeypatch.setattr("training.trainer._make_model", _make_mock)

    model, result = _train_single("AAPL", "arima", processed, overwrite=True)

    assert result.success is True
    assert result.ticker == "AAPL"
    assert result.model_name == "arima"
    assert result.duration_s >= 0
    assert model is not None
    assert model.is_fitted


def test_train_single_failure(processed, tmp_path, monkeypatch):
    """If _make_model raises, _train_single captures it and returns None."""
    monkeypatch.setattr("training.trainer.SAVED_MODELS_DIR", tmp_path)

    def _bad_model(name, ticker):
        raise RuntimeError("Simulated failure")

    monkeypatch.setattr("training.trainer._make_model", _bad_model)

    model, result = _train_single("AAPL", "arima", processed, overwrite=True)

    assert result.success is False
    assert "Simulated failure" in result.error
    assert model is None


def test_train_single_saves_model(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("training.trainer.SAVED_MODELS_DIR", tmp_path)
    monkeypatch.setattr("training.trainer._make_model", _make_mock)

    _train_single("AAPL", "arima", processed, overwrite=True)

    save_dir = tmp_path / "AAPL" / "arima"
    assert save_dir.exists()


# ══════════════════════════════════════════════════════════════════════════════
# train_all
# ══════════════════════════════════════════════════════════════════════════════

def test_train_all_returns_correct_structure(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("training.trainer.SAVED_MODELS_DIR", tmp_path)
    monkeypatch.setattr("training.trainer._make_model", _make_mock)
    monkeypatch.setattr("training.trainer.PREDICTIONS_DIR", tmp_path)

    data = {"AAPL": processed, "MSFT": processed}
    fitted, results = train_all(data, model_names=["arima", "prophet"])

    assert set(fitted.keys()) == {"AAPL", "MSFT"}
    for ticker_models in fitted.values():
        assert set(ticker_models.keys()) == {"arima", "prophet"}


def test_train_all_result_count(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("training.trainer.SAVED_MODELS_DIR", tmp_path)
    monkeypatch.setattr("training.trainer._make_model", _make_mock)
    monkeypatch.setattr("training.trainer.PREDICTIONS_DIR", tmp_path)

    data = {"AAPL": processed}
    _, results = train_all(data, model_names=["arima", "lstm"])

    # 1 ticker × 2 models = 2 results
    assert len(results) == 2


def test_train_all_isolates_failures(processed, tmp_path, monkeypatch):
    """One failing model should not abort the others."""
    monkeypatch.setattr("training.trainer.SAVED_MODELS_DIR", tmp_path)
    monkeypatch.setattr("training.trainer.PREDICTIONS_DIR", tmp_path)

    call_count = {"n": 0}

    def _flaky_model(name, ticker):
        call_count["n"] += 1
        if call_count["n"] == 1:   # first call fails
            raise RuntimeError("Forced failure")
        return _make_mock(name, ticker)

    monkeypatch.setattr("training.trainer._make_model", _flaky_model)

    data = {"AAPL": processed}
    fitted, results = train_all(data, model_names=["arima", "prophet"])

    n_ok   = sum(1 for r in results if r.success)
    n_fail = sum(1 for r in results if not r.success)
    assert n_ok   == 1
    assert n_fail == 1


def test_train_all_writes_report(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("training.trainer.SAVED_MODELS_DIR", tmp_path)
    monkeypatch.setattr("training.trainer.PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr("training.trainer._make_model", _make_mock)

    data = {"AAPL": processed}
    train_all(data, model_names=["arima"])

    report_path = tmp_path / "training_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert "total" in report
    assert "success" in report
    assert "results" in report


# ══════════════════════════════════════════════════════════════════════════════
# evaluate_naive
# ══════════════════════════════════════════════════════════════════════════════

def test_evaluate_naive_returns_metrics(processed):
    m = evaluate_naive(processed)
    assert set(m.keys()) == {"RMSE", "MAPE", "MAE"}
    assert m["RMSE"] > 0
    assert m["MAE"]  > 0


def test_evaluate_naive_mape_percent_range(processed):
    """MAPE should be a percentage (typically 0–100 for stocks)."""
    m = evaluate_naive(processed)
    assert 0 <= m["MAPE"] <= 100


# ══════════════════════════════════════════════════════════════════════════════
# build_benchmark_table
# ══════════════════════════════════════════════════════════════════════════════

def _make_fitted_dict(processed, tickers, model_names):
    """Helper: build a fitted_models dict using MockModel."""
    fitted = {}
    for ticker in tickers:
        fitted[ticker] = {}
        for name in model_names:
            m = MockModel(ticker=ticker, model_name=name)
            m.fit(processed.train, processed.val)
            fitted[ticker][name] = m
    return fitted


def test_build_benchmark_table_shape(processed):
    tickers      = ["AAPL", "MSFT"]
    model_names  = ["arima", "prophet"]
    fitted       = _make_fitted_dict(processed, tickers, model_names)
    proc_data    = {t: processed for t in tickers}

    df = build_benchmark_table(fitted, proc_data, model_names=model_names)

    # 2 tickers × (2 models + 1 naive) = 6 rows
    assert len(df) == 6


def test_build_benchmark_table_columns(processed):
    fitted    = _make_fitted_dict(processed, ["AAPL"], ["arima"])
    proc_data = {"AAPL": processed}

    df = build_benchmark_table(fitted, proc_data, model_names=["arima"])

    expected_cols = {"ticker", "model", "RMSE", "MAPE", "MAE", "rmse_vs_naive_pct"}
    assert expected_cols.issubset(set(df.columns))


def test_build_benchmark_naive_pct_zero(processed):
    """Naive row should have rmse_vs_naive_pct == 0.0."""
    fitted    = _make_fitted_dict(processed, ["AAPL"], ["arima"])
    proc_data = {"AAPL": processed}

    df = build_benchmark_table(fitted, proc_data, model_names=["arima"])
    naive_row = df[df["model"] == "naive"]

    assert not naive_row.empty
    assert naive_row.iloc[0]["rmse_vs_naive_pct"] == 0.0


def test_build_benchmark_skips_missing_model(processed):
    """If a model is missing from fitted_models, it's skipped gracefully."""
    fitted    = {"AAPL": {}}    # no models at all
    proc_data = {"AAPL": processed}

    df = build_benchmark_table(fitted, proc_data, model_names=["arima"])

    # Should only have naive row
    assert len(df) == 1
    assert df.iloc[0]["model"] == "naive"


# ══════════════════════════════════════════════════════════════════════════════
# compute_aggregate_summary
# ══════════════════════════════════════════════════════════════════════════════

def test_aggregate_summary_keys(processed):
    fitted    = _make_fitted_dict(processed, ["AAPL"], ["arima", "lstm"])
    proc_data = {"AAPL": processed}

    df      = build_benchmark_table(fitted, proc_data, model_names=["arima", "lstm"])
    summary = compute_aggregate_summary(df)

    expected_keys = {
        "avg_rmse_reduction_pct",
        "best_model_by_rmse",
        "best_model_by_mape",
        "naive_avg_rmse",
        "model_avg_metrics",
        "n_tickers",
        "n_models",
    }
    assert expected_keys.issubset(set(summary.keys()))


def test_aggregate_summary_n_tickers(processed):
    fitted    = _make_fitted_dict(processed, ["AAPL", "MSFT"], ["arima"])
    proc_data = {t: processed for t in ["AAPL", "MSFT"]}

    df      = build_benchmark_table(fitted, proc_data, model_names=["arima"])
    summary = compute_aggregate_summary(df)

    assert summary["n_tickers"] == 2


def test_aggregate_summary_empty_df():
    summary = compute_aggregate_summary(pd.DataFrame())
    assert summary == {}


# ══════════════════════════════════════════════════════════════════════════════
# run_benchmark
# ══════════════════════════════════════════════════════════════════════════════

def test_run_benchmark_saves_csv(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("training.benchmarking.BENCHMARK_CSV", tmp_path / "bench.csv")
    monkeypatch.setattr("training.benchmarking.BENCHMARK_JSON", tmp_path / "bench.json")

    fitted    = _make_fitted_dict(processed, ["AAPL"], ["arima"])
    proc_data = {"AAPL": processed}

    df, summary = run_benchmark(fitted, proc_data, model_names=["arima"], save=True)

    assert (tmp_path / "bench.csv").exists()
    assert (tmp_path / "bench.json").exists()
    loaded = pd.read_csv(tmp_path / "bench.csv")
    assert len(loaded) == len(df)


def test_run_benchmark_no_save(processed, tmp_path, monkeypatch):
    csv_path = tmp_path / "bench.csv"
    monkeypatch.setattr("training.benchmarking.BENCHMARK_CSV", csv_path)

    fitted    = _make_fitted_dict(processed, ["AAPL"], ["arima"])
    proc_data = {"AAPL": processed}

    run_benchmark(fitted, proc_data, model_names=["arima"], save=False)

    assert not csv_path.exists()


def test_run_benchmark_returns_tuple(processed):
    fitted    = _make_fitted_dict(processed, ["AAPL"], ["arima"])
    proc_data = {"AAPL": processed}

    result = run_benchmark(fitted, proc_data, model_names=["arima"], save=False)

    assert isinstance(result, tuple)
    assert len(result) == 2
    df, summary = result
    assert isinstance(df, pd.DataFrame)
    assert isinstance(summary, dict)
