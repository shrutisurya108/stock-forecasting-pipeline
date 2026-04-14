"""
tests/test_models.py
====================
Unit tests for all three models: ARIMA, Prophet, LSTM.

Design:
- All tests use the same synthetic 500-row DataFrame fixture.
- No real network calls, no real stock data needed.
- LSTM epochs are set to 3 in tests so the suite stays fast.
- Each model is tested for: fit, predict shape, CI keys+shapes,
  evaluate keys, unfitted guard, save/load round-trip.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.preprocessing import preprocess_ticker, ProcessedData
from models.base_model import BaseModel, _compute_metrics, naive_baseline_metrics
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
from models.lstm_model import LSTMModel
from config.settings import LSTM_CONFIG


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def processed() -> ProcessedData:
    """
    Single preprocessed dataset shared across all model tests.
    scope="module" means it's built once per test file — saves ~1s.
    """
    np.random.seed(42)
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
    assert result is not None, "Fixture preprocessing failed"
    return result


@pytest.fixture(scope="module")
def fitted_arima(processed) -> ARIMAModel:
    model = ARIMAModel("TEST")
    model.fit(processed.train, processed.val)
    return model


@pytest.fixture(scope="module")
def fitted_prophet(processed) -> ProphetModel:
    model = ProphetModel("TEST")
    model.fit(processed.train, processed.val)
    return model


@pytest.fixture(scope="module")
def fitted_lstm(processed) -> LSTMModel:
    # Use minimal epochs so the test suite stays fast
    orig_epochs   = LSTM_CONFIG["epochs"]
    orig_patience = LSTM_CONFIG["early_stopping_patience"]
    LSTM_CONFIG["epochs"]                   = 3
    LSTM_CONFIG["early_stopping_patience"]  = 2
    model = LSTMModel("TEST")
    model.fit(processed.train, processed.val)
    LSTM_CONFIG["epochs"]                   = orig_epochs
    LSTM_CONFIG["early_stopping_patience"]  = orig_patience
    return model


# ══════════════════════════════════════════════════════════════════════════════
# _compute_metrics
# ══════════════════════════════════════════════════════════════════════════════

def test_compute_metrics_perfect_forecast():
    y = np.array([100.0, 200.0, 300.0])
    m = _compute_metrics(y, y)
    assert m["RMSE"] == pytest.approx(0.0)
    assert m["MAE"]  == pytest.approx(0.0)
    assert m["MAPE"] == pytest.approx(0.0)


def test_compute_metrics_keys():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    m = _compute_metrics(y_true, y_pred)
    assert set(m.keys()) == {"RMSE", "MAPE", "MAE"}


def test_compute_metrics_values():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 90.0])
    m = _compute_metrics(y_true, y_pred)
    assert m["RMSE"] == pytest.approx(10.0)
    assert m["MAE"]  == pytest.approx(10.0)
    assert m["MAPE"] == pytest.approx(10.0)


# ══════════════════════════════════════════════════════════════════════════════
# naive_baseline_metrics
# ══════════════════════════════════════════════════════════════════════════════

def test_naive_baseline_returns_metrics(processed):
    m = naive_baseline_metrics(processed.test, processed.scaler)
    assert set(m.keys()) == {"RMSE", "MAPE", "MAE"}
    assert m["RMSE"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# ARIMA
# ══════════════════════════════════════════════════════════════════════════════

class TestARIMAModel:

    def test_is_fitted(self, fitted_arima):
        assert fitted_arima.is_fitted is True

    def test_order_is_tuple(self, fitted_arima):
        assert isinstance(fitted_arima.order, tuple)
        assert len(fitted_arima.order) == 3

    def test_predict_shape(self, fitted_arima):
        fc = fitted_arima.predict(10)
        assert fc.shape == (10,)

    def test_predict_finite(self, fitted_arima):
        fc = fitted_arima.predict(10)
        assert np.all(np.isfinite(fc))

    def test_predict_with_ci_keys(self, fitted_arima):
        result = fitted_arima.predict_with_ci(10)
        assert set(result.keys()) == {"forecast", "lower", "upper"}

    def test_predict_with_ci_shapes(self, fitted_arima):
        result = fitted_arima.predict_with_ci(10)
        for arr in result.values():
            assert arr.shape == (10,)

    def test_predict_with_ci_ordering(self, fitted_arima):
        """lower ≤ forecast ≤ upper for all steps."""
        result = fitted_arima.predict_with_ci(10)
        assert np.all(result["lower"] <= result["forecast"] + 1e-6)
        assert np.all(result["forecast"] <= result["upper"] + 1e-6)

    def test_evaluate_keys(self, fitted_arima, processed):
        m = fitted_arima.evaluate(processed.test, processed.scaler)
        assert set(m.keys()) == {"RMSE", "MAPE", "MAE"}

    def test_evaluate_positive_values(self, fitted_arima, processed):
        m = fitted_arima.evaluate(processed.test, processed.scaler)
        assert m["RMSE"] > 0
        assert m["MAE"]  > 0

    def test_unfitted_guard(self):
        model = ARIMAModel("TEST")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(5)

    def test_save_load_roundtrip(self, fitted_arima):
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_arima.save(Path(tmpdir))
            model2 = ARIMAModel("TEST")
            model2.load(Path(tmpdir))
            assert model2.is_fitted
            fc1 = fitted_arima.predict(5)
            fc2 = model2.predict(5)
            assert np.allclose(fc1, fc2, atol=1e-5)

    def test_repr(self, fitted_arima):
        r = repr(fitted_arima)
        assert "ARIMA" in r
        assert "fitted" in r


# ══════════════════════════════════════════════════════════════════════════════
# Prophet
# ══════════════════════════════════════════════════════════════════════════════

class TestProphetModel:

    def test_is_fitted(self, fitted_prophet):
        assert fitted_prophet.is_fitted is True

    def test_predict_shape(self, fitted_prophet):
        fc = fitted_prophet.predict(10)
        assert fc.shape == (10,)

    def test_predict_finite(self, fitted_prophet):
        fc = fitted_prophet.predict(10)
        assert np.all(np.isfinite(fc))

    def test_predict_with_ci_keys(self, fitted_prophet):
        result = fitted_prophet.predict_with_ci(10)
        assert set(result.keys()) == {"forecast", "lower", "upper"}

    def test_predict_with_ci_shapes(self, fitted_prophet):
        result = fitted_prophet.predict_with_ci(10)
        for arr in result.values():
            assert arr.shape == (10,)

    def test_predict_with_ci_ordering(self, fitted_prophet):
        result = fitted_prophet.predict_with_ci(10)
        assert np.all(result["lower"] <= result["forecast"] + 1e-6)
        assert np.all(result["forecast"] <= result["upper"] + 1e-6)

    def test_evaluate_keys(self, fitted_prophet, processed):
        m = fitted_prophet.evaluate(processed.test, processed.scaler)
        assert set(m.keys()) == {"RMSE", "MAPE", "MAE"}

    def test_evaluate_positive_values(self, fitted_prophet, processed):
        m = fitted_prophet.evaluate(processed.test, processed.scaler)
        assert m["RMSE"] > 0

    def test_unfitted_guard(self):
        model = ProphetModel("TEST")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(5)

    def test_save_load_roundtrip(self, fitted_prophet):
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_prophet.save(Path(tmpdir))
            model2 = ProphetModel("TEST")
            model2.load(Path(tmpdir))
            assert model2.is_fitted
            fc1 = fitted_prophet.predict(5)
            fc2 = model2.predict(5)
            assert np.allclose(fc1, fc2, atol=1e-5)

    def test_last_date_set(self, fitted_prophet):
        assert fitted_prophet._last_date is not None


# ══════════════════════════════════════════════════════════════════════════════
# LSTM
# ══════════════════════════════════════════════════════════════════════════════

class TestLSTMModel:

    def test_is_fitted(self, fitted_lstm):
        assert fitted_lstm.is_fitted is True

    def test_predict_shape(self, fitted_lstm):
        fc = fitted_lstm.predict(10)
        assert fc.shape == (10,)

    def test_predict_finite(self, fitted_lstm):
        fc = fitted_lstm.predict(10)
        assert np.all(np.isfinite(fc))

    def test_predict_with_ci_keys(self, fitted_lstm):
        result = fitted_lstm.predict_with_ci(10)
        assert set(result.keys()) == {"forecast", "lower", "upper"}

    def test_predict_with_ci_shapes(self, fitted_lstm):
        result = fitted_lstm.predict_with_ci(10)
        for arr in result.values():
            assert arr.shape == (10,)

    def test_evaluate_keys(self, fitted_lstm, processed):
        m = fitted_lstm.evaluate(processed.test, processed.scaler)
        assert set(m.keys()) == {"RMSE", "MAPE", "MAE"}

    def test_evaluate_positive_values(self, fitted_lstm, processed):
        m = fitted_lstm.evaluate(processed.test, processed.scaler)
        assert m["RMSE"] > 0

    def test_unfitted_guard(self):
        model = LSTMModel("TEST")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(5)

    def test_save_load_roundtrip(self, fitted_lstm):
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_lstm.save(Path(tmpdir))
            model2 = LSTMModel("TEST")
            model2.load(Path(tmpdir))
            assert model2.is_fitted
            fc1 = fitted_lstm.predict(5)
            fc2 = model2.predict(5)
            # LSTM deterministic after load (eval mode, no MC dropout)
            assert np.allclose(fc1, fc2, atol=1e-4)

    def test_feature_cols_stored(self, fitted_lstm):
        assert fitted_lstm._feature_cols is not None
        assert len(fitted_lstm._feature_cols) > 0

    def test_last_seq_shape(self, fitted_lstm):
        seq = fitted_lstm._last_seq
        assert seq is not None
        assert seq.shape == (
            LSTM_CONFIG["sequence_length"],
            len(fitted_lstm._feature_cols),
        )

    def test_summary_contains_key_info(self, fitted_lstm):
        s = fitted_lstm.summary()
        assert "TEST" in s
        assert "features" in s
        assert "params" in s
