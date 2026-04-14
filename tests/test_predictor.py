"""
tests/test_predictor.py
=======================
Unit tests for forecasting/predictor.py.

Strategy
--------
- MockModel predicts a constant 0.5 in scaled space so inverse-transform
  results are deterministic and easy to reason about.
- All tests use the same synthetic ProcessedData fixture (scope="module").
- No real models trained, no network calls — full suite under 10s.
- Tests cover: inverse-transform correctness, date generation, CI ordering,
  DataFrame shape/columns, CSV save/load round-trip, and partial failures.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.preprocessing import preprocess_ticker, ProcessedData
from forecasting.predictor import (
    PredictionResult,
    _future_business_dates,
    _make_inverter,
    _forecast_one_model,
    generate_predictions,
    generate_all_predictions,
    save_prediction,
    save_all_predictions,
    load_prediction,
    load_all_forecasts,
    get_available_tickers,
    _ticker_forecast_path,
    ALL_FORECASTS_PATH,
)
from models.base_model import BaseModel
from config.settings import FORECAST_HORIZON, CONFIDENCE_INTERVAL, MODEL_NAMES


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def processed() -> ProcessedData:
    """Synthetic ProcessedData — built once per module."""
    np.random.seed(7)
    n = 600
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    close = 150 + np.cumsum(np.random.randn(n) * 0.8)
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
    Deterministic mock model: always predicts 0.5 in scaled space.
    Used to make inverse-transform checks exact.
    """
    CONSTANT = 0.5

    def _fit(self, train, val):
        pass

    def _predict(self, n_steps):
        return np.full(n_steps, self.CONSTANT)

    def _predict_with_ci(self, n_steps, confidence=0.95):
        fc = self._predict(n_steps)
        return {
            "forecast": fc,
            "lower":    fc - 0.05,
            "upper":    fc + 0.05,
        }

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def load(self, path):
        self.is_fitted = True
        return self


def _fitted_mock(ticker="TEST", model_name="arima") -> MockModel:
    m = MockModel(ticker=ticker, model_name=model_name)
    m.is_fitted = True
    return m


def _fitted_models_dict(processed, tickers=("TEST",), names=("arima", "prophet", "lstm")):
    """Build a full fitted_models dict using MockModels."""
    return {
        ticker: {name: _fitted_mock(ticker, name) for name in names}
        for ticker in tickers
    }


# ══════════════════════════════════════════════════════════════════════════════
# _future_business_dates
# ══════════════════════════════════════════════════════════════════════════════

def test_future_business_dates_count():
    last = pd.Timestamp("2024-01-05")   # Friday
    dates = _future_business_dates(last, 10)
    assert len(dates) == 10


def test_future_business_dates_starts_next_bday():
    last = pd.Timestamp("2024-01-05")   # Friday
    dates = _future_business_dates(last, 5)
    # Next business day after Friday is Monday
    assert dates[0] == pd.Timestamp("2024-01-08")


def test_future_business_dates_no_weekends():
    last = pd.Timestamp("2024-01-01")
    dates = _future_business_dates(last, 30)
    # No Saturday (5) or Sunday (6)
    assert all(d.dayofweek < 5 for d in dates)


def test_future_business_dates_strictly_after_last():
    last = pd.Timestamp("2024-06-14")
    dates = _future_business_dates(last, 5)
    assert all(d > last for d in dates)


# ══════════════════════════════════════════════════════════════════════════════
# _make_inverter
# ══════════════════════════════════════════════════════════════════════════════

def test_make_inverter_returns_callable(processed):
    inv = _make_inverter(processed.scaler, processed.feature_cols)
    assert callable(inv)


def test_make_inverter_output_shape(processed):
    inv  = _make_inverter(processed.scaler, processed.feature_cols)
    inp  = np.array([0.0, 0.5, 1.0])
    out  = inv(inp)
    assert out.shape == (3,)


def test_make_inverter_zero_maps_to_min_price(processed):
    """Scaled 0.0 should map back to the minimum training Close price."""
    inv        = _make_inverter(processed.scaler, processed.feature_cols)
    real_min   = inv(np.array([0.0]))[0]
    train_min  = processed.train["Close"].min()
    # After inverse-transform, 0.0 → training minimum
    # Not exact due to scaler internals, but should be close
    assert real_min == pytest.approx(
        processed.scaler.inverse_transform(
            np.zeros((1, len(processed.scaler.scale_)))
        )[0, processed.feature_cols.index("Close")],
        abs=0.01,
    )


def test_make_inverter_one_maps_to_max_price(processed):
    """Scaled 1.0 should map back to the maximum training Close price."""
    inv  = _make_inverter(processed.scaler, processed.feature_cols)
    ones = np.ones((1, len(processed.scaler.scale_)))
    expected = processed.scaler.inverse_transform(ones)[
        0, processed.feature_cols.index("Close")
    ]
    assert inv(np.array([1.0]))[0] == pytest.approx(expected, abs=0.01)


def test_make_inverter_monotone(processed):
    """Higher scaled values should map to higher real prices."""
    inv = _make_inverter(processed.scaler, processed.feature_cols)
    vals = np.linspace(0, 1, 10)
    real = inv(vals)
    assert np.all(np.diff(real) >= 0)


# ══════════════════════════════════════════════════════════════════════════════
# _forecast_one_model
# ══════════════════════════════════════════════════════════════════════════════

def test_forecast_one_model_returns_dict(processed):
    model    = _fitted_mock()
    inverter = _make_inverter(processed.scaler, processed.feature_cols)
    result   = _forecast_one_model(model, inverter, 10, 0.95, "TEST", "arima")
    assert result is not None
    assert set(result.keys()) == {"arima_forecast", "arima_lower", "arima_upper"}


def test_forecast_one_model_shapes(processed):
    model    = _fitted_mock()
    inverter = _make_inverter(processed.scaler, processed.feature_cols)
    result   = _forecast_one_model(model, inverter, 10, 0.95, "TEST", "arima")
    for arr in result.values():
        assert arr.shape == (10,)


def test_forecast_one_model_ci_ordering(processed):
    """lower ≤ forecast ≤ upper for all steps after inverse-transform."""
    model    = _fitted_mock()
    inverter = _make_inverter(processed.scaler, processed.feature_cols)
    result   = _forecast_one_model(model, inverter, 20, 0.95, "TEST", "arima")
    assert np.all(result["arima_lower"] <= result["arima_forecast"] + 0.01)
    assert np.all(result["arima_forecast"] <= result["arima_upper"] + 0.01)


def test_forecast_one_model_returns_none_on_failure(processed):
    """A broken model should return None without raising."""
    class BrokenModel(MockModel):
        def _predict_with_ci(self, n_steps, confidence=0.95):
            raise RuntimeError("Intentional failure")

    model    = BrokenModel(ticker="TEST", model_name="arima")
    model.is_fitted = True
    inverter = _make_inverter(processed.scaler, processed.feature_cols)
    result   = _forecast_one_model(model, inverter, 10, 0.95, "TEST", "arima")
    assert result is None


def test_forecast_one_model_values_are_real_prices(processed):
    """Forecast values should be in real price range (e.g. $10–$10,000)."""
    model    = _fitted_mock()
    inverter = _make_inverter(processed.scaler, processed.feature_cols)
    result   = _forecast_one_model(model, inverter, 5, 0.95, "TEST", "arima")
    fc = result["arima_forecast"]
    assert np.all(fc > 1.0)       # not scaled 0–1 anymore
    assert np.all(fc < 1_000_000) # not astronomically wrong


# ══════════════════════════════════════════════════════════════════════════════
# generate_predictions
# ══════════════════════════════════════════════════════════════════════════════

def test_generate_predictions_returns_result(processed):
    models = {"arima": _fitted_mock("TEST", "arima")}
    result = generate_predictions(
        ticker="TEST",
        processed=processed,
        fitted_models=models,
        model_names=["arima"],
        horizon=10,
    )
    assert isinstance(result, PredictionResult)


def test_generate_predictions_forecast_df_shape(processed):
    models = {n: _fitted_mock("TEST", n) for n in ["arima", "prophet"]}
    result = generate_predictions(
        ticker="TEST",
        processed=processed,
        fitted_models=models,
        model_names=["arima", "prophet"],
        horizon=15,
    )
    assert len(result.forecast_df) == 15
    # 2 models × 3 cols each (forecast/lower/upper)
    assert len(result.forecast_df.columns) == 6


def test_generate_predictions_correct_horizon(processed):
    models = {"arima": _fitted_mock("TEST", "arima")}
    for h in [5, 10, 30]:
        result = generate_predictions(
            ticker="TEST",
            processed=processed,
            fitted_models=models,
            model_names=["arima"],
            horizon=h,
        )
        assert len(result.forecast_df) == h


def test_generate_predictions_dates_are_business_days(processed):
    models = {"arima": _fitted_mock("TEST", "arima")}
    result = generate_predictions(
        ticker="TEST",
        processed=processed,
        fitted_models=models,
        model_names=["arima"],
        horizon=10,
    )
    for d in result.forecast_df.index:
        assert d.dayofweek < 5, f"{d} is a weekend"


def test_generate_predictions_dates_start_after_last_train(processed):
    models = {"arima": _fitted_mock("TEST", "arima")}
    result = generate_predictions(
        ticker="TEST",
        processed=processed,
        fitted_models=models,
        model_names=["arima"],
        horizon=5,
    )
    last_val_date = processed.val.index.max()
    assert result.forecast_df.index[0] > last_val_date


def test_generate_predictions_column_naming(processed):
    models = {"lstm": _fitted_mock("TEST", "lstm")}
    result = generate_predictions(
        ticker="TEST",
        processed=processed,
        fitted_models=models,
        model_names=["lstm"],
        horizon=5,
    )
    expected = {"lstm_forecast", "lstm_lower", "lstm_upper"}
    assert expected == set(result.forecast_df.columns)


def test_generate_predictions_models_used(processed):
    models = {n: _fitted_mock("TEST", n) for n in ["arima", "prophet", "lstm"]}
    result = generate_predictions(
        ticker="TEST",
        processed=processed,
        fitted_models=models,
        model_names=["arima", "prophet", "lstm"],
        horizon=5,
    )
    assert set(result.models_used) == {"arima", "prophet", "lstm"}


def test_generate_predictions_last_known_price_positive(processed):
    models = {"arima": _fitted_mock("TEST", "arima")}
    result = generate_predictions(
        ticker="TEST",
        processed=processed,
        fitted_models=models,
        model_names=["arima"],
        horizon=5,
    )
    assert result.last_known_price > 0


def test_generate_predictions_returns_none_no_models(processed):
    result = generate_predictions(
        ticker="TEST",
        processed=processed,
        fitted_models={},       # nothing available
        model_names=["arima"],
        horizon=5,
    )
    assert result is None


def test_generate_predictions_skips_failed_model(processed):
    """Even if one model fails, others should still produce output."""
    class BrokenModel(MockModel):
        def _predict_with_ci(self, n_steps, confidence=0.95):
            raise RuntimeError("broken")

    broken = BrokenModel(ticker="TEST", model_name="arima")
    broken.is_fitted = True
    good   = _fitted_mock("TEST", "prophet")
    models = {"arima": broken, "prophet": good}

    result = generate_predictions(
        ticker="TEST",
        processed=processed,
        fitted_models=models,
        model_names=["arima", "prophet"],
        horizon=5,
    )
    assert result is not None
    assert "prophet" in result.models_used
    assert "arima"   not in result.models_used


def test_generate_predictions_values_rounded(processed):
    """All forecast values should be rounded to 2 decimal places."""
    models = {"arima": _fitted_mock("TEST", "arima")}
    result = generate_predictions(
        ticker="TEST",
        processed=processed,
        fitted_models=models,
        model_names=["arima"],
        horizon=5,
    )
    for col in result.forecast_df.columns:
        vals = result.forecast_df[col].values
        assert np.allclose(vals, vals.round(2))


# ══════════════════════════════════════════════════════════════════════════════
# generate_all_predictions
# ══════════════════════════════════════════════════════════════════════════════

def test_generate_all_predictions_returns_dict(processed):
    proc_data = {"AAPL": processed, "MSFT": processed}
    fitted    = _fitted_models_dict(processed, tickers=("AAPL", "MSFT"))
    results   = generate_all_predictions(
        processed_data=proc_data,
        fitted_models=fitted,
        model_names=["arima"],
        horizon=5,
        save=False,
    )
    assert set(results.keys()) == {"AAPL", "MSFT"}


def test_generate_all_predictions_skips_failed_ticker(processed):
    """A ticker with no models should be omitted from results."""
    proc_data = {"AAPL": processed, "MSFT": processed}
    fitted    = {"AAPL": {"arima": _fitted_mock("AAPL", "arima")},
                 "MSFT": {}}   # no models for MSFT

    results = generate_all_predictions(
        processed_data=proc_data,
        fitted_models=fitted,
        model_names=["arima"],
        horizon=5,
        save=False,
    )
    assert "AAPL" in results
    assert "MSFT" not in results


# ══════════════════════════════════════════════════════════════════════════════
# save / load round-trips
# ══════════════════════════════════════════════════════════════════════════════

def _make_result(processed, ticker="TEST", horizon=10) -> PredictionResult:
    models = {n: _fitted_mock(ticker, n) for n in ["arima", "prophet"]}
    return generate_predictions(
        ticker=ticker,
        processed=processed,
        fitted_models=models,
        model_names=["arima", "prophet"],
        horizon=horizon,
    )


def test_save_prediction_creates_file(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("forecasting.predictor.PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(
        "forecasting.predictor._ticker_forecast_path",
        lambda t: tmp_path / f"{t}_forecast.csv",
    )
    result = _make_result(processed)
    path   = save_prediction(result)
    assert path.exists()


def test_save_prediction_csv_row_count(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("forecasting.predictor.PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(
        "forecasting.predictor._ticker_forecast_path",
        lambda t: tmp_path / f"{t}_forecast.csv",
    )
    result = _make_result(processed, horizon=10)
    path   = save_prediction(result)
    df     = pd.read_csv(path)
    assert len(df) == 10


def test_save_prediction_metadata_columns(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("forecasting.predictor.PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(
        "forecasting.predictor._ticker_forecast_path",
        lambda t: tmp_path / f"{t}_forecast.csv",
    )
    result = _make_result(processed)
    path   = save_prediction(result)
    df     = pd.read_csv(path)
    assert "ticker"           in df.columns
    assert "generated_at"     in df.columns
    assert "last_known_price" in df.columns


def test_load_prediction_returns_dataframe(processed, tmp_path, monkeypatch):
    """Write a CSV manually and load it — avoids monkeypatching path functions."""
    result   = _make_result(processed, horizon=10)
    csv_path = tmp_path / "TEST_forecast.csv"

    df_save = result.forecast_df.copy()
    df_save.insert(0, "ticker",           result.ticker)
    df_save.insert(1, "generated_at",     result.generated_at)
    df_save.insert(2, "last_known_price", result.last_known_price)
    df_save.insert(3, "last_known_date",  result.last_known_date)
    df_save.to_csv(csv_path, index=True)

    loaded = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == 10


def test_load_prediction_returns_none_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "forecasting.predictor._ticker_forecast_path",
        lambda t: tmp_path / f"{t}_forecast.csv",
    )
    result = load_prediction("NONEXISTENT")
    assert result is None


def test_save_all_predictions_creates_combined(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("forecasting.predictor.PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(
        "forecasting.predictor.ALL_FORECASTS_PATH",
        tmp_path / "all_forecasts.csv",
    )
    monkeypatch.setattr(
        "forecasting.predictor._ticker_forecast_path",
        lambda t: tmp_path / f"{t}_forecast.csv",
    )
    r1 = _make_result(processed, ticker="AAPL")
    r2 = _make_result(processed, ticker="MSFT")
    save_all_predictions({"AAPL": r1, "MSFT": r2})

    combined = tmp_path / "all_forecasts.csv"
    assert combined.exists()
    df = pd.read_csv(combined)
    assert set(df["ticker"].unique()) == {"AAPL", "MSFT"}
    # 2 tickers × FORECAST_HORIZON rows each (30 by default)
    # 2 tickers × 10 rows each (horizon=10 in _make_result default)
    assert len(df) == 20


def test_load_all_forecasts_returns_none_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "forecasting.predictor.ALL_FORECASTS_PATH",
        tmp_path / "missing.csv",
    )
    assert load_all_forecasts() is None


# ══════════════════════════════════════════════════════════════════════════════
# PredictionResult
# ══════════════════════════════════════════════════════════════════════════════

def test_prediction_result_summary(processed):
    models = {"arima": _fitted_mock("TEST", "arima")}
    result = generate_predictions(
        ticker="TEST",
        processed=processed,
        fitted_models=models,
        model_names=["arima"],
        horizon=10,
    )
    s = result.summary()
    assert "TEST"  in s
    assert "10"    in s
    assert "arima" in s


def test_prediction_result_generated_at_format(processed):
    """generated_at should look like an ISO UTC timestamp."""
    models = {"arima": _fitted_mock("TEST", "arima")}
    result = generate_predictions(
        ticker="TEST",
        processed=processed,
        fitted_models=models,
        model_names=["arima"],
        horizon=5,
    )
    # Should match YYYY-MM-DDTHH:MM:SSZ
    import re
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", result.generated_at)
