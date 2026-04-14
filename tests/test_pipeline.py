"""
tests/test_pipeline.py
======================
Unit tests for pipeline/run_pipeline.py and pipeline/lambda_handler.py.

Strategy:
- All external I/O (fetch, train, forecast, benchmark) is mocked.
- Tests verify orchestration logic: stage sequencing, mode behaviour,
  error isolation, PipelineResult fields, Lambda response structure.
- Suite completes in under 3 seconds — no network, no ML training.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline.run_pipeline import (
    PipelineResult,
    _parse_args,
    main,
    stage_fetch,
    stage_preprocess,
    VALID_MODES,
    _load_existing_models,
)
from pipeline.lambda_handler import (
    _parse_event,
    _success_response,
    _error_response,
    handler,
    _MockContext,
)
from config.settings import MODEL_NAMES, STOCK_TICKERS


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures & helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mock_processed():
    """Minimal ProcessedData-like mock."""
    m = MagicMock()
    m.train = pd.DataFrame({"Close": [0.5] * 50})
    m.val   = pd.DataFrame({"Close": [0.5] * 10})
    m.test  = pd.DataFrame({"Close": [0.5] * 10})
    return m


def _mock_raw_data(tickers=None):
    tickers = tickers or ["AAPL", "MSFT"]
    return {t: pd.DataFrame({"Close": [100.0] * 300}) for t in tickers}


def _mock_processed_data(tickers=None):
    tickers = tickers or ["AAPL", "MSFT"]
    return {t: _mock_processed() for t in tickers}


def _mock_fitted(tickers=None, models=None):
    tickers = tickers or ["AAPL", "MSFT"]
    models  = models  or ["arima"]
    return {t: {m: MagicMock(is_fitted=True) for m in models} for t in tickers}


def _mock_prediction_result(ticker):
    m = MagicMock()
    m.ticker = ticker
    return m


# ══════════════════════════════════════════════════════════════════════════════
# PipelineResult
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_result_defaults():
    r = PipelineResult(run_id="2026-01-01", mode="full")
    assert r.success is False
    assert r.tickers_fetched == 0
    assert r.errors == []


def test_pipeline_result_to_dict():
    r = PipelineResult(run_id="2026-01-01", mode="full", success=True)
    d = r.to_dict()
    assert d["run_id"]  == "2026-01-01"
    assert d["success"] is True


def test_pipeline_result_summary_lines_contains_mode():
    r = PipelineResult(run_id="2026-01-01", mode="forecast_only")
    lines = r.summary_lines()
    assert any("forecast_only" in l for l in lines)


def test_pipeline_result_summary_shows_status():
    r = PipelineResult(run_id="2026-01-01", mode="full", success=True)
    lines = "\n".join(r.summary_lines())
    assert "SUCCESS" in lines


def test_pipeline_result_summary_shows_failure():
    r = PipelineResult(run_id="2026-01-01", mode="full", success=False)
    lines = "\n".join(r.summary_lines())
    assert "FAILED" in lines


# ══════════════════════════════════════════════════════════════════════════════
# VALID_MODES
# ══════════════════════════════════════════════════════════════════════════════

def test_valid_modes_contains_all_three():
    assert "full"          in VALID_MODES
    assert "forecast_only" in VALID_MODES
    assert "single"        in VALID_MODES


# ══════════════════════════════════════════════════════════════════════════════
# main() — full mode
# ══════════════════════════════════════════════════════════════════════════════

@patch("pipeline.run_pipeline.generate_all_predictions")
@patch("pipeline.run_pipeline.run_benchmark")
@patch("pipeline.run_pipeline.train_all")
@patch("pipeline.run_pipeline.preprocess_all")
@patch("pipeline.run_pipeline.fetch_all_tickers")
def test_main_full_mode_success(
    mock_fetch, mock_preprocess, mock_train,
    mock_benchmark, mock_forecast,
):
    mock_fetch.return_value     = _mock_raw_data()
    mock_preprocess.return_value = _mock_processed_data()

    train_result               = MagicMock()
    train_result.success       = True
    mock_train.return_value    = (_mock_fitted(), [train_result, train_result])

    mock_benchmark.return_value = (
        pd.DataFrame(), {"avg_rmse_reduction_pct": -15.0}
    )
    mock_forecast.return_value  = {
        "AAPL": _mock_prediction_result("AAPL"),
        "MSFT": _mock_prediction_result("MSFT"),
    }

    result = main(mode="full", tickers=["AAPL", "MSFT"], model_names=["arima"])

    assert result.success           is True
    assert result.tickers_fetched   == 2
    assert result.tickers_forecast  == 2
    assert result.mode              == "full"


@patch("pipeline.run_pipeline.generate_all_predictions")
@patch("pipeline.run_pipeline.run_benchmark")
@patch("pipeline.run_pipeline.train_all")
@patch("pipeline.run_pipeline.preprocess_all")
@patch("pipeline.run_pipeline.fetch_all_tickers")
def test_main_full_mode_records_benchmark(
    mock_fetch, mock_preprocess, mock_train,
    mock_benchmark, mock_forecast,
):
    mock_fetch.return_value      = _mock_raw_data()
    mock_preprocess.return_value = _mock_processed_data()
    train_result                 = MagicMock(success=True)
    mock_train.return_value      = (_mock_fitted(), [train_result])
    mock_benchmark.return_value  = (
        pd.DataFrame(), {"avg_rmse_reduction_pct": -18.5}
    )
    mock_forecast.return_value   = {"AAPL": _mock_prediction_result("AAPL")}

    result = main(mode="full", tickers=["AAPL"], model_names=["arima"])

    assert result.benchmark_avg_rmse_reduction == pytest.approx(-18.5)


@patch("pipeline.run_pipeline.fetch_all_tickers")
def test_main_aborts_when_no_data_fetched(mock_fetch):
    mock_fetch.return_value = {}

    result = main(mode="full", tickers=["AAPL"])

    assert result.success is False
    assert any("No data" in e for e in result.errors)


@patch("pipeline.run_pipeline.generate_all_predictions")
@patch("pipeline.run_pipeline.preprocess_all")
@patch("pipeline.run_pipeline.fetch_all_tickers")
def test_main_aborts_when_preprocess_empty(
    mock_fetch, mock_preprocess, mock_forecast
):
    mock_fetch.return_value      = _mock_raw_data()
    mock_preprocess.return_value = {}   # no tickers survived preprocessing

    result = main(mode="full", tickers=["AAPL"])

    assert result.success is False


# ══════════════════════════════════════════════════════════════════════════════
# main() — forecast_only mode
# ══════════════════════════════════════════════════════════════════════════════

@patch("pipeline.run_pipeline.generate_all_predictions")
@patch("pipeline.run_pipeline._load_existing_models")
@patch("pipeline.run_pipeline.preprocess_all")
@patch("pipeline.run_pipeline.fetch_all_tickers")
def test_main_forecast_only_skips_training(
    mock_fetch, mock_preprocess, mock_load, mock_forecast
):
    mock_fetch.return_value      = _mock_raw_data()
    mock_preprocess.return_value = _mock_processed_data()
    mock_load.return_value       = _mock_fitted()
    mock_forecast.return_value   = {
        "AAPL": _mock_prediction_result("AAPL"),
        "MSFT": _mock_prediction_result("MSFT"),
    }

    with patch("pipeline.run_pipeline.train_all") as mock_train:
        result = main(
            mode="forecast_only",
            tickers=["AAPL", "MSFT"],
            model_names=["arima"],
        )
        mock_train.assert_not_called()

    assert result.tickers_forecast == 2


@patch("pipeline.run_pipeline.generate_all_predictions")
@patch("pipeline.run_pipeline._load_existing_models")
@patch("pipeline.run_pipeline.preprocess_all")
@patch("pipeline.run_pipeline.fetch_all_tickers")
def test_main_forecast_only_does_not_benchmark(
    mock_fetch, mock_preprocess, mock_load, mock_forecast
):
    mock_fetch.return_value      = _mock_raw_data(["AAPL"])
    mock_preprocess.return_value = _mock_processed_data(["AAPL"])
    mock_load.return_value       = _mock_fitted(["AAPL"])
    mock_forecast.return_value   = {"AAPL": _mock_prediction_result("AAPL")}

    with patch("pipeline.run_pipeline.run_benchmark") as mock_bench:
        main(mode="forecast_only", tickers=["AAPL"], model_names=["arima"])
        mock_bench.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# main() — single mode
# ══════════════════════════════════════════════════════════════════════════════

@patch("pipeline.run_pipeline.generate_all_predictions")
@patch("pipeline.run_pipeline.run_benchmark")
@patch("pipeline.run_pipeline.train_all")
@patch("pipeline.run_pipeline.preprocess_all")
@patch("pipeline.run_pipeline.fetch_all_tickers")
def test_main_single_mode(
    mock_fetch, mock_preprocess, mock_train,
    mock_benchmark, mock_forecast,
):
    mock_fetch.return_value      = {"AAPL": pd.DataFrame({"Close": [100.0]*300})}
    mock_preprocess.return_value = {"AAPL": _mock_processed()}
    mock_train.return_value      = (
        {"AAPL": {"arima": MagicMock(is_fitted=True)}},
        [MagicMock(success=True)],
    )
    mock_benchmark.return_value  = (pd.DataFrame(), {"avg_rmse_reduction_pct": -10.0})
    mock_forecast.return_value   = {"AAPL": _mock_prediction_result("AAPL")}

    result = main(mode="single", tickers=["AAPL"], model_names=["arima"])

    assert result.tickers_fetched  == 1
    assert result.tickers_forecast == 1


# ══════════════════════════════════════════════════════════════════════════════
# main() — error isolation
# ══════════════════════════════════════════════════════════════════════════════

@patch("pipeline.run_pipeline.generate_all_predictions")
@patch("pipeline.run_pipeline.run_benchmark")
@patch("pipeline.run_pipeline.train_all")
@patch("pipeline.run_pipeline.preprocess_all")
@patch("pipeline.run_pipeline.fetch_all_tickers")
def test_main_benchmark_failure_is_non_fatal(
    mock_fetch, mock_preprocess, mock_train,
    mock_benchmark, mock_forecast,
):
    """A benchmark crash must not abort the forecast stage."""
    mock_fetch.return_value      = _mock_raw_data(["AAPL"])
    mock_preprocess.return_value = _mock_processed_data(["AAPL"])
    mock_train.return_value      = (
        _mock_fitted(["AAPL"]),
        [MagicMock(success=True)],
    )
    mock_benchmark.side_effect   = RuntimeError("Benchmark exploded")
    mock_forecast.return_value   = {"AAPL": _mock_prediction_result("AAPL")}

    result = main(
        mode="full",
        tickers=["AAPL"],
        model_names=["arima"],
        run_benchmark_flag=True,
    )

    # Forecast still ran
    assert result.tickers_forecast == 1
    # Error was recorded but pipeline still succeeded
    assert any("Benchmark" in e for e in result.errors)


@patch("pipeline.run_pipeline.generate_all_predictions")
@patch("pipeline.run_pipeline.run_benchmark")
@patch("pipeline.run_pipeline.train_all")
@patch("pipeline.run_pipeline.preprocess_all")
@patch("pipeline.run_pipeline.fetch_all_tickers")
def test_main_invalid_mode_raises(
    mock_fetch, mock_preprocess, mock_train, mock_benchmark, mock_forecast
):
    with pytest.raises(ValueError, match="Invalid mode"):
        main(mode="invalid_mode", tickers=["AAPL"])


# ══════════════════════════════════════════════════════════════════════════════
# _load_existing_models
# ══════════════════════════════════════════════════════════════════════════════

def test_load_existing_models_returns_dict():
    with patch("pipeline.run_pipeline.load_model", return_value=None):
        result = _load_existing_models(["AAPL"], ["arima"])
    assert "AAPL" in result
    assert isinstance(result["AAPL"], dict)


def test_load_existing_models_skips_missing():
    with patch("pipeline.run_pipeline.load_model", return_value=None):
        result = _load_existing_models(["AAPL"], ["arima", "lstm"])
    # Both missing → empty inner dicts
    assert result["AAPL"] == {}


def test_load_existing_models_includes_loaded():
    mock_model = MagicMock(is_fitted=True)
    with patch("pipeline.run_pipeline.load_model", return_value=mock_model):
        result = _load_existing_models(["AAPL"], ["arima"])
    assert result["AAPL"]["arima"] is mock_model


# ══════════════════════════════════════════════════════════════════════════════
# _parse_event
# ══════════════════════════════════════════════════════════════════════════════

def test_parse_event_empty_uses_defaults():
    config = _parse_event({})
    assert config["mode"]           == "full"
    assert config["tickers"]        is None
    assert config["model_names"]    is None
    assert config["run_benchmark_flag"] is True


def test_parse_event_reads_mode():
    config = _parse_event({"mode": "forecast_only"})
    assert config["mode"] == "forecast_only"


def test_parse_event_invalid_mode_defaults_to_full():
    config = _parse_event({"mode": "nonsense"})
    assert config["mode"] == "full"


def test_parse_event_reads_tickers():
    config = _parse_event({"tickers": ["AAPL", "TSLA"]})
    assert config["tickers"] == ["AAPL", "TSLA"]


def test_parse_event_reads_models():
    config = _parse_event({"models": ["arima", "lstm"]})
    assert config["model_names"] == ["arima", "lstm"]


def test_parse_event_reads_benchmark_flag():
    config = _parse_event({"benchmark": False})
    assert config["run_benchmark_flag"] is False


def test_parse_event_eventbridge_wrapped():
    """EventBridge wraps payload in 'detail' key."""
    event = {
        "source":      "aws.events",
        "detail-type": "Scheduled Event",
        "detail": {
            "mode":    "forecast_only",
            "tickers": ["AAPL"],
        },
    }
    config = _parse_event(event)
    assert config["mode"]    == "forecast_only"
    assert config["tickers"] == ["AAPL"]


def test_parse_event_non_list_tickers_ignored():
    config = _parse_event({"tickers": "AAPL"})   # string, not list
    assert config["tickers"] is None


# ══════════════════════════════════════════════════════════════════════════════
# Response builders
# ══════════════════════════════════════════════════════════════════════════════

def test_success_response_status_code():
    r = _success_response({"key": "val"})
    assert r["statusCode"] == 200


def test_success_response_body_is_json():
    r = _success_response({"run_id": "2026-01-01"})
    body = json.loads(r["body"])
    assert body["run_id"] == "2026-01-01"


def test_error_response_status_code():
    r = _error_response(400, "Bad request")
    assert r["statusCode"] == 400


def test_error_response_body_has_error_key():
    r = _error_response(500, "Internal error", "traceback here")
    body = json.loads(r["body"])
    assert "error"  in body
    assert "detail" in body


# ══════════════════════════════════════════════════════════════════════════════
# handler()
# ══════════════════════════════════════════════════════════════════════════════

@patch("pipeline.lambda_handler.run_pipeline_main")
def test_handler_returns_200_on_success(mock_run):
    mock_run.return_value = PipelineResult(
        run_id="2026-01-01",
        mode="forecast_only",
        success=True,
        tickers_fetched=2,
        tickers_forecast=2,
    )
    response = handler({"mode": "forecast_only"}, _MockContext())
    assert response["statusCode"] == 200


@patch("pipeline.lambda_handler.run_pipeline_main")
def test_handler_returns_500_on_pipeline_failure(mock_run):
    mock_run.return_value = PipelineResult(
        run_id="2026-01-01",
        mode="full",
        success=False,
        errors=["Something went wrong"],
    )
    response = handler({}, _MockContext())
    assert response["statusCode"] == 500


@patch("pipeline.lambda_handler.run_pipeline_main")
def test_handler_returns_500_on_exception(mock_run):
    mock_run.side_effect = RuntimeError("Unexpected crash")
    response = handler({}, _MockContext())
    assert response["statusCode"] == 500


@patch("pipeline.lambda_handler.run_pipeline_main")
def test_handler_response_body_has_required_keys(mock_run):
    mock_run.return_value = PipelineResult(
        run_id="2026-01-01",
        mode="forecast_only",
        success=True,
        tickers_fetched=2,
        tickers_forecast=2,
    )
    response = handler({"mode": "forecast_only"}, _MockContext())
    body     = json.loads(response["body"])

    required = {
        "run_id", "mode", "success",
        "tickers_fetched", "tickers_forecast",
        "duration_s", "errors", "request_id",
    }
    assert required.issubset(set(body.keys()))


@patch("pipeline.lambda_handler.run_pipeline_main")
def test_handler_passes_mode_to_pipeline(mock_run):
    mock_run.return_value = PipelineResult(
        run_id="2026-01-01",
        mode="forecast_only",
        success=True,
        tickers_fetched=1,
        tickers_forecast=1,
    )
    handler({"mode": "forecast_only", "tickers": ["AAPL"]}, _MockContext())

    call_kwargs = mock_run.call_args[1]
    assert call_kwargs["mode"]    == "forecast_only"
    assert call_kwargs["tickers"] == ["AAPL"]


@patch("pipeline.lambda_handler.run_pipeline_main")
def test_handler_invalid_event_payload_uses_defaults(mock_run):
    mock_run.return_value = PipelineResult(
        run_id="2026-01-01",
        mode="full",
        success=True,
        tickers_fetched=10,
        tickers_forecast=10,
    )
    handler({}, _MockContext())   # empty event

    call_kwargs = mock_run.call_args[1]
    assert call_kwargs["mode"]    == "full"
    assert call_kwargs["tickers"] is None
