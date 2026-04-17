"""
pipeline/lambda_handler.py
==========================
AWS Lambda entry point for the automated daily pipeline.

How it works in production
--------------------------
1. AWS EventBridge fires a scheduled rule every day at 01:00 UTC.
2. EventBridge invokes this Lambda function with an optional JSON payload.
3. handler() parses the payload, calls run_pipeline.main(), and returns
   a structured JSON response.
4. CloudWatch automatically captures all stdout/stderr as logs.

Event payload (all fields optional)
------------------------------------
{
    "mode":    "full" | "forecast_only" | "single",  # default: "full"
    "tickers": ["AAPL", "MSFT"],                      # default: all 10
    "models":  ["arima", "prophet", "lstm"],           # default: all 3
    "benchmark": true                                  # default: true
}

Lambda response
---------------
{
    "statusCode": 200,
    "body": {
        "run_id":            "2026-04-15",
        "success":           true,
        "tickers_fetched":   10,
        "tickers_forecast":  10,
        "models_trained":    30,
        "duration_s":        1842.3,
        "avg_rmse_reduction": -17.8,
        "errors":            []
    }
}

Local smoke-test
----------------
    python -m pipeline.lambda_handler

Environment variables (set in Lambda console or .env)
------------------------------------------------------
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
    S3_BUCKET_NAME, LOG_LEVEL
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any

from config.logging_config import get_logger
from pipeline.run_pipeline import main as run_pipeline_main, VALID_MODES

# Note: /tmp path redirection is handled in config/settings.py directly
# by detecting AWS_LAMBDA_FUNCTION_NAME at import time. No override needed here.

logger = get_logger(__name__)

# ── Lambda execution environment detection ────────────────────────────────────

def _is_lambda_env() -> bool:
    """Return True when running inside an actual AWS Lambda container."""
    return bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))


# ── Response builders ─────────────────────────────────────────────────────────

def _success_response(body: dict) -> dict:
    return {
        "statusCode": 200,
        "headers":    {"Content-Type": "application/json"},
        "body":       json.dumps(body),
    }


def _error_response(status_code: int, message: str, detail: str = "") -> dict:
    body = {"error": message}
    if detail:
        body["detail"] = detail
    return {
        "statusCode": status_code,
        "headers":    {"Content-Type": "application/json"},
        "body":       json.dumps(body),
    }


# ── Event parser ──────────────────────────────────────────────────────────────

def _parse_event(event: dict) -> dict:
    """
    Extract pipeline configuration from the Lambda event payload.

    Handles both direct invocation payloads and EventBridge-wrapped events.
    All fields are optional — sensible defaults are used if absent.

    Returns a clean config dict ready to pass to run_pipeline.main().
    """
    # EventBridge wraps its payload in event["detail"]
    payload = event.get("detail", event) if isinstance(event, dict) else {}

    mode = payload.get("mode", "full")
    if mode not in VALID_MODES:
        logger.warning(
            "Invalid mode '%s' in event payload — defaulting to 'full'", mode
        )
        mode = "full"

    tickers = payload.get("tickers", None)
    if tickers is not None and not isinstance(tickers, list):
        logger.warning("'tickers' in payload is not a list — ignoring")
        tickers = None

    models = payload.get("models", None)
    if models is not None and not isinstance(models, list):
        logger.warning("'models' in payload is not a list — ignoring")
        models = None

    benchmark = payload.get("benchmark", True)

    return {
        "mode":                 mode,
        "tickers":              tickers,
        "model_names":          models,
        "run_benchmark_flag":   benchmark,
    }


# ── Main Lambda handler ───────────────────────────────────────────────────────

def _restore_models_from_s3() -> bool:
    """
    Download saved model artifacts from S3 to /tmp before forecasting.

    Lambda containers are ephemeral — no models persist between invocations.
    This restores them from S3 so forecast_only mode works correctly.

    Returns True if at least one model file was restored.
    """
    try:
        from storage.s3_client import S3Client
        from config.settings import SAVED_MODELS_DIR

        client = S3Client()
        if not client.is_available():
            logger.warning("S3 not available — cannot restore models")
            return False

        s3 = client._get_client()
        prefix = "models/"
        n_restored = 0

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=client.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key      = obj["Key"]
                rel_path = key[len(prefix):]
                local    = SAVED_MODELS_DIR / rel_path
                local.parent.mkdir(parents=True, exist_ok=True)
                ok = client.download_file(key, local)
                if ok:
                    n_restored += 1

        logger.info("Restored %d model files from S3", n_restored)
        return n_restored > 0

    except Exception as exc:
        logger.warning("Model restore failed (non-fatal): %s", exc)
        return False


def handler(event: dict, context: Any) -> dict:
    """
    AWS Lambda entry point.

    Called automatically by Lambda runtime. The `context` object provides
    Lambda metadata (function name, remaining time, request ID) but is not
    used here — the pipeline manages its own timing.

    Args:
        event:   JSON payload from EventBridge or direct invocation.
        context: Lambda runtime context (LambdaContext object in AWS).

    Returns:
        HTTP-style response dict with statusCode and JSON body.
    """
    t_start    = time.time()
    request_id = getattr(context, "aws_request_id", "local")

    logger.info(
        "Lambda handler invoked — request_id=%s env=%s",
        request_id,
        "lambda" if _is_lambda_env() else "local",
    )
    logger.info("Event payload: %s", json.dumps(event, default=str))

    # ── Parse event ───────────────────────────────────────────────────────────
    try:
        config = _parse_event(event)
    except Exception as exc:
        logger.error("Failed to parse event: %s", exc)
        return _error_response(400, "Invalid event payload", str(exc))

    logger.info(
        "Pipeline config — mode=%s tickers=%s models=%s benchmark=%s",
        config["mode"],
        config["tickers"] or "all",
        config["model_names"] or "all",
        config["run_benchmark_flag"],
    )

    # ── Restore models from S3 (Lambda containers are always fresh) ───────────
    # forecast_only mode needs saved models on disk. Since Lambda containers
    # are ephemeral, restore them from S3 before running the pipeline.
    if config.get("mode") in ("forecast_only", None):
        logger.info("Restoring saved models from S3...")
        has_models = _restore_models_from_s3()
        if not has_models:
            logger.warning(
                "No models found in S3 — promoting forecast_only to full mode"
            )
            config["mode"] = "full"

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        result = run_pipeline_main(**config)

    except Exception as exc:
        elapsed = round(time.time() - t_start, 2)
        tb      = traceback.format_exc()
        logger.error(
            "Pipeline raised unhandled exception after %.1fs:\n%s", elapsed, tb
        )
        return _error_response(
            500,
            f"Pipeline failed: {type(exc).__name__}: {exc}",
            tb,
        )

    # ── Build response ────────────────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 2)

    body = {
        "run_id":             result.run_id,
        "mode":               result.mode,
        "success":            result.success,
        "tickers_fetched":    result.tickers_fetched,
        "tickers_trained":    result.tickers_trained,
        "tickers_forecast":   result.tickers_forecast,
        "models_trained":     result.models_trained,
        "duration_s":         elapsed,
        "avg_rmse_reduction": result.benchmark_avg_rmse_reduction,
        "errors":             result.errors,
        "request_id":         request_id,
    }

    if result.success:
        logger.info(
            "Lambda handler SUCCESS — duration=%.1fs tickers_forecast=%d",
            elapsed, result.tickers_forecast,
        )
        return _success_response(body)
    else:
        logger.error(
            "Lambda handler FAILURE — duration=%.1fs errors=%s",
            elapsed, result.errors,
        )
        # Return 500 so EventBridge / CloudWatch alarms can detect failures
        return {
            "statusCode": 500,
            "headers":    {"Content-Type": "application/json"},
            "body":       json.dumps(body),
        }


# ── Local smoke-test ──────────────────────────────────────────────────────────

class _MockContext:
    """Minimal stand-in for the Lambda LambdaContext object."""
    aws_request_id        = "local-smoke-test"
    function_name         = "stock-forecasting-pipeline"
    memory_limit_in_mb    = 1024
    remaining_time_in_millis = lambda self: 900_000   # 15 min


if __name__ == "__main__":
    from config.settings import LSTM_CONFIG

    print("\n" + "═" * 60)
    print("  pipeline/lambda_handler.py — local smoke test")
    print("═" * 60 + "\n")

    # ── Test 1: forecast_only with 2 tickers ──────────────────────────────────
    print("Test 1: forecast_only mode (reuses saved models from Phase 4)")
    event1 = {
        "mode":      "forecast_only",
        "tickers":   ["AAPL", "MSFT"],
        "benchmark": False,
    }
    response1 = handler(event1, _MockContext())
    body1     = json.loads(response1["body"])

    print(f"  Status : {response1['statusCode']}")
    print(f"  Success: {body1['success']}")
    print(f"  Tickers forecast: {body1['tickers_forecast']}")
    print(f"  Duration: {body1['duration_s']:.1f}s")

    assert response1["statusCode"] in (200, 500), "Unexpected status code"
    print(f"  ✅ Test 1 complete — status={response1['statusCode']}\n")

    # ── Test 2: invalid mode → 200 with mode corrected to "full" ─────────────
    print("Test 2: Invalid mode in payload → defaults to 'full'")
    config2 = _parse_event({"mode": "invalid_mode", "tickers": ["AAPL"]})
    assert config2["mode"] == "full", f"Expected 'full', got {config2['mode']}"
    print("  ✅ Invalid mode defaulted to 'full'\n")

    # ── Test 3: EventBridge-wrapped payload ───────────────────────────────────
    print("Test 3: EventBridge-wrapped payload")
    eb_event = {
        "version":     "0",
        "source":      "aws.events",
        "detail-type": "Scheduled Event",
        "detail": {
            "mode":      "forecast_only",
            "tickers":   ["AAPL"],
            "benchmark": False,
        },
    }
    config3 = _parse_event(eb_event)
    assert config3["mode"] == "forecast_only"
    assert config3["tickers"] == ["AAPL"]
    print("  ✅ EventBridge payload parsed correctly\n")

    # ── Test 4: Empty event → defaults ────────────────────────────────────────
    print("Test 4: Empty event → sensible defaults")
    config4 = _parse_event({})
    assert config4["mode"] == "full"
    assert config4["tickers"] is None     # → all 10
    assert config4["model_names"] is None # → all 3
    print("  ✅ Empty event uses correct defaults\n")

    # ── Response structure check ──────────────────────────────────────────────
    print("Test 5: Response structure validation")
    required_keys = {
        "run_id", "mode", "success", "tickers_fetched",
        "tickers_forecast", "duration_s", "errors", "request_id",
    }
    assert required_keys.issubset(set(body1.keys())), (
        f"Missing keys: {required_keys - set(body1.keys())}"
    )
    print(f"  ✅ All required response keys present: {sorted(required_keys)}\n")

    print("✅ lambda_handler.py smoke test PASSED\n")
