"""
tests/test_s3_client.py
========================
Unit tests for storage/s3_client.py.

All boto3 calls are mocked — no AWS credentials or real S3 needed.
Tests cover: availability check, upload/download, graceful degradation,
batch operations, UploadResult/DownloadResult correctness.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from storage.s3_client import (
    S3Client,
    UploadResult,
    DownloadResult,
    get_client,
    upload_predictions,
    download_predictions,
    s3_available,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def client() -> S3Client:
    """Fresh S3Client with no cached state."""
    c = S3Client(bucket_name="test-bucket", region="us-east-1")
    c._client    = None
    c._available = None
    return c


@pytest.fixture
def mock_boto3_client():
    """Patch boto3.client to return a MagicMock (boto3 imported inside method)."""
    mock_s3 = MagicMock()
    with patch("boto3.client", return_value=mock_s3):
        yield mock_s3


@pytest.fixture
def tmp_predictions(tmp_path) -> Path:
    """Write a handful of fake prediction CSVs into a temp directory."""
    (tmp_path / "AAPL_forecast.csv").write_text(
        "date,arima_forecast\n2026-01-01,150.0\n"
    )
    (tmp_path / "MSFT_forecast.csv").write_text(
        "date,arima_forecast\n2026-01-01,300.0\n"
    )
    (tmp_path / "all_forecasts.csv").write_text(
        "date,ticker,arima_forecast\n2026-01-01,AAPL,150.0\n"
    )
    (tmp_path / "benchmark_results.csv").write_text(
        "ticker,model,RMSE\nAAPL,arima,3.5\n"
    )
    return tmp_path


# ══════════════════════════════════════════════════════════════════════════════
# UploadResult / DownloadResult
# ══════════════════════════════════════════════════════════════════════════════

def test_upload_result_defaults():
    r = UploadResult(success=True)
    assert r.files_uploaded == 0
    assert r.files_failed   == 0
    assert r.keys_uploaded  == []
    assert r.errors         == []


def test_upload_result_str():
    r = UploadResult(success=True, files_uploaded=3, files_failed=0)
    s = str(r)
    assert "uploaded=3" in s
    assert "failed=0"   in s


def test_download_result_defaults():
    r = DownloadResult(success=True)
    assert r.files_downloaded == 0
    assert r.paths_written    == []


# ══════════════════════════════════════════════════════════════════════════════
# is_available()
# ══════════════════════════════════════════════════════════════════════════════

def test_is_available_false_when_no_credentials(client, monkeypatch):
    monkeypatch.setattr("storage.s3_client.AWS_ACCESS_KEY_ID", "")
    monkeypatch.setattr("storage.s3_client.AWS_SECRET_ACCESS_KEY", "")
    assert client.is_available() is False


def test_is_available_true_when_bucket_reachable(client, mock_boto3_client):
    client._available = None
    mock_boto3_client.head_bucket.return_value = {}
    with patch("storage.s3_client.AWS_ACCESS_KEY_ID", "key"),          patch("storage.s3_client.AWS_SECRET_ACCESS_KEY", "secret"):
        result = client.is_available()
    assert result is True


def test_is_available_false_when_bucket_unreachable(client, mock_boto3_client):
    client._available = None
    mock_boto3_client.head_bucket.side_effect = Exception("NoSuchBucket")
    with patch("storage.s3_client.AWS_ACCESS_KEY_ID", "key"),          patch("storage.s3_client.AWS_SECRET_ACCESS_KEY", "secret"):
        result = client.is_available()
    assert result is False


def test_is_available_cached(client):
    """Second call should not re-check — use cached value."""
    client._available = True
    result = client.is_available()
    assert result is True   # from cache, no boto3 call needed


# ══════════════════════════════════════════════════════════════════════════════
# upload_file()
# ══════════════════════════════════════════════════════════════════════════════

def test_upload_file_success(client, mock_boto3_client, tmp_path):
    test_file = tmp_path / "test.csv"
    test_file.write_text("a,b\n1,2\n")

    ok = client.upload_file(test_file, "predictions/test.csv")

    assert ok is True
    mock_boto3_client.upload_file.assert_called_once()
    call_kwargs = mock_boto3_client.upload_file.call_args
    assert call_kwargs[1]["Key"]    == "predictions/test.csv"
    assert call_kwargs[1]["Bucket"] == "test-bucket"


def test_upload_file_missing_local(client, mock_boto3_client, tmp_path):
    """Uploading a non-existent file returns False without raising."""
    ok = client.upload_file(tmp_path / "nonexistent.csv", "predictions/x.csv")
    assert ok is False
    mock_boto3_client.upload_file.assert_not_called()


def test_upload_file_boto3_error(client, mock_boto3_client, tmp_path):
    """boto3 exception is caught and returns False."""
    test_file = tmp_path / "test.csv"
    test_file.write_text("data")
    mock_boto3_client.upload_file.side_effect = Exception("S3 error")

    ok = client.upload_file(test_file, "predictions/test.csv")
    assert ok is False


def test_upload_file_with_extra_args(client, mock_boto3_client, tmp_path):
    test_file = tmp_path / "test.csv"
    test_file.write_text("data")

    client.upload_file(
        test_file, "predictions/test.csv",
        extra_args={"ContentType": "text/csv"},
    )

    call_kwargs = mock_boto3_client.upload_file.call_args[1]
    assert call_kwargs["ExtraArgs"]["ContentType"] == "text/csv"


# ══════════════════════════════════════════════════════════════════════════════
# download_file()
# ══════════════════════════════════════════════════════════════════════════════

def test_download_file_success(client, mock_boto3_client, tmp_path):
    dest = tmp_path / "downloaded.csv"
    ok   = client.download_file("predictions/AAPL_forecast.csv", dest)

    assert ok is True
    mock_boto3_client.download_file.assert_called_once_with(
        Bucket="test-bucket",
        Key="predictions/AAPL_forecast.csv",
        Filename=str(dest),
    )


def test_download_file_creates_parent_dirs(client, mock_boto3_client, tmp_path):
    dest = tmp_path / "nested" / "dir" / "file.csv"
    client.download_file("predictions/test.csv", dest)
    assert dest.parent.exists()


def test_download_file_boto3_error(client, mock_boto3_client, tmp_path):
    mock_boto3_client.download_file.side_effect = Exception("NoSuchKey")
    ok = client.download_file("predictions/missing.csv", tmp_path / "out.csv")
    assert ok is False


# ══════════════════════════════════════════════════════════════════════════════
# key_exists()
# ══════════════════════════════════════════════════════════════════════════════

def test_key_exists_true(client, mock_boto3_client):
    mock_boto3_client.head_object.return_value = {}
    assert client.key_exists("predictions/AAPL_forecast.csv") is True


def test_key_exists_false_when_exception(client, mock_boto3_client):
    mock_boto3_client.head_object.side_effect = Exception("NoSuchKey")
    assert client.key_exists("predictions/missing.csv") is False


# ══════════════════════════════════════════════════════════════════════════════
# list_keys()
# ══════════════════════════════════════════════════════════════════════════════

def test_list_keys_returns_keys(client, mock_boto3_client):
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "predictions/AAPL_forecast.csv"}]},
        {"Contents": [{"Key": "predictions/MSFT_forecast.csv"}]},
    ]
    mock_boto3_client.get_paginator.return_value = mock_paginator

    keys = client.list_keys(prefix="predictions/")
    assert "predictions/AAPL_forecast.csv" in keys
    assert "predictions/MSFT_forecast.csv" in keys


def test_list_keys_empty_on_error(client, mock_boto3_client):
    mock_boto3_client.get_paginator.side_effect = Exception("Error")
    keys = client.list_keys("predictions/")
    assert keys == []


def test_list_keys_handles_empty_pages(client, mock_boto3_client):
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [{}]   # no Contents key
    mock_boto3_client.get_paginator.return_value = mock_paginator

    keys = client.list_keys()
    assert keys == []


# ══════════════════════════════════════════════════════════════════════════════
# upload_all_predictions()
# ══════════════════════════════════════════════════════════════════════════════

def test_upload_all_predictions_skips_when_unavailable(client):
    client._available = False
    result = client.upload_all_predictions()
    assert result.success       is False
    assert result.files_uploaded == 0
    assert "S3 not available" in result.errors


def test_upload_all_predictions_uploads_csv_files(
    client, mock_boto3_client, tmp_predictions, monkeypatch
):
    client._available = True
    monkeypatch.setattr("storage.s3_client.STOCK_TICKERS", ["AAPL", "MSFT"])

    result = client.upload_all_predictions(predictions_dir=tmp_predictions)

    # 2 ticker CSVs + all_forecasts + benchmark_results = 4
    assert result.files_uploaded >= 4
    assert result.files_failed   == 0
    assert result.success        is True


def test_upload_all_predictions_counts_bytes(
    client, mock_boto3_client, tmp_predictions, monkeypatch
):
    client._available = True
    monkeypatch.setattr("storage.s3_client.STOCK_TICKERS", ["AAPL", "MSFT"])

    result = client.upload_all_predictions(predictions_dir=tmp_predictions)
    assert result.bytes_uploaded > 0


def test_upload_all_predictions_partial_failure(
    client, mock_boto3_client, tmp_predictions, monkeypatch
):
    """If some files fail, success=False but others still upload."""
    client._available = True
    monkeypatch.setattr("storage.s3_client.STOCK_TICKERS", ["AAPL", "MSFT"])

    call_count = {"n": 0}
    original   = mock_boto3_client.upload_file.side_effect

    def flaky(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("First upload fails")
        # rest succeed (default MagicMock returns None)

    mock_boto3_client.upload_file.side_effect = flaky

    result = client.upload_all_predictions(predictions_dir=tmp_predictions)

    assert result.files_failed   >= 1
    assert result.files_uploaded >= 1
    assert result.success        is False


# ══════════════════════════════════════════════════════════════════════════════
# download_all_predictions()
# ══════════════════════════════════════════════════════════════════════════════

def test_download_all_predictions_skips_when_unavailable(client, tmp_path):
    client._available = False
    result = client.download_all_predictions(dest_dir=tmp_path)
    assert result.success          is False
    assert result.files_downloaded == 0


def test_download_all_predictions_success(
    client, mock_boto3_client, tmp_path, monkeypatch
):
    client._available = True
    monkeypatch.setattr("storage.s3_client.STOCK_TICKERS", ["AAPL", "MSFT"])

    result = client.download_all_predictions(
        dest_dir=tmp_path, tickers=["AAPL", "MSFT"]
    )

    # AAPL + MSFT + all_forecasts + benchmark_results = 4
    assert mock_boto3_client.download_file.call_count == 4
    assert result.files_downloaded == 4


def test_download_all_predictions_creates_dest_dir(
    client, mock_boto3_client, tmp_path
):
    client._available = True
    dest = tmp_path / "new_subdir"

    with patch("storage.s3_client.STOCK_TICKERS", ["AAPL"]):
        client.download_all_predictions(dest_dir=dest, tickers=["AAPL"])

    assert dest.exists()


def test_download_all_predictions_all_fail_sets_success_false(
    client, mock_boto3_client, tmp_path, monkeypatch
):
    client._available = True
    monkeypatch.setattr("storage.s3_client.STOCK_TICKERS", ["AAPL"])
    mock_boto3_client.download_file.side_effect = Exception("NoSuchKey")

    result = client.download_all_predictions(
        dest_dir=tmp_path, tickers=["AAPL"]
    )
    assert result.success is False


# ══════════════════════════════════════════════════════════════════════════════
# ensure_bucket_exists()
# ══════════════════════════════════════════════════════════════════════════════

def test_ensure_bucket_exists_when_already_exists(client, mock_boto3_client):
    mock_boto3_client.head_bucket.return_value = {}
    result = client.ensure_bucket_exists()
    assert result is True
    mock_boto3_client.create_bucket.assert_not_called()


def test_ensure_bucket_creates_when_missing(client, mock_boto3_client):
    mock_boto3_client.head_bucket.side_effect = Exception("NoSuchBucket")
    mock_boto3_client.create_bucket.return_value = {}

    result = client.ensure_bucket_exists()

    assert result is True
    mock_boto3_client.create_bucket.assert_called_once()


def test_ensure_bucket_returns_false_on_create_error(client, mock_boto3_client):
    mock_boto3_client.head_bucket.side_effect  = Exception("NoSuchBucket")
    mock_boto3_client.create_bucket.side_effect = Exception("AccessDenied")

    result = client.ensure_bucket_exists()
    assert result is False


# ══════════════════════════════════════════════════════════════════════════════
# Module-level convenience functions
# ══════════════════════════════════════════════════════════════════════════════

def test_get_client_returns_s3_client():
    # Reset the module-level singleton
    import storage.s3_client as mod
    mod._default_client = None
    c = get_client()
    assert isinstance(c, S3Client)


def test_get_client_returns_same_instance():
    import storage.s3_client as mod
    mod._default_client = None
    c1 = get_client()
    c2 = get_client()
    assert c1 is c2


def test_s3_available_returns_bool():
    import storage.s3_client as mod
    mod._default_client = None
    with patch("storage.s3_client.AWS_ACCESS_KEY_ID", ""):
        with patch("storage.s3_client.AWS_SECRET_ACCESS_KEY", ""):
            result = s3_available()
    assert isinstance(result, bool)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline integration — S3 stage in run_pipeline
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_skips_s3_when_unavailable():
    """
    When S3 is not available, the pipeline should still complete successfully
    and log a skip message — it must never fail because of S3.
    """
    from pipeline.run_pipeline import main
    from unittest.mock import patch, MagicMock
    import pandas as pd

    def _mock_processed():
        m = MagicMock()
        m.train = pd.DataFrame({"Close": [0.5] * 50})
        m.val   = pd.DataFrame({"Close": [0.5] * 10})
        m.test  = pd.DataFrame({"Close": [0.5] * 10})
        return m

    with patch("pipeline.run_pipeline.fetch_all_tickers",
               return_value={"AAPL": pd.DataFrame({"Close": [100.0]*300})}), \
         patch("pipeline.run_pipeline.preprocess_all",
               return_value={"AAPL": _mock_processed()}), \
         patch("pipeline.run_pipeline._load_existing_models",
               return_value={"AAPL": {"arima": MagicMock(is_fitted=True)}}), \
         patch("pipeline.run_pipeline.generate_all_predictions",
               return_value={"AAPL": MagicMock()}), \
         patch("pipeline.run_pipeline.S3Client") as MockS3:

        mock_s3_instance      = MagicMock()
        mock_s3_instance.is_available.return_value = False
        MockS3.return_value   = mock_s3_instance

        result = main(
            mode="forecast_only",
            tickers=["AAPL"],
            model_names=["arima"],
        )

    # Pipeline succeeds even though S3 was skipped
    assert result.tickers_forecast == 1
    mock_s3_instance.upload_all_predictions.assert_not_called()


def test_pipeline_uploads_when_s3_available():
    """When S3 is available, upload_all_predictions should be called."""
    from pipeline.run_pipeline import main
    import pandas as pd

    def _mock_processed():
        m = MagicMock()
        m.train = pd.DataFrame({"Close": [0.5] * 50})
        m.val   = pd.DataFrame({"Close": [0.5] * 10})
        m.test  = pd.DataFrame({"Close": [0.5] * 10})
        return m

    with patch("pipeline.run_pipeline.fetch_all_tickers",
               return_value={"AAPL": pd.DataFrame({"Close": [100.0]*300})}), \
         patch("pipeline.run_pipeline.preprocess_all",
               return_value={"AAPL": _mock_processed()}), \
         patch("pipeline.run_pipeline._load_existing_models",
               return_value={"AAPL": {"arima": MagicMock(is_fitted=True)}}), \
         patch("pipeline.run_pipeline.generate_all_predictions",
               return_value={"AAPL": MagicMock()}), \
         patch("pipeline.run_pipeline.S3Client") as MockS3:

        mock_s3_instance = MagicMock()
        mock_s3_instance.is_available.return_value = True
        mock_s3_instance.upload_all_predictions.return_value = UploadResult(
            success=True, files_uploaded=5, bytes_uploaded=10240
        )
        # upload_models is also called by stage_upload — must return UploadResult
        mock_s3_instance.upload_models.return_value = UploadResult(
            success=True, files_uploaded=3, bytes_uploaded=5120
        )
        MockS3.return_value = mock_s3_instance

        result = main(
            mode="forecast_only",
            tickers=["AAPL"],
            model_names=["arima"],
        )

    mock_s3_instance.upload_all_predictions.assert_called_once()
    assert result.success is True
