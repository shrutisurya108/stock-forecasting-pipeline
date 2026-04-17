"""
storage/s3_client.py
====================
AWS S3 storage client for the forecasting pipeline.

Responsibilities
----------------
- Upload prediction CSVs and benchmark results to S3 after each pipeline run.
- Upload fitted model artifacts so Lambda containers can share model state.
- Download predictions back to disk for the Streamlit dashboard.
- Check bucket reachability and credential validity at startup.

Graceful degradation
--------------------
Every public function catches all exceptions and returns a result object
instead of raising. If AWS credentials are missing or S3 is unreachable,
the pipeline logs a warning and continues — local-only operation is fully
supported without an AWS account.

Bucket structure
----------------
stock-forecasting-pipeline/           ← S3_BUCKET_NAME
├── predictions/
│   ├── AAPL_forecast.csv
│   ├── all_forecasts.csv
│   ├── benchmark_results.csv
│   └── benchmark_summary.json
└── models/
    ├── AAPL/arima/arima_model.pkl
    ├── AAPL/lstm/lstm_model.pt
    └── ...

Usage
-----
    from storage.s3_client import S3Client
    client = S3Client()
    if client.is_available():
        client.upload_all_predictions()
        client.upload_benchmark()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.logging_config import get_logger
from config.settings import (
    AWS_ACCESS_KEY_ID,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    PREDICTIONS_DIR,
    S3_BUCKET_NAME,
    S3_MODELS_PREFIX,
    S3_PREDICTIONS_PREFIX,
    SAVED_MODELS_DIR,
    STOCK_TICKERS,
)

logger = get_logger(__name__)


# ── Upload result container ───────────────────────────────────────────────────

@dataclass
class UploadResult:
    """Outcome of a batch upload operation."""
    success:        bool
    files_uploaded: int              = 0
    files_failed:   int              = 0
    keys_uploaded:  list[str]        = field(default_factory=list)
    errors:         list[str]        = field(default_factory=list)
    bytes_uploaded: int              = 0

    def __str__(self) -> str:
        return (
            f"UploadResult(success={self.success}, "
            f"uploaded={self.files_uploaded}, "
            f"failed={self.files_failed}, "
            f"bytes={self.bytes_uploaded:,})"
        )


@dataclass
class DownloadResult:
    """Outcome of a batch download operation."""
    success:          bool
    files_downloaded: int       = 0
    files_failed:     int       = 0
    paths_written:    list[str] = field(default_factory=list)
    errors:           list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"DownloadResult(success={self.success}, "
            f"downloaded={self.files_downloaded}, "
            f"failed={self.files_failed})"
        )


# ── S3 Client ─────────────────────────────────────────────────────────────────

class S3Client:
    """
    Thin wrapper around boto3 S3 for the forecasting pipeline.

    Instantiate once and reuse — the boto3 session is created lazily on
    first use so importing this module never raises even if boto3 is not
    installed or credentials are absent.

    Args:
        bucket_name: S3 bucket name. Defaults to S3_BUCKET_NAME from settings.
        region:      AWS region. Defaults to AWS_REGION from settings.
    """

    def __init__(
        self,
        bucket_name: str = S3_BUCKET_NAME,
        region:      str = AWS_REGION,
    ):
        self.bucket_name = bucket_name
        self.region      = region
        self._client     = None   # lazy-initialised
        self._available: Optional[bool] = None   # cached after first check

    # ── boto3 client (lazy) ───────────────────────────────────────────────────

    def _get_client(self):
        """Return (or create) the boto3 S3 client."""
        if self._client is not None:
            return self._client
        try:
            import boto3
            import os
            # In Lambda, credentials come from the IAM role automatically.
            # Only pass explicit credentials when running locally with .env.
            if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
                # Lambda environment — use role-based credentials
                self._client = boto3.client("s3", region_name=self.region)
            else:
                # Local environment — use explicit credentials from .env
                self._client = boto3.client(
                    "s3",
                    region_name=self.region,
                    aws_access_key_id=AWS_ACCESS_KEY_ID or None,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY or None,
                )
            return self._client
        except Exception as exc:
            logger.error("Failed to create boto3 S3 client: %s", exc)
            return None

    # ── Availability check ────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Return True if S3 is reachable and credentials are valid.

        Result is cached after the first check so subsequent calls are free.
        Returns False (never raises) if anything goes wrong.
        """
        if self._available is not None:
            return self._available

        import os
        # In Lambda, credentials come from IAM role — skip explicit key check
        in_lambda = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
        if not in_lambda and (not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY):
            logger.warning(
                "AWS credentials not set — S3 upload/download disabled. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env"
            )
            self._available = False
            return False

        client = self._get_client()
        if client is None:
            self._available = False
            return False

        try:
            client.head_bucket(Bucket=self.bucket_name)
            logger.info(
                "S3 available — bucket=%s region=%s",
                self.bucket_name, self.region,
            )
            self._available = True
        except Exception as exc:
            logger.warning(
                "S3 not available (bucket=%s): %s", self.bucket_name, exc
            )
            self._available = False

        return self._available

    # ── Single-file helpers ───────────────────────────────────────────────────

    def upload_file(
        self,
        local_path: Path,
        s3_key:     str,
        extra_args: dict | None = None,
    ) -> bool:
        """
        Upload one local file to S3.

        Args:
            local_path: Path to the local file.
            s3_key:     Destination key inside the bucket.
            extra_args: Optional boto3 ExtraArgs dict (e.g. ContentType).

        Returns:
            True on success, False on any error.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            logger.warning("upload_file: %s does not exist", local_path)
            return False

        client = self._get_client()
        if client is None:
            return False

        try:
            client.upload_file(
                Filename=str(local_path),
                Bucket=self.bucket_name,
                Key=s3_key,
                ExtraArgs=extra_args or {},
            )
            size = local_path.stat().st_size
            logger.debug(
                "Uploaded %s → s3://%s/%s (%d bytes)",
                local_path.name, self.bucket_name, s3_key, size,
            )
            return True
        except Exception as exc:
            logger.error(
                "Failed to upload %s → %s: %s", local_path, s3_key, exc
            )
            return False

    def download_file(
        self,
        s3_key:     str,
        local_path: Path,
    ) -> bool:
        """
        Download one S3 object to a local path.

        Args:
            s3_key:     Source key inside the bucket.
            local_path: Destination local path (parent dirs auto-created).

        Returns:
            True on success, False on any error.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        client = self._get_client()
        if client is None:
            return False

        try:
            client.download_file(
                Bucket=self.bucket_name,
                Key=s3_key,
                Filename=str(local_path),
            )
            logger.debug(
                "Downloaded s3://%s/%s → %s",
                self.bucket_name, s3_key, local_path,
            )
            return True
        except Exception as exc:
            logger.error(
                "Failed to download %s → %s: %s", s3_key, local_path, exc
            )
            return False

    def key_exists(self, s3_key: str) -> bool:
        """Return True if the given S3 key exists in the bucket."""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except Exception:
            return False

    def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all S3 keys under a given prefix.

        Returns an empty list on any error.
        """
        client = self._get_client()
        if client is None:
            return []
        try:
            paginator = client.get_paginator("list_objects_v2")
            keys = []
            for page in paginator.paginate(
                Bucket=self.bucket_name, Prefix=prefix
            ):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return keys
        except Exception as exc:
            logger.error("list_keys failed (prefix=%s): %s", prefix, exc)
            return []

    # ── Batch prediction upload ───────────────────────────────────────────────

    def upload_all_predictions(
        self,
        predictions_dir: Path = PREDICTIONS_DIR,
    ) -> UploadResult:
        """
        Upload all prediction CSVs and the benchmark files to S3.

        Uploads:
          - predictions/{TICKER}_forecast.csv  for every ticker
          - predictions/all_forecasts.csv
          - predictions/benchmark_results.csv
          - predictions/benchmark_summary.json
          - predictions/training_report.json

        Args:
            predictions_dir: Local directory containing the CSV files.

        Returns:
            UploadResult with counts and any error messages.
        """
        if not self.is_available():
            logger.warning("S3 not available — skipping prediction upload")
            return UploadResult(success=False, errors=["S3 not available"])

        result = UploadResult(success=True)
        files_to_upload: list[tuple[Path, str]] = []

        # Per-ticker forecast CSVs
        for ticker in STOCK_TICKERS:
            local = predictions_dir / f"{ticker}_forecast.csv"
            if local.exists():
                key = f"{S3_PREDICTIONS_PREFIX}{ticker}_forecast.csv"
                files_to_upload.append((local, key))

        # Combined and benchmark files
        for filename in [
            "all_forecasts.csv",
            "benchmark_results.csv",
            "benchmark_summary.json",
            "training_report.json",
        ]:
            local = predictions_dir / filename
            if local.exists():
                key = f"{S3_PREDICTIONS_PREFIX}{filename}"
                files_to_upload.append((local, key))

        logger.info(
            "Uploading %d prediction files to s3://%s/%s",
            len(files_to_upload), self.bucket_name, S3_PREDICTIONS_PREFIX,
        )

        for local_path, s3_key in files_to_upload:
            content_type = (
                "text/csv" if local_path.suffix == ".csv" else "application/json"
            )
            ok = self.upload_file(
                local_path, s3_key,
                extra_args={"ContentType": content_type},
            )
            if ok:
                result.files_uploaded += 1
                result.keys_uploaded.append(s3_key)
                result.bytes_uploaded += local_path.stat().st_size
            else:
                result.files_failed += 1
                result.errors.append(f"Failed: {local_path.name}")
                result.success = False

        logger.info(
            "Prediction upload complete — %d uploaded, %d failed, %d bytes",
            result.files_uploaded, result.files_failed, result.bytes_uploaded,
        )
        return result

    # ── Model artifact upload ─────────────────────────────────────────────────

    def upload_models(
        self,
        tickers:      list[str] | None = None,
        model_names:  list[str] | None = None,
        saved_models_dir: Path = SAVED_MODELS_DIR,
    ) -> UploadResult:
        """
        Upload all saved model artifacts to S3.

        Walks the predictions/models/{ticker}/{model}/ directory tree and
        uploads every file found there. This lets Lambda containers share
        fitted models across invocations.

        Args:
            tickers:          Subset of tickers to upload. Defaults to all.
            model_names:      Subset of models to upload. Defaults to all.
            saved_models_dir: Local root of saved models.

        Returns:
            UploadResult with counts.
        """
        if not self.is_available():
            logger.warning("S3 not available — skipping model upload")
            return UploadResult(success=False, errors=["S3 not available"])

        from config.settings import MODEL_NAMES
        tickers     = tickers     or STOCK_TICKERS
        model_names = model_names or MODEL_NAMES

        result = UploadResult(success=True)

        for ticker in tickers:
            for model_name in model_names:
                model_dir = saved_models_dir / ticker / model_name
                if not model_dir.exists():
                    continue

                for file_path in model_dir.iterdir():
                    if file_path.is_file():
                        s3_key = (
                            f"{S3_MODELS_PREFIX}"
                            f"{ticker}/{model_name}/{file_path.name}"
                        )
                        ok = self.upload_file(file_path, s3_key)
                        if ok:
                            result.files_uploaded += 1
                            result.keys_uploaded.append(s3_key)
                            result.bytes_uploaded += file_path.stat().st_size
                        else:
                            result.files_failed += 1
                            result.errors.append(
                                f"Failed: {ticker}/{model_name}/{file_path.name}"
                            )

        if result.files_failed > 0:
            result.success = False

        logger.info(
            "Model upload complete — %d files, %d failed",
            result.files_uploaded, result.files_failed,
        )
        return result

    # ── Prediction download ───────────────────────────────────────────────────

    def download_all_predictions(
        self,
        dest_dir: Path = PREDICTIONS_DIR,
        tickers:  list[str] | None = None,
    ) -> DownloadResult:
        """
        Download prediction CSVs from S3 to local disk.

        Used by the Streamlit dashboard when it starts up to get the
        latest forecasts.

        Args:
            dest_dir: Local directory to write files into.
            tickers:  Subset of tickers to download. Defaults to all.

        Returns:
            DownloadResult with counts.
        """
        if not self.is_available():
            logger.warning("S3 not available — skipping prediction download")
            return DownloadResult(success=False, errors=["S3 not available"])

        tickers = tickers or STOCK_TICKERS
        result  = DownloadResult(success=True)
        dest_dir.mkdir(parents=True, exist_ok=True)

        files_to_download: list[tuple[str, Path]] = []

        for ticker in tickers:
            key   = f"{S3_PREDICTIONS_PREFIX}{ticker}_forecast.csv"
            local = dest_dir / f"{ticker}_forecast.csv"
            files_to_download.append((key, local))

        for filename in ["all_forecasts.csv", "benchmark_results.csv"]:
            key   = f"{S3_PREDICTIONS_PREFIX}{filename}"
            local = dest_dir / filename
            files_to_download.append((key, local))

        logger.info(
            "Downloading %d prediction files from s3://%s/%s",
            len(files_to_download), self.bucket_name, S3_PREDICTIONS_PREFIX,
        )

        for s3_key, local_path in files_to_download:
            ok = self.download_file(s3_key, local_path)
            if ok:
                result.files_downloaded += 1
                result.paths_written.append(str(local_path))
            else:
                result.files_failed += 1
                result.errors.append(f"Failed: {s3_key}")

        if result.files_failed == len(files_to_download):
            result.success = False

        logger.info(
            "Download complete — %d downloaded, %d failed",
            result.files_downloaded, result.files_failed,
        )
        return result

    # ── Bucket management ─────────────────────────────────────────────────────

    def ensure_bucket_exists(self) -> bool:
        """
        Create the S3 bucket if it does not already exist.

        Safe to call multiple times — no-ops if bucket already exists.
        Returns True on success, False on failure.
        """
        client = self._get_client()
        if client is None:
            return False

        try:
            client.head_bucket(Bucket=self.bucket_name)
            logger.info("Bucket already exists: %s", self.bucket_name)
            return True
        except Exception:
            pass   # bucket doesn't exist — create it

        try:
            if self.region == "us-east-1":
                client.create_bucket(Bucket=self.bucket_name)
            else:
                client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={
                        "LocationConstraint": self.region
                    },
                )
            logger.info(
                "Created S3 bucket: %s (region=%s)",
                self.bucket_name, self.region,
            )
            return True
        except Exception as exc:
            logger.error("Failed to create bucket %s: %s", self.bucket_name, exc)
            return False


# ── Module-level convenience functions ────────────────────────────────────────
# These wrap a shared default client so callers don't need to instantiate one.

_default_client: Optional[S3Client] = None


def get_client() -> S3Client:
    """Return the shared default S3Client instance (created lazily)."""
    global _default_client
    if _default_client is None:
        _default_client = S3Client()
    return _default_client


def upload_predictions(predictions_dir: Path = PREDICTIONS_DIR) -> UploadResult:
    """Upload all local prediction files to S3. Convenience wrapper."""
    return get_client().upload_all_predictions(predictions_dir)


def download_predictions(
    dest_dir: Path = PREDICTIONS_DIR,
    tickers: list[str] | None = None,
) -> DownloadResult:
    """Download prediction files from S3. Convenience wrapper."""
    return get_client().download_all_predictions(dest_dir, tickers)


def s3_available() -> bool:
    """Return True if S3 is reachable. Convenience wrapper."""
    return get_client().is_available()


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("\n" + "═" * 60)
    print("  storage/s3_client.py — smoke test")
    print("═" * 60 + "\n")

    client = S3Client()

    # ── Step 1: Check availability ────────────────────────────────────────────
    print("Step 1: Checking S3 availability…")
    available = client.is_available()
    if not available:
        print("  ⚠️  S3 not available — check your .env credentials")
        print("  Remaining tests require a live S3 connection.\n")
        print("  To test locally without AWS, run the unit tests instead:")
        print("  pytest tests/test_s3_client.py -v\n")
        raise SystemExit(0)

    print(f"  ✅ S3 available — bucket={client.bucket_name}\n")

    # ── Step 2: Upload a test file ────────────────────────────────────────────
    print("Step 2: Upload / download / delete round-trip…")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Write a tiny test CSV
        test_file = tmp / "smoke_test.csv"
        test_file.write_text("date,value\n2026-01-01,123.45\n2026-01-02,124.00\n")
        test_key  = "predictions/_smoke_test.csv"

        ok = client.upload_file(test_file, test_key,
                                extra_args={"ContentType": "text/csv"})
        assert ok, "Upload failed"
        print(f"  ✅ Uploaded {test_file.name} → s3://{client.bucket_name}/{test_key}")

        # Verify key exists
        assert client.key_exists(test_key), "key_exists() returned False"
        print(f"  ✅ key_exists() confirmed")

        # Download it back
        downloaded = tmp / "smoke_test_downloaded.csv"
        ok = client.download_file(test_key, downloaded)
        assert ok, "Download failed"
        assert downloaded.read_text() == test_file.read_text(), "Content mismatch"
        print(f"  ✅ Downloaded and content matches original")

        # Cleanup: delete test key
        client._get_client().delete_object(
            Bucket=client.bucket_name, Key=test_key
        )
        assert not client.key_exists(test_key), "Delete failed"
        print(f"  ✅ Deleted test key from S3\n")

    # ── Step 3: Upload real prediction files ──────────────────────────────────
    print("Step 3: Uploading prediction files…")
    upload_result = client.upload_all_predictions()
    print(f"  Files uploaded : {upload_result.files_uploaded}")
    print(f"  Files failed   : {upload_result.files_failed}")
    print(f"  Bytes uploaded : {upload_result.bytes_uploaded:,}")
    if upload_result.files_uploaded > 0:
        print(f"  ✅ Upload succeeded")
    else:
        print(f"  ⚠️  No prediction files found locally — run pipeline first")

    # ── Step 4: List keys ─────────────────────────────────────────────────────
    print("\nStep 4: Listing S3 prediction keys…")
    keys = client.list_keys(prefix=S3_PREDICTIONS_PREFIX)
    if keys:
        for k in keys[:5]:
            print(f"  📄 {k}")
        if len(keys) > 5:
            print(f"  ... and {len(keys) - 5} more")
        print(f"  ✅ {len(keys)} keys found\n")
    else:
        print("  ⚠️  No keys found (bucket may be empty)\n")

    print("✅ S3 smoke test PASSED\n")
