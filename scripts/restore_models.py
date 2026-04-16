"""
scripts/restore_models.py
=========================
Downloads saved model artifacts from S3 to local disk.

Used by GitHub Actions before running the pipeline in forecast_only mode.
GitHub Actions containers are ephemeral — models don't persist between runs.
This script restores them from S3 so retraining can be skipped.

Writes a sentinel file `.models_restored` if at least one model file was
downloaded, so the workflow YAML can detect whether restore succeeded.

Usage:
    python3 scripts/restore_models.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config.logging_config import get_logger
from config.settings import SAVED_MODELS_DIR
from storage.s3_client import S3Client

logger = get_logger(__name__)


def restore_models() -> int:
    """
    Download all model files from S3 into predictions/models/.

    Returns the number of files downloaded.
    """
    client = S3Client()

    if not client.is_available():
        logger.warning("S3 not available — skipping model restore")
        return 0

    s3 = client._get_client()
    if s3 is None:
        logger.error("Could not create S3 client")
        return 0

    prefix          = "models/"
    files_restored  = 0

    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages     = paginator.paginate(Bucket=client.bucket_name, Prefix=prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                key      = obj["Key"]
                rel_path = key[len(prefix):]          # strip "models/" prefix
                local    = SAVED_MODELS_DIR / rel_path
                local.parent.mkdir(parents=True, exist_ok=True)

                ok = client.download_file(key, local)
                if ok:
                    files_restored += 1
                    logger.debug("Restored: %s", local)

    except Exception as exc:
        logger.warning("Model restore encountered error: %s", exc)

    logger.info("Model restore complete — %d files downloaded", files_restored)
    return files_restored


if __name__ == "__main__":
    n = restore_models()
    print(f"Restored {n} model files from S3")

    # Write sentinel file so workflow YAML can detect success
    sentinel = _ROOT / ".models_restored"
    if n > 0:
        sentinel.write_text(str(n))
        print(f"✅ Sentinel written: {sentinel}")
    else:
        # Remove sentinel if it exists from a previous run
        if sentinel.exists():
            sentinel.unlink()
        print("ℹ️  No models restored — full training will run")