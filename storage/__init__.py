"""
storage/__init__.py
===================
Clean exports for the storage layer.
"""
from storage.s3_client import (
    S3Client,
    UploadResult,
    DownloadResult,
    get_client,
    upload_predictions,
    download_predictions,
    s3_available,
)

__all__ = [
    "S3Client",
    "UploadResult",
    "DownloadResult",
    "get_client",
    "upload_predictions",
    "download_predictions",
    "s3_available",
]
