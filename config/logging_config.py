"""
config/logging_config.py
========================
Centralized logging configuration for the entire pipeline.

Features:
- Rotating file handler  → logs/pipeline.log  (auto-rotates at 10 MB, keeps 5 files)
- Console (StreamHandler) → colour-aware output to stdout
- Per-module named loggers → clean, traceable log lines
- One-call setup: call `get_logger(__name__)` in any module

Usage:
    from config.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Starting ingestion for AAPL")
"""

import logging
import logging.handlers
import sys
from pathlib import Path

# Pull settings without triggering a circular import
# (settings imports Path, not logging_config)
from config.settings import (
    LOGS_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_ROTATION_BYTES,
    LOG_BACKUP_COUNT,
)

# ── Internal sentinel: only configure root logger once ────────────────────────
_CONFIGURED = False


def _build_formatter() -> logging.Formatter:
    """Return a shared Formatter instance."""
    return logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


def _build_file_handler(log_file: Path) -> logging.Handler:
    """
    RotatingFileHandler that writes to logs/pipeline.log.
    Rotates when the file hits LOG_ROTATION_BYTES; keeps LOG_BACKUP_COUNT backups.
    """
    handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=LOG_ROTATION_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(_build_formatter())
    handler.setLevel(logging.DEBUG)   # file captures everything
    return handler


def _build_console_handler() -> logging.Handler:
    """StreamHandler writing to stdout with the shared format."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_build_formatter())
    handler.setLevel(logging.INFO)    # console shows INFO and above
    return handler


def setup_logging() -> None:
    """
    Configure the root logger exactly once.
    Called automatically by `get_logger()` — you rarely need to call this directly.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_file = LOGS_DIR / "pipeline.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)       # root level: let handlers filter

    # Avoid adding duplicate handlers if setup_logging is somehow called twice
    if not root.handlers:
        root.addHandler(_build_file_handler(log_file))
        root.addHandler(_build_console_handler())

    # Silence noisy third-party loggers
    for noisy_lib in ["urllib3", "boto3", "botocore", "s3transfer",
                       "prophet", "cmdstanpy", "numexpr"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger, triggering logging setup if not yet done.

    Args:
        name: Typically `__name__` of the calling module.

    Returns:
        A configured Logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Pipeline started")
    """
    setup_logging()
    return logging.getLogger(name)


def get_run_logger(run_id: str) -> logging.Logger:
    """
    Return a logger scoped to a specific pipeline run ID.
    Writes to a separate per-run log file: logs/run_<run_id>.log

    Args:
        run_id: A unique identifier for the pipeline run (e.g. "2024-01-15").

    Returns:
        A Logger that writes to both the main log and a dedicated run file.
    """
    setup_logging()

    run_log_file = LOGS_DIR / f"run_{run_id}.log"
    logger = logging.getLogger(f"pipeline.run.{run_id}")

    if not logger.handlers:
        run_handler = logging.handlers.RotatingFileHandler(
            filename=run_log_file,
            maxBytes=LOG_ROTATION_BYTES,
            backupCount=2,
            encoding="utf-8",
        )
        run_handler.setFormatter(_build_formatter())
        run_handler.setLevel(logging.DEBUG)
        logger.addHandler(run_handler)
        logger.propagate = True   # also goes to root (file + console)

    return logger
