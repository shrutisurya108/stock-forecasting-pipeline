"""
tests/test_ingestion.py
=======================
Unit tests for data/ingestion.py.
Uses pytest-mock to avoid real network calls.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.ingestion import (
    _cache_path,
    _is_cache_fresh,
    _validate_dataframe,
    fetch_ticker,
    fetch_all_tickers,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Minimal valid OHLCV DataFrame (300 rows to pass validation)."""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    return pd.DataFrame(
        {
            "Open":   [100.0] * 300,
            "High":   [105.0] * 300,
            "Low":    [95.0]  * 300,
            "Close":  [102.0] * 300,
            "Volume": [1_000_000] * 300,
        },
        index=dates,
    )


# ── _validate_dataframe ───────────────────────────────────────────────────────

def test_validate_dataframe_valid(sample_ohlcv):
    assert _validate_dataframe(sample_ohlcv, "TEST") is True


def test_validate_dataframe_empty():
    assert _validate_dataframe(pd.DataFrame(), "TEST") is False


def test_validate_dataframe_missing_column(sample_ohlcv):
    df = sample_ohlcv.drop(columns=["Close"])
    assert _validate_dataframe(df, "TEST") is False


def test_validate_dataframe_all_nan_close(sample_ohlcv):
    sample_ohlcv["Close"] = float("nan")
    assert _validate_dataframe(sample_ohlcv, "TEST") is False


def test_validate_dataframe_too_few_rows(sample_ohlcv):
    assert _validate_dataframe(sample_ohlcv.iloc[:10], "TEST") is False


# ── _is_cache_fresh ───────────────────────────────────────────────────────────

def test_cache_fresh_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("data.ingestion.DATA_DIR", tmp_path)
    assert _is_cache_fresh("AAPL") is False


def test_cache_fresh_recent_file(tmp_path, monkeypatch):
    monkeypatch.setattr("data.ingestion.DATA_DIR", tmp_path)
    p = tmp_path / "AAPL.csv"
    p.write_text("dummy")
    # File was just created — should be fresh
    assert _is_cache_fresh("AAPL") is True


def test_cache_stale_old_file(tmp_path, monkeypatch):
    monkeypatch.setattr("data.ingestion.DATA_DIR", tmp_path)
    p = tmp_path / "AAPL.csv"
    p.write_text("dummy")
    # Backdate the modification time by 25 hours
    old_mtime = time.time() - 25 * 3600
    import os
    os.utime(p, (old_mtime, old_mtime))
    assert _is_cache_fresh("AAPL") is False


# ── fetch_ticker ──────────────────────────────────────────────────────────────

def test_fetch_ticker_uses_cache(tmp_path, monkeypatch, sample_ohlcv):
    """fetch_ticker should return cached data without calling yfinance."""
    monkeypatch.setattr("data.ingestion.DATA_DIR", tmp_path)

    # Write a fresh cache file
    sample_ohlcv.index.name = "Date"
    sample_ohlcv.to_csv(tmp_path / "AAPL.csv")

    with patch("data.ingestion.yf.download") as mock_dl:
        result = fetch_ticker("AAPL")
        mock_dl.assert_not_called()

    assert result is not None
    assert len(result) == 300


def test_fetch_ticker_downloads_on_cache_miss(tmp_path, monkeypatch, sample_ohlcv):
    """fetch_ticker should call yf.download when no cache exists."""
    monkeypatch.setattr("data.ingestion.DATA_DIR", tmp_path)

    with patch("data.ingestion.yf.download", return_value=sample_ohlcv) as mock_dl:
        result = fetch_ticker("AAPL")
        mock_dl.assert_called_once()

    assert result is not None


def test_fetch_ticker_returns_none_on_failure(tmp_path, monkeypatch):
    """fetch_ticker should return None when all retries fail."""
    monkeypatch.setattr("data.ingestion.DATA_DIR", tmp_path)
    monkeypatch.setattr("data.ingestion.MAX_RETRIES", 1)
    monkeypatch.setattr("data.ingestion.RETRY_BASE_DELAY", 0)

    with patch("data.ingestion.yf.download", side_effect=Exception("network error")):
        result = fetch_ticker("AAPL")

    assert result is None


# ── fetch_all_tickers ─────────────────────────────────────────────────────────

def test_fetch_all_tickers_returns_dict(tmp_path, monkeypatch, sample_ohlcv):
    monkeypatch.setattr("data.ingestion.DATA_DIR", tmp_path)
    tickers = ["AAPL", "MSFT"]

    with patch("data.ingestion.yf.download", return_value=sample_ohlcv):
        with patch("data.ingestion.time.sleep"):  # skip delays in tests
            result = fetch_all_tickers(tickers=tickers)

    assert set(result.keys()) == {"AAPL", "MSFT"}
    for df in result.values():
        assert "Close" in df.columns


def test_fetch_all_tickers_skips_failed(tmp_path, monkeypatch):
    monkeypatch.setattr("data.ingestion.DATA_DIR", tmp_path)
    monkeypatch.setattr("data.ingestion.MAX_RETRIES", 1)
    monkeypatch.setattr("data.ingestion.RETRY_BASE_DELAY", 0)

    with patch("data.ingestion.yf.download", side_effect=Exception("fail")):
        with patch("data.ingestion.time.sleep"):
            result = fetch_all_tickers(tickers=["AAPL", "MSFT"])

    assert len(result) == 0   # both failed
