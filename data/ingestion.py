"""
data/ingestion.py
=================
Responsible for ONE thing: reliably fetching raw OHLCV stock data from
Yahoo Finance and persisting it as CSV files in data/raw/.

Features:
- Per-ticker CSV caching (skips download if file < 24 hours old)
- Exponential-backoff retry on network failures
- Post-download validation (shape, missing columns, all-NaN guards)
- Structured logging at every step
- Returns a dict  {ticker: DataFrame}  for downstream use

Usage (standalone smoke-test):
    python -m data.ingestion
"""

import time
import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config.logging_config import get_logger
from config.settings import (
    DATA_DIR,
    DATA_INTERVAL,
    DATA_YEARS,
    STOCK_TICKERS,
    TARGET_COL,
)

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CACHE_MAX_AGE_HOURS: int = 24          # re-download if CSV is older than this
REQUIRED_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]
MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 2.0          # seconds; doubles on each retry


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_path(ticker: str) -> Path:
    """Return the expected CSV path for a ticker."""
    return DATA_DIR / f"{ticker}.csv"


def _is_cache_fresh(ticker: str) -> bool:
    """
    Return True if a cached CSV exists and is younger than CACHE_MAX_AGE_HOURS.
    """
    path = _cache_path(ticker)
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    fresh = age_hours < CACHE_MAX_AGE_HOURS
    if fresh:
        logger.debug(
            "Cache HIT for %s (%.1f h old, limit %d h)",
            ticker, age_hours, CACHE_MAX_AGE_HOURS,
        )
    else:
        logger.debug(
            "Cache STALE for %s (%.1f h old, limit %d h) — will re-download",
            ticker, age_hours, CACHE_MAX_AGE_HOURS,
        )
    return fresh


def _calculate_date_range() -> tuple[str, str]:
    """Return (start_date, end_date) strings for the configured DATA_YEARS window."""
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=DATA_YEARS * 365)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _validate_dataframe(df: pd.DataFrame, ticker: str) -> bool:
    """
    Run basic sanity checks on a freshly downloaded DataFrame.
    Logs a warning for every issue found.
    Returns True if the DataFrame is usable, False otherwise.
    """
    if df is None or df.empty:
        logger.warning("[%s] Validation FAILED — DataFrame is empty", ticker)
        return False

    # Check required columns exist
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning(
            "[%s] Validation FAILED — missing columns: %s", ticker, missing_cols
        )
        return False

    # Check target column is not entirely NaN
    if df[TARGET_COL].isna().all():
        logger.warning(
            "[%s] Validation FAILED — '%s' column is all NaN", ticker, TARGET_COL
        )
        return False

    # Warn (but don't fail) if there are some NaNs
    nan_pct = df[TARGET_COL].isna().mean() * 100
    if nan_pct > 0:
        logger.warning(
            "[%s] %.1f%% NaN values in '%s' — will be handled in preprocessing",
            ticker, nan_pct, TARGET_COL,
        )

    # Minimum row count: at least 252 trading days (≈ 1 year)
    if len(df) < 252:
        logger.warning(
            "[%s] Validation FAILED — only %d rows (need ≥ 252)", ticker, len(df)
        )
        return False

    logger.debug(
        "[%s] Validation OK — %d rows, %.1f%% NaN in Close",
        ticker, len(df), nan_pct,
    )
    return True


# ── Core download logic ───────────────────────────────────────────────────────

def _download_ticker(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for one ticker from Yahoo Finance with retry logic.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        start:  Start date string "YYYY-MM-DD".
        end:    End date string "YYYY-MM-DD".

    Returns:
        A clean DataFrame indexed by Date, or None on persistent failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "[%s] Downloading (attempt %d/%d) — %s to %s",
                ticker, attempt, MAX_RETRIES, start, end,
            )
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval=DATA_INTERVAL,
                auto_adjust=True,   # adjusts for splits/dividends automatically
                progress=False,
                threads=False,      # single-threaded avoids rate-limit issues
            )

            # yfinance may return a MultiIndex when downloading a single ticker
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw.index = pd.to_datetime(raw.index)
            raw.index.name = "Date"

            if _validate_dataframe(raw, ticker):
                logger.info(
                    "[%s] Downloaded %d rows successfully", ticker, len(raw)
                )
                return raw

            # Validation failed — don't retry, the data itself is bad
            return None

        except Exception as exc:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "[%s] Download error on attempt %d: %s. Retrying in %.0fs…",
                ticker, attempt, exc, delay,
            )
            if attempt < MAX_RETRIES:
                time.sleep(delay)

    logger.error("[%s] All %d download attempts failed", ticker, MAX_RETRIES)
    return None


def _load_from_cache(ticker: str) -> Optional[pd.DataFrame]:
    """Load a ticker's DataFrame from its CSV cache file."""
    path = _cache_path(ticker)
    try:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        logger.info("[%s] Loaded %d rows from cache: %s", ticker, len(df), path)
        return df
    except Exception as exc:
        logger.warning("[%s] Failed to read cache file %s: %s", ticker, path, exc)
        return None


def _save_to_cache(df: pd.DataFrame, ticker: str) -> None:
    """Persist a DataFrame to CSV."""
    path = _cache_path(ticker)
    try:
        df.to_csv(path)
        logger.debug("[%s] Saved to cache: %s", ticker, path)
    except Exception as exc:
        logger.warning("[%s] Failed to save cache: %s", ticker, exc)


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_ticker(
    ticker: str,
    force_download: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a single ticker.

    Checks the local CSV cache first (unless force_download=True).
    Downloads from Yahoo Finance on cache miss or stale cache.

    Args:
        ticker:         Stock symbol, e.g. "AAPL".
        force_download: Bypass cache and always re-download.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] indexed by Date,
        or None if fetching failed.
    """
    if not force_download and _is_cache_fresh(ticker):
        df = _load_from_cache(ticker)
        if df is not None:
            return df
        logger.warning("[%s] Cache read failed — falling back to download", ticker)

    start, end = _calculate_date_range()
    df = _download_ticker(ticker, start, end)

    if df is not None:
        _save_to_cache(df, ticker)

    return df


def fetch_all_tickers(
    tickers: list[str] | None = None,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all tickers (defaults to STOCK_TICKERS from settings).

    Args:
        tickers:        List of ticker symbols. Defaults to settings.STOCK_TICKERS.
        force_download: Bypass cache for all tickers.

    Returns:
        Dict mapping ticker → DataFrame.
        Tickers that fail to download are omitted with a logged error.
    """
    tickers = tickers or STOCK_TICKERS
    results: dict[str, pd.DataFrame] = {}

    logger.info(
        "Starting data fetch for %d tickers: %s", len(tickers), tickers
    )

    for i, ticker in enumerate(tickers, 1):
        logger.info("── Ticker %d/%d: %s", i, len(tickers), ticker)
        df = fetch_ticker(ticker, force_download=force_download)
        if df is not None:
            results[ticker] = df
        else:
            logger.error("[%s] Skipping — fetch returned None", ticker)

        # Polite delay between API calls to avoid rate limiting
        if i < len(tickers):
            time.sleep(0.5)

    success = len(results)
    failed  = len(tickers) - success
    logger.info(
        "Fetch complete — %d/%d succeeded, %d failed",
        success, len(tickers), failed,
    )

    if failed > 0:
        failed_tickers = [t for t in tickers if t not in results]
        logger.warning("Failed tickers: %s", failed_tickers)

    return results


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  data/ingestion.py — smoke test")
    print("═" * 60 + "\n")

    data = fetch_all_tickers()

    print("\n── Results ──────────────────────────────────────────────")
    print(f"{'Ticker':<8} {'Rows':>6} {'Start':>12} {'End':>12} {'NaN%':>6}")
    print("─" * 50)
    for ticker, df in data.items():
        nan_pct = df["Close"].isna().mean() * 100
        print(
            f"{ticker:<8} {len(df):>6} "
            f"{str(df.index.min().date()):>12} "
            f"{str(df.index.max().date()):>12} "
            f"{nan_pct:>5.1f}%"
        )

    print(f"\n✅ Fetched {len(data)}/10 tickers successfully")
    if len(data) < 10:
        missing = [t for t in STOCK_TICKERS if t not in data]
        print(f"❌ Failed: {missing}")
