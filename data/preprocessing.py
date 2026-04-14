"""
data/preprocessing.py
=====================
Takes raw OHLCV DataFrames (from ingestion.py) and produces model-ready splits.

Pipeline (in order):
  1. Sort by date, remove duplicate indices
  2. Fill missing trading days (forward-fill then back-fill)
  3. Remove outliers (IQR method on Close price)
  4. Engineer features: log returns, rolling mean/std, volatility
  5. Drop rows with NaN (created by rolling windows)
  6. Split into train / val / test (time-ordered — NO shuffle)
  7. Fit MinMaxScaler on train only, apply to val and test
  8. Return a clean ProcessedData dict per ticker

Key guarantee: NO DATA LEAKAGE.
  - Scaler is fit on train set only.
  - Val and test are transformed (not fit) with the train scaler.

Usage (standalone smoke-test):
    python -m data.preprocessing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.logging_config import get_logger
from config.settings import (
    STOCK_TICKERS,
    TARGET_COL,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)

logger = get_logger(__name__)

# ── Feature engineering config ────────────────────────────────────────────────
ROLLING_WINDOWS: list[int] = [5, 10, 20]      # days for rolling stats
OUTLIER_IQR_FACTOR: float  = 3.0              # anything beyond 3×IQR is clipped

# Columns that will be scaled (Close + all engineered features)
# Raw OHLCV columns that are NOT scaled (kept for reference)
OHLCV_COLS: list[str] = ["Open", "High", "Low", "Close", "Volume"]


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class ProcessedData:
    """
    Container for a single ticker's preprocessed data.

    Attributes:
        ticker:        Stock symbol.
        train:         Training DataFrame (scaled).
        val:           Validation DataFrame (scaled).
        test:          Test DataFrame (scaled).
        scaler:        MinMaxScaler fitted on train — use to inverse-transform.
        feature_cols:  Names of the scaled feature columns.
        raw_test:      Unscaled test DataFrame (for computing real-price metrics).
        split_dates:   Dict with 'train_end', 'val_end' date strings.
    """
    ticker:      str
    train:       pd.DataFrame
    val:         pd.DataFrame
    test:        pd.DataFrame
    scaler:      MinMaxScaler
    feature_cols: list[str]
    raw_test:    pd.DataFrame
    split_dates: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.ticker}: "
            f"train={len(self.train)} | val={len(self.val)} | test={len(self.test)} | "
            f"features={len(self.feature_cols)}"
        )


# ── Step 1 & 2: Sort + fill gaps ──────────────────────────────────────────────

def _sort_and_fill(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Sort by date, remove duplicates, forward-fill then back-fill missing values.
    """
    df = df.sort_index()
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        logger.warning("[%s] Removing %d duplicate index entries", ticker, n_dupes)
        df = df[~df.index.duplicated(keep="first")]

    # Reindex to complete calendar range, then fill trading gaps
    # (yfinance already returns only trading days, but gaps can exist)
    before = len(df)
    df = df.ffill().bfill()
    after_nan = df[OHLCV_COLS].isna().sum().sum()
    if after_nan > 0:
        logger.warning(
            "[%s] %d NaN values remain after fill — dropping those rows",
            ticker, after_nan,
        )
        df = df.dropna(subset=OHLCV_COLS)

    logger.debug("[%s] Sort+fill: %d → %d rows", ticker, before, len(df))
    return df


# ── Step 3: Outlier clipping ──────────────────────────────────────────────────

def _clip_outliers(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clip extreme Close prices using the IQR method.
    Values beyond OUTLIER_IQR_FACTOR × IQR are clipped (not dropped).
    """
    q1 = df[TARGET_COL].quantile(0.25)
    q3 = df[TARGET_COL].quantile(0.75)
    iqr = q3 - q1
    lo  = q1 - OUTLIER_IQR_FACTOR * iqr
    hi  = q3 + OUTLIER_IQR_FACTOR * iqr

    n_outliers = ((df[TARGET_COL] < lo) | (df[TARGET_COL] > hi)).sum()
    if n_outliers > 0:
        logger.warning(
            "[%s] Clipping %d outliers in Close (IQR bounds: %.2f – %.2f)",
            ticker, n_outliers, lo, hi,
        )
        df[TARGET_COL] = df[TARGET_COL].clip(lower=lo, upper=hi)

    return df


# ── Step 4: Feature engineering ──────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Add features that help time-series models detect patterns:

    - log_return:          log(Close_t / Close_{t-1})  — stationarises price series
    - rolling_mean_{w}:    w-day rolling mean of Close
    - rolling_std_{w}:     w-day rolling std of Close (volatility proxy)
    - rolling_return_{w}:  cumulative log-return over w days
    - high_low_spread:     (High - Low) / Close  — intraday volatility
    - volume_change:       log(Volume_t / Volume_{t-1})
    """
    df = df.copy()

    # Log return (primary feature for ARIMA stationarity)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Rolling statistics on Close
    for w in ROLLING_WINDOWS:
        df[f"rolling_mean_{w}"] = df["Close"].rolling(w).mean()
        df[f"rolling_std_{w}"]  = df["Close"].rolling(w).std()
        df[f"rolling_return_{w}"] = (
            df["log_return"].rolling(w).sum()          # cumulative log-return
        )

    # Intraday spread
    df["high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]

    # Volume change (log-differenced to handle large magnitudes)
    df["volume_change"] = np.log(df["Volume"] / df["Volume"].shift(1))
    # Replace ±inf from zero-volume days
    df["volume_change"] = df["volume_change"].replace([np.inf, -np.inf], np.nan)

    # Drop rows created by rolling windows (NaN in new features)
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped > 0:
        logger.debug(
            "[%s] Dropped %d warm-up rows after feature engineering", ticker, dropped
        )

    logger.debug(
        "[%s] Features engineered: %d rows, %d columns", ticker, len(df), len(df.columns)
    )
    return df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return columns that will be scaled (everything except raw OHLCV except Close)."""
    exclude = {"Open", "High", "Low", "Volume"}   # keep Close — it IS a feature
    return [c for c in df.columns if c not in exclude]


# ── Step 5 & 6: Split ─────────────────────────────────────────────────────────

def _time_split(
    df: pd.DataFrame,
    ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """
    Split DataFrame into train / val / test in chronological order.
    No shuffling — future data must never appear in the training set.
    """
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, \
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    split_dates = {
        "train_start": str(train.index.min().date()),
        "train_end":   str(train.index.max().date()),
        "val_start":   str(val.index.min().date()),
        "val_end":     str(val.index.max().date()),
        "test_start":  str(test.index.min().date()),
        "test_end":    str(test.index.max().date()),
    }

    logger.debug(
        "[%s] Split: train=%d | val=%d | test=%d",
        ticker, len(train), len(val), len(test),
    )
    return train, val, test, split_dates


# ── Step 7: Scaling ───────────────────────────────────────────────────────────

def _scale_splits(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    feature_cols: list[str],
    ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Fit MinMaxScaler on train feature columns only.
    Transform val and test with the same scaler.
    Non-feature columns (Open, High, Low, Volume) are kept but not scaled.

    Returns scaled copies of each split plus the fitted scaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_scaled = train.copy()
    val_scaled   = val.copy()
    test_scaled  = test.copy()

    # Fit ONLY on train
    train_scaled[feature_cols] = scaler.fit_transform(train[feature_cols])

    # Transform val and test
    val_scaled[feature_cols]  = scaler.transform(val[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test[feature_cols])

    logger.debug(
        "[%s] Scaled %d feature columns (fit on train only)", ticker, len(feature_cols)
    )
    return train_scaled, val_scaled, test_scaled, scaler


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_ticker(
    df: pd.DataFrame,
    ticker: str,
) -> Optional[ProcessedData]:
    """
    Full preprocessing pipeline for one ticker.

    Args:
        df:     Raw OHLCV DataFrame from ingestion.py.
        ticker: Stock symbol (used for logging).

    Returns:
        ProcessedData with train/val/test splits and fitted scaler,
        or None if preprocessing fails.
    """
    logger.info("[%s] Starting preprocessing (%d raw rows)", ticker, len(df))

    try:
        # Step 1–2: Sort and fill
        df = _sort_and_fill(df, ticker)

        # Step 3: Outlier clipping
        df = _clip_outliers(df, ticker)

        # Step 4: Feature engineering
        df = _engineer_features(df, ticker)

        if len(df) < 100:
            logger.error(
                "[%s] Too few rows after preprocessing (%d) — skipping",
                ticker, len(df),
            )
            return None

        # Step 5: Identify feature columns
        feature_cols = _get_feature_cols(df)

        # Step 6: Time-ordered split
        train, val, test, split_dates = _time_split(df, ticker)

        # Keep raw (unscaled) test for inverse-transform during evaluation
        raw_test = test.copy()

        # Step 7: Scale
        train, val, test, scaler = _scale_splits(
            train, val, test, feature_cols, ticker
        )

        result = ProcessedData(
            ticker=ticker,
            train=train,
            val=val,
            test=test,
            scaler=scaler,
            feature_cols=feature_cols,
            raw_test=raw_test,
            split_dates=split_dates,
        )

        logger.info("[%s] Preprocessing complete — %s", ticker, result.summary())
        return result

    except Exception as exc:
        logger.exception("[%s] Preprocessing failed: %s", ticker, exc)
        return None


def preprocess_all(
    data: dict[str, pd.DataFrame],
) -> dict[str, ProcessedData]:
    """
    Run preprocessing for every ticker in the provided dict.

    Args:
        data: Output of ingestion.fetch_all_tickers().

    Returns:
        Dict mapping ticker → ProcessedData (failed tickers are omitted).
    """
    results: dict[str, ProcessedData] = {}

    logger.info("Preprocessing %d tickers…", len(data))

    for ticker, df in data.items():
        processed = preprocess_ticker(df, ticker)
        if processed is not None:
            results[ticker] = processed
        else:
            logger.error("[%s] Dropped from pipeline after preprocessing failure", ticker)

    logger.info(
        "Preprocessing complete — %d/%d tickers ready",
        len(results), len(data),
    )
    return results


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.ingestion import fetch_all_tickers

    print("\n" + "═" * 70)
    print("  data/preprocessing.py — smoke test")
    print("═" * 70 + "\n")

    # 1. Fetch raw data
    print("Step 1: Fetching raw data…")
    raw_data = fetch_all_tickers()
    print(f"        → {len(raw_data)} tickers fetched\n")

    # 2. Preprocess
    print("Step 2: Preprocessing…")
    processed = preprocess_all(raw_data)
    print(f"        → {len(processed)} tickers preprocessed\n")

    # 3. Print summary table
    print("── Summary ──────────────────────────────────────────────────────────")
    header = (
        f"{'Ticker':<7} {'Train':>6} {'Val':>6} {'Test':>6} "
        f"{'Features':>9} {'Train Start':>12} {'Test End':>12}"
    )
    print(header)
    print("─" * 65)

    all_ok = True
    for ticker, pd_obj in processed.items():
        # Verify no NaNs leaked through
        nan_train = pd_obj.train[pd_obj.feature_cols].isna().sum().sum()
        nan_val   = pd_obj.val[pd_obj.feature_cols].isna().sum().sum()
        nan_test  = pd_obj.test[pd_obj.feature_cols].isna().sum().sum()
        has_nan   = (nan_train + nan_val + nan_test) > 0

        if has_nan:
            all_ok = False

        flag = "⚠️ NaN!" if has_nan else "✅"
        print(
            f"{ticker:<7} {len(pd_obj.train):>6} {len(pd_obj.val):>6} "
            f"{len(pd_obj.test):>6} {len(pd_obj.feature_cols):>9} "
            f"{pd_obj.split_dates['train_start']:>12} "
            f"{pd_obj.split_dates['test_end']:>12}  {flag}"
        )

    print()

    # 4. Feature list (same for all tickers)
    if processed:
        sample = next(iter(processed.values()))
        print(f"── Feature columns ({len(sample.feature_cols)}) ──────────────")
        for col in sample.feature_cols:
            print(f"   • {col}")

    print()
    if all_ok and len(processed) == len(raw_data):
        print(f"✅ Phase 2 PASSED — {len(processed)}/10 tickers ready, 0 NaN values")
    else:
        issues = []
        if not all_ok:
            issues.append("NaN values detected")
        if len(processed) < len(raw_data):
            issues.append(f"only {len(processed)}/{len(raw_data)} tickers preprocessed")
        print(f"❌ Phase 2 issues: {', '.join(issues)}")
