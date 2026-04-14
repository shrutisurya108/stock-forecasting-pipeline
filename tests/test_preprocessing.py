"""
tests/test_preprocessing.py
============================
Unit tests for data/preprocessing.py.
All tests use synthetic DataFrames — no network calls.
"""

import numpy as np
import pandas as pd
import pytest

from data.preprocessing import (
    ProcessedData,
    _clip_outliers,
    _engineer_features,
    _get_feature_cols,
    _sort_and_fill,
    _time_split,
    preprocess_ticker,
    preprocess_all,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """500-row synthetic OHLCV DataFrame — enough for all rolling windows."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Open":   close * 0.99,
            "High":   close * 1.02,
            "Low":    close * 0.97,
            "Close":  close,
            "Volume": np.random.randint(500_000, 2_000_000, n).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


# ── _sort_and_fill ────────────────────────────────────────────────────────────

def test_sort_and_fill_removes_duplicates(raw_df):
    df_dup = pd.concat([raw_df, raw_df.iloc[:5]])
    result = _sort_and_fill(df_dup, "TEST")
    assert result.index.duplicated().sum() == 0


def test_sort_and_fill_sorts_ascending(raw_df):
    shuffled = raw_df.sample(frac=1, random_state=0)
    result   = _sort_and_fill(shuffled, "TEST")
    assert result.index.is_monotonic_increasing


def test_sort_and_fill_no_nans_after_fill(raw_df):
    raw_df.iloc[5:8, raw_df.columns.get_loc("Close")] = np.nan
    result = _sort_and_fill(raw_df, "TEST")
    assert result["Close"].isna().sum() == 0


# ── _clip_outliers ────────────────────────────────────────────────────────────

def test_clip_outliers_reduces_extremes(raw_df):
    raw_df = raw_df.copy()
    raw_df.iloc[0, raw_df.columns.get_loc("Close")] = 1_000_000   # extreme high
    result = _clip_outliers(raw_df, "TEST")
    assert result["Close"].max() < 1_000_000


def test_clip_outliers_preserves_row_count(raw_df):
    result = _clip_outliers(raw_df.copy(), "TEST")
    assert len(result) == len(raw_df)


# ── _engineer_features ────────────────────────────────────────────────────────

def test_engineer_features_adds_log_return(raw_df):
    result = _engineer_features(raw_df, "TEST")
    assert "log_return" in result.columns


def test_engineer_features_adds_rolling_cols(raw_df):
    result = _engineer_features(raw_df, "TEST")
    for w in [5, 10, 20]:
        assert f"rolling_mean_{w}" in result.columns
        assert f"rolling_std_{w}"  in result.columns


def test_engineer_features_no_nans(raw_df):
    result = _engineer_features(raw_df, "TEST")
    # After dropna(), no NaN should remain
    assert result.isna().sum().sum() == 0


def test_engineer_features_row_count_reduced(raw_df):
    # Rolling window of 20 removes warm-up rows
    result = _engineer_features(raw_df, "TEST")
    assert len(result) < len(raw_df)


# ── _time_split ───────────────────────────────────────────────────────────────

def test_time_split_no_overlap(raw_df):
    df = _engineer_features(raw_df, "TEST")
    train, val, test, _ = _time_split(df, "TEST")
    assert train.index.max() < val.index.min()
    assert val.index.max()   < test.index.min()


def test_time_split_covers_all_rows(raw_df):
    df = _engineer_features(raw_df, "TEST")
    train, val, test, _ = _time_split(df, "TEST")
    assert len(train) + len(val) + len(test) == len(df)


def test_time_split_returns_split_dates(raw_df):
    df = _engineer_features(raw_df, "TEST")
    _, _, _, dates = _time_split(df, "TEST")
    assert "train_start" in dates
    assert "test_end"    in dates


# ── preprocess_ticker ─────────────────────────────────────────────────────────

def test_preprocess_ticker_returns_processed_data(raw_df):
    result = preprocess_ticker(raw_df, "TEST")
    assert isinstance(result, ProcessedData)


def test_preprocess_ticker_no_nans_in_features(raw_df):
    result = preprocess_ticker(raw_df, "TEST")
    assert result is not None
    for split in [result.train, result.val, result.test]:
        assert split[result.feature_cols].isna().sum().sum() == 0


def test_preprocess_ticker_scaler_fitted(raw_df):
    result = preprocess_ticker(raw_df, "TEST")
    assert result is not None
    # scaler should have data_min_ after fitting
    assert hasattr(result.scaler, "data_min_")


def test_preprocess_ticker_scaled_range(raw_df):
    result = preprocess_ticker(raw_df, "TEST")
    assert result is not None
    # Scaled Close in train should be in [0, 1]
    close_min = result.train["Close"].min()
    close_max = result.train["Close"].max()
    assert close_min >= -0.01    # small tolerance
    assert close_max <= 1.01


def test_preprocess_ticker_returns_none_on_tiny_df():
    """Very small DataFrame should fail gracefully."""
    tiny = pd.DataFrame(
        {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [1000]},
        index=pd.DatetimeIndex(["2020-01-01"], name="Date"),
    )
    result = preprocess_ticker(tiny, "TINY")
    assert result is None


# ── preprocess_all ────────────────────────────────────────────────────────────

def test_preprocess_all_returns_dict(raw_df):
    data = {"AAPL": raw_df.copy(), "MSFT": raw_df.copy()}
    result = preprocess_all(data)
    assert set(result.keys()) == {"AAPL", "MSFT"}


def test_preprocess_all_skips_failed():
    tiny = pd.DataFrame(
        {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [1000]},
        index=pd.DatetimeIndex(["2020-01-01"], name="Date"),
    )
    result = preprocess_all({"BAD": tiny})
    assert len(result) == 0


def test_preprocess_all_no_data_leakage(raw_df):
    """Scaler fitted on train must not have seen val/test dates."""
    result = preprocess_all({"TEST": raw_df.copy()})
    pd_obj = result["TEST"]
    # All test index dates must be strictly after the last train date
    assert pd_obj.test.index.min() > pd_obj.train.index.max()
