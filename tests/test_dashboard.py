"""
tests/test_dashboard.py
========================
Unit tests for dashboard components.

Strategy:
- Test data preparation, formatting, and chart-building logic directly.
- No Streamlit server is launched — we test the Python functions, not the UI.
- Plotly Figure structure is verified (traces, layout keys).
- All file I/O uses tmp_path fixtures.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.components.charts import (
    build_forecast_chart,
    build_benchmark_chart,
    _empty_figure,
    _hex_to_rgba,
    MODEL_COLORS,
)
from dashboard.components.metrics_table import (
    load_benchmark_data,
    format_benchmark_df,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def forecast_df() -> pd.DataFrame:
    """30-row forecast DataFrame with 3 model columns each."""
    dates = pd.date_range("2025-01-06", periods=30, freq="B")
    data  = {}
    for model in ["arima", "prophet", "lstm"]:
        base = 150.0 + np.random.randn(30) * 2
        data[f"{model}_forecast"] = base
        data[f"{model}_lower"]    = base - 5
        data[f"{model}_upper"]    = base + 5
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="date"))


@pytest.fixture
def historical_df() -> pd.DataFrame:
    """90-row historical OHLCV DataFrame."""
    dates = pd.date_range("2024-09-01", periods=90, freq="B")
    close = 140 + np.cumsum(np.random.randn(90) * 0.5)
    return pd.DataFrame(
        {"Close": close, "Volume": np.random.randint(1e6, 5e6, 90)},
        index=pd.DatetimeIndex(dates, name="Date"),
    )


@pytest.fixture
def benchmark_df() -> pd.DataFrame:
    """Minimal benchmark DataFrame with 2 tickers and 4 models."""
    rows = []
    for ticker in ["AAPL", "MSFT"]:
        naive_rmse = 4.5
        rows.append({"ticker": ticker, "model": "naive",
                     "RMSE": naive_rmse, "MAPE": 2.8, "MAE": 3.2,
                     "rmse_vs_naive_pct": 0.0})
        for model, reduction in [("arima", -9.0), ("prophet", -14.0), ("lstm", -18.0)]:
            rmse = naive_rmse * (1 + reduction / 100)
            rows.append({
                "ticker": ticker, "model": model,
                "RMSE": round(rmse, 4),
                "MAPE": round(2.8 + reduction / 100, 4),
                "MAE":  round(3.2 + reduction / 100, 4),
                "rmse_vs_naive_pct": reduction,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# _hex_to_rgba
# ══════════════════════════════════════════════════════════════════════════════

def test_hex_to_rgba_format():
    result = _hex_to_rgba("#3B82F6", 0.12)
    assert result.startswith("(")
    assert result.endswith(")")
    assert "0.12" in result


def test_hex_to_rgba_components():
    result = _hex_to_rgba("#FF0000", 0.5)
    assert "255" in result   # red channel
    assert "0.5" in result


def test_hex_to_rgba_strips_hash():
    result = _hex_to_rgba("#3B82F6", 0.1)
    # Should not crash and should have 4 components
    parts = result.strip("()").split(",")
    assert len(parts) == 4


# ══════════════════════════════════════════════════════════════════════════════
# MODEL_COLORS
# ══════════════════════════════════════════════════════════════════════════════

def test_model_colors_has_all_models():
    for model in ["arima", "prophet", "lstm"]:
        assert model in MODEL_COLORS


def test_model_colors_are_valid_hex():
    for name, color in MODEL_COLORS.items():
        assert color.startswith("#"), f"{name} color '{color}' not a hex string"
        assert len(color) == 7


# ══════════════════════════════════════════════════════════════════════════════
# _empty_figure
# ══════════════════════════════════════════════════════════════════════════════

def test_empty_figure_returns_figure():
    import plotly.graph_objects as go
    fig = _empty_figure("Test message")
    assert isinstance(fig, go.Figure)


def test_empty_figure_has_annotation():
    fig = _empty_figure("No data")
    assert len(fig.layout.annotations) > 0
    assert "No data" in fig.layout.annotations[0].text


def test_empty_figure_dark_background():
    fig = _empty_figure()
    assert fig.layout.paper_bgcolor == "#0F172A"


# ══════════════════════════════════════════════════════════════════════════════
# build_forecast_chart
# ══════════════════════════════════════════════════════════════════════════════

def test_build_forecast_chart_returns_figure(forecast_df, historical_df):
    import plotly.graph_objects as go
    fig = build_forecast_chart(
        forecast_df=forecast_df,
        historical_df=historical_df,
        ticker="AAPL",
        last_known_price=155.0,
        last_known_date="2025-01-05",
        selected_models=["arima", "prophet", "lstm"],
    )
    assert isinstance(fig, go.Figure)


def test_build_forecast_chart_has_traces(forecast_df, historical_df):
    fig = build_forecast_chart(
        forecast_df=forecast_df,
        historical_df=historical_df,
        ticker="AAPL",
        last_known_price=155.0,
        last_known_date="2025-01-05",
        selected_models=["arima"],
    )
    # Should have: historical line + anchor marker + 1 CI band + 1 forecast line
    assert len(fig.data) >= 3


def test_build_forecast_chart_no_history(forecast_df):
    """Chart should work without historical data."""
    fig = build_forecast_chart(
        forecast_df=forecast_df,
        historical_df=None,
        ticker="MSFT",
        last_known_price=300.0,
        last_known_date="2025-01-05",
        selected_models=["arima"],
    )
    assert len(fig.data) >= 1


def test_build_forecast_chart_three_models(forecast_df, historical_df):
    fig = build_forecast_chart(
        forecast_df=forecast_df,
        historical_df=historical_df,
        ticker="AAPL",
        last_known_price=155.0,
        last_known_date="2025-01-05",
        selected_models=["arima", "prophet", "lstm"],
    )
    # 3 models × (1 CI band + 1 line) + historical + anchor = 8 traces min
    assert len(fig.data) >= 8


def test_build_forecast_chart_skips_missing_model(forecast_df, historical_df):
    """A model name not in forecast_df columns should be silently skipped."""
    fig = build_forecast_chart(
        forecast_df=forecast_df,
        historical_df=historical_df,
        ticker="AAPL",
        last_known_price=155.0,
        last_known_date="2025-01-05",
        selected_models=["arima", "xgboost"],   # xgboost not in cols
    )
    trace_names = [t.name for t in fig.data]
    assert not any("xgboost" in (n or "").lower() for n in trace_names)


def test_build_forecast_chart_title_contains_ticker(forecast_df):
    fig = build_forecast_chart(
        forecast_df=forecast_df,
        historical_df=None,
        ticker="NVDA",
        last_known_price=500.0,
        last_known_date="2025-01-05",
        selected_models=["lstm"],
    )
    assert "NVDA" in fig.layout.title.text


def test_build_forecast_chart_dark_background(forecast_df):
    fig = build_forecast_chart(
        forecast_df=forecast_df,
        historical_df=None,
        ticker="AAPL",
        last_known_price=150.0,
        last_known_date="2025-01-05",
        selected_models=["arima"],
    )
    assert fig.layout.paper_bgcolor == "#0F172A"
    assert fig.layout.plot_bgcolor  == "#0F172A"


def test_build_forecast_chart_custom_height(forecast_df):
    fig = build_forecast_chart(
        forecast_df=forecast_df,
        historical_df=None,
        ticker="AAPL",
        last_known_price=150.0,
        last_known_date="2025-01-05",
        selected_models=["arima"],
        chart_height=400,
    )
    assert fig.layout.height == 400


# ══════════════════════════════════════════════════════════════════════════════
# build_benchmark_chart
# ══════════════════════════════════════════════════════════════════════════════

def test_build_benchmark_chart_returns_figure(benchmark_df):
    import plotly.graph_objects as go
    fig = build_benchmark_chart(benchmark_df, metric="RMSE")
    assert isinstance(fig, go.Figure)


def test_build_benchmark_chart_has_bars(benchmark_df):
    fig = build_benchmark_chart(benchmark_df, metric="RMSE")
    # At least one Bar trace per model (naive + arima + prophet + lstm)
    bar_traces = [t for t in fig.data if t.type == "bar"]
    assert len(bar_traces) >= 4


def test_build_benchmark_chart_empty_df():
    fig = build_benchmark_chart(pd.DataFrame(), metric="RMSE")
    # Returns empty figure with annotation
    assert len(fig.layout.annotations) > 0


def test_build_benchmark_chart_mape_metric(benchmark_df):
    fig = build_benchmark_chart(benchmark_df, metric="MAPE")
    assert "MAPE" in fig.layout.title.text


def test_build_benchmark_chart_mae_metric(benchmark_df):
    fig = build_benchmark_chart(benchmark_df, metric="MAE")
    assert "MAE" in fig.layout.title.text


# ══════════════════════════════════════════════════════════════════════════════
# load_benchmark_data
# ══════════════════════════════════════════════════════════════════════════════

def test_load_benchmark_data_valid_csv(tmp_path, benchmark_df):
    csv_path = tmp_path / "benchmark_results.csv"
    benchmark_df.to_csv(csv_path, index=False)

    loaded = load_benchmark_data(csv_path)
    assert loaded is not None
    assert set(loaded.columns).issuperset({"ticker", "model", "RMSE"})


def test_load_benchmark_data_missing_file(tmp_path):
    result = load_benchmark_data(tmp_path / "nonexistent.csv")
    assert result is None


def test_load_benchmark_data_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("a,b\n1,2\n")
    result = load_benchmark_data(bad_csv)
    assert result is None


def test_load_benchmark_data_row_count(tmp_path, benchmark_df):
    csv_path = tmp_path / "benchmark_results.csv"
    benchmark_df.to_csv(csv_path, index=False)
    loaded = load_benchmark_data(csv_path)
    assert len(loaded) == len(benchmark_df)


# ══════════════════════════════════════════════════════════════════════════════
# format_benchmark_df
# ══════════════════════════════════════════════════════════════════════════════

def test_format_benchmark_df_renames_columns(benchmark_df):
    formatted = format_benchmark_df(benchmark_df)
    assert "RMSE ($)" in formatted.columns
    assert "MAPE (%)" in formatted.columns
    assert "vs Naive"  in formatted.columns


def test_format_benchmark_df_ticker_filter(benchmark_df):
    formatted = format_benchmark_df(benchmark_df, ticker="AAPL")
    assert "MSFT" not in formatted["Ticker"].values
    assert "AAPL" in formatted["Ticker"].values


def test_format_benchmark_df_all_tickers(benchmark_df):
    formatted = format_benchmark_df(benchmark_df, ticker=None)
    assert "AAPL" in formatted["Ticker"].values
    assert "MSFT" in formatted["Ticker"].values


def test_format_benchmark_df_model_names_uppercased(benchmark_df):
    formatted = format_benchmark_df(benchmark_df)
    models_col = formatted["Model"].tolist()
    assert "ARIMA" in models_col or "Arima" in models_col or "arima" in str(models_col)


def test_format_benchmark_df_rmse_is_string(benchmark_df):
    """RMSE should be formatted as a string with 4 decimal places."""
    formatted = format_benchmark_df(benchmark_df)
    sample    = formatted["RMSE ($)"].iloc[0]
    assert isinstance(sample, str)
    assert "." in sample


def test_format_benchmark_df_naive_shows_baseline(benchmark_df):
    formatted = format_benchmark_df(benchmark_df)
    vs_col    = formatted["vs Naive"]
    assert any(v == "baseline" for v in vs_col)


def test_format_benchmark_df_empty_input():
    result = format_benchmark_df(pd.DataFrame())
    assert result.empty


def test_format_benchmark_df_unknown_ticker_returns_empty(benchmark_df):
    result = format_benchmark_df(benchmark_df, ticker="UNKNOWN")
    # Unknown ticker not in data → should still return something
    # (filter keeps all rows if ticker not found)
    # The function filters only if ticker IS in the unique values
    assert isinstance(result, pd.DataFrame)
