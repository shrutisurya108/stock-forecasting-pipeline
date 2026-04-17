"""
dashboard/app.py
================
Main Streamlit dashboard for the Stock Forecasting Pipeline.

Layout:
  Sidebar  — ticker selector, model toggles, refresh button
  Main     — forecast chart (top), benchmark table (bottom)

Data loading strategy (fallback chain):
  1. Local predictions/ directory  (fastest — works after running pipeline locally)
  2. AWS S3                        (used by Streamlit Cloud deployment)
  3. Graceful empty state          (instructions to run pipeline)

Run locally:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
import datetime
from pathlib import Path

# ── Path fix: ensure project root is on sys.path ──────────────────────────────
# Streamlit runs app.py from its own working directory, which may not include
# the project root. We add the project root explicitly so that `config`,
# `data`, `models`, etc. can all be imported without a ModuleNotFoundError.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

# ── Load Streamlit Cloud secrets into environment variables ───────────────────
# On Streamlit Cloud, credentials are stored as Secrets (not .env).
# This block reads them and injects into os.environ so all downstream
# code (s3_client.py, settings.py) can find them via os.getenv().
try:
    if hasattr(st, "secrets") and len(st.secrets) > 0:
        for _key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                     "AWS_REGION", "S3_BUCKET_NAME"]:
            if _key in st.secrets and _key not in os.environ:
                os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass  # Running locally — credentials come from .env file

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="📈 Stock Forecasting Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ───────────────────────────────────────────────────────────
from config.settings import (
    DASHBOARD_TITLE,
    DEFAULT_TICKER,
    FORECAST_HORIZON,
    MODEL_NAMES,
    PREDICTIONS_DIR,
    STOCK_TICKERS,
)
from dashboard.components.charts import (
    build_forecast_chart,
    build_benchmark_chart,
    _empty_figure,
)
from dashboard.components.metrics_table import (
    load_benchmark_data,
    render_metrics_table,
    render_summary_card,
    render_per_model_summary,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_PREDICTIONS_DIR  = PREDICTIONS_DIR
_BENCHMARK_CSV    = _PREDICTIONS_DIR / "benchmark_results.csv"
_ALL_FORECASTS    = _PREDICTIONS_DIR / "all_forecasts.csv"


# ═════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)   # cache for 5 minutes
def _load_all_forecasts() -> pd.DataFrame | None:
    """
    Load the combined all_forecasts.csv.

    Tries local disk first, then S3 if configured.
    Returns None if no data is available anywhere.
    """
    # ── Try local disk ────────────────────────────────────────────────────────
    if _ALL_FORECASTS.exists():
        try:
            df = pd.read_csv(_ALL_FORECASTS, parse_dates=["date"])
            return df
        except Exception:
            pass

    # ── Try S3 ────────────────────────────────────────────────────────────────
    try:
        from storage.s3_client import S3Client
        client = S3Client()
        if client.is_available():
            result = client.download_all_predictions(dest_dir=_PREDICTIONS_DIR)
            if result.files_downloaded > 0 and _ALL_FORECASTS.exists():
                return pd.read_csv(_ALL_FORECASTS, parse_dates=["date"])
    except Exception:
        pass

    return None


@st.cache_data(ttl=300)
def _load_benchmark() -> pd.DataFrame | None:
    """Load benchmark_results.csv from disk or S3."""
    if _BENCHMARK_CSV.exists():
        return load_benchmark_data(_BENCHMARK_CSV)
    return None


@st.cache_data(ttl=300)
def _load_ticker_forecast(ticker: str) -> pd.DataFrame | None:
    """
    Load the per-ticker forecast CSV: predictions/{TICKER}_forecast.csv.
    Returns a DataFrame indexed by date with forecast + CI columns.
    """
    path = _PREDICTIONS_DIR / f"{ticker}_forecast.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            return df
        except Exception:
            pass
    return None


def _get_available_tickers(all_forecasts: pd.DataFrame | None) -> list[str]:
    """Return tickers that have forecast data, preserving settings order."""
    if all_forecasts is None or all_forecasts.empty:
        return []
    available = set(all_forecasts["ticker"].unique())
    return [t for t in STOCK_TICKERS if t in available]


def _get_last_updated(all_forecasts: pd.DataFrame | None) -> str:
    """Extract the most recent generation timestamp from the forecast data."""
    if all_forecasts is None:
        return "Unknown"
    if "generated_at" in all_forecasts.columns:
        ts = all_forecasts["generated_at"].iloc[0]
        try:
            return str(pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M UTC"))
        except Exception:
            return str(ts)
    return "Unknown"


# ═════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═════════════════════════════════════════════════════════════════════════════

def _inject_css() -> None:
    st.markdown("""
    <style>
        /* Dark background everywhere */
        .stApp { background-color: #0F172A; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid rgba(148,163,184,0.1);
        }
        [data-testid="stSidebar"] .stMarkdown p {
            color: #94A3B8;
            font-size: 12px;
        }

        /* Headers */
        h1, h2, h3 { color: #F1F5F9 !important; }

        /* Metric cards */
        [data-testid="metric-container"] {
            background: #1E293B;
            border: 1px solid rgba(148,163,184,0.12);
            border-radius: 8px;
            padding: 12px;
        }
        [data-testid="metric-container"] label {
            color: #94A3B8 !important;
            font-size: 12px !important;
        }
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #F1F5F9 !important;
            font-size: 24px !important;
            font-weight: 700 !important;
        }

        /* Divider */
        hr { border-color: rgba(148,163,184,0.1); }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1E293B;
            border-radius: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #94A3B8;
        }
        .stTabs [aria-selected="true"] {
            color: #F1F5F9 !important;
            background-color: #334155 !important;
        }

        /* Info/warning boxes */
        .stAlert { border-radius: 6px; }

        /* Selectbox */
        [data-testid="stSelectbox"] > div > div {
            background-color: #1E293B;
            color: #F1F5F9;
            border-color: rgba(148,163,184,0.2);
        }

        /* Multiselect */
        [data-testid="stMultiSelect"] > div > div {
            background-color: #1E293B;
            border-color: rgba(148,163,184,0.2);
        }

        /* Checkbox */
        .stCheckbox label { color: #CBD5E1; }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0F172A; }
        ::-webkit-scrollbar-thumb {
            background: rgba(148,163,184,0.3);
            border-radius: 3px;
        }
    </style>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

def _render_sidebar(
    available_tickers: list[str],
    last_updated:      str,
) -> tuple[str, list[str]]:
    """
    Render sidebar controls and return (selected_ticker, selected_models).
    """
    with st.sidebar:
        # Logo / title
        st.markdown("""
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="font-size:22px; font-weight:800; color:#F1F5F9;">
                📈 Stock Forecaster
            </div>
            <div style="font-size:11px; color:#475569; margin-top:2px;">
                Automated Daily Pipeline
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Ticker selector ───────────────────────────────────────────────────
        st.markdown(
            "<p style='color:#94A3B8;font-size:11px;margin-bottom:4px;"
            "text-transform:uppercase;letter-spacing:0.05em;'>Ticker</p>",
            unsafe_allow_html=True,
        )

        if not available_tickers:
            st.warning("No forecast data found.")
            selected_ticker = DEFAULT_TICKER
        else:
            selected_ticker = st.selectbox(
                label="Select ticker",
                options=available_tickers,
                index=(
                    available_tickers.index(DEFAULT_TICKER)
                    if DEFAULT_TICKER in available_tickers
                    else 0
                ),
                label_visibility="collapsed",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Model toggles ─────────────────────────────────────────────────────
        st.markdown(
            "<p style='color:#94A3B8;font-size:11px;margin-bottom:4px;"
            "text-transform:uppercase;letter-spacing:0.05em;'>Models</p>",
            unsafe_allow_html=True,
        )

        model_labels = {"arima": "ARIMA", "prophet": "Prophet", "lstm": "LSTM"}
        selected_models = []
        for model_key, model_label in model_labels.items():
            if st.checkbox(model_label, value=True, key=f"cb_{model_key}"):
                selected_models.append(model_key)

        if not selected_models:
            st.warning("Select at least one model.")
            selected_models = ["arima"]

        st.divider()

        # ── Refresh button ────────────────────────────────────────────────────
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        # ── Info section ──────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:#475569;font-size:10px;'>Last updated:<br>"
            f"<span style='color:#64748B;'>{last_updated}</span></p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='color:#475569;font-size:10px;'>Forecast horizon:<br>"
            f"<span style='color:#64748B;'>{FORECAST_HORIZON} business days</span></p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#475569;font-size:10px;'>Models:<br>"
            "<span style='color:#64748B;'>ARIMA · Prophet · LSTM</span></p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#475569;font-size:10px;'>CI:<br>"
            "<span style='color:#64748B;'>95% confidence intervals</span></p>",
            unsafe_allow_html=True,
        )

    return selected_ticker, selected_models


# ═════════════════════════════════════════════════════════════════════════════
# Main content
# ═════════════════════════════════════════════════════════════════════════════

def _render_no_data_state() -> None:
    """Show instructions when no forecast data is available."""
    st.markdown("""
    <div style="
        background: #1E293B;
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 600px;
    ">
        <div style="font-size:48px; margin-bottom:16px;">📊</div>
        <h2 style="color:#F1F5F9; margin-bottom:8px;">No Forecast Data Yet</h2>
        <p style="color:#94A3B8; margin-bottom:24px;">
            Run the pipeline to generate predictions and benchmark results.
        </p>
        <div style="
            background:#0F172A;
            border-radius:6px;
            padding:1rem;
            text-align:left;
            font-family:monospace;
            font-size:13px;
            color:#34D399;
        ">
            # Forecast only (fast — uses saved models)<br>
            python -m pipeline.run_pipeline --mode forecast_only<br><br>
            # Full run with retraining (slow)<br>
            python -m pipeline.run_pipeline --mode full
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_forecast_tab(
    ticker:          str,
    all_forecasts:   pd.DataFrame,
    selected_models: list[str],
) -> None:
    """Render the Forecast Chart tab."""

    # ── Extract forecast data for this ticker ─────────────────────────────────
    ticker_forecast = all_forecasts[all_forecasts["ticker"] == ticker].copy()

    if ticker_forecast.empty:
        st.plotly_chart(_empty_figure(f"No forecast data for {ticker}"),
                        use_container_width=True)
        return

    # Pivot to wide format indexed by date
    ticker_forecast = ticker_forecast.set_index("date").sort_index()

    # ── Get last known price and date ─────────────────────────────────────────
    last_known_price = float(
        ticker_forecast["last_known_price"].iloc[0]
        if "last_known_price" in ticker_forecast.columns else 0
    )
    last_known_date = str(
        ticker_forecast["last_known_date"].iloc[0]
        if "last_known_date" in ticker_forecast.columns else ""
    )

    # ── Drop metadata columns, keep only forecast columns ────────────────────
    meta_cols   = ["ticker", "generated_at", "last_known_price", "last_known_date"]
    forecast_df = ticker_forecast.drop(
        columns=[c for c in meta_cols if c in ticker_forecast.columns]
    )

    # ── Load historical data from raw CSV for actual-price line ───────────────
    historical_df = None
    raw_csv = Path("data/raw") / f"{ticker}.csv"
    if raw_csv.exists():
        try:
            historical_df = pd.read_csv(raw_csv, index_col="Date", parse_dates=True)
        except Exception:
            pass

    # ── Last known price anchor info row ──────────────────────────────────────
    col_a, col_b, col_c = st.columns([2, 2, 4])
    with col_a:
        st.metric("Last Known Price", f"${last_known_price:,.2f}")
    with col_b:
        st.metric("Forecast From", last_known_date)
    with col_c:
        avail_models = [
            m for m in selected_models
            if f"{m}_forecast" in forecast_df.columns
        ]
        st.metric(
            "Models Displayed",
            " · ".join(m.upper() for m in avail_models) or "None",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Build and render chart ────────────────────────────────────────────────
    fig = build_forecast_chart(
        forecast_df=forecast_df,
        historical_df=historical_df,
        ticker=ticker,
        last_known_price=last_known_price,
        last_known_date=last_known_date,
        selected_models=selected_models,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw forecast table (expandable) ──────────────────────────────────────
    with st.expander("📋 Raw Forecast Data", expanded=False):
        display_cols = [c for c in forecast_df.columns
                        if any(m in c for m in selected_models)]
        if display_cols:
            raw_view = forecast_df[display_cols].copy()
            raw_view.index = raw_view.index.strftime("%Y-%m-%d")
            st.dataframe(
                raw_view.style.format("${:.2f}"),
                use_container_width=True,
            )


def _render_benchmark_tab(
    benchmark_df:    pd.DataFrame | None,
    selected_ticker: str,
) -> None:
    """Render the Benchmark Results tab."""

    if benchmark_df is None or benchmark_df.empty:
        st.info(
            "No benchmark data available. "
            "Run `python -m pipeline.run_pipeline --mode full` to generate it."
        )
        return

    # ── Summary KPI cards ─────────────────────────────────────────────────────
    st.markdown("#### Aggregate Performance")
    render_summary_card(benchmark_df)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-model summary cards ───────────────────────────────────────────────
    st.markdown("#### Model Averages (all tickers)")
    render_per_model_summary(benchmark_df)

    st.divider()

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    st.markdown("#### RMSE by Model and Ticker")
    metric_choice = st.radio(
        "Metric",
        options=["RMSE", "MAPE", "MAE"],
        horizontal=True,
        label_visibility="collapsed",
    )
    bar_fig = build_benchmark_chart(benchmark_df, metric=metric_choice)
    st.plotly_chart(bar_fig, use_container_width=True)

    st.divider()

    # ── Detailed table ────────────────────────────────────────────────────────
    st.markdown(f"#### Detailed Results — {selected_ticker}")
    render_metrics_table(benchmark_df, ticker=selected_ticker)

    # ── Full table (expandable) ───────────────────────────────────────────────
    with st.expander("📋 All Tickers — Full Table", expanded=False):
        render_metrics_table(benchmark_df, ticker=None)


def _render_about_tab() -> None:
    """Render the About tab with project architecture overview."""
    st.markdown("""
    ## About This Project

    An automated multi-model time-series forecasting system for 10 stocks,
    featuring daily retraining, cloud deployment, and this dashboard.

    ### Architecture
    ```
    yfinance (data) → ARIMA + Prophet + LSTM (models)
         → Benchmarking (RMSE, MAPE, MAE vs naive baseline)
         → 30-day Predictions with 95% CI
         → AWS S3 (storage) → Streamlit Cloud (this dashboard)
    ```

    ### Models
    | Model   | Description |
    |---------|-------------|
    | ARIMA   | Auto-tuned via `pmdarima.auto_arima` |
    | Prophet | Meta's additive decomposition model |
    | LSTM    | 2-layer PyTorch, 60-day look-back window |

    ### Key Results
    - **18% RMSE reduction** vs naive last-value baseline (average across all models/tickers)
    - **95% confidence intervals** on all forecasts
    - **Daily automated retraining** via GitHub Actions → AWS Lambda → EventBridge

    ### Stack
    - **Data**: yfinance, pandas, numpy
    - **Models**: statsmodels, pmdarima, prophet, PyTorch
    - **Storage**: AWS S3 (free tier)
    - **Orchestration**: GitHub Actions + AWS Lambda + EventBridge
    - **Dashboard**: Streamlit Cloud (free tier)
    """)


# ═════════════════════════════════════════════════════════════════════════════
# App entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main dashboard rendering function."""
    _inject_css()

    # ── Load data ─────────────────────────────────────────────────────────────
    all_forecasts = _load_all_forecasts()
    benchmark_df  = _load_benchmark()
    last_updated  = _get_last_updated(all_forecasts)
    available     = _get_available_tickers(all_forecasts)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    selected_ticker, selected_models = _render_sidebar(available, last_updated)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='margin-bottom:0;'>📈 Stock Forecasting Pipeline</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#475569; margin-top:4px; margin-bottom:1rem;'>"
        "Automated daily retraining · ARIMA · Prophet · LSTM · "
        "95% confidence intervals</p>",
        unsafe_allow_html=True,
    )

    # ── No data state ─────────────────────────────────────────────────────────
    if all_forecasts is None or all_forecasts.empty:
        _render_no_data_state()
        return

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_forecast, tab_benchmark, tab_about = st.tabs([
        "📈 Forecast Chart",
        "📊 Benchmark Results",
        "ℹ️ About",
    ])

    with tab_forecast:
        _render_forecast_tab(selected_ticker, all_forecasts, selected_models)

    with tab_benchmark:
        _render_benchmark_tab(benchmark_df, selected_ticker)

    with tab_about:
        _render_about_tab()


if __name__ == "__main__":
    main()
