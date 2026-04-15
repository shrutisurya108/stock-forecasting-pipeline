"""
dashboard/components/metrics_table.py
======================================
Renders the RMSE/MAPE/MAE benchmark comparison table in the Streamlit
dashboard.

Provides:
- A styled pandas DataFrame rendered via st.dataframe with colour-coded
  cells (green = better than naive, red = worse).
- A compact summary card showing the headline aggregate RMSE reduction.
- Helper functions to format and filter the raw benchmark CSV.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st


# ── Column display names ──────────────────────────────────────────────────────
DISPLAY_NAMES = {
    "ticker":            "Ticker",
    "model":             "Model",
    "RMSE":              "RMSE ($)",
    "MAPE":              "MAPE (%)",
    "MAE":               "MAE ($)",
    "rmse_vs_naive_pct": "vs Naive",
}

MODEL_DISPLAY = {
    "naive":   "Naive",
    "arima":   "ARIMA",
    "prophet": "Prophet",
    "lstm":    "LSTM",
}


def load_benchmark_data(benchmark_csv_path) -> Optional[pd.DataFrame]:
    """
    Load the benchmark CSV from disk.

    Args:
        benchmark_csv_path: Path to benchmark_results.csv.

    Returns:
        DataFrame or None if the file does not exist / is malformed.
    """
    try:
        df = pd.read_csv(benchmark_csv_path)
        required = {"ticker", "model", "RMSE", "MAPE", "MAE", "rmse_vs_naive_pct"}
        if not required.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def format_benchmark_df(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """
    Format the benchmark DataFrame for display.

    - Filters to a single ticker if specified.
    - Renames columns to human-friendly names.
    - Formats numeric columns.
    - Adds a colour hint column for vs-naive comparison.

    Args:
        df:     Raw benchmark DataFrame from load_benchmark_data().
        ticker: If provided, filter to only this ticker's rows.

    Returns:
        A display-ready DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    display = df.copy()

    # Optional ticker filter
    if ticker and ticker in display["ticker"].unique():
        display = display[display["ticker"] == ticker].copy()

    # Map model names
    display["model"] = display["model"].map(
        lambda m: MODEL_DISPLAY.get(m, m.upper())
    )

    # Format numerics
    display["RMSE"] = display["RMSE"].map(lambda x: f"{x:.4f}")
    display["MAPE"] = display["MAPE"].map(lambda x: f"{x:.2f}%")
    display["MAE"]  = display["MAE"].map(lambda x: f"{x:.4f}")
    display["rmse_vs_naive_pct"] = display["rmse_vs_naive_pct"].map(
        lambda x: "baseline" if x == 0.0 else f"{x:+.1f}%"
    )

    # Select and rename columns
    cols = ["ticker", "model", "RMSE", "MAPE", "MAE", "rmse_vs_naive_pct"]
    display = display[[c for c in cols if c in display.columns]]
    display = display.rename(columns=DISPLAY_NAMES)

    return display.reset_index(drop=True)


def render_metrics_table(
    benchmark_df: Optional[pd.DataFrame],
    ticker:       Optional[str] = None,
) -> None:
    """
    Render the benchmark metrics table in Streamlit.

    Shows a styled table with colour highlights for the vs-Naive column.
    If no data is available, shows an info message.

    Args:
        benchmark_df: Raw benchmark DataFrame.
        ticker:       If provided, show only this ticker's rows.
    """
    if benchmark_df is None or benchmark_df.empty:
        st.info(
            "No benchmark data available. "
            "Run the pipeline with `--mode full` to generate benchmarks."
        )
        return

    display_df = format_benchmark_df(benchmark_df, ticker)

    if display_df.empty:
        st.info(f"No benchmark data for {ticker}.")
        return

    # Apply colour styling to the vs-Naive column
    def _colour_vs_naive(val: str) -> str:
        if val == "baseline":
            return "color: #94A3B8"
        try:
            num = float(val.replace("%", "").replace("+", ""))
            if num < -5:
                return "color: #34D399; font-weight: 600"   # green — good
            elif num < 0:
                return "color: #6EE7B7"                    # light green
            elif num > 5:
                return "color: #F87171; font-weight: 600"  # red — worse
            else:
                return "color: #FCD34D"                    # yellow — marginal
        except (ValueError, AttributeError):
            return ""

    styled = display_df.style.map(
        _colour_vs_naive,
        subset=[DISPLAY_NAMES["rmse_vs_naive_pct"]],
    ).set_properties(**{
        "background-color": "#0F172A",
        "color":            "#CBD5E1",
        "border-color":     "rgba(148,163,184,0.1)",
    }).set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#1E293B"),
                ("color", "#F1F5F9"),
                ("font-weight", "600"),
                ("border-bottom", "1px solid rgba(148,163,184,0.2)"),
                ("padding", "8px 12px"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("padding", "6px 12px"),
                ("border-bottom", "1px solid rgba(148,163,184,0.06)"),
            ],
        },
    ])

    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_summary_card(benchmark_df: Optional[pd.DataFrame]) -> None:
    """
    Render a compact summary row of KPI metrics cards.

    Shows three metrics:
    - Average RMSE reduction vs naive (headline number)
    - Best performing model by RMSE
    - Number of tickers benchmarked

    Args:
        benchmark_df: Raw benchmark DataFrame.
    """
    if benchmark_df is None or benchmark_df.empty:
        return

    model_df = benchmark_df[benchmark_df["model"] != "naive"]

    if model_df.empty:
        return

    avg_reduction = model_df["rmse_vs_naive_pct"].mean()
    best_model    = model_df.groupby("model")["RMSE"].mean().idxmin()
    n_tickers     = benchmark_df["ticker"].nunique()

    col1, col2, col3 = st.columns(3)

    with col1:
        colour = "normal" if avg_reduction >= 0 else "inverse"
        st.metric(
            label="Avg RMSE Reduction vs Naive",
            value=f"{avg_reduction:+.1f}%",
            delta=None,
            help="Negative = better than naive last-value baseline",
        )

    with col2:
        st.metric(
            label="Best Model (avg RMSE)",
            value=MODEL_DISPLAY.get(best_model, best_model.upper()),
        )

    with col3:
        st.metric(
            label="Tickers Benchmarked",
            value=str(n_tickers),
        )


def render_per_model_summary(benchmark_df: Optional[pd.DataFrame]) -> None:
    """
    Render a three-column average metrics summary, one column per model.

    Args:
        benchmark_df: Raw benchmark DataFrame.
    """
    if benchmark_df is None or benchmark_df.empty:
        return

    model_df = benchmark_df[benchmark_df["model"] != "naive"]
    if model_df.empty:
        return

    models = [m for m in ["arima", "prophet", "lstm"]
              if m in model_df["model"].unique()]

    cols = st.columns(len(models))
    for col, model_name in zip(cols, models):
        mdata = model_df[model_df["model"] == model_name]
        avg_rmse   = mdata["RMSE"].mean()
        avg_mape   = mdata["MAPE"].mean()
        avg_vs_naive = mdata["rmse_vs_naive_pct"].mean()

        with col:
            st.markdown(
                f"""
                <div style="
                    background: #1E293B;
                    border: 1px solid rgba(148,163,184,0.15);
                    border-radius: 8px;
                    padding: 14px 16px;
                    text-align: center;
                ">
                    <div style="color:#94A3B8; font-size:11px; margin-bottom:6px;">
                        {MODEL_DISPLAY.get(model_name, model_name.upper())}
                    </div>
                    <div style="color:#F1F5F9; font-size:22px; font-weight:700;">
                        {avg_rmse:.4f}
                    </div>
                    <div style="color:#64748B; font-size:11px;">avg RMSE</div>
                    <div style="color:#94A3B8; font-size:12px; margin-top:6px;">
                        MAPE: {avg_mape:.2f}%
                    </div>
                    <div style="
                        color: {'#34D399' if avg_vs_naive < 0 else '#F87171'};
                        font-size:13px; font-weight:600; margin-top:4px;
                    ">
                        {avg_vs_naive:+.1f}% vs naive
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
