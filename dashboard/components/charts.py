"""
dashboard/components/charts.py
================================
Plotly chart components for the forecasting dashboard.

Main chart: shows the last N days of actual historical Close prices
as a solid line, then overlays the 30-day forward forecast from each
model as a dashed line with a shaded confidence interval band.

This is the "prediction-vs-actual visualizations with confidence intervals"
chart referenced on the resume.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

# Model colours — consistent across all charts in the dashboard
MODEL_COLORS = {
    "arima":   "#3B82F6",   # blue
    "prophet": "#10B981",   # green
    "lstm":    "#F59E0B",   # amber
    "actual":  "#E2E8F0",   # light grey (historical prices)
    "naive":   "#6B7280",   # grey
}

# Shaded CI band opacity
CI_OPACITY = 0.12


def build_forecast_chart(
    forecast_df:       pd.DataFrame,
    historical_df:     Optional[pd.DataFrame],
    ticker:            str,
    last_known_price:  float,
    last_known_date:   str,
    selected_models:   list[str],
    history_days:      int = 90,
    chart_height:      int = 520,
) -> go.Figure:
    """
    Build the main forecast chart for one ticker.

    Args:
        forecast_df:      DataFrame with future dates as index and columns
                          {model}_forecast, {model}_lower, {model}_upper.
        historical_df:    DataFrame with historical Close prices (raw test data
                          or full OHLCV). Index = Date, must have 'Close' col.
                          Pass None to skip the historical line.
        ticker:           Stock symbol shown in chart title.
        last_known_price: Last actual Close price (anchor point).
        last_known_date:  Date of that price (string or Timestamp).
        selected_models:  List of model names to overlay (subset of columns).
        history_days:     How many calendar days of history to show.
        chart_height:     Pixel height of the chart.

    Returns:
        A fully configured Plotly Figure object.
    """
    fig = go.Figure()

    # ── 1. Historical actual prices ───────────────────────────────────────────
    if historical_df is not None and not historical_df.empty:
        hist = historical_df.tail(history_days).copy()
        if "Close" in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                name="Actual",
                line=dict(color=MODEL_COLORS["actual"], width=2),
                hovertemplate="%{x|%b %d, %Y}<br>Actual: $%{y:.2f}<extra></extra>",
            ))

    # ── 2. Anchor point (last known price) ────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[last_known_date],
        y=[last_known_price],
        mode="markers",
        name="Last Known",
        marker=dict(
            color=MODEL_COLORS["actual"],
            size=8,
            symbol="circle",
            line=dict(color="white", width=2),
        ),
        hovertemplate=f"%{{x|%b %d, %Y}}<br>Last known: ${last_known_price:.2f}<extra></extra>",
        showlegend=True,
    ))

    # ── 3. Model forecasts with CI bands ──────────────────────────────────────
    for model_name in selected_models:
        fc_col    = f"{model_name}_forecast"
        lower_col = f"{model_name}_lower"
        upper_col = f"{model_name}_upper"

        if fc_col not in forecast_df.columns:
            continue

        color = MODEL_COLORS.get(model_name, "#8B5CF6")

        # CI shaded band (filled area between lower and upper)
        if lower_col in forecast_df.columns and upper_col in forecast_df.columns:
            # Combine upper + reversed lower to create a closed polygon
            idx    = [str(d) for d in forecast_df.index]  # strings for Plotly
            upper  = list(forecast_df[upper_col])
            lower  = list(forecast_df[lower_col])
            x_band = idx + list(reversed(idx))
            y_band = upper + list(reversed(lower))
            fig.add_trace(go.Scatter(
                x=x_band,
                y=y_band,
                fill="toself",
                fillcolor=f"rgba{_hex_to_rgba(color, CI_OPACITY)}",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{model_name.upper()} 95% CI",
                showlegend=True,
                hoverinfo="skip",
            ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=[str(d) for d in forecast_df.index],
            y=forecast_df[fc_col],
            mode="lines",
            name=f"{model_name.upper()} Forecast",
            line=dict(color=color, width=2.5, dash="dash"),
            hovertemplate=(
                f"%{{x|%b %d, %Y}}<br>"
                f"{model_name.upper()}: $%{{y:.2f}}<extra></extra>"
            ),
        ))

    # ── 4. Vertical divider at forecast start ─────────────────────────────────
    if not forecast_df.empty:
        forecast_start = str(forecast_df.index[0])
        fig.update_layout(shapes=[dict(
            type="line",
            x0=forecast_start, x1=forecast_start,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="rgba(148,163,184,0.5)", width=1.5, dash="dot"),
        )])

    # ── 5. Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{ticker}</b> — 30-Day Price Forecast",
            font=dict(size=20, color="#F1F5F9"),
            x=0.02,
        ),
        height=chart_height,
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#94A3B8", family="monospace"),
        legend=dict(
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="rgba(148,163,184,0.2)",
            borderwidth=1,
            font=dict(size=12),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            zerolinecolor="rgba(148,163,184,0.15)",
            tickfont=dict(size=11),
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="Price (USD)", font=dict(size=12)),
            gridcolor="rgba(148,163,184,0.08)",
            zerolinecolor="rgba(148,163,184,0.15)",
            tickprefix="$",
            tickfont=dict(size=11),
            showgrid=True,
        ),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=80, b=40),
    )

    return fig


def build_benchmark_chart(
    benchmark_df: pd.DataFrame,
    metric:       str = "RMSE",
    chart_height: int = 350,
) -> go.Figure:
    """
    Build a grouped bar chart comparing RMSE/MAPE/MAE across models and tickers.

    Args:
        benchmark_df: Output of benchmarking.build_benchmark_table().
        metric:       Column to plot ("RMSE", "MAPE", or "MAE").
        chart_height: Pixel height of the chart.

    Returns:
        A Plotly Figure with grouped bars — one group per ticker.
    """
    if benchmark_df.empty:
        return _empty_figure(f"No benchmark data for {metric}")

    model_order = ["naive", "arima", "prophet", "lstm"]
    fig = go.Figure()

    for model_name in model_order:
        model_df = benchmark_df[benchmark_df["model"] == model_name]
        if model_df.empty:
            continue

        color = MODEL_COLORS.get(model_name, "#8B5CF6")
        if model_name == "naive":
            color = MODEL_COLORS["naive"]

        fig.add_trace(go.Bar(
            name=model_name.upper(),
            x=model_df["ticker"],
            y=model_df[metric],
            marker_color=color,
            hovertemplate=(
                f"<b>%{{x}}</b><br>"
                f"{model_name.upper()} {metric}: %{{y:.4f}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{metric} by Model and Ticker</b>",
            font=dict(size=16, color="#F1F5F9"),
            x=0.02,
        ),
        height=chart_height,
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#94A3B8", family="monospace"),
        barmode="group",
        bargap=0.2,
        bargroupgap=0.05,
        legend=dict(
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="rgba(148,163,184,0.2)",
            borderwidth=1,
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(text=metric, font=dict(size=12)),
            gridcolor="rgba(148,163,184,0.08)",
            tickfont=dict(size=11),
        ),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig


def _empty_figure(message: str = "No data available") -> go.Figure:
    """Return a blank figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#94A3B8"),
    )
    fig.update_layout(
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex colour string to an rgba tuple string for Plotly fills."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"({r},{g},{b},{alpha})"
