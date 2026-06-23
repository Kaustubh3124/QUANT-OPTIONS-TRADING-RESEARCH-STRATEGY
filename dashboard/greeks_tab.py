# =============================================================================
# Tab 6: Greeks Dashboard
# =============================================================================
# Per-strategy Greek values, Greek exposure through time,
# sensitivity heatmaps, and theta decay curve.
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from core.options import list_strategies, get_strategy
from core.greeks import (
    strategy_greeks, strategy_greeks_table,
    greeks_through_time, greek_sensitivity_grid,
)
from core.market_data import fetch_ohlcv, get_close_prices
from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def render(ticker_name: str, start: str, end: str):
    """Render the Greeks tab."""

    st.markdown("## 🔬 Greeks Analysis")

    # --- Parameters ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        strategy_name = st.selectbox(
            "Strategy",
            list_strategies(),
            format_func=lambda x: get_strategy(x).name,
            key="greeks_strategy",
        )
    with col2:
        spot = st.number_input("Spot Price (₹)", value=22000.0, step=100.0, key="greeks_spot")
    with col3:
        sigma = st.slider("Implied Vol (%)", 5, 60, 15, key="greeks_sigma") / 100
    with col4:
        dte = st.slider("Days to Expiry", 5, 90, 30, key="greeks_dte")

    tte = dte / 252
    r = RISK_FREE_RATE
    strategy = get_strategy(strategy_name)

    # --- Greek Values ---
    st.markdown(f"### {strategy.name} — Current Greeks")

    greeks = strategy_greeks(strategy, spot, sigma, tte, r)

    gcols = st.columns(5)
    greek_descriptions = {
        "Delta": ("Directional exposure", "Asset-equivalent"),
        "Gamma": ("Delta sensitivity", "Acceleration"),
        "Theta": ("Time decay/day", "₹/day"),
        "Vega": ("Vol sensitivity", "₹/1% vol"),
        "Rho": ("Rate sensitivity", "₹/1% rate"),
    }

    for i, (name, value) in enumerate(greeks.items()):
        desc, unit = greek_descriptions[name]
        gcols[i].metric(
            name,
            f"{value:.4f}" if name in ("Delta", "Gamma") else f"₹{value:.2f}",
            help=f"{desc} ({unit})",
        )

    # --- Per-Leg Greeks Table ---
    st.markdown("### Per-Leg Breakdown")
    greeks_table = strategy_greeks_table(strategy, spot, sigma, tte, r)
    st.dataframe(greeks_table, use_container_width=True, hide_index=True)

    # --- Theta Decay Curve ---
    st.markdown("### Theta Decay Through Time")

    days_range = list(range(dte, 0, -1))
    theta_values = []
    delta_values = []
    gamma_values = []
    vega_values = []

    for d in days_range:
        t = d / TRADING_DAYS_PER_YEAR
        g = strategy_greeks(strategy, spot, sigma, t, r)
        theta_values.append(g["Theta"])
        delta_values.append(g["Delta"])
        gamma_values.append(g["Gamma"])
        vega_values.append(g["Vega"])

    fig_decay = go.Figure()

    fig_decay.add_trace(go.Scatter(
        x=days_range,
        y=theta_values,
        mode="lines",
        name="Theta (₹/day)",
        line=dict(color="#ff6348", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(255, 99, 72, 0.1)",
    ))

    fig_decay.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Days to Expiry",
        yaxis_title="Theta (₹/day)",
        xaxis=dict(autorange="reversed"),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_decay.update_xaxes(gridcolor="rgba(255,255,255,0.03)")
    fig_decay.update_yaxes(gridcolor="rgba(255,255,255,0.03)")

    st.plotly_chart(fig_decay, use_container_width=True)

    # --- Greek Exposure Through DTE ---
    st.markdown("### All Greeks Through Time")

    fig_all_greeks = make_greeks_timeline(
        days_range, delta_values, gamma_values, theta_values, vega_values
    )
    st.plotly_chart(fig_all_greeks, use_container_width=True)

    # --- Sensitivity Heatmaps ---
    st.markdown("### Greek Sensitivity Heatmaps")

    greek_choice = st.selectbox(
        "Select Greek to visualize",
        ["Delta", "Gamma", "Theta", "Vega"],
        key="greeks_heatmap_choice",
    )

    with st.spinner(f"Computing {greek_choice} sensitivity grid..."):
        grid = greek_sensitivity_grid(
            strategy, spot, sigma, tte, r,
            greek_name=greek_choice,
            n_points=15,
        )

    fig_heat = go.Figure(data=go.Heatmap(
        z=grid.values,
        x=[f"{v:.1%}" for v in grid.columns.astype(float)],
        y=[f"₹{v:,.0f}" for v in grid.index.astype(float)],
        colorscale="RdYlGn",
        colorbar=dict(title=greek_choice),
    ))

    fig_heat.update_layout(
        height=450,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Implied Volatility",
        yaxis_title="Spot Price",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig_heat, use_container_width=True)


def make_greeks_timeline(days, deltas, gammas, thetas, vegas):
    """Create a multi-panel Greek timeline chart."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Delta", "Gamma", "Theta", "Vega"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    configs = [
        (deltas, "#8b5cf6", 1, 1),
        (gammas, "#ffa502", 1, 2),
        (thetas, "#ff6348", 2, 1),
        (vegas, "#3742fa", 2, 2),
    ]

    for values, color, row, col in configs:
        fig.add_trace(
            go.Scatter(
                x=days, y=values,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        height=500,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # Reverse x-axis (DTE decreasing)
    for i in range(1, 5):
        fig.update_xaxes(autorange="reversed", gridcolor="rgba(255,255,255,0.03)", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.03)", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)

    return fig
