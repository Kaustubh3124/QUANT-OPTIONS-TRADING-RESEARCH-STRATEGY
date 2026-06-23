# =============================================================================
# Tab 4: Backtest Results
# =============================================================================
# Equity curve, trade log, monthly returns heatmap, strategy comparison,
# rolling Sharpe ratio.
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np

from core.backtester import run_backtest, run_dynamic_backtest, run_all_strategies_backtest
from core.risk_analytics import (
    equity_curve, drawdown_series, compute_all_metrics,
    monthly_returns, metrics_comparison_table,
)
from core.options import list_strategies, get_strategy
from config import INITIAL_CAPITAL, DEFAULT_DTE


def render(ticker_name: str, start: str, end: str):
    """Render the Backtest Results tab."""

    st.markdown("## 🧪 Backtest Results")

    # --- Controls ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        strategy_name = st.selectbox(
            "Strategy",
            ["Dynamic (Auto-Select)"] + list_strategies(),
            key="bt_strategy",
        )
    with col2:
        dte = st.slider("Days to Expiry", 7, 90, DEFAULT_DTE, key="bt_dte")
    with col3:
        frequency = st.selectbox(
            "Entry Frequency",
            ["monthly", "biweekly", "weekly"],
            key="bt_freq",
        )
    with col4:
        capital = st.number_input(
            "Initial Capital (₹)",
            value=INITIAL_CAPITAL,
            step=100000,
            key="bt_capital",
        )

    run_btn = st.button("🚀 Run Backtest", type="primary", key="bt_run")

    if run_btn:
        with st.spinner("Running backtest... This may take a moment."):
            if strategy_name == "Dynamic (Auto-Select)":
                trades = run_dynamic_backtest(
                    ticker_name, start, end, dte, frequency,
                )
            else:
                trades = run_backtest(
                    ticker_name, strategy_name, start, end, dte, frequency,
                )

        if trades.empty:
            st.warning("No trades generated. Try adjusting parameters.")
            return

        st.session_state["bt_trades"] = trades


    # --- Display results ---
    if "bt_trades" not in st.session_state:
        st.info("👆 Configure parameters and click **Run Backtest** to see results.")
        return

    trades = st.session_state["bt_trades"]
    capital = st.session_state["bt_capital"]

    # Compute metrics
    metrics = compute_all_metrics(trades, capital)
    eq = equity_curve(trades, capital)

    # --- Summary Metrics ---
    st.markdown("### Performance Summary")

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Total P&L", f"₹{metrics['total_pnl']:,.0f}")
    mc2.metric("CAGR", f"{metrics['cagr_pct']:.1f}%")
    mc3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    mc4.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.1f}%")
    mc5.metric("Win Rate", f"{metrics['win_rate_pct']:.0f}%")
    mc6.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")

    # --- Equity Curve + Drawdown ---
    st.markdown("### Equity Curve")

    fig_eq = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Portfolio Equity", "Drawdown"),
    )

    # Equity line
    fig_eq.add_trace(
        go.Scatter(
            x=eq.index, y=eq["Equity"],
            name="Equity",
            line=dict(color="#00d4aa", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.08)",
        ),
        row=1, col=1,
    )

    # Capital baseline
    fig_eq.add_hline(
        y=capital, line_dash="dash", line_color="rgba(255,255,255,0.3)",
        row=1, col=1,
    )

    # Drawdown
    dd = drawdown_series(eq["Equity"])
    fig_eq.add_trace(
        go.Scatter(
            x=dd.index, y=dd["Drawdown_Pct"],
            name="Drawdown %",
            line=dict(color="#ff4757", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 71, 87, 0.15)",
        ),
        row=2, col=1,
    )

    fig_eq.update_layout(
        height=550,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig_eq.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
    fig_eq.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

    st.plotly_chart(fig_eq, use_container_width=True)

    # --- Monthly Returns Heatmap ---
    st.markdown("### Monthly Returns Heatmap")

    mr = monthly_returns(trades, capital)
    if not mr.empty:
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        # Rename columns to month names
        mr.columns = [month_names[int(c) - 1] for c in mr.columns]

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=mr.values,
            x=mr.columns,
            y=mr.index.astype(str),
            colorscale=[
                [0, "#ff4757"],
                [0.5, "#2f3542"],
                [1, "#00d4aa"],
            ],
            zmid=0,
            text=[[f"{v:.1f}%" for v in row] for row in mr.values],
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar=dict(title="Return %"),
        ))

        fig_heatmap.update_layout(
            height=max(250, len(mr) * 45),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- Trade Log ---
    st.markdown("### Trade Log")

    display_cols = [
        "Entry_Date", "Exit_Date", "Strategy", "Spot_Entry",
        "Spot_Exit", "Sigma", "net_pnl", "return_pct",
    ]
    available_cols = [c for c in display_cols if c in trades.columns]

    # Add vol percentile and regime if dynamic
    extra_cols = ["Vol_Percentile", "Regime"]
    for ec in extra_cols:
        if ec in trades.columns:
            available_cols.insert(3, ec)

    trades_display = trades[available_cols].copy()
    trades_display = trades_display.sort_values("Entry_Date", ascending=False)

    st.dataframe(
        trades_display.style.applymap(
            lambda v: "color: #00d4aa" if isinstance(v, (int, float)) and v > 0
            else "color: #ff4757" if isinstance(v, (int, float)) and v < 0
            else "",
            subset=["net_pnl", "return_pct"],
        ),
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    # --- Detailed Metrics ---
    with st.expander("📋 Detailed Risk Metrics"):
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ["Value"]
        st.dataframe(metrics_df, use_container_width=True)
