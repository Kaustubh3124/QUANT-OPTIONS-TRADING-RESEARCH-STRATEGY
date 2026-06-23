# =============================================================================
# Tab 7: Portfolio Simulator
# =============================================================================
# Allocation sliders, combined equity curve, drawdown, attribution,
# correlation matrix, allocation drift.
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from core.backtester import run_backtest
from core.portfolio import (
    simulate_portfolio, strategy_correlation_matrix, attribution_analysis,
)
from core.risk_analytics import drawdown_series
from core.options import list_strategies, get_strategy
from config import INITIAL_CAPITAL, DEFAULT_DTE, PORTFOLIO_WEIGHTS


def render(ticker_name: str, start: str, end: str):
    """Render the Portfolio Simulator tab."""

    st.markdown("## 💼 Portfolio Simulator")

    # --- Allocation Sliders ---
    st.markdown("### Portfolio Allocation")

    strategies_for_portfolio = ["IronCondor", "LongStraddle", "BullCallSpread", "BearPutSpread"]

    cols = st.columns(len(strategies_for_portfolio))
    allocations = {}
    for i, name in enumerate(strategies_for_portfolio):
        default = int(PORTFOLIO_WEIGHTS.get(name, 25) * 100)
        allocations[name] = cols[i].slider(
            get_strategy(name).name,
            0, 100, default,
            step=5,
            key=f"port_alloc_{name}",
        )

    total_alloc = sum(allocations.values())

    if total_alloc == 0:
        st.warning("Total allocation is 0%. Adjust the sliders above.")
        return

    if total_alloc != 100:
        st.warning(f"⚠️ Allocations sum to **{total_alloc}%** (will be normalized to 100%).")

    # Normalize
    alloc_normalized = {k: v / total_alloc for k, v in allocations.items() if v > 0}

    # --- Display allocation pie ---
    col_pie, col_params = st.columns([1, 2])

    with col_pie:
        active = {k: v for k, v in allocations.items() if v > 0}
        fig_pie = go.Figure(data=[go.Pie(
            labels=[get_strategy(k).name for k in active],
            values=list(active.values()),
            hole=0.45,
            marker_colors=["#00d4aa", "#ffa502", "#3742fa", "#ff6348", "#e056fd"],
            textinfo="label+percent",
        )])
        fig_pie.update_layout(
            height=250,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_params:
        capital = st.number_input(
            "Initial Capital (₹)", value=INITIAL_CAPITAL,
            step=100000, key="port_capital",
        )
        dte = st.slider("Days to Expiry", 7, 90, DEFAULT_DTE, key="port_dte")

    # --- Run Portfolio Simulation ---
    run_btn = st.button("🚀 Run Portfolio Simulation", type="primary", key="port_run")

    if run_btn:
        with st.spinner("Running strategy backtests and combining portfolio..."):
            results = {}
            for name in alloc_normalized:
                try:
                    results[name] = run_backtest(
                        ticker_name, name, start, end, dte, "monthly"
                    )
                except Exception as e:
                    st.warning(f"Skipping {name}: {e}")

            portfolio = simulate_portfolio(results, alloc_normalized, capital)
            st.session_state["portfolio"] = portfolio
            st.session_state["port_results"] = results
            st.session_state["port_alloc"] = alloc_normalized

    if "portfolio" not in st.session_state:
        st.info("👆 Set allocations and click **Run Portfolio Simulation**.")
        return

    portfolio = st.session_state["portfolio"]
    results = st.session_state["port_results"]
    alloc_used = st.session_state["port_alloc"]

    # --- Portfolio Metrics ---
    st.markdown("### Portfolio Performance")

    metrics = portfolio["metrics"]
    mc = st.columns(6)
    mc[0].metric("Total P&L", f"₹{metrics.get('total_pnl', 0):,.0f}")
    mc[1].metric("CAGR", f"{metrics.get('cagr_pct', 0):.1f}%")
    mc[2].metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
    mc[3].metric("Max DD", f"{metrics.get('max_drawdown_pct', 0):.1f}%")
    mc[4].metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.0f}%")
    mc[5].metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

    # --- Combined Equity Curve ---
    st.markdown("### Portfolio Equity Curve")

    eq = portfolio["portfolio_equity"]
    strat_eqs = portfolio["strategy_equities"]

    if not eq.empty:
        fig_eq = go.Figure()

        # Individual strategy equity lines
        colors = ["rgba(0, 212, 170, 0.31)", "rgba(255, 165, 2, 0.31)", "rgba(55, 66, 250, 0.31)", "rgba(255, 99, 72, 0.31)"]
        for i, (name, seq) in enumerate(strat_eqs.items()):
            if not seq.empty:
                fig_eq.add_trace(go.Scatter(
                    x=seq.index, y=seq["Equity"],
                    name=get_strategy(name).name,
                    line=dict(color=colors[i % len(colors)], width=1, dash="dot"),
                ))

        # Combined portfolio line
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq["Equity"],
            name="Portfolio",
            line=dict(color="#ffffff", width=2.5),
        ))

        fig_eq.add_hline(
            y=capital, line_dash="dash", line_color="rgba(255,255,255,0.2)",
        )

        fig_eq.update_layout(
            height=450,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Portfolio Value (₹)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        fig_eq.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
        fig_eq.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

        st.plotly_chart(fig_eq, use_container_width=True)

    # --- Attribution ---
    st.markdown("### Strategy Attribution")

    attr = attribution_analysis(results, alloc_used)
    if not attr.empty:
        col_attr, col_bar = st.columns([1, 1])

        with col_attr:
            st.dataframe(attr, use_container_width=True, hide_index=True)

        with col_bar:
            fig_bar = go.Figure(data=[go.Bar(
                x=attr["Strategy"].apply(lambda x: get_strategy(x).name),
                y=attr["Total_PnL"],
                marker_color=[
                    "#00d4aa" if v > 0 else "#ff4757"
                    for v in attr["Total_PnL"]
                ],
            )])
            fig_bar.update_layout(
                height=300,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Total P&L (₹)",
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_bar.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
            fig_bar.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

            st.plotly_chart(fig_bar, use_container_width=True)

    # --- Correlation Matrix ---
    st.markdown("### Strategy Correlation Matrix")

    corr = strategy_correlation_matrix(results)
    if not corr.empty:
        corr_display = corr.copy()
        corr_display.index = [get_strategy(n).name for n in corr_display.index]
        corr_display.columns = [get_strategy(n).name for n in corr_display.columns]

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_display.values,
            x=corr_display.columns,
            y=corr_display.index,
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in corr_display.values],
            texttemplate="%{text}",
            colorbar=dict(title="Correlation"),
        ))
        fig_corr.update_layout(
            height=350,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
