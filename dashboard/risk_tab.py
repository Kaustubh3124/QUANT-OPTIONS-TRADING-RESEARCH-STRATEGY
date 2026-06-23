# =============================================================================
# Tab 5: Risk Analytics
# =============================================================================
# Metric cards, return distribution, drawdown chart, risk-return scatter,
# VaR/CVaR visualization.
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from core.backtester import run_backtest, run_all_strategies_backtest
from core.risk_analytics import (
    compute_all_metrics, equity_curve, drawdown_series,
    metrics_comparison_table, var_95, cvar_95,
)
from core.options import list_strategies, get_strategy
from config import INITIAL_CAPITAL


def render(ticker_name: str, start: str, end: str):
    """Render the Risk Analytics tab."""

    st.markdown("## ⚠️ Risk Analytics")

    # Check if backtest has been run
    if "bt_trades" not in st.session_state:
        st.info(
            "📌 Run a backtest first in the **Backtest Results** tab. "
            "Risk analytics will be computed from those results."
        )

        # Option to run all strategies
        if st.button("🔄 Run All Strategies Comparison", key="risk_run_all"):
            with st.spinner("Running backtests for all strategies..."):
                results = run_all_strategies_backtest(ticker_name, start, end)
                st.session_state["risk_all_results"] = results
                # Use the first strategy's trades as primary
                if results:
                    first_key = list(results.keys())[0]
                    st.session_state["bt_trades"] = results[first_key]
                    if "bt_capital" not in st.session_state:
                        st.session_state["bt_capital"] = INITIAL_CAPITAL

        if "risk_all_results" not in st.session_state:
            return

    trades = st.session_state["bt_trades"]
    capital = st.session_state.get("bt_capital", INITIAL_CAPITAL)

    metrics = compute_all_metrics(trades, capital)

    # --- Metric Cards ---
    st.markdown("### Key Risk Metrics")

    row1 = st.columns(4)
    row1[0].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}",
                   help="Risk-adjusted return (>1 good, >2 great)")
    row1[1].metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}",
                   help="Downside risk-adjusted return")
    row1[2].metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}",
                   help="CAGR / Max Drawdown")
    row1[3].metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.1f}%",
                   help="Largest peak-to-trough decline")

    row2 = st.columns(4)
    row2[0].metric("Win Rate", f"{metrics['win_rate_pct']:.0f}%")
    row2[1].metric("Profit Factor", f"{metrics['profit_factor']:.2f}",
                   help="Gross profits / Gross losses (>1 profitable)")
    row2[2].metric("Expected Value", f"₹{metrics['expected_value']:,.0f}/trade")
    row2[3].metric("CAGR", f"{metrics['cagr_pct']:.1f}%")

    row3 = st.columns(4)
    row3[0].metric("VaR (95%)", f"₹{metrics['var_95']:,.0f}",
                   help="5% chance of losing more than this per trade")
    row3[1].metric("CVaR (95%)", f"₹{metrics['cvar_95']:,.0f}",
                   help="Expected loss when losses exceed VaR")
    row3[2].metric("Best Trade", f"₹{metrics['best_trade']:,.0f}")
    row3[3].metric("Worst Trade", f"₹{metrics['worst_trade']:,.0f}")

    # --- Return Distribution ---
    st.markdown("### Return Distribution")

    fig_dist = go.Figure()

    pnl_data = trades["net_pnl"]

    fig_dist.add_trace(go.Histogram(
        x=pnl_data,
        nbinsx=30,
        marker_color="#8b5cf6",
        opacity=0.7,
        name="P&L Distribution",
    ))

    # VaR and CVaR lines
    var = var_95(trades)
    cvar = cvar_95(trades)

    fig_dist.add_vline(
        x=var, line_dash="dash", line_color="#ffa502",
        annotation_text=f"VaR 95%: ₹{var:,.0f}",
        annotation_position="top",
    )
    fig_dist.add_vline(
        x=cvar, line_dash="dash", line_color="#f43f5e",
        annotation_text=f"CVaR 95%: ₹{cvar:,.0f}",
        annotation_position="top",
    )
    fig_dist.add_vline(
        x=0, line_dash="solid", line_color="rgba(255,255,255,0.3)",
    )

    fig_dist.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Trade P&L (₹)",
        yaxis_title="Frequency",
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_dist.update_xaxes(gridcolor="rgba(255,255,255,0.03)")
    fig_dist.update_yaxes(gridcolor="rgba(255,255,255,0.03)")

    st.plotly_chart(fig_dist, use_container_width=True)

    # --- Drawdown Analysis ---
    st.markdown("### Drawdown Analysis")

    eq = equity_curve(trades, capital)
    if not eq.empty:
        dd = drawdown_series(eq["Equity"])

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd["Drawdown_Pct"],
            fill="tozeroy",
            fillcolor="rgba(244, 63, 94, 0.2)",
            line=dict(color="#f43f5e", width=1.5),
            name="Drawdown %", line_shape="spline",
        ))

        fig_dd.update_layout(
            height=300,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Drawdown (%)",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_dd.update_xaxes(gridcolor="rgba(255,255,255,0.03)")
        fig_dd.update_yaxes(gridcolor="rgba(255,255,255,0.03)")

        st.plotly_chart(fig_dd, use_container_width=True)

    # --- Strategy Comparison ---
    if "risk_all_results" in st.session_state:
        st.markdown("### Strategy Comparison")

        all_results = st.session_state["risk_all_results"]
        comp = metrics_comparison_table(all_results, capital)

        if not comp.empty:
            st.dataframe(
                comp.style.format({
                    "total_pnl": "₹{:,.0f}",
                    "cagr_pct": "{:.1f}%",
                    "sharpe_ratio": "{:.2f}",
                    "sortino_ratio": "{:.2f}",
                    "max_drawdown_pct": "{:.1f}%",
                    "win_rate_pct": "{:.0f}%",
                    "profit_factor": "{:.2f}",
                }),
                use_container_width=True,
            )

            # Risk-Return Scatter
            st.markdown("### Risk-Return Trade-off")

            fig_scatter = go.Figure()
            colors = ["#8b5cf6", "#ff6348", "#3742fa", "#ffa502", "#e056fd", "#2ed573", "#f8a5c2"]

            for i, (name, row) in enumerate(comp.iterrows()):
                fig_scatter.add_trace(go.Scatter(
                    x=[abs(row["max_drawdown_pct"])],
                    y=[row["cagr_pct"]],
                    mode="markers+text",
                    marker=dict(
                        size=max(row["sharpe_ratio"] * 15, 8),
                        color=colors[i % len(colors)],
                    ),
                    text=[name],
                    textposition="top center",
                    textfont=dict(size=11),
                    name=name,
                ))

            fig_scatter.update_layout(
                height=400,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Max Drawdown (%) — Risk →",
                yaxis_title="CAGR (%) — Return →",
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_scatter.update_xaxes(gridcolor="rgba(255,255,255,0.03)")
            fig_scatter.update_yaxes(gridcolor="rgba(255,255,255,0.03)")

            st.plotly_chart(fig_scatter, use_container_width=True)
