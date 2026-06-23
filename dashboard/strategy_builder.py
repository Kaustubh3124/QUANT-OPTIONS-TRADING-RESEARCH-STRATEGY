# =============================================================================
# Tab 3: Strategy Builder
# =============================================================================
# Interactive strategy selection, payoff diagrams, Greeks display,
# breakeven analysis, and side-by-side comparison.
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from core.options import list_strategies, get_strategy, bs_call_price, bs_put_price
from core.greeks import strategy_greeks, strategy_greeks_table
from config import RISK_FREE_RATE


def render(ticker_name: str, start: str, end: str):
    """Render the Strategy Builder tab."""

    st.markdown("## 🔧 Strategy Builder")

    # --- Sidebar inputs ---
    col_input, col_chart = st.columns([1, 3])

    with col_input:
        st.markdown("### Parameters")

        strategy_name = st.selectbox(
            "Strategy",
            list_strategies(),
            format_func=lambda x: get_strategy(x).name,
            key="sb_strategy",
        )

        spot = st.number_input("Spot Price (₹)", value=22000.0, step=100.0, key="sb_spot")
        sigma = st.slider(
            "Implied Volatility (%)", 5, 60, 15, key="sb_sigma"
        ) / 100
        dte = st.slider("Days to Expiry", 5, 90, 30, key="sb_dte")
        r = st.number_input(
            "Risk-Free Rate (%)", value=RISK_FREE_RATE * 100,
            step=0.1, key="sb_rfr",
        ) / 100

        tte = dte / 252  # Convert to years

        strategy = get_strategy(strategy_name)

    with col_chart:
        # --- Payoff Diagram ---
        st.markdown(f"### {strategy.name} — Payoff Diagram")

        curve = strategy.payoff_curve(spot, sigma, tte, r, pct_range=0.15)

        fig = go.Figure()

        # PnL line
        fig.add_trace(go.Scatter(
            x=curve["Spot"], y=curve["PnL"],
            mode="lines",
            name="P&L at Expiry",
            line=dict(color="#00d4aa", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.1)",
        ))

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

        # Spot price line
        fig.add_vline(
            x=spot, line_dash="dot", line_color="#ffa502",
            annotation_text=f"Spot: ₹{spot:,.0f}",
            annotation_position="top",
        )

        # Breakeven points
        breakevens = strategy.breakeven_points(spot, sigma, tte, r)
        for be in breakevens:
            fig.add_vline(
                x=be, line_dash="dash", line_color="#ff6348",
                annotation_text=f"BE: ₹{be:,.0f}",
                annotation_position="bottom",
            )

        fig.update_layout(
            height=450,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Spot Price at Expiry (₹)",
            yaxis_title="Profit / Loss (₹)",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

        st.plotly_chart(fig, use_container_width=True)

    # --- Strategy Details ---
    col1, col2, col3 = st.columns(3)

    entry_cost = strategy.entry_cost(spot, sigma, tte, r)
    max_prof = strategy.max_profit(spot, sigma, tte, r)
    max_los = strategy.max_loss(spot, sigma, tte, r)

    col1.metric(
        "Entry Cost",
        f"₹{abs(entry_cost):,.2f}",
        "Credit" if entry_cost < 0 else "Debit",
    )
    col2.metric("Max Profit", f"₹{max_prof:,.2f}")
    col3.metric("Max Loss", f"₹{max_los:,.2f}")

    # --- Legs Detail ---
    st.markdown("### Strategy Legs")
    legs_info = strategy.get_legs_info(spot, sigma, tte, r)
    legs_df = pd.DataFrame(legs_info)
    st.dataframe(legs_df, use_container_width=True, hide_index=True)

    # --- Greeks ---
    st.markdown("### Greeks Exposure")

    greeks = strategy_greeks(strategy, spot, sigma, tte, r)

    gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
    gcol1.metric("Delta", f"{greeks['Delta']:.4f}")
    gcol2.metric("Gamma", f"{greeks['Gamma']:.6f}")
    gcol3.metric("Theta", f"₹{greeks['Theta']:.2f}/day")
    gcol4.metric("Vega", f"₹{greeks['Vega']:.2f}/1%σ")
    gcol5.metric("Rho", f"₹{greeks['Rho']:.2f}/1%r")

    # Per-leg Greeks table
    with st.expander("📋 Per-Leg Greeks Breakdown"):
        greeks_table = strategy_greeks_table(strategy, spot, sigma, tte, r)
        st.dataframe(greeks_table, use_container_width=True, hide_index=True)

    # --- Strategy Comparison ---
    st.markdown("### 📊 Strategy Comparison")
    compare_names = st.multiselect(
        "Select strategies to compare",
        list_strategies(),
        default=["LongStraddle", "IronCondor"],
        format_func=lambda x: get_strategy(x).name,
        key="sb_compare",
    )

    if compare_names:
        fig_compare = go.Figure()
        colors = ["#00d4aa", "#ff6348", "#3742fa", "#ffa502", "#e056fd", "#2ed573", "#f8a5c2"]

        for i, name in enumerate(compare_names):
            strat = get_strategy(name)
            curve = strat.payoff_curve(spot, sigma, tte, r, pct_range=0.15)
            fig_compare.add_trace(go.Scatter(
                x=curve["Spot"], y=curve["PnL"],
                mode="lines",
                name=strat.name,
                line=dict(color=colors[i % len(colors)], width=2),
            ))

        fig_compare.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig_compare.add_vline(x=spot, line_dash="dot", line_color="rgba(255,255,255,0.3)")

        fig_compare.update_layout(
            height=400,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Spot Price at Expiry (₹)",
            yaxis_title="Profit / Loss (₹)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig_compare.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
        fig_compare.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

        st.plotly_chart(fig_compare, use_container_width=True)
