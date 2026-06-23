# =============================================================================
# Tab 2: Volatility Analysis
# =============================================================================
# HV time series, volatility cone, percentile gauge, GARCH forecasts,
# realized vs historical vol comparison, term structure.
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from core.market_data import fetch_ohlcv, get_returns
from core.volatility import (
    historical_volatility, realized_volatility, volatility_percentile,
    current_vol_percentile, volatility_term_structure, volatility_cone,
    fit_garch, forecast_volatility, garch_conditional_volatility,
)


def render(ticker_name: str, start: str, end: str):
    """Render the Volatility Analysis tab."""

    st.markdown("## 📊 Volatility Analysis")

    with st.spinner("Computing volatility metrics..."):
        df = fetch_ohlcv(ticker_name, start, end)
        returns = get_returns(df)

        hv_21 = historical_volatility(returns, window=21)
        hv_10 = historical_volatility(returns, window=10)
        hv_63 = historical_volatility(returns, window=63)
        rv_21 = realized_volatility(returns, window=21)
        vol_pct = current_vol_percentile(hv_21)

    # --- Volatility Metrics ---
    col1, col2, col3, col4 = st.columns(4)

    current_hv = hv_21.iloc[-1] if not hv_21.empty else 0
    col1.metric("21-Day HV", f"{current_hv * 100:.1f}%")

    current_rv = rv_21.iloc[-1] if not rv_21.empty else 0
    col2.metric("21-Day RV", f"{current_rv * 100:.1f}%")

    vol_pct_display = vol_pct if not np.isnan(vol_pct) else 0
    col3.metric("Vol Percentile", f"{vol_pct_display:.0f}th")

    # Vol regime indicator
    if vol_pct_display > 80:
        vol_regime = "🔴 HIGH"
    elif vol_pct_display < 20:
        vol_regime = "🟢 LOW"
    else:
        vol_regime = "🟡 NORMAL"
    col4.metric("Vol Regime", vol_regime)

    # --- Multi-Window HV Time Series ---
    st.markdown("### Historical Volatility (Multi-Window)")

    fig_hv = go.Figure()
    for name, series, color in [
        ("10-Day HV", hv_10, "#ffa502"),
        ("21-Day HV", hv_21, "#00d4aa"),
        ("63-Day HV", hv_63, "#3742fa"),
    ]:
        fig_hv.add_trace(go.Scatter(
            x=series.index, y=series * 100,
            name=name, line=dict(color=color, width=1.5),
        ))

    fig_hv.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Annualized Volatility (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_hv.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
    fig_hv.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

    st.plotly_chart(fig_hv, use_container_width=True)

    # --- HV vs RV Comparison ---
    st.markdown("### Historical vs Realized Volatility")

    fig_hvrv = go.Figure()
    fig_hvrv.add_trace(go.Scatter(
        x=hv_21.index, y=hv_21 * 100,
        name="21-Day HV", line=dict(color="#00d4aa", width=2),
    ))
    fig_hvrv.add_trace(go.Scatter(
        x=rv_21.index, y=rv_21 * 100,
        name="21-Day RV", line=dict(color="#ff6348", width=2),
    ))

    fig_hvrv.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Annualized Volatility (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_hvrv.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
    fig_hvrv.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

    st.plotly_chart(fig_hvrv, use_container_width=True)

    # --- Volatility Cone ---
    col_cone, col_pct = st.columns([3, 2])

    with col_cone:
        st.markdown("### Volatility Cone")

        cone = volatility_cone(returns)
        if not cone.empty:
            fig_cone = go.Figure()

            # Shaded area: P25 to P75
            fig_cone.add_trace(go.Scatter(
                x=cone["Window"], y=cone["P75"] * 100,
                mode="lines", line=dict(width=0),
                showlegend=False,
            ))
            fig_cone.add_trace(go.Scatter(
                x=cone["Window"], y=cone["P25"] * 100,
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(0, 212, 170, 0.15)",
                name="P25–P75 Range",
            ))

            # Lines
            for name, col, color, dash in [
                ("Max", "Max", "#ff4757", "dot"),
                ("Median", "Median", "#ffa502", "dash"),
                ("Min", "Min", "#2ed573", "dot"),
                ("Current", "Current", "#ffffff", "solid"),
            ]:
                fig_cone.add_trace(go.Scatter(
                    x=cone["Window"], y=cone[col] * 100,
                    name=name, line=dict(color=color, width=2, dash=dash),
                ))

            fig_cone.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Lookback Window (Days)",
                yaxis_title="Annualized Volatility (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_cone.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
            fig_cone.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

            st.plotly_chart(fig_cone, use_container_width=True)

    with col_pct:
        st.markdown("### Volatility Percentile")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=vol_pct_display,
            number={"suffix": "th", "font": {"size": 36, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": "#00d4aa"},
                "bgcolor": "rgba(255,255,255,0.05)",
                "steps": [
                    {"range": [0, 20], "color": "rgba(46, 213, 115, 0.3)"},
                    {"range": [20, 80], "color": "rgba(255, 165, 2, 0.2)"},
                    {"range": [80, 100], "color": "rgba(255, 71, 87, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": vol_pct_display,
                },
            },
        ))
        fig_gauge.update_layout(
            height=300,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Interpretation
        if vol_pct_display > 80:
            st.error(
                "⚠️ Volatility is **unusually high** — "
                "premium selling strategies (Iron Condor, Short Straddle) may be favorable."
            )
        elif vol_pct_display < 20:
            st.success(
                "✅ Volatility is **unusually low** — "
                "options are cheap. Long Straddle may capture vol expansion."
            )
        else:
            st.info(
                "ℹ️ Volatility is in the **normal range**. "
                "Directional strategies aligned with regime may work best."
            )

    # --- GARCH Forecast ---
    st.markdown("### GARCH(1,1) Volatility Forecast")

    try:
        with st.spinner("Fitting GARCH model..."):
            garch_result = fit_garch(returns)
            forecast = forecast_volatility(garch_result, horizon=10)
            cond_vol = garch_conditional_volatility(garch_result)

        col_gf1, col_gf2 = st.columns([2, 1])

        with col_gf1:
            # Conditional vol plot (last 252 days) + forecast
            recent_cv = cond_vol.tail(252)

            fig_garch = go.Figure()
            fig_garch.add_trace(go.Scatter(
                x=recent_cv.index, y=recent_cv * 100,
                name="GARCH Conditional Vol",
                line=dict(color="#e056fd", width=2),
            ))
            fig_garch.add_trace(go.Scatter(
                x=hv_21.tail(252).index, y=hv_21.tail(252) * 100,
                name="21-Day HV",
                line=dict(color="#00d4aa", width=1.5, dash="dash"),
            ))

            fig_garch.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Annualized Volatility (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_garch.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
            fig_garch.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

            st.plotly_chart(fig_garch, use_container_width=True)

        with col_gf2:
            st.markdown("**Forward Forecast**")
            forecast_display = forecast.copy()
            forecast_display["Forecast_Annual_Vol"] = (
                forecast_display["Forecast_Annual_Vol"] * 100
            ).round(2)
            forecast_display.columns = ["Day", "Daily Vol", "Annual Vol (%)"]
            st.dataframe(forecast_display, use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"GARCH model could not be fitted: {e}")
