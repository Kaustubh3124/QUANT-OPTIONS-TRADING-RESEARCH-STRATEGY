# =============================================================================
# Tab 8: Strategy Recommendation
# =============================================================================
# Current market state, ML signal confidence, top-N strategy recommendations
# with explanations and historical performance under similar conditions.
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from core.market_data import fetch_ohlcv, get_returns, get_close_prices
from core.volatility import (
    historical_volatility, current_vol_percentile,
    fit_garch, forecast_volatility,
)
from core.regime import detect_regime, get_current_regime
from core.strategy_selector import recommend_top_n, select_strategy
from core.options import get_strategy
from config import RISK_FREE_RATE


def render(ticker_name: str, start: str, end: str):
    """Render the Strategy Recommendation tab."""

    st.markdown("## 🎯 Strategy Recommendation Engine")

    with st.spinner("Analyzing current market state..."):
        df = fetch_ohlcv(ticker_name, start, end)
        returns = get_returns(df)
        close = get_close_prices(df)

        hv_21 = historical_volatility(returns, window=21)
        vol_pct = current_vol_percentile(hv_21)
        regime = get_current_regime(df)

        current_hv = hv_21.iloc[-1] if not hv_21.empty else 0.15
        current_spot = close.iloc[-1] if not close.empty else 0

    # --- Current Market State ---
    st.markdown("### 📡 Current Market State")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Spot Price", f"₹{current_spot:,.2f}")

    regime_emoji = {"BULL": "🟢", "BEAR": "🔴", "SIDEWAYS": "🟡"}.get(regime, "⚪")
    col2.metric("Market Regime", f"{regime_emoji} {regime}")

    col3.metric("21-Day HV", f"{current_hv * 100:.1f}%")

    vol_pct_display = vol_pct if not np.isnan(vol_pct) else 50
    if vol_pct_display > 80:
        pct_color = "🔴"
    elif vol_pct_display < 20:
        pct_color = "🟢"
    else:
        pct_color = "🟡"
    col4.metric("Vol Percentile", f"{pct_color} {vol_pct_display:.0f}th")

    # --- GARCH Forecast Summary ---
    try:
        garch_result = fit_garch(returns)
        forecast = forecast_volatility(garch_result, horizon=5)
        avg_forecast = forecast["Forecast_Annual_Vol"].mean()

        vol_direction = "📈 Expanding" if avg_forecast > current_hv else "📉 Contracting"
        st.info(f"**GARCH Forecast**: 5-day average annualized vol = "
                f"**{avg_forecast * 100:.1f}%** ({vol_direction} vs current {current_hv * 100:.1f}%)")
    except Exception:
        avg_forecast = current_hv
        st.warning("GARCH forecast unavailable.")

    # --- ML Signal ---
    st.markdown("### 🤖 ML Volatility Signal")

    try:
        from core.ml_signals import build_features, build_target, train_model, predict_signal

        with st.spinner("Training ML model (walk-forward)... This may take a moment."):
            volume = df["Volume"] if "Volume" in df.columns else None
            regimes = detect_regime(df)
            features = build_features(returns, close, volume, regimes)
            target = build_target(returns)

            result = train_model(features, target, model_type="random_forest")

        if result.get("model") is not None:
            signal = predict_signal(result, features)

            col_sig1, col_sig2, col_sig3 = st.columns(3)

            signal_emoji = "📈" if signal["signal"] == "VOL_EXPANSION" else "📉"
            col_sig1.metric("Signal", f"{signal_emoji} {signal['signal'].replace('_', ' ')}")
            col_sig2.metric("Confidence", f"{signal['probability'] * 100:.0f}%")
            col_sig3.metric("Model Accuracy", f"{result['accuracy'] * 100:.1f}%")

            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=signal["probability"] * 100,
                number={"suffix": "%", "font": {"size": 32, "color": "white"}},
                title={"text": "Vol Expansion Probability", "font": {"size": 14, "color": "rgba(255,255,255,0.7)"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e056fd"},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(46, 213, 115, 0.2)"},
                        {"range": [30, 70], "color": "rgba(255, 165, 2, 0.15)"},
                        {"range": [70, 100], "color": "rgba(244, 63, 94, 0.2)"},
                    ],
                },
            ))
            fig_gauge.update_layout(
                height=250,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=50, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Feature importance
            with st.expander("📊 Feature Importance"):
                top_features = result["feature_importance"].head(10)
                fig_feat = go.Figure(data=[go.Bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation="h",
                    marker_color="#e056fd",
                )])
                fig_feat.update_layout(
                    height=350,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Importance",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                fig_feat.update_xaxes(gridcolor="rgba(255,255,255,0.03)")
                st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.warning(f"ML model could not be trained: {result.get('error', 'Unknown error')}")

    except Exception as e:
        st.warning(f"ML signal unavailable: {e}")

    # --- Top Strategy Recommendations ---
    st.markdown("### 🏆 Recommended Strategies")

    dte_input = st.slider("Planned DTE (days)", 7, 90, 30, key="rec_dte")

    recommendations = recommend_top_n(vol_pct_display, regime, dte_input, n=5)

    for i, rec in enumerate(recommendations):
        rank = i + 1
        score = rec["score"]
        strat = rec["strategy"]
        name = rec["name"]
        reason = rec["reason"]

        # Color based on rank
        if rank == 1:
            border_color = "#8b5cf6"
            badge = "🥇 TOP PICK"
        elif rank == 2:
            border_color = "#ffa502"
            badge = "🥈"
        elif rank == 3:
            border_color = "#3742fa"
            badge = "🥉"
        else:
            border_color = "rgba(255,255,255,0.2)"
            badge = f"#{rank}"

        with st.container():
            st.markdown(
                f"""
                <div style="
                    border-left: 4px solid {border_color};
                    padding: 12px 16px;
                    margin: 8px 0;
                    background: rgba(255,255,255,0.03);
                    border-radius: 0 8px 8px 0;
                ">
                    <span style="font-size: 1.1em; font-weight: 600;">
                        {badge} {strat.name}
                    </span>
                    <span style="float: right; color: {border_color}; font-weight: 700;">
                        Score: {score}/100
                    </span>
                    <br/>
                    <span style="color: rgba(255,255,255,0.7); font-size: 0.9em;">
                        {reason}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Decision Logic Explanation ---
    with st.expander("🧠 How the Decision Engine Works"):
        st.markdown("""
        The strategy recommendation engine uses a **two-layer approach**:

        **Layer 1: Rules-Based Selection**
        - **High Vol (>80th percentile)** → Sell premium (Iron Condor, Short Straddle)
        - **Low Vol (<20th percentile)** → Buy cheap options (Long Straddle)
        - **Bull regime** → Bull Call Spread
        - **Bear regime** → Bear Put Spread
        - **Sideways** → Iron Condor or Calendar Spread

        **Layer 2: Scoring Matrix**
        - Each strategy is scored (0–100) based on a (regime × vol_bucket) matrix
        - TTE bonuses: theta-selling strategies get a boost with <30 DTE
        - Strategies are ranked by composite score

        **ML Enhancement** (when available)
        - Random Forest / XGBoost predicts volatility expansion vs. crush
        - Uses walk-forward validation to avoid look-ahead bias
        - Signal confidence adjusts the weight of vol-directional strategies
        """)
