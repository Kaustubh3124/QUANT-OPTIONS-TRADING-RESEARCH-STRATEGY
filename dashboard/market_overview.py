# =============================================================================
# Tab 1: Market Overview
# =============================================================================
# Candlestick chart with SMA overlays, volume bars, regime shading,
# and key market metrics.
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from core.market_data import fetch_ohlcv
from core.regime import detect_regime, regime_summary, compute_moving_averages
from config import SMA_SHORT, SMA_LONG


def render(ticker_name: str, start: str, end: str):
    """Render the Market Overview tab."""

    st.markdown("## 📈 Market Overview")

    # Fetch data
    with st.spinner(f"Loading {ticker_name} data..."):
        df = fetch_ohlcv(ticker_name, start, end)
        df_sma = compute_moving_averages(df, SMA_SHORT, SMA_LONG)
        regimes = detect_regime(df)

    # --- Key Metrics Row ---
    col1, col2, col3, col4 = st.columns(4)

    current_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2] if len(df) > 1 else current_price
    daily_change = current_price - prev_price
    daily_change_pct = (daily_change / prev_price) * 100

    high_52w = df["Close"].rolling(252).max().iloc[-1]
    low_52w = df["Close"].rolling(252).min().iloc[-1]

    col1.metric(
        "Current Price",
        f"₹{current_price:,.2f}",
        f"{daily_change:+,.2f} ({daily_change_pct:+.2f}%)",
    )
    col2.metric("52-Week High", f"₹{high_52w:,.2f}")
    col3.metric("52-Week Low", f"₹{low_52w:,.2f}")

    current_regime = regimes.iloc[-1] if not regimes.empty else "UNKNOWN"
    regime_color = {"BULL": "🟢", "BEAR": "🔴", "SIDEWAYS": "🟡"}.get(
        current_regime, "⚪"
    )
    col4.metric("Current Regime", f"{regime_color} {current_regime}")

    # --- Candlestick + SMA Chart ---
    st.markdown("### Price Chart with Moving Averages & Regime")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=("", "Volume"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#8b5cf6",
            decreasing_line_color="#f43f5e",
        ),
        row=1, col=1,
    )

    # SMAs
    sma_s = df_sma[f"SMA_{SMA_SHORT}"]
    sma_l = df_sma[f"SMA_{SMA_LONG}"]

    fig.add_trace(
        go.Scatter(
            x=sma_s.index, y=sma_s,
            name=f"SMA {SMA_SHORT}",
            line=dict(color="#ffa502", width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sma_l.index, y=sma_l,
            name=f"SMA {SMA_LONG}",
            line=dict(color="#3742fa", width=1.5),
        ),
        row=1, col=1,
    )

    # Regime shading
    regime_colors = {"BULL": "rgba(139, 92, 246, 0.08)", "BEAR": "rgba(244, 63, 94, 0.08)", "SIDEWAYS": "rgba(255, 165, 2, 0.05)"}

    # Add regime annotations at the top
    prev_regime = None
    start_idx = 0
    for i in range(len(regimes)):
        r = regimes.iloc[i]
        if r != prev_regime:
            if prev_regime is not None and prev_regime in regime_colors:
                fig.add_vrect(
                    x0=regimes.index[start_idx],
                    x1=regimes.index[i - 1],
                    fillcolor=regime_colors[prev_regime],
                    layer="below",
                    line_width=0,
                    row=1, col=1,
                )
            prev_regime = r
            start_idx = i

    # Final segment
    if prev_regime in regime_colors:
        fig.add_vrect(
            x0=regimes.index[start_idx],
            x1=regimes.index[-1],
            fillcolor=regime_colors[prev_regime],
            layer="below",
            line_width=0,
            row=1, col=1,
        )

    # Volume bars
    colors = [
        "#8b5cf6" if c >= o else "#f43f5e"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.6,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.03)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.03)")

    st.plotly_chart(fig, use_container_width=True)

    # --- Regime Summary Table ---
    st.markdown("### Regime Distribution")

    summary = regime_summary(regimes)
    summary = summary[summary["Regime"] != "UNKNOWN"]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(
            summary.style.format({
                "Percentage": "{:.1f}%",
                "Avg_Streak_Days": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        fig_regime = go.Figure(data=[
            go.Pie(
                labels=summary["Regime"],
                values=summary["Days"],
                hole=0.45,
                marker_colors=["#8b5cf6", "#f43f5e", "#ffa502"],
                textinfo="label+percent",
                textfont_size=14,
            )
        ])
        fig_regime.update_layout(
            height=300,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_regime, use_container_width=True)
