# =============================================================================
# Quant Options Trading Strategy Research Platform
# =============================================================================
# Streamlit entry point — main application with 8 interactive tabs.
#
# Usage:
#   streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    STREAMLIT_PAGE_TITLE, STREAMLIT_PAGE_ICON, STREAMLIT_LAYOUT,
    TICKERS, START_DATE, END_DATE, DEFAULT_TICKER,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded",
)


# =============================================================================
# Custom Styling
# =============================================================================

st.markdown("""
<style>
    /* Hide Streamlit top header and footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(8, 11, 20) 90%);
        color: #e2e8f0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(0, 212, 170, 0.15) !important;
    }
    
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.4);
        padding: 6px;
        border-radius: 12px;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: #94a3b8;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 170, 0.1);
        color: #f8fafc;
        transform: translateY(-1px);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.2), rgba(0, 184, 148, 0.1)) !important;
        border-bottom: 2px solid #00d4aa !important;
        color: #fff !important;
        box-shadow: 0 4px 12px rgba(0, 212, 170, 0.1) !important;
    }

    /* Metric cards - Glassmorphism */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
        position: relative;
        overflow: hidden;
    }
    
    /* Glow effect on hover */
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 8px 30px rgba(0, 212, 170, 0.15) !important;
        border-color: rgba(0, 212, 170, 0.3) !important;
    }
    
    /* Subtle top highlight for metrics */
    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        background: linear-gradient(135deg, #ffffff, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 4px;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00d4aa, #00b894);
        color: #020617;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(0, 212, 170, 0.2);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 170, 0.4);
        filter: brightness(1.1);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(30, 41, 59, 0.6);
        border-color: rgba(0, 212, 170, 0.2);
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        letter-spacing: -0.02em;
    }
    
    /* Global Font */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Remove default padding */
    .block-container {
        padding-top: 1.5rem;
        max-width: 1400px;
    }

    /* Slider track color */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #00d4aa;
        box-shadow: 0 0 10px rgba(0, 212, 170, 0.5);
    }
</style>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="
            font-size: 1.5rem;
            background: linear-gradient(135deg, #00d4aa, #00b894, #55efc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        ">📊 QuantOptions</h1>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin: 0;">
            Strategy Research Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Ticker selection
    ticker_name = st.selectbox(
        "📌 Select Ticker",
        list(TICKERS.keys()),
        index=list(TICKERS.keys()).index(DEFAULT_TICKER),
        key="global_ticker",
    )

    # Date range
    st.markdown("##### 📅 Date Range")
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "Start",
            value=pd.Timestamp(START_DATE),
            key="global_start",
        )
    with col_end:
        end_date = st.date_input(
            "End",
            value=pd.Timestamp(END_DATE),
            key="global_end",
        )

    start_str = str(start_date)
    end_str = str(end_date)

    st.markdown("---")

    st.markdown("""
    <div style="
        background: rgba(0, 212, 170, 0.05);
        border: 1px solid rgba(0, 212, 170, 0.15);
        border-radius: 8px;
        padding: 12px;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.6);
    ">
        <strong style="color: #00d4aa;">Modules</strong><br/>
        ✅ Market Data (yfinance)<br/>
        ✅ Volatility Analysis + GARCH<br/>
        ✅ Regime Detection (SMA + ADX)<br/>
        ✅ 7 Option Strategies<br/>
        ✅ Strategy Decision Engine<br/>
        ✅ Backtesting Engine<br/>
        ✅ Risk Analytics (10+ metrics)<br/>
        ✅ Greeks Dashboard<br/>
        ✅ Portfolio Simulator<br/>
        ✅ ML Volatility Signal<br/>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Main Content — Tabs
# =============================================================================

st.markdown("""
<h1 style="
    text-align: center;
    font-size: 2rem;
    background: linear-gradient(135deg, #00d4aa, #00b894, #55efc4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
">Quant Options Trading Research Platform</h1>
<p style="text-align: center; color: rgba(255,255,255,0.5); font-size: 0.9rem; margin-top: 4px;">
    Systematic strategy generation, backtesting, and risk analysis for options trading
</p>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Market Overview",
    "📊 Volatility",
    "🔧 Strategy Builder",
    "🧪 Backtest",
    "⚠️ Risk Analytics",
    "🔬 Greeks",
    "💼 Portfolio",
    "🎯 Recommendations",
])


# Lazy-load each tab to avoid importing everything upfront
with tab1:
    try:
        from dashboard.market_overview import render as render_market
        render_market(ticker_name, start_str, end_str)
    except Exception as e:
        st.error(f"Error loading Market Overview: {e}")

with tab2:
    try:
        from dashboard.volatility_tab import render as render_vol
        render_vol(ticker_name, start_str, end_str)
    except Exception as e:
        st.error(f"Error loading Volatility Analysis: {e}")

with tab3:
    try:
        from dashboard.strategy_builder import render as render_strat
        render_strat(ticker_name, start_str, end_str)
    except Exception as e:
        st.error(f"Error loading Strategy Builder: {e}")

with tab4:
    try:
        from dashboard.backtest_tab import render as render_bt
        render_bt(ticker_name, start_str, end_str)
    except Exception as e:
        st.error(f"Error loading Backtest Results: {e}")

with tab5:
    try:
        from dashboard.risk_tab import render as render_risk
        render_risk(ticker_name, start_str, end_str)
    except Exception as e:
        st.error(f"Error loading Risk Analytics: {e}")

with tab6:
    try:
        from dashboard.greeks_tab import render as render_greeks
        render_greeks(ticker_name, start_str, end_str)
    except Exception as e:
        st.error(f"Error loading Greeks: {e}")

with tab7:
    try:
        from dashboard.portfolio_tab import render as render_portfolio
        render_portfolio(ticker_name, start_str, end_str)
    except Exception as e:
        st.error(f"Error loading Portfolio Simulator: {e}")

with tab8:
    try:
        from dashboard.recommendation_tab import render as render_rec
        render_rec(ticker_name, start_str, end_str)
    except Exception as e:
        st.error(f"Error loading Recommendations: {e}")


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.3); font-size: 0.75rem; padding: 10px 0;">
    Quant Options Trading Strategy Research Platform • Built with Python, Streamlit & Plotly
    <br/>
    Data sourced from Yahoo Finance via yfinance • Pricing via Black-Scholes model
</div>
""", unsafe_allow_html=True)
