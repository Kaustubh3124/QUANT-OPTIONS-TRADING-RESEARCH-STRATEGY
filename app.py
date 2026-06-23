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

    /* Global fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Main background - Deep Midnight FinTech Theme */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a, #020617);
        color: #e2e8f0;
        animation: fadeIn 0.8s ease-out forwards;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.15) !important;
    }
    
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Segmented Control Tab styling (macOS/iOS inspired) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(30, 41, 59, 0.5);
        padding: 4px;
        border-radius: 14px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        color: #64748b;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(139, 92, 246, 0.15) !important;
        color: #a78bfa !important;
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.2) !important;
    }

    /* Premium Metric cards - Glowing Glassmorphism */
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        position: relative;
        overflow: hidden;
    }
    
    /* Multicolored Glow effect on hover */
    [data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.02) !important;
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.2) !important;
        border-color: rgba(139, 92, 246, 0.4) !important;
    }
    
    /* Subtle top gradient line for metrics */
    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.8), transparent);
        opacity: 0.5;
        transition: opacity 0.3s;
    }
    [data-testid="stMetric"]:hover::before {
        opacity: 1;
        background: linear-gradient(90deg, transparent, #06b6d4, transparent);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        background: linear-gradient(135deg, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 8px;
        letter-spacing: -1px;
    }

    /* Sleek Dataframe styling */
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .stDataFrame [data-testid="stTable"] {
        background: transparent !important;
    }

    /* Premium Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.5px;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5);
        filter: brightness(1.1);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.3);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.2s ease;
        font-weight: 600;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(30, 41, 59, 0.6);
        border-color: rgba(139, 92, 246, 0.3);
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Global Font */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }

    /* Slider track color */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #8b5cf6;
        box-shadow: 0 0 12px rgba(139, 92, 246, 0.6);
        border-color: #ffffff !important;
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
            background: linear-gradient(135deg, #8b5cf6, #3b82f6, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
            letter-spacing: -1px;
        ">📊 QuantOptions</h1>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin: 0; letter-spacing: 0.5px; text-transform: uppercase;">
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
        background: rgba(139, 92, 246, 0.05);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 16px;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
        box-shadow: inset 0 0 20px rgba(139, 92, 246, 0.05);
    ">
        <strong style="color: #a78bfa; font-size: 0.95rem; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <span style="display: inline-block; width: 8px; height: 8px; background: #06b6d4; border-radius: 50%; box-shadow: 0 0 8px #06b6d4;"></span>
            Active Modules
        </strong>
        <div style="line-height: 1.6; padding-left: 16px; border-left: 1px solid rgba(255,255,255,0.1);">
            <span style="color: #06b6d4;">✓</span> Market Data (yfinance)<br/>
            <span style="color: #06b6d4;">✓</span> Volatility Analysis + GARCH<br/>
            <span style="color: #06b6d4;">✓</span> Regime Detection (SMA + ADX)<br/>
            <span style="color: #06b6d4;">✓</span> 7 Option Strategies<br/>
            <span style="color: #06b6d4;">✓</span> Strategy Decision Engine<br/>
            <span style="color: #06b6d4;">✓</span> Backtesting Engine<br/>
            <span style="color: #06b6d4;">✓</span> Risk Analytics (10+ metrics)<br/>
            <span style="color: #06b6d4;">✓</span> Greeks Dashboard<br/>
            <span style="color: #06b6d4;">✓</span> Portfolio Simulator<br/>
            <span style="color: #06b6d4;">✓</span> ML Volatility Signal
        </div>
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
