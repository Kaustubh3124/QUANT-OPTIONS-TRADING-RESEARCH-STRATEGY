# =============================================================================
# Quant Options Trading Strategy Research Platform — Configuration
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Ticker Definitions (yfinance format)
# ---------------------------------------------------------------------------
TICKERS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
}

DEFAULT_TICKER = "NIFTY"

# ---------------------------------------------------------------------------
# Date Range
# ---------------------------------------------------------------------------
START_DATE = "2019-01-01"
END_DATE = "2025-06-23"

# ---------------------------------------------------------------------------
# Financial Constants
# ---------------------------------------------------------------------------
RISK_FREE_RATE = 0.07          # India 10-year government bond yield (~7%)
TRADING_DAYS_PER_YEAR = 252
DEFAULT_DTE = 30               # Default days to expiry for backtests

# ---------------------------------------------------------------------------
# Capital & Portfolio
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 10_00_000    # ₹10,00,000

PORTFOLIO_WEIGHTS = {
    "IronCondor": 0.40,
    "LongStraddle": 0.30,
    "BullCallSpread": 0.15,
    "BearPutSpread": 0.15,
}

# ---------------------------------------------------------------------------
# Volatility Thresholds (for strategy selection)
# ---------------------------------------------------------------------------
VOL_HIGH_PERCENTILE = 80       # Above this → sell volatility strategies
VOL_LOW_PERCENTILE = 20        # Below this → buy volatility strategies

# ---------------------------------------------------------------------------
# Backtesting Parameters
# ---------------------------------------------------------------------------
TRANSACTION_COST = 20          # ₹20 per trade (brokerage)
SLIPPAGE_PCT = 0.001           # 0.1% slippage
ENTRY_FREQUENCY = "monthly"    # Trade entry frequency
LOT_SIZE = {                   # NSE lot sizes (approximate)
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "RELIANCE": 250,
    "HDFCBANK": 550,
}

# ---------------------------------------------------------------------------
# Strike Selection
# ---------------------------------------------------------------------------
OTM_OFFSET_PCT = 0.05          # 5% away from spot for OTM strikes
FAR_OTM_OFFSET_PCT = 0.10      # 10% away for far-OTM (iron condor wings)

# ---------------------------------------------------------------------------
# Regime Detection
# ---------------------------------------------------------------------------
SMA_SHORT = 50
SMA_LONG = 200
ADX_THRESHOLD = 20             # Below this → sideways
REGIME_BAND_PCT = 0.01         # SMA50 within 1% of SMA200 → sideways

# ---------------------------------------------------------------------------
# ML Signal Parameters
# ---------------------------------------------------------------------------
ML_LOOKBACK_WINDOWS = [5, 10, 21, 63]
ML_FORWARD_WINDOW = 10         # Predict vol expansion over next 10 days
ML_TRAIN_MIN_SAMPLES = 504     # Minimum 2 years of data before training

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
STREAMLIT_PAGE_TITLE = "Quant Options Trading Research Platform"
STREAMLIT_PAGE_ICON = "📊"
STREAMLIT_LAYOUT = "wide"
