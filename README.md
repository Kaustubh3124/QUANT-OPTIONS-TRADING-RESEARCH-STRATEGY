# 📊 Quant Options Trading Strategy Research Platform

> A quantitative research platform for options trading that dynamically selects and backtests multi-leg strategies based on volatility regimes and market conditions.

---

## 🎯 Goal

> **"Can we systematically generate profits using options under different volatility and market conditions?"**

This platform answers that question by combining volatility analysis, regime detection, strategy selection, backtesting, and risk analytics into a single interactive research environment.

---

## 🏗️ Architecture

```
Market Data (yfinance)
    │
    ▼
Volatility Analysis (HV, RV, GARCH)
    │
    ▼
Market Regime Detection (SMA + ADX)
    │
    ▼
Strategy Selection Engine (Rules + Scoring)
    │
    ▼
Backtesting Engine (2019–2025)
    │
    ▼
Risk Analytics (Sharpe, Drawdown, VaR, ...)
    │
    ▼
Interactive Dashboard (Streamlit + Plotly)
```

---

## 📦 Modules

| Module | File | Description |
|--------|------|-------------|
| **Market Data** | `core/market_data.py` | OHLCV data via yfinance with CSV caching |
| **Volatility** | `core/volatility.py` | HV, RV, percentile, cone, GARCH(1,1) |
| **Regime Detection** | `core/regime.py` | Bull/Bear/Sideways via SMA crossover + ADX |
| **Options Engine** | `core/options.py` | 7 strategies with Black-Scholes pricing |
| **Greeks** | `core/greeks.py` | Delta, Gamma, Theta, Vega, Rho analytics |
| **Strategy Selector** | `core/strategy_selector.py` | Rules + scoring decision engine |
| **Backtester** | `core/backtester.py` | Static & dynamic backtesting with costs |
| **Risk Analytics** | `core/risk_analytics.py` | 10+ metrics: Sharpe, Sortino, VaR, etc. |
| **Portfolio** | `core/portfolio.py` | Multi-strategy portfolio simulation |
| **ML Signals** | `core/ml_signals.py` | XGBoost/RF volatility prediction |

---

## 🔧 Strategies Implemented

| Strategy | Type | Best When |
|----------|------|-----------|
| Long Straddle | Vol Buying | Low vol, expect expansion |
| Short Straddle | Vol Selling | High vol, expect crush |
| Iron Condor | Defined Risk | High vol, sideways market |
| Bull Call Spread | Directional | Bullish regime |
| Bear Put Spread | Directional | Bearish regime |
| Covered Call | Income | Mild bull / neutral |
| Calendar Spread | Time Value | Sideways, longer DTE |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd QUANT-OPTIONS-TRADING-RESEARCH-STRATEGY
pip install -r requirements.txt
```

### 2. Launch the dashboard

```bash
streamlit run app.py
```

### 3. Run tests

```bash
python -m pytest tests/ -v
```

---

## 📊 Dashboard Tabs

1. **📈 Market Overview** — Candlestick charts, SMA overlays, regime shading
2. **📊 Volatility** — Multi-window HV, volatility cone, GARCH forecasts
3. **🔧 Strategy Builder** — Interactive payoff diagrams, Greeks, comparison
4. **🧪 Backtest** — Equity curves, monthly returns heatmap, trade log
5. **⚠️ Risk Analytics** — Sharpe/Sortino/VaR, return distribution, drawdown
6. **🔬 Greeks** — Theta decay, sensitivity heatmaps, time-series exposure
7. **💼 Portfolio** — Multi-strategy allocation, attribution, correlation
8. **🎯 Recommendations** — ML-powered strategy suggestions

---

## 🧠 Advanced Features

### GARCH Volatility Forecasting
- GARCH(1,1) with Student-t distribution
- 5–10 day forward volatility forecasts
- Conditional volatility time series

### Machine Learning Signal
- XGBoost / Random Forest classifiers
- Predicts volatility expansion vs. crush
- Walk-forward cross-validation (no look-ahead bias)
- Feature importance analysis

### Strategy Recommendation System
- Input: Current volatility, regime, DTE
- Output: Top-N strategies with scores and explanations
- Scoring matrix: 3 vol buckets × 3 regimes × 7 strategies

---

## 📈 Risk Metrics

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| CAGR | Compound Annual Growth Rate |
| Win Rate | % of profitable trades |
| Profit Factor | Gross profits / Gross losses |
| Expected Value | Average expected P&L per trade |
| VaR (95%) | Value at Risk |
| CVaR (95%) | Conditional VaR (Expected Shortfall) |
| Calmar Ratio | CAGR / Max Drawdown |

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas / NumPy / SciPy** — Data manipulation & statistics
- **yfinance** — Market data
- **Plotly** — Interactive visualizations
- **Streamlit** — Dashboard framework
- **arch** — GARCH volatility models
- **XGBoost / scikit-learn** — Machine learning

---

## 📁 Project Structure

```
├── app.py                  # Streamlit entry point
├── config.py               # Global configuration
├── requirements.txt
├── data/                   # Cached market data
├── core/
│   ├── market_data.py      # Data collection
│   ├── volatility.py       # Volatility analysis
│   ├── regime.py           # Regime detection
│   ├── options.py          # Strategy engine
│   ├── greeks.py           # Greeks calculation
│   ├── strategy_selector.py # Decision engine
│   ├── backtester.py       # Backtest loop
│   ├── risk_analytics.py   # Risk metrics
│   ├── portfolio.py        # Portfolio simulator
│   └── ml_signals.py       # ML volatility signal
├── dashboard/
│   ├── market_overview.py  # Tab 1
│   ├── volatility_tab.py   # Tab 2
│   ├── strategy_builder.py # Tab 3
│   ├── backtest_tab.py     # Tab 4
│   ├── risk_tab.py         # Tab 5
│   ├── greeks_tab.py       # Tab 6
│   ├── portfolio_tab.py    # Tab 7
│   └── recommendation_tab.py # Tab 8
└── tests/
    ├── test_volatility.py
    ├── test_options.py
    ├── test_backtester.py
    └── test_risk_analytics.py
```


