# =============================================================================
# Module 6: Backtesting Engine
# =============================================================================
# Runs option strategies through historical data (2019–2025).
# Supports static single-strategy backtests and dynamic strategy selection
# based on volatility + regime signals.
# =============================================================================

import numpy as np
import pandas as pd
from datetime import timedelta

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RISK_FREE_RATE, TRADING_DAYS_PER_YEAR, DEFAULT_DTE,
    TRANSACTION_COST, SLIPPAGE_PCT, START_DATE, END_DATE,
)
from core.market_data import fetch_ohlcv, get_returns
from core.volatility import historical_volatility, current_vol_percentile
from core.regime import detect_regime, get_current_regime
from core.options import (
    OptionStrategy, get_strategy, list_strategies, STRATEGY_REGISTRY,
)
from core.strategy_selector import select_strategy


# =============================================================================
# Trade Entry Point Generator
# =============================================================================

def _generate_entry_dates(
    dates: pd.DatetimeIndex,
    frequency: str = "monthly",
) -> list[pd.Timestamp]:
    """
    Generate trade entry dates from a DatetimeIndex.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Available trading dates.
    frequency : str
        'monthly' → first trading day of each month
        'weekly'  → first trading day of each week
        'biweekly' → every 2 weeks

    Returns
    -------
    list[pd.Timestamp]
    """
    if frequency == "monthly":
        # Group by year-month, take first date
        grouped = pd.Series(dates, index=dates).groupby(
            [dates.year, dates.month]
        ).first()
        return grouped.tolist()
    elif frequency == "weekly":
        grouped = pd.Series(dates, index=dates).groupby(
            [dates.year, dates.isocalendar().week]
        ).first()
        return grouped.tolist()
    elif frequency == "biweekly":
        monthly = _generate_entry_dates(dates, "weekly")
        return monthly[::2]
    else:
        raise ValueError(f"Unknown frequency '{frequency}'")


def _find_exit_date(
    dates: pd.DatetimeIndex,
    entry_date: pd.Timestamp,
    dte: int = DEFAULT_DTE,
) -> pd.Timestamp | None:
    """
    Find the nearest trading date to entry_date + dte days.
    Returns None if exit date would be beyond available data.
    """
    target = entry_date + timedelta(days=dte)
    future = dates[dates >= target]
    if future.empty:
        return None
    return future[0]


# =============================================================================
# Single Trade Simulation
# =============================================================================

def _simulate_trade(
    strategy: OptionStrategy,
    spot_entry: float,
    spot_exit: float,
    sigma: float,
    tte_years: float,
    r: float = RISK_FREE_RATE,
    transaction_cost: float = TRANSACTION_COST,
    slippage_pct: float = SLIPPAGE_PCT,
) -> dict:
    """
    Simulate a single options trade.

    Returns
    -------
    dict
        Trade details: entry_cost, pnl, return_pct, etc.
    """
    # Entry cost (premium paid or received)
    entry_cost = strategy.entry_cost(spot_entry, sigma, tte_years, r)

    # Apply slippage to entry cost
    slippage = abs(entry_cost) * slippage_pct
    entry_cost_adjusted = entry_cost + slippage  # Slippage always costs money

    # P&L at expiry
    raw_pnl = strategy.pnl_at_expiry(spot_exit, spot_entry, sigma, tte_years, r)

    # Deduct transaction costs (per leg × 2 for entry + exit)
    n_legs = len(strategy.build_legs(spot_entry, sigma, tte_years, r))
    total_txn_cost = transaction_cost * n_legs * 2  # Entry + exit

    net_pnl = raw_pnl - slippage - total_txn_cost

    # Return percentage (based on capital at risk)
    capital_at_risk = abs(entry_cost_adjusted) + total_txn_cost
    if capital_at_risk > 0:
        return_pct = (net_pnl / capital_at_risk) * 100
    else:
        return_pct = 0.0

    return {
        "entry_cost": round(entry_cost, 2),
        "raw_pnl": round(raw_pnl, 2),
        "transaction_costs": round(total_txn_cost, 2),
        "slippage": round(slippage, 2),
        "net_pnl": round(net_pnl, 2),
        "capital_at_risk": round(capital_at_risk, 2),
        "return_pct": round(return_pct, 2),
    }


# =============================================================================
# Static Backtest (single strategy throughout)
# =============================================================================

def run_backtest(
    ticker: str,
    strategy_name: str,
    start: str = START_DATE,
    end: str = END_DATE,
    dte: int = DEFAULT_DTE,
    frequency: str = "monthly",
    hv_window: int = 21,
    r: float = RISK_FREE_RATE,
    transaction_cost: float = TRANSACTION_COST,
    slippage_pct: float = SLIPPAGE_PCT,
) -> pd.DataFrame:
    """
    Run a backtest for a single strategy across historical data.

    Parameters
    ----------
    ticker : str
        Ticker name (e.g. 'NIFTY').
    strategy_name : str
        Strategy to backtest (e.g. 'IronCondor').
    start, end : str
        Date range.
    dte : int
        Days to expiry per trade.
    frequency : str
        Entry frequency: 'monthly', 'weekly', 'biweekly'.

    Returns
    -------
    pd.DataFrame
        Trade log with columns: Entry_Date, Exit_Date, Spot_Entry, Spot_Exit,
        Sigma, Strategy, Entry_Cost, Net_PnL, Return_Pct, etc.
    """
    # Fetch data
    df = fetch_ohlcv(ticker, start, end)
    returns = get_returns(df)
    hv = historical_volatility(returns, window=hv_window)

    strategy = get_strategy(strategy_name)
    tte_years = dte / TRADING_DAYS_PER_YEAR

    # Generate entry dates
    entry_dates = _generate_entry_dates(df.index, frequency)

    trades = []
    for entry_date in entry_dates:
        # Find exit date
        exit_date = _find_exit_date(df.index, entry_date, dte)
        if exit_date is None:
            continue  # Skip if exit beyond data range

        # Get spot prices
        spot_entry = df.loc[entry_date, "Close"]
        spot_exit = df.loc[exit_date, "Close"]

        # Handle scalar extraction from potential Series
        if hasattr(spot_entry, 'iloc'):
            spot_entry = spot_entry.iloc[0]
        if hasattr(spot_exit, 'iloc'):
            spot_exit = spot_exit.iloc[0]

        # Get volatility at entry
        if entry_date in hv.index and not np.isnan(hv.loc[entry_date]):
            sigma = hv.loc[entry_date]
            if hasattr(sigma, 'iloc'):
                sigma = sigma.iloc[0]
        else:
            sigma = 0.20  # Default fallback

        # Simulate trade
        trade = _simulate_trade(
            strategy, spot_entry, spot_exit, sigma, tte_years,
            r, transaction_cost, slippage_pct,
        )

        trades.append({
            "Entry_Date": entry_date,
            "Exit_Date": exit_date,
            "Spot_Entry": round(float(spot_entry), 2),
            "Spot_Exit": round(float(spot_exit), 2),
            "Sigma": round(float(sigma), 4),
            "Strategy": strategy.name,
            **trade,
        })

    return pd.DataFrame(trades)


# =============================================================================
# Dynamic Backtest (strategy selected per trade based on conditions)
# =============================================================================

def run_dynamic_backtest(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
    dte: int = DEFAULT_DTE,
    frequency: str = "monthly",
    hv_window: int = 21,
    vol_lookback: int = 252,
    r: float = RISK_FREE_RATE,
    transaction_cost: float = TRANSACTION_COST,
    slippage_pct: float = SLIPPAGE_PCT,
) -> pd.DataFrame:
    """
    Run a dynamic backtest where strategy is selected at each entry
    based on volatility percentile and market regime.
    """
    df = fetch_ohlcv(ticker, start, end)
    returns = get_returns(df)
    hv = historical_volatility(returns, window=hv_window)
    regimes = detect_regime(df)

    tte_years = dte / TRADING_DAYS_PER_YEAR
    entry_dates = _generate_entry_dates(df.index, frequency)

    trades = []
    for entry_date in entry_dates:
        exit_date = _find_exit_date(df.index, entry_date, dte)
        if exit_date is None:
            continue

        spot_entry = df.loc[entry_date, "Close"]
        spot_exit = df.loc[exit_date, "Close"]
        if hasattr(spot_entry, 'iloc'):
            spot_entry = spot_entry.iloc[0]
        if hasattr(spot_exit, 'iloc'):
            spot_exit = spot_exit.iloc[0]

        # Get volatility
        if entry_date in hv.index and not np.isnan(hv.loc[entry_date]):
            sigma = hv.loc[entry_date]
            if hasattr(sigma, 'iloc'):
                sigma = sigma.iloc[0]
        else:
            sigma = 0.20

        # Compute vol percentile up to entry date
        hv_to_date = hv.loc[:entry_date].dropna()
        if len(hv_to_date) > vol_lookback:
            window = hv_to_date.values[-(vol_lookback + 1):-1]
            vol_pct = float(
                (window < hv_to_date.values[-1]).sum() / len(window) * 100
            )
        else:
            vol_pct = 50.0  # Default to medium

        # Get regime at entry
        regime = regimes.get(entry_date, "UNKNOWN")

        # Select strategy dynamically
        strategy = select_strategy(vol_pct, regime, dte)

        # Simulate trade
        trade = _simulate_trade(
            strategy, spot_entry, spot_exit, sigma, tte_years,
            r, transaction_cost, slippage_pct,
        )

        trades.append({
            "Entry_Date": entry_date,
            "Exit_Date": exit_date,
            "Spot_Entry": round(float(spot_entry), 2),
            "Spot_Exit": round(float(spot_exit), 2),
            "Sigma": round(float(sigma), 4),
            "Vol_Percentile": round(vol_pct, 1),
            "Regime": regime,
            "Strategy": strategy.name,
            **trade,
        })

    return pd.DataFrame(trades)


# =============================================================================
# Backtest Comparison (all strategies side by side)
# =============================================================================

def run_all_strategies_backtest(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
    dte: int = DEFAULT_DTE,
    frequency: str = "monthly",
) -> dict[str, pd.DataFrame]:
    """
    Run backtests for all registered strategies and return results.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of strategy_name → trade log DataFrame.
    """
    results = {}
    for name in list_strategies():
        try:
            result = run_backtest(ticker, name, start, end, dte, frequency)
            if not result.empty:
                results[name] = result
        except Exception as e:
            print(f"[WARN] Backtest failed for {name}: {e}")
    return results


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("Running Iron Condor backtest on NIFTY (2019-2025)...")
    trades = run_backtest("NIFTY", "IronCondor")
    print(f"Total trades: {len(trades)}")
    if not trades.empty:
        print(f"Total P&L: ₹{trades['net_pnl'].sum():.2f}")
        print(f"Win rate: {(trades['net_pnl'] > 0).mean() * 100:.1f}%")
        print(f"\nLast 5 trades:\n{trades.tail()}")

    print("\n\nRunning dynamic backtest on NIFTY...")
    dynamic = run_dynamic_backtest("NIFTY")
    print(f"Total trades: {len(dynamic)}")
    if not dynamic.empty:
        print(f"Total P&L: ₹{dynamic['net_pnl'].sum():.2f}")
        print(f"Strategies used: {dynamic['Strategy'].value_counts().to_dict()}")
