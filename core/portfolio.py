# =============================================================================
# Module 9: Portfolio Simulator
# =============================================================================
# Combines multiple strategy backtests into a portfolio with configurable
# allocations, rebalancing, correlation analysis, and attribution.
# =============================================================================

import numpy as np
import pandas as pd

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INITIAL_CAPITAL, PORTFOLIO_WEIGHTS
from core.risk_analytics import (
    equity_curve, compute_all_metrics, drawdown_series, max_drawdown,
)


# =============================================================================
# Portfolio Construction
# =============================================================================

def simulate_portfolio(
    backtest_results: dict[str, pd.DataFrame],
    allocations: dict[str, float] | None = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """
    Combine multiple strategy backtests into a portfolio.

    Parameters
    ----------
    backtest_results : dict[str, pd.DataFrame]
        Mapping of strategy_name → trade log DataFrame.
    allocations : dict[str, float]
        Strategy allocation weights (must sum to 1.0).
        If None, uses equal weighting.
    initial_capital : float
        Total portfolio initial capital.

    Returns
    -------
    dict with keys:
        - portfolio_equity: pd.DataFrame (combined equity curve)
        - strategy_equities: dict of per-strategy equity curves
        - portfolio_trades: pd.DataFrame (all trades combined)
        - metrics: dict (portfolio-level risk metrics)
        - allocations: dict (weights used)
    """
    if not backtest_results:
        return {
            "portfolio_equity": pd.DataFrame(),
            "strategy_equities": {},
            "portfolio_trades": pd.DataFrame(),
            "metrics": {},
            "allocations": {},
        }

    # Default to equal weights if not specified
    if allocations is None:
        n = len(backtest_results)
        allocations = {name: 1.0 / n for name in backtest_results}

    # Normalize weights to sum to 1.0
    total_weight = sum(allocations.get(name, 0) for name in backtest_results)
    if total_weight > 0:
        allocations = {
            name: allocations.get(name, 0) / total_weight
            for name in backtest_results
        }

    # Build per-strategy equity curves with allocated capital
    strategy_equities = {}
    all_trades = []

    for name, trades in backtest_results.items():
        weight = allocations.get(name, 0)
        if weight <= 0 or trades.empty:
            continue

        allocated_capital = initial_capital * weight

        # Scale P&L proportionally to allocation
        scaled_trades = trades.copy()
        scaled_trades["allocation_weight"] = weight
        scaled_trades["portfolio_strategy"] = name

        eq = equity_curve(scaled_trades, allocated_capital)
        strategy_equities[name] = eq

        all_trades.append(scaled_trades)

    # Combine all trades
    if all_trades:
        portfolio_trades = pd.concat(all_trades, ignore_index=True)
        portfolio_trades.sort_values("Exit_Date", inplace=True)
    else:
        portfolio_trades = pd.DataFrame()

    # Build combined portfolio equity curve
    portfolio_equity = _build_combined_equity(
        strategy_equities, initial_capital
    )

    # Compute portfolio-level metrics
    metrics = compute_all_metrics(portfolio_trades, initial_capital)

    return {
        "portfolio_equity": portfolio_equity,
        "strategy_equities": strategy_equities,
        "portfolio_trades": portfolio_trades,
        "metrics": metrics,
        "allocations": allocations,
    }


def _build_combined_equity(
    strategy_equities: dict[str, pd.DataFrame],
    initial_capital: float,
) -> pd.DataFrame:
    """
    Merge per-strategy equity curves into a single portfolio equity curve.
    """
    if not strategy_equities:
        return pd.DataFrame(columns=["Equity", "Cumulative_PnL"])

    # Collect all equity series
    all_eq = pd.DataFrame()
    for name, eq in strategy_equities.items():
        if not eq.empty:
            all_eq[name] = eq["Cumulative_PnL"]

    if all_eq.empty:
        return pd.DataFrame(columns=["Equity", "Cumulative_PnL"])

    # Forward-fill and sum across strategies
    all_eq = all_eq.sort_index()
    all_eq.ffill(inplace=True)
    all_eq.fillna(0, inplace=True)

    combined_pnl = all_eq.sum(axis=1)
    combined_equity = initial_capital + combined_pnl

    result = pd.DataFrame({
        "Equity": combined_equity,
        "Cumulative_PnL": combined_pnl,
    })

    return result


# =============================================================================
# Correlation Analysis
# =============================================================================

def strategy_correlation_matrix(
    backtest_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute correlation between strategy returns.

    Returns
    -------
    pd.DataFrame
        Correlation matrix (strategy × strategy).
    """
    returns_dict = {}
    for name, trades in backtest_results.items():
        if not trades.empty:
            # Create a return series indexed by exit date
            r = trades.set_index("Exit_Date")["return_pct"]
            r.index = pd.to_datetime(r.index)
            # Resample to monthly to align across strategies
            monthly = r.resample("ME").sum()
            returns_dict[name] = monthly

    if not returns_dict:
        return pd.DataFrame()

    returns_df = pd.DataFrame(returns_dict)
    returns_df.ffill(inplace=True)
    returns_df.fillna(0, inplace=True)

    return returns_df.corr()


# =============================================================================
# Attribution Analysis
# =============================================================================

def attribution_analysis(
    backtest_results: dict[str, pd.DataFrame],
    allocations: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Break down portfolio P&L contribution by strategy.

    Returns
    -------
    pd.DataFrame
        Columns: Strategy, Allocation, Total_PnL, Contribution_Pct,
                 Win_Rate, Num_Trades
    """
    if not backtest_results:
        return pd.DataFrame()

    if allocations is None:
        n = len(backtest_results)
        allocations = {name: 1.0 / n for name in backtest_results}

    total_pnl = sum(
        trades["net_pnl"].sum()
        for trades in backtest_results.values()
        if not trades.empty
    )

    rows = []
    for name, trades in backtest_results.items():
        if trades.empty:
            continue
        strat_pnl = trades["net_pnl"].sum()
        rows.append({
            "Strategy": name,
            "Allocation": f"{allocations.get(name, 0) * 100:.0f}%",
            "Total_PnL": round(strat_pnl, 2),
            "Contribution_Pct": round(
                (strat_pnl / total_pnl * 100) if total_pnl != 0 else 0, 1
            ),
            "Win_Rate": f"{(trades['net_pnl'] > 0).mean() * 100:.1f}%",
            "Num_Trades": len(trades),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Allocation Drift
# =============================================================================

def allocation_drift(
    strategy_equities: dict[str, pd.DataFrame],
    target_allocations: dict[str, float],
) -> pd.DataFrame:
    """
    Track how allocation weights drift from target over time.

    Returns
    -------
    pd.DataFrame
        Index = Date, Columns = strategy names, Values = actual weight.
    """
    if not strategy_equities:
        return pd.DataFrame()

    equity_df = pd.DataFrame()
    for name, eq in strategy_equities.items():
        if not eq.empty:
            equity_df[name] = eq["Equity"]

    if equity_df.empty:
        return pd.DataFrame()

    equity_df.ffill(inplace=True)
    equity_df.fillna(method="bfill", inplace=True)

    total = equity_df.sum(axis=1)
    weights = equity_df.div(total, axis=0)

    return weights


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    from core.backtester import run_backtest

    print("Running backtests for portfolio simulation...")
    results = {}
    for strat in ["IronCondor", "LongStraddle", "BullCallSpread"]:
        print(f"  Backtesting {strat}...")
        results[strat] = run_backtest("NIFTY", strat)

    allocations = {"IronCondor": 0.4, "LongStraddle": 0.3, "BullCallSpread": 0.3}
    portfolio = simulate_portfolio(results, allocations)

    print(f"\nPortfolio metrics:")
    for k, v in portfolio["metrics"].items():
        print(f"  {k}: {v}")

    print(f"\nAttribution:")
    attr = attribution_analysis(results, allocations)
    print(attr.to_string(index=False))

    corr = strategy_correlation_matrix(results)
    print(f"\nCorrelation matrix:\n{corr}")
