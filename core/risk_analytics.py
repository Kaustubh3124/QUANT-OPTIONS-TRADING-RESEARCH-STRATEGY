# =============================================================================
# Module 7: Risk Analytics
# =============================================================================
# Computes comprehensive risk metrics from trade-level P&L:
# Sharpe, Sortino, Max Drawdown, CAGR, Win Rate, Profit Factor,
# Expected Value, Calmar Ratio, VaR, CVaR.
# =============================================================================

import numpy as np
import pandas as pd

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR, INITIAL_CAPITAL


# =============================================================================
# Equity Curve
# =============================================================================

def equity_curve(
    trade_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    pnl_column: str = "net_pnl",
) -> pd.DataFrame:
    """
    Build a cumulative equity curve from trade P&L.

    Returns
    -------
    pd.DataFrame
        Columns: Date, Equity, Cumulative_PnL, Trade_PnL
    """
    if trade_df.empty:
        return pd.DataFrame(columns=["Date", "Equity", "Cumulative_PnL", "Trade_PnL"])

    df = trade_df.copy()
    df = df.sort_values("Exit_Date")

    cum_pnl = df[pnl_column].cumsum()
    equity = initial_capital + cum_pnl

    result = pd.DataFrame({
        "Date": df["Exit_Date"].values,
        "Equity": equity.values,
        "Cumulative_PnL": cum_pnl.values,
        "Trade_PnL": df[pnl_column].values,
    })
    result.set_index("Date", inplace=True)
    return result


# =============================================================================
# Drawdown
# =============================================================================

def drawdown_series(equity_series: pd.Series) -> pd.DataFrame:
    """
    Compute drawdown at each point in the equity curve.

    Returns
    -------
    pd.DataFrame
        Columns: Equity, Peak, Drawdown, Drawdown_Pct
    """
    peak = equity_series.cummax()
    dd = equity_series - peak
    dd_pct = (dd / peak) * 100

    return pd.DataFrame({
        "Equity": equity_series,
        "Peak": peak,
        "Drawdown": dd,
        "Drawdown_Pct": dd_pct,
    })


def max_drawdown(equity_series: pd.Series) -> float:
    """Maximum peak-to-trough decline in percentage."""
    peak = equity_series.cummax()
    dd_pct = ((equity_series - peak) / peak) * 100
    return dd_pct.min()


def max_drawdown_amount(equity_series: pd.Series) -> float:
    """Maximum peak-to-trough decline in absolute currency."""
    peak = equity_series.cummax()
    return (equity_series - peak).min()


# =============================================================================
# Return Metrics
# =============================================================================

def cagr(
    trade_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> float:
    """
    Compound Annual Growth Rate.
    CAGR = (final / initial) ^ (1 / years) - 1
    """
    if trade_df.empty:
        return 0.0

    total_pnl = trade_df["net_pnl"].sum()
    final = initial_capital + total_pnl

    start = pd.Timestamp(trade_df["Entry_Date"].min())
    end = pd.Timestamp(trade_df["Exit_Date"].max())
    years = (end - start).days / 365.25

    if years <= 0 or final <= 0:
        return 0.0

    return (final / initial_capital) ** (1 / years) - 1


def total_return(
    trade_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> float:
    """Total return percentage over the entire period."""
    if trade_df.empty:
        return 0.0
    total_pnl = trade_df["net_pnl"].sum()
    return (total_pnl / initial_capital) * 100


# =============================================================================
# Risk-Adjusted Returns
# =============================================================================

def sharpe_ratio(
    trade_df: pd.DataFrame,
    rf: float = RISK_FREE_RATE,
) -> float:
    """
    Annualized Sharpe Ratio.
    Sharpe = (mean_return - rf) / std(returns) × √(trades_per_year)
    """
    if trade_df.empty or len(trade_df) < 2:
        return 0.0

    returns = trade_df["return_pct"] / 100  # Convert from percentage

    # Estimate trades per year
    start = pd.Timestamp(trade_df["Entry_Date"].min())
    end = pd.Timestamp(trade_df["Exit_Date"].max())
    years = max((end - start).days / 365.25, 0.01)
    trades_per_year = len(trade_df) / years

    excess = returns.mean() - (rf / trades_per_year)
    std = returns.std()

    if std == 0:
        return 0.0

    return (excess / std) * np.sqrt(trades_per_year)


def sortino_ratio(
    trade_df: pd.DataFrame,
    rf: float = RISK_FREE_RATE,
) -> float:
    """
    Annualized Sortino Ratio (uses downside deviation only).
    """
    if trade_df.empty or len(trade_df) < 2:
        return 0.0

    returns = trade_df["return_pct"] / 100

    start = pd.Timestamp(trade_df["Entry_Date"].min())
    end = pd.Timestamp(trade_df["Exit_Date"].max())
    years = max((end - start).days / 365.25, 0.01)
    trades_per_year = len(trade_df) / years

    excess = returns.mean() - (rf / trades_per_year)
    downside = returns[returns < 0]

    if downside.empty or downside.std() == 0:
        return 0.0

    return (excess / downside.std()) * np.sqrt(trades_per_year)


def calmar_ratio(
    trade_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> float:
    """
    Calmar Ratio = CAGR / |Max Drawdown %|
    """
    c = cagr(trade_df, initial_capital)
    eq = equity_curve(trade_df, initial_capital)

    if eq.empty:
        return 0.0

    mdd = abs(max_drawdown(eq["Equity"]))

    if mdd == 0:
        return 0.0

    return (c * 100) / mdd  # Both in percentage terms


# =============================================================================
# Trade Statistics
# =============================================================================

def win_rate(trade_df: pd.DataFrame) -> float:
    """Percentage of profitable trades."""
    if trade_df.empty:
        return 0.0
    return (trade_df["net_pnl"] > 0).mean() * 100


def profit_factor(trade_df: pd.DataFrame) -> float:
    """
    Profit Factor = Gross Profits / Gross Losses.
    > 1.0 is profitable. > 2.0 is strong.
    """
    if trade_df.empty:
        return 0.0
    profits = trade_df.loc[trade_df["net_pnl"] > 0, "net_pnl"].sum()
    losses = abs(trade_df.loc[trade_df["net_pnl"] < 0, "net_pnl"].sum())

    if losses == 0:
        return float("inf") if profits > 0 else 0.0

    return profits / losses


def expected_value(trade_df: pd.DataFrame) -> float:
    """
    Expected Value per trade.
    EV = win_rate × avg_win - loss_rate × avg_loss
    """
    if trade_df.empty:
        return 0.0

    winners = trade_df[trade_df["net_pnl"] > 0]["net_pnl"]
    losers = trade_df[trade_df["net_pnl"] < 0]["net_pnl"]

    w_rate = len(winners) / len(trade_df)
    l_rate = len(losers) / len(trade_df)

    avg_win = winners.mean() if not winners.empty else 0
    avg_loss = abs(losers.mean()) if not losers.empty else 0

    return w_rate * avg_win - l_rate * avg_loss


# =============================================================================
# Value at Risk
# =============================================================================

def var_95(trade_df: pd.DataFrame) -> float:
    """Value at Risk (95%) — 5th percentile of P&L distribution."""
    if trade_df.empty:
        return 0.0
    return trade_df["net_pnl"].quantile(0.05)


def cvar_95(trade_df: pd.DataFrame) -> float:
    """Conditional VaR (Expected Shortfall) — mean of losses below VaR."""
    if trade_df.empty:
        return 0.0
    var = var_95(trade_df)
    tail = trade_df[trade_df["net_pnl"] <= var]["net_pnl"]
    return tail.mean() if not tail.empty else var


# =============================================================================
# All Metrics
# =============================================================================

def compute_all_metrics(
    trade_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """
    Compute all risk analytics metrics.

    Returns
    -------
    dict
        Comprehensive metrics dictionary.
    """
    eq = equity_curve(trade_df, initial_capital)

    metrics = {
        "total_trades": len(trade_df),
        "total_pnl": round(trade_df["net_pnl"].sum(), 2) if not trade_df.empty else 0,
        "total_return_pct": round(total_return(trade_df, initial_capital), 2),
        "cagr_pct": round(cagr(trade_df, initial_capital) * 100, 2),
        "sharpe_ratio": round(sharpe_ratio(trade_df), 2),
        "sortino_ratio": round(sortino_ratio(trade_df), 2),
        "calmar_ratio": round(calmar_ratio(trade_df, initial_capital), 2),
        "max_drawdown_pct": round(
            max_drawdown(eq["Equity"]) if not eq.empty else 0, 2
        ),
        "max_drawdown_amount": round(
            max_drawdown_amount(eq["Equity"]) if not eq.empty else 0, 2
        ),
        "win_rate_pct": round(win_rate(trade_df), 1),
        "profit_factor": round(profit_factor(trade_df), 2),
        "expected_value": round(expected_value(trade_df), 2),
        "var_95": round(var_95(trade_df), 2),
        "cvar_95": round(cvar_95(trade_df), 2),
        "avg_pnl_per_trade": round(
            trade_df["net_pnl"].mean() if not trade_df.empty else 0, 2
        ),
        "best_trade": round(
            trade_df["net_pnl"].max() if not trade_df.empty else 0, 2
        ),
        "worst_trade": round(
            trade_df["net_pnl"].min() if not trade_df.empty else 0, 2
        ),
    }

    return metrics


def metrics_comparison_table(
    results: dict[str, pd.DataFrame],
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """
    Compare metrics across multiple strategy backtests.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Mapping of strategy_name → trade log.

    Returns
    -------
    pd.DataFrame
        One row per strategy, columns = metrics.
    """
    rows = []
    for name, trades in results.items():
        metrics = compute_all_metrics(trades, initial_capital)
        metrics["strategy"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if "strategy" in df.columns:
        df.set_index("strategy", inplace=True)
    return df


# =============================================================================
# Monthly Returns Heatmap Data
# =============================================================================

def monthly_returns(
    trade_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """
    Aggregate P&L by month for heatmap visualization.

    Returns
    -------
    pd.DataFrame
        Index = Year, Columns = Month (1–12), Values = Return %
    """
    if trade_df.empty:
        return pd.DataFrame()

    df = trade_df.copy()
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["Year"] = df["Exit_Date"].dt.year
    df["Month"] = df["Exit_Date"].dt.month

    monthly = df.groupby(["Year", "Month"])["net_pnl"].sum().unstack(fill_value=0)
    # Convert to return percentage
    monthly = (monthly / initial_capital) * 100

    return monthly


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    from core.backtester import run_backtest

    print("Running Iron Condor backtest for risk analysis...")
    trades = run_backtest("NIFTY", "IronCondor")

    if not trades.empty:
        metrics = compute_all_metrics(trades)
        print("\n=== Risk Analytics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        eq = equity_curve(trades)
        print(f"\nEquity curve (last 5):\n{eq.tail()}")

        mr = monthly_returns(trades)
        print(f"\nMonthly returns:\n{mr}")
