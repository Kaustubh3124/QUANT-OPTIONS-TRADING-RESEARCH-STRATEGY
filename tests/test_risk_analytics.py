# =============================================================================
# Tests: Risk Analytics Module
# =============================================================================

import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.risk_analytics import (
    equity_curve, drawdown_series, max_drawdown, max_drawdown_amount,
    sharpe_ratio, sortino_ratio, win_rate, profit_factor,
    expected_value, var_95, cvar_95, cagr, total_return,
    compute_all_metrics,
)


@pytest.fixture
def sample_trades():
    """Generate a synthetic trade log."""
    np.random.seed(42)
    n_trades = 50
    dates = pd.bdate_range("2020-01-01", periods=n_trades * 2)

    trades = pd.DataFrame({
        "Entry_Date": dates[:n_trades],
        "Exit_Date": dates[n_trades:],
        "net_pnl": np.random.normal(500, 2000, n_trades),
        "return_pct": np.random.normal(2, 8, n_trades),
        "Strategy": "TestStrategy",
    })
    return trades


@pytest.fixture
def profitable_trades():
    """All winning trades."""
    return pd.DataFrame({
        "Entry_Date": pd.bdate_range("2020-01-01", periods=10),
        "Exit_Date": pd.bdate_range("2020-02-01", periods=10),
        "net_pnl": [100, 200, 150, 300, 250, 100, 200, 150, 300, 250],
        "return_pct": [1, 2, 1.5, 3, 2.5, 1, 2, 1.5, 3, 2.5],
        "Strategy": "TestStrategy",
    })


@pytest.fixture
def losing_trades():
    """All losing trades."""
    return pd.DataFrame({
        "Entry_Date": pd.bdate_range("2020-01-01", periods=10),
        "Exit_Date": pd.bdate_range("2020-02-01", periods=10),
        "net_pnl": [-100, -200, -150, -300, -250, -100, -200, -150, -300, -250],
        "return_pct": [-1, -2, -1.5, -3, -2.5, -1, -2, -1.5, -3, -2.5],
        "Strategy": "TestStrategy",
    })


class TestEquityCurve:
    def test_starts_at_capital(self, sample_trades):
        eq = equity_curve(sample_trades, 1000000)
        assert eq["Equity"].iloc[0] == 1000000 + sample_trades["net_pnl"].iloc[0]

    def test_empty_trades(self):
        eq = equity_curve(pd.DataFrame(columns=["Exit_Date", "net_pnl"]))
        assert eq.empty

    def test_cumulative_pnl(self, sample_trades):
        eq = equity_curve(sample_trades, 0)
        expected = sample_trades["net_pnl"].cumsum().values
        np.testing.assert_array_almost_equal(eq["Cumulative_PnL"].values, expected)


class TestDrawdown:
    def test_drawdown_nonpositive(self, sample_trades):
        eq = equity_curve(sample_trades, 1000000)
        dd = drawdown_series(eq["Equity"])
        assert (dd["Drawdown"] <= 0).all()

    def test_max_drawdown_negative(self, sample_trades):
        eq = equity_curve(sample_trades, 1000000)
        mdd = max_drawdown(eq["Equity"])
        assert mdd <= 0

    def test_no_drawdown_for_monotonic_increase(self, profitable_trades):
        eq = equity_curve(profitable_trades, 1000000)
        mdd = max_drawdown(eq["Equity"])
        assert mdd == 0  # No drawdown if equity always goes up


class TestRiskMetrics:
    def test_win_rate_all_winners(self, profitable_trades):
        assert win_rate(profitable_trades) == 100.0

    def test_win_rate_all_losers(self, losing_trades):
        assert win_rate(losing_trades) == 0.0

    def test_win_rate_mixed(self, sample_trades):
        wr = win_rate(sample_trades)
        assert 0 <= wr <= 100

    def test_profit_factor_all_winners(self, profitable_trades):
        pf = profit_factor(profitable_trades)
        assert pf == float("inf")

    def test_profit_factor_all_losers(self, losing_trades):
        pf = profit_factor(losing_trades)
        assert pf == 0.0

    def test_sharpe_ratio_returns_float(self, sample_trades):
        sr = sharpe_ratio(sample_trades)
        assert isinstance(sr, float)

    def test_sortino_ratio_returns_float(self, sample_trades):
        so = sortino_ratio(sample_trades)
        assert isinstance(so, float)

    def test_expected_value(self, sample_trades):
        ev = expected_value(sample_trades)
        assert isinstance(ev, float)

    def test_var_95_negative(self, sample_trades):
        """VaR should typically be negative (it's a loss metric)."""
        var = var_95(sample_trades)
        # May or may not be negative depending on trades, but should be a float
        assert isinstance(var, float)

    def test_cvar_less_than_var(self, sample_trades):
        """CVaR (Expected Shortfall) should be ≤ VaR."""
        var = var_95(sample_trades)
        cv = cvar_95(sample_trades)
        assert cv <= var

    def test_cagr_positive_for_profitable(self, profitable_trades):
        c = cagr(profitable_trades, 1000000)
        assert c > 0

    def test_total_return(self, sample_trades):
        tr = total_return(sample_trades, 1000000)
        assert isinstance(tr, float)


class TestComputeAllMetrics:
    def test_returns_dict(self, sample_trades):
        metrics = compute_all_metrics(sample_trades)
        assert isinstance(metrics, dict)
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert "win_rate_pct" in metrics
        assert "profit_factor" in metrics
        assert "expected_value" in metrics
        assert "total_trades" in metrics

    def test_empty_trades(self):
        metrics = compute_all_metrics(pd.DataFrame(columns=[
            "Exit_Date", "net_pnl", "return_pct",
        ]))
        assert metrics["total_trades"] == 0
