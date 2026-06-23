# =============================================================================
# Tests: Backtester Module
# =============================================================================

import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.backtester import _generate_entry_dates, _find_exit_date, _simulate_trade
from core.options import LongStraddle, IronCondor


class TestEntryDateGeneration:
    @pytest.fixture
    def trading_dates(self):
        return pd.bdate_range("2020-01-01", "2020-12-31")

    def test_monthly_entries(self, trading_dates):
        entries = _generate_entry_dates(trading_dates, "monthly")
        # Should have roughly 12 entries (one per month)
        assert 10 <= len(entries) <= 14

    def test_weekly_entries(self, trading_dates):
        entries = _generate_entry_dates(trading_dates, "weekly")
        # Should have roughly 52 entries
        assert 45 <= len(entries) <= 55

    def test_entries_are_within_range(self, trading_dates):
        entries = _generate_entry_dates(trading_dates, "monthly")
        for e in entries:
            assert e >= trading_dates[0]
            assert e <= trading_dates[-1]


class TestFindExitDate:
    def test_exit_date_after_entry(self):
        dates = pd.bdate_range("2020-01-01", "2020-06-30")
        entry = pd.Timestamp("2020-01-02")
        exit_date = _find_exit_date(dates, entry, dte=30)
        assert exit_date is not None
        assert exit_date > entry

    def test_exit_date_approximately_dte_days(self):
        dates = pd.bdate_range("2020-01-01", "2020-06-30")
        entry = pd.Timestamp("2020-01-02")
        exit_date = _find_exit_date(dates, entry, dte=30)
        diff = (exit_date - entry).days
        assert 28 <= diff <= 35  # Some slack for weekends

    def test_exit_date_beyond_data_returns_none(self):
        dates = pd.bdate_range("2020-01-01", "2020-01-15")
        entry = pd.Timestamp("2020-01-02")
        exit_date = _find_exit_date(dates, entry, dte=30)
        assert exit_date is None


class TestSimulateTrade:
    def test_pnl_calculation(self):
        strat = LongStraddle()
        result = _simulate_trade(
            strat,
            spot_entry=100,
            spot_exit=110,
            sigma=0.20,
            tte_years=30 / 252,
            r=0.05,
            transaction_cost=0,
            slippage_pct=0,
        )
        # Should have all required keys
        assert "net_pnl" in result
        assert "return_pct" in result
        assert "entry_cost" in result
        assert "capital_at_risk" in result

    def test_transaction_costs_reduce_pnl(self):
        strat = LongStraddle()
        pnl_no_cost = _simulate_trade(
            strat, 100, 110, 0.20, 30 / 252, 0.05, 0, 0,
        )["net_pnl"]

        pnl_with_cost = _simulate_trade(
            strat, 100, 110, 0.20, 30 / 252, 0.05, 20, 0.001,
        )["net_pnl"]

        assert pnl_with_cost < pnl_no_cost

    def test_iron_condor_in_range_positive(self):
        """IC should be profitable when spot stays near entry."""
        strat = IronCondor()
        result = _simulate_trade(
            strat, 100, 100, 0.20, 30 / 252, 0.05, 0, 0,
        )
        assert result["net_pnl"] > 0
