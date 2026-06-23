# =============================================================================
# Tests: Options Module
# =============================================================================

import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.options import (
    bs_call_price, bs_put_price,
    LongStraddle, ShortStraddle, IronCondor,
    BullCallSpread, BearPutSpread, CoveredCall, CalendarSpread,
    get_strategy, list_strategies,
)


class TestBlackScholes:
    """Test Black-Scholes pricing against known values."""

    def test_atm_call_price_positive(self):
        price = bs_call_price(100, 100, 1.0, 0.05, 0.20)
        assert price > 0

    def test_atm_put_price_positive(self):
        price = bs_put_price(100, 100, 1.0, 0.05, 0.20)
        assert price > 0

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT)"""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        C = bs_call_price(S, K, T, r, sigma)
        P = bs_put_price(S, K, T, r, sigma)
        parity = C - P - (S - K * np.exp(-r * T))
        assert abs(parity) < 1e-10

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S - K*exp(-rT)"""
        S, K, T, r, sigma = 200, 100, 1.0, 0.05, 0.20
        C = bs_call_price(S, K, T, r, sigma)
        intrinsic = S - K * np.exp(-r * T)
        assert abs(C - intrinsic) < 1.0

    def test_deep_otm_call(self):
        """Deep OTM call → 0"""
        C = bs_call_price(100, 200, 0.1, 0.05, 0.20)
        assert C < 0.01

    def test_expired_call(self):
        """At expiry, call = max(S-K, 0)"""
        assert bs_call_price(110, 100, 0, 0.05, 0.20) == 10
        assert bs_call_price(90, 100, 0, 0.05, 0.20) == 0

    def test_expired_put(self):
        """At expiry, put = max(K-S, 0)"""
        assert bs_put_price(90, 100, 0, 0.05, 0.20) == 10
        assert bs_put_price(110, 100, 0, 0.05, 0.20) == 0

    def test_higher_vol_higher_price(self):
        """Higher vol → higher option price."""
        C_low = bs_call_price(100, 100, 1.0, 0.05, 0.10)
        C_high = bs_call_price(100, 100, 1.0, 0.05, 0.40)
        assert C_high > C_low

    def test_known_value(self):
        """Test against known BS value.
        S=100, K=100, T=1, r=0.05, σ=0.20 → Call ≈ 10.4506
        """
        C = bs_call_price(100, 100, 1.0, 0.05, 0.20)
        assert abs(C - 10.4506) < 0.01


class TestStrategyPayoffs:
    """Test strategy payoff logic at expiry."""

    S = 22000
    sigma = 0.15
    tte = 30 / 252
    r = 0.07

    def test_long_straddle_at_spot(self):
        """At-the-money at expiry, long straddle loses the premium."""
        strat = LongStraddle()
        pnl = strat.pnl_at_expiry(self.S, self.S, self.sigma, self.tte, self.r)
        # Should lose approximately the premium paid (negative)
        assert pnl < 0

    def test_long_straddle_big_move_up(self):
        """Large upward move should be profitable for long straddle."""
        strat = LongStraddle()
        pnl = strat.pnl_at_expiry(
            self.S * 1.15, self.S, self.sigma, self.tte, self.r,
        )
        assert pnl > 0

    def test_short_straddle_at_spot(self):
        """At expiry near spot, short straddle profits (collects premium)."""
        strat = ShortStraddle()
        pnl = strat.pnl_at_expiry(self.S, self.S, self.sigma, self.tte, self.r)
        assert pnl > 0

    def test_iron_condor_in_range(self):
        """Inside the condor wings, should be profitable."""
        strat = IronCondor()
        pnl = strat.pnl_at_expiry(self.S, self.S, self.sigma, self.tte, self.r)
        assert pnl > 0

    def test_iron_condor_outside_range(self):
        """Far outside wings, iron condor has defined max loss."""
        strat = IronCondor()
        pnl = strat.pnl_at_expiry(
            self.S * 1.20, self.S, self.sigma, self.tte, self.r,
        )
        assert pnl < 0

    def test_bull_call_spread_profit_above_strikes(self):
        """Bull call spread profits when spot rises above long strike."""
        strat = BullCallSpread()
        pnl = strat.pnl_at_expiry(
            self.S * 1.08, self.S, self.sigma, self.tte, self.r,
        )
        assert pnl > 0

    def test_bear_put_spread_profit_below_strikes(self):
        """Bear put spread profits when spot falls below long strike."""
        strat = BearPutSpread()
        pnl = strat.pnl_at_expiry(
            self.S * 0.92, self.S, self.sigma, self.tte, self.r,
        )
        assert pnl > 0


class TestStrategyInterface:
    """Test strategy common interface methods."""

    def test_all_strategies_have_name(self):
        for name in list_strategies():
            strat = get_strategy(name)
            assert hasattr(strat, "name")
            assert len(strat.name) > 0

    def test_payoff_curve_shape(self):
        strat = LongStraddle()
        curve = strat.payoff_curve(100, 0.20, 30 / 252, 0.05)
        assert "Spot" in curve.columns
        assert "PnL" in curve.columns
        assert len(curve) == 200

    def test_breakeven_returns_list(self):
        strat = LongStraddle()
        be = strat.breakeven_points(100, 0.20, 30 / 252, 0.05)
        assert isinstance(be, list)
        assert len(be) >= 2  # Straddle has 2 breakevens

    def test_max_profit_greater_than_max_loss_for_credit(self):
        """For credit strategies, max loss should be limited."""
        strat = IronCondor()
        max_p = strat.max_profit(22000, 0.15, 30 / 252, 0.07)
        max_l = strat.max_loss(22000, 0.15, 30 / 252, 0.07)
        assert max_p > 0
        assert max_l < 0

    def test_get_strategy_invalid_name(self):
        with pytest.raises(ValueError):
            get_strategy("NonExistentStrategy")

    def test_get_legs_info(self):
        strat = IronCondor()
        legs = strat.get_legs_info(22000, 0.15, 30 / 252, 0.07)
        assert len(legs) == 4  # Iron condor has 4 legs
        for leg in legs:
            assert "Type" in leg
            assert "Strike" in leg
            assert "Position" in leg
            assert "Premium" in leg
