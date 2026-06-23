# =============================================================================
# Module 4: Option Strategy Engine
# =============================================================================
# Black-Scholes pricing + 7 multi-leg option strategies, each with payoff,
# P&L, breakeven, max profit/loss calculations.
#
# Strategies: Long Straddle, Short Straddle, Iron Condor, Bull Call Spread,
#             Bear Put Spread, Covered Call, Calendar Spread
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import norm
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR, OTM_OFFSET_PCT, FAR_OTM_OFFSET_PCT


# =============================================================================
# Black-Scholes Pricing
# =============================================================================

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 in the Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 in the Black-Scholes formula."""
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """
    Black-Scholes European call option price.

    Parameters
    ----------
    S : float — Spot price
    K : float — Strike price
    T : float — Time to expiry in years
    r : float — Risk-free rate (annualized)
    sigma : float — Volatility (annualized)
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """
    Black-Scholes European put option price.
    """
    if T <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# =============================================================================
# Leg descriptor
# =============================================================================

@dataclass
class OptionLeg:
    """Represents a single option leg in a strategy."""
    option_type: str        # 'call' or 'put'
    strike: float
    position: int           # +1 = long, -1 = short
    premium: float = 0.0    # Computed from BS model
    tte: float = 0.0        # Time to expiry in years

    def intrinsic_value(self, spot_at_expiry: float) -> float:
        if self.option_type == "call":
            return max(spot_at_expiry - self.strike, 0.0)
        else:
            return max(self.strike - spot_at_expiry, 0.0)

    def payoff(self, spot_at_expiry: float) -> float:
        return self.position * self.intrinsic_value(spot_at_expiry)

    def pnl(self, spot_at_expiry: float) -> float:
        return self.payoff(spot_at_expiry) - self.position * self.premium


# =============================================================================
# Abstract Strategy Base
# =============================================================================

class OptionStrategy(ABC):
    """Abstract base class for all option strategies."""

    name: str = "Base"

    @abstractmethod
    def build_legs(
        self, spot: float, sigma: float, tte: float, r: float,
    ) -> list[OptionLeg]:
        """Construct the strategy legs and price them."""
        ...

    def entry_cost(self, spot: float, sigma: float, tte: float, r: float) -> float:
        """
        Net premium paid (positive = debit, negative = credit).
        """
        legs = self.build_legs(spot, sigma, tte, r)
        return sum(leg.position * leg.premium for leg in legs)

    def payoff_at_expiry(
        self, spot_at_expiry: float, spot: float, sigma: float, tte: float, r: float,
    ) -> float:
        """Gross payoff at expiry (before premium)."""
        legs = self.build_legs(spot, sigma, tte, r)
        return sum(leg.payoff(spot_at_expiry) for leg in legs)

    def pnl_at_expiry(
        self, spot_at_expiry: float, spot: float, sigma: float, tte: float, r: float,
    ) -> float:
        """Net P&L at expiry (after premium)."""
        legs = self.build_legs(spot, sigma, tte, r)
        return sum(leg.pnl(spot_at_expiry) for leg in legs)

    def payoff_curve(
        self, spot: float, sigma: float, tte: float, r: float,
        pct_range: float = 0.20, n_points: int = 200,
    ) -> pd.DataFrame:
        """
        Generate a payoff/PnL curve across a range of spot prices.
        """
        low = spot * (1 - pct_range)
        high = spot * (1 + pct_range)
        spots = np.linspace(low, high, n_points)

        legs = self.build_legs(spot, sigma, tte, r)
        payoffs = []
        pnls = []
        for s in spots:
            payoff = sum(leg.payoff(s) for leg in legs)
            pnl = sum(leg.pnl(s) for leg in legs)
            payoffs.append(payoff)
            pnls.append(pnl)

        return pd.DataFrame({
            "Spot": spots,
            "Payoff": payoffs,
            "PnL": pnls,
        })

    def breakeven_points(
        self, spot: float, sigma: float, tte: float, r: float,
        pct_range: float = 0.20,
    ) -> list[float]:
        """Find approximate breakeven spot prices."""
        curve = self.payoff_curve(spot, sigma, tte, r, pct_range, 1000)
        pnl = curve["PnL"].values
        spots = curve["Spot"].values

        breakevens = []
        for i in range(1, len(pnl)):
            if pnl[i - 1] * pnl[i] < 0:  # Sign change
                # Linear interpolation
                frac = pnl[i - 1] / (pnl[i - 1] - pnl[i])
                be = spots[i - 1] + frac * (spots[i] - spots[i - 1])
                breakevens.append(round(be, 2))

        return breakevens

    def max_profit(
        self, spot: float, sigma: float, tte: float, r: float,
        pct_range: float = 0.30,
    ) -> float:
        curve = self.payoff_curve(spot, sigma, tte, r, pct_range)
        return curve["PnL"].max()

    def max_loss(
        self, spot: float, sigma: float, tte: float, r: float,
        pct_range: float = 0.30,
    ) -> float:
        curve = self.payoff_curve(spot, sigma, tte, r, pct_range)
        return curve["PnL"].min()

    def get_legs_info(
        self, spot: float, sigma: float, tte: float, r: float,
    ) -> list[dict]:
        """Return leg details as list of dicts for display."""
        legs = self.build_legs(spot, sigma, tte, r)
        return [
            {
                "Type": leg.option_type.upper(),
                "Strike": leg.strike,
                "Position": "Long" if leg.position > 0 else "Short",
                "Premium": round(leg.premium, 2),
            }
            for leg in legs
        ]


# =============================================================================
# Strategy Implementations
# =============================================================================

class LongStraddle(OptionStrategy):
    """Buy ATM Call + Buy ATM Put. Profits from large moves."""
    name = "Long Straddle"

    def build_legs(self, spot, sigma, tte, r):
        K = round(spot)  # ATM strike
        call_prem = bs_call_price(spot, K, tte, r, sigma)
        put_prem = bs_put_price(spot, K, tte, r, sigma)
        return [
            OptionLeg("call", K, +1, call_prem, tte),
            OptionLeg("put", K, +1, put_prem, tte),
        ]


class ShortStraddle(OptionStrategy):
    """Sell ATM Call + Sell ATM Put. Profits from low realized vol."""
    name = "Short Straddle"

    def build_legs(self, spot, sigma, tte, r):
        K = round(spot)
        call_prem = bs_call_price(spot, K, tte, r, sigma)
        put_prem = bs_put_price(spot, K, tte, r, sigma)
        return [
            OptionLeg("call", K, -1, call_prem, tte),
            OptionLeg("put", K, -1, put_prem, tte),
        ]


class IronCondor(OptionStrategy):
    """
    Sell OTM Call + Sell OTM Put + Buy far-OTM Call + Buy far-OTM Put.
    Defined-risk, range-bound strategy.
    """
    name = "Iron Condor"

    def __init__(self, otm_pct: float = OTM_OFFSET_PCT, far_otm_pct: float = FAR_OTM_OFFSET_PCT):
        self.otm_pct = otm_pct
        self.far_otm_pct = far_otm_pct

    def build_legs(self, spot, sigma, tte, r):
        K_sell_call = round(spot * (1 + self.otm_pct))
        K_sell_put = round(spot * (1 - self.otm_pct))
        K_buy_call = round(spot * (1 + self.far_otm_pct))
        K_buy_put = round(spot * (1 - self.far_otm_pct))

        return [
            OptionLeg("call", K_sell_call, -1, bs_call_price(spot, K_sell_call, tte, r, sigma), tte),
            OptionLeg("put", K_sell_put, -1, bs_put_price(spot, K_sell_put, tte, r, sigma), tte),
            OptionLeg("call", K_buy_call, +1, bs_call_price(spot, K_buy_call, tte, r, sigma), tte),
            OptionLeg("put", K_buy_put, +1, bs_put_price(spot, K_buy_put, tte, r, sigma), tte),
        ]


class BullCallSpread(OptionStrategy):
    """Buy ATM Call + Sell OTM Call. Bullish directional."""
    name = "Bull Call Spread"

    def __init__(self, otm_pct: float = OTM_OFFSET_PCT):
        self.otm_pct = otm_pct

    def build_legs(self, spot, sigma, tte, r):
        K_buy = round(spot)  # ATM
        K_sell = round(spot * (1 + self.otm_pct))

        return [
            OptionLeg("call", K_buy, +1, bs_call_price(spot, K_buy, tte, r, sigma), tte),
            OptionLeg("call", K_sell, -1, bs_call_price(spot, K_sell, tte, r, sigma), tte),
        ]


class BearPutSpread(OptionStrategy):
    """Buy ATM Put + Sell OTM Put. Bearish directional."""
    name = "Bear Put Spread"

    def __init__(self, otm_pct: float = OTM_OFFSET_PCT):
        self.otm_pct = otm_pct

    def build_legs(self, spot, sigma, tte, r):
        K_buy = round(spot)  # ATM
        K_sell = round(spot * (1 - self.otm_pct))

        return [
            OptionLeg("put", K_buy, +1, bs_put_price(spot, K_buy, tte, r, sigma), tte),
            OptionLeg("put", K_sell, -1, bs_put_price(spot, K_sell, tte, r, sigma), tte),
        ]


class CoveredCall(OptionStrategy):
    """
    Long underlying + Sell OTM Call.
    Income generation / mild bullish.
    The 'spot' leg is modeled as payoff = spot_at_expiry - spot_at_entry.
    """
    name = "Covered Call"

    def __init__(self, otm_pct: float = OTM_OFFSET_PCT):
        self.otm_pct = otm_pct

    def build_legs(self, spot, sigma, tte, r):
        K_sell = round(spot * (1 + self.otm_pct))
        call_prem = bs_call_price(spot, K_sell, tte, r, sigma)
        return [
            OptionLeg("call", K_sell, -1, call_prem, tte),
        ]

    def pnl_at_expiry(self, spot_at_expiry, spot, sigma, tte, r):
        """Override to include the underlying position."""
        legs = self.build_legs(spot, sigma, tte, r)
        options_pnl = sum(leg.pnl(spot_at_expiry) for leg in legs)
        stock_pnl = spot_at_expiry - spot
        return options_pnl + stock_pnl

    def payoff_curve(self, spot, sigma, tte, r, pct_range=0.20, n_points=200):
        low = spot * (1 - pct_range)
        high = spot * (1 + pct_range)
        spots = np.linspace(low, high, n_points)

        legs = self.build_legs(spot, sigma, tte, r)
        pnls = []
        for s in spots:
            opt_pnl = sum(leg.pnl(s) for leg in legs)
            stock_pnl = s - spot
            pnls.append(opt_pnl + stock_pnl)

        return pd.DataFrame({
            "Spot": spots,
            "Payoff": pnls,  # For covered call, payoff ≈ PnL
            "PnL": pnls,
        })


class CalendarSpread(OptionStrategy):
    """
    Sell near-term ATM Call + Buy far-term ATM Call.
    Profits from time decay differential.

    Simplified model: near-term = tte, far-term = 2 × tte.
    At near-term expiry, the far-term option still has tte remaining
    and is valued via BS model.
    """
    name = "Calendar Spread"

    def __init__(self, far_tte_multiplier: float = 2.0):
        self.far_tte_multiplier = far_tte_multiplier

    def build_legs(self, spot, sigma, tte, r):
        K = round(spot)  # ATM
        near_prem = bs_call_price(spot, K, tte, r, sigma)
        far_tte = tte * self.far_tte_multiplier
        far_prem = bs_call_price(spot, K, far_tte, r, sigma)
        return [
            OptionLeg("call", K, -1, near_prem, tte),
            OptionLeg("call", K, +1, far_prem, far_tte),
        ]

    def pnl_at_expiry(self, spot_at_expiry, spot, sigma, tte, r):
        """
        At near-term expiry:
        - Near-term call has intrinsic value
        - Far-term call still has time value (BS price with remaining tte)
        """
        K = round(spot)
        near_prem = bs_call_price(spot, K, tte, r, sigma)
        far_tte = tte * self.far_tte_multiplier
        far_prem = bs_call_price(spot, K, far_tte, r, sigma)

        # Near-term short call P&L
        near_intrinsic = max(spot_at_expiry - K, 0)
        near_pnl = near_prem - near_intrinsic  # Sold, so premium received minus payoff

        # Far-term long call: value at near-term expiry (remaining time = far_tte - tte)
        remaining_tte = far_tte - tte
        far_value_now = bs_call_price(spot_at_expiry, K, remaining_tte, r, sigma)
        far_pnl = far_value_now - far_prem

        return near_pnl + far_pnl

    def payoff_curve(self, spot, sigma, tte, r, pct_range=0.20, n_points=200):
        low = spot * (1 - pct_range)
        high = spot * (1 + pct_range)
        spots = np.linspace(low, high, n_points)

        pnls = [
            self.pnl_at_expiry(s, spot, sigma, tte, r)
            for s in spots
        ]

        return pd.DataFrame({
            "Spot": spots,
            "Payoff": pnls,
            "PnL": pnls,
        })


# =============================================================================
# Strategy Registry
# =============================================================================

STRATEGY_REGISTRY: dict[str, type[OptionStrategy]] = {
    "LongStraddle": LongStraddle,
    "ShortStraddle": ShortStraddle,
    "IronCondor": IronCondor,
    "BullCallSpread": BullCallSpread,
    "BearPutSpread": BearPutSpread,
    "CoveredCall": CoveredCall,
    "CalendarSpread": CalendarSpread,
}


def get_strategy(name: str, **kwargs) -> OptionStrategy:
    """Instantiate a strategy by name."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}")
    return cls(**kwargs)


def list_strategies() -> list[str]:
    """Return all available strategy names."""
    return list(STRATEGY_REGISTRY.keys())


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    S = 22000  # NIFTY spot
    sigma = 0.15
    tte = 30 / 365  # 30 days
    r = 0.07

    print("=" * 60)
    print("Black-Scholes Test")
    print(f"Spot={S}, Vol={sigma}, TTE={tte:.4f}yr, r={r}")
    print(f"ATM Call: ₹{bs_call_price(S, S, tte, r, sigma):.2f}")
    print(f"ATM Put:  ₹{bs_put_price(S, S, tte, r, sigma):.2f}")

    for name in list_strategies():
        strat = get_strategy(name)
        print(f"\n{'=' * 60}")
        print(f"Strategy: {strat.name}")
        print(f"Legs: {strat.get_legs_info(S, sigma, tte, r)}")
        print(f"Entry cost: ₹{strat.entry_cost(S, sigma, tte, r):.2f}")
        print(f"Breakevens: {strat.breakeven_points(S, sigma, tte, r)}")
        print(f"Max Profit: ₹{strat.max_profit(S, sigma, tte, r):.2f}")
        print(f"Max Loss:   ₹{strat.max_loss(S, sigma, tte, r):.2f}")
