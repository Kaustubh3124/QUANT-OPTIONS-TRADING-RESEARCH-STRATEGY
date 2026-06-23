# =============================================================================
# Module 8: Greeks Calculation
# =============================================================================
# Analytical Black-Scholes Greeks for individual legs and multi-leg strategies.
# Delta, Gamma, Theta, Vega, Rho — plus aggregate strategy-level Greeks
# and time-series Greek exposure tracking.
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import norm

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_DAYS_PER_YEAR
from core.options import _d1, _d2, OptionLeg, OptionStrategy


# =============================================================================
# Individual Greeks (analytical, Black-Scholes)
# =============================================================================

def delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """
    Option delta: ∂V/∂S.
    Call delta ∈ [0, 1], Put delta ∈ [-1, 0].
    """
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def gamma(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """
    Option gamma: ∂²V/∂S² (same for calls and puts).
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """
    Option theta: ∂V/∂t (time decay per calendar day).
    Returns a negative value (options lose value as time passes).
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)

    common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        th = common - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        th = common + r * K * np.exp(-r * T) * norm.cdf(-d2)

    # Convert from per-year to per-day
    return th / TRADING_DAYS_PER_YEAR


def vega(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """
    Option vega: ∂V/∂σ (same for calls and puts).
    Returns the change in option price for a 1-unit (100%) change in vol.
    Commonly reported per 1% vol change → divide by 100.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    # Per 1% change in vol
    return S * norm.pdf(d1) * np.sqrt(T) / 100


def rho(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """
    Option rho: ∂V/∂r.
    Returns the change in option price for a 1% change in interest rate.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d2 = _d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


# =============================================================================
# Compute all Greeks for a single option
# =============================================================================

def compute_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> dict[str, float]:
    """Compute all Greeks for a single option."""
    return {
        "Delta": delta(S, K, T, r, sigma, option_type),
        "Gamma": gamma(S, K, T, r, sigma),
        "Theta": theta(S, K, T, r, sigma, option_type),
        "Vega": vega(S, K, T, r, sigma),
        "Rho": rho(S, K, T, r, sigma, option_type),
    }


# =============================================================================
# Aggregate Strategy-level Greeks
# =============================================================================

def strategy_greeks(
    strategy: OptionStrategy,
    spot: float,
    sigma: float,
    tte: float,
    r: float,
) -> dict[str, float]:
    """
    Compute aggregate Greeks for a multi-leg strategy.

    Each leg's Greeks are weighted by the leg's position (+1 or -1).
    """
    legs = strategy.build_legs(spot, sigma, tte, r)

    total = {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0, "Rho": 0}
    for leg in legs:
        g = compute_greeks(spot, leg.strike, leg.tte, r, sigma, leg.option_type)
        for key in total:
            total[key] += leg.position * g[key]

    # Round for display
    return {k: round(v, 6) for k, v in total.items()}


def strategy_greeks_table(
    strategy: OptionStrategy,
    spot: float,
    sigma: float,
    tte: float,
    r: float,
) -> pd.DataFrame:
    """
    Return per-leg Greeks as a DataFrame for display.
    """
    legs = strategy.build_legs(spot, sigma, tte, r)
    rows = []
    for leg in legs:
        g = compute_greeks(spot, leg.strike, leg.tte, r, sigma, leg.option_type)
        rows.append({
            "Leg": f"{'Long' if leg.position > 0 else 'Short'} {leg.option_type.upper()} @ {leg.strike}",
            "Position": leg.position,
            **{k: round(leg.position * v, 6) for k, v in g.items()},
        })

    # Add totals row
    total = strategy_greeks(strategy, spot, sigma, tte, r)
    rows.append({"Leg": "TOTAL", "Position": "", **total})

    return pd.DataFrame(rows)


# =============================================================================
# Greeks Through Time
# =============================================================================

def greeks_through_time(
    strategy: OptionStrategy,
    spot_series: pd.Series,
    sigma: float,
    tte_start: float,
    r: float,
) -> pd.DataFrame:
    """
    Compute strategy-level Greeks at each date as TTE decreases.

    Parameters
    ----------
    spot_series : pd.Series
        Spot price at each date.
    sigma : float
        Assumed constant vol (or could be extended to vol series).
    tte_start : float
        Time to expiry at first date (in years).
    r : float
        Risk-free rate.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with Delta, Gamma, Theta, Vega columns.
    """
    n_days = len(spot_series)
    daily_decrement = 1 / TRADING_DAYS_PER_YEAR

    records = []
    for i, (date, spot) in enumerate(spot_series.items()):
        tte = max(tte_start - i * daily_decrement, 0.001)  # Prevent zero
        greeks = strategy_greeks(strategy, spot, sigma, tte, r)
        records.append({"Date": date, "TTE": tte, **greeks})

    df = pd.DataFrame(records)
    df.set_index("Date", inplace=True)
    return df


# =============================================================================
# Greek Sensitivity Heatmaps
# =============================================================================

def greek_sensitivity_grid(
    strategy: OptionStrategy,
    spot_center: float,
    sigma_center: float,
    tte: float,
    r: float,
    greek_name: str = "Delta",
    spot_range_pct: float = 0.10,
    sigma_range: float = 0.10,
    n_points: int = 20,
) -> pd.DataFrame:
    """
    Generate a 2D grid of a Greek across spot and vol ranges.
    Useful for heatmap visualization.

    Returns
    -------
    pd.DataFrame
        Index = spot values, columns = sigma values, cells = Greek value.
    """
    greek_funcs = {
        "Delta": lambda s, sig: strategy_greeks(strategy, s, sig, tte, r)["Delta"],
        "Gamma": lambda s, sig: strategy_greeks(strategy, s, sig, tte, r)["Gamma"],
        "Theta": lambda s, sig: strategy_greeks(strategy, s, sig, tte, r)["Theta"],
        "Vega": lambda s, sig: strategy_greeks(strategy, s, sig, tte, r)["Vega"],
    }

    func = greek_funcs.get(greek_name)
    if func is None:
        raise ValueError(f"Unknown greek '{greek_name}'")

    spots = np.linspace(
        spot_center * (1 - spot_range_pct),
        spot_center * (1 + spot_range_pct),
        n_points,
    )
    sigmas = np.linspace(
        max(sigma_center - sigma_range, 0.01),
        sigma_center + sigma_range,
        n_points,
    )

    grid = np.zeros((n_points, n_points))
    for i, s in enumerate(spots):
        for j, sig in enumerate(sigmas):
            grid[i, j] = func(s, sig)

    return pd.DataFrame(
        grid,
        index=np.round(spots, 1),
        columns=np.round(sigmas, 3),
    )


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    from core.options import LongStraddle, IronCondor

    S, sigma, tte, r = 22000, 0.15, 30 / 365, 0.07

    print("Single ATM Call Greeks:")
    g = compute_greeks(S, S, tte, r, sigma, "call")
    for k, v in g.items():
        print(f"  {k}: {v:.6f}")

    print("\nLong Straddle Greeks:")
    strat = LongStraddle()
    sg = strategy_greeks(strat, S, sigma, tte, r)
    for k, v in sg.items():
        print(f"  {k}: {v:.6f}")

    print("\nIron Condor Greeks Table:")
    ic = IronCondor()
    table = strategy_greeks_table(ic, S, sigma, tte, r)
    print(table.to_string(index=False))
