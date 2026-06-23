# =============================================================================
# Module 3: Market Regime Detection
# =============================================================================
# Classifies each trading day into Bull, Bear, or Sideways using:
#   - SMA crossover (50 vs 200 day)
#   - ADX (Average Directional Index) for trend strength
#   - SMA proximity band for sideways detection
# =============================================================================

import numpy as np
import pandas as pd

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SMA_SHORT, SMA_LONG, ADX_THRESHOLD, REGIME_BAND_PCT


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def compute_moving_averages(
    df: pd.DataFrame,
    short: int = SMA_SHORT,
    long: int = SMA_LONG,
) -> pd.DataFrame:
    """
    Compute short and long SMAs on Close price.

    Returns the original DataFrame with added SMA columns.
    """
    result = df.copy()
    result[f"SMA_{short}"] = result["Close"].rolling(window=short).mean()
    result[f"SMA_{long}"] = result["Close"].rolling(window=long).mean()
    return result


# ---------------------------------------------------------------------------
# ADX (Average Directional Index)
# ---------------------------------------------------------------------------

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute the Average Directional Index (ADX) for trend strength.

    ADX < 20 → weak/no trend (sideways)
    ADX 20–40 → developing trend
    ADX > 40 → strong trend

    Parameters
    ----------
    df : pd.DataFrame
        Must contain High, Low, Close columns.
    period : int
        Lookback period (default 14).

    Returns
    -------
    pd.Series
        ADX values.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=df.index,
    )

    # Smoothed averages (Wilder's smoothing)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)

    # ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()

    return adx


# ---------------------------------------------------------------------------
# Regime Detection
# ---------------------------------------------------------------------------

def detect_regime(
    df: pd.DataFrame,
    short: int = SMA_SHORT,
    long: int = SMA_LONG,
    adx_threshold: float = ADX_THRESHOLD,
    band_pct: float = REGIME_BAND_PCT,
) -> pd.Series:
    """
    Classify each trading day into a market regime.

    Rules
    -----
    1. If ADX < adx_threshold → SIDEWAYS (weak trend)
    2. If SMA_short and SMA_long are within band_pct of each other → SIDEWAYS
    3. If SMA_short > SMA_long → BULL
    4. If SMA_short < SMA_long → BEAR

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Close, High, Low columns.

    Returns
    -------
    pd.Series
        Regime labels: 'BULL', 'BEAR', or 'SIDEWAYS'.
    """
    data = compute_moving_averages(df, short, long)
    adx = compute_adx(df)

    sma_short_col = f"SMA_{short}"
    sma_long_col = f"SMA_{long}"

    sma_s = data[sma_short_col]
    sma_l = data[sma_long_col]

    # Proximity check: SMA_short within band_pct of SMA_long
    proximity = ((sma_s - sma_l).abs() / sma_l) < band_pct

    regime = pd.Series(index=df.index, dtype="object")

    # Default: classify by SMA crossover
    regime[sma_s > sma_l] = "BULL"
    regime[sma_s < sma_l] = "BEAR"
    regime[sma_s == sma_l] = "SIDEWAYS"

    # Override: weak trend or SMA proximity → SIDEWAYS
    regime[adx < adx_threshold] = "SIDEWAYS"
    regime[proximity] = "SIDEWAYS"

    # Fill NaN (early dates without enough data for SMA200) with UNKNOWN
    regime.fillna("UNKNOWN", inplace=True)

    return regime


# ---------------------------------------------------------------------------
# Regime Summary Statistics
# ---------------------------------------------------------------------------

def regime_summary(regimes: pd.Series) -> pd.DataFrame:
    """
    Compute summary statistics for each regime.

    Returns
    -------
    pd.DataFrame
        Columns: Regime, Count, Percentage, Avg_Duration_Days
    """
    # Count days per regime
    counts = regimes.value_counts()
    total = len(regimes)

    # Compute average streak duration
    streaks = []
    current_regime = regimes.iloc[0]
    current_count = 1

    for r in regimes.iloc[1:]:
        if r == current_regime:
            current_count += 1
        else:
            streaks.append({"Regime": current_regime, "Duration": current_count})
            current_regime = r
            current_count = 1
    streaks.append({"Regime": current_regime, "Duration": current_count})

    streak_df = pd.DataFrame(streaks)
    avg_duration = streak_df.groupby("Regime")["Duration"].mean()

    summary = pd.DataFrame({
        "Regime": counts.index,
        "Days": counts.values,
        "Percentage": (counts.values / total * 100).round(1),
        "Avg_Streak_Days": [
            avg_duration.get(r, 0) for r in counts.index
        ],
    })

    return summary.reset_index(drop=True)


def get_current_regime(df: pd.DataFrame) -> str:
    """Return the most recent regime classification."""
    regimes = detect_regime(df)
    valid = regimes[regimes != "UNKNOWN"]
    if valid.empty:
        return "UNKNOWN"
    return valid.iloc[-1]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from market_data import fetch_ohlcv

    print("Fetching NIFTY data for regime detection...")
    nifty = fetch_ohlcv("NIFTY")

    regime = detect_regime(nifty)
    print(f"\nRegime (last 10 days):\n{regime.tail(10)}")
    print(f"\nCurrent regime: {get_current_regime(nifty)}")

    summary = regime_summary(regime)
    print(f"\nRegime summary:\n{summary}")
