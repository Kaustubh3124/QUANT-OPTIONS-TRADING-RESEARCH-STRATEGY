# =============================================================================
# Module 2: Volatility Analysis
# =============================================================================
# Historical volatility, realized volatility, volatility percentile,
# volatility cone, term structure, and GARCH(1,1) forecasting.
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Historical Volatility (rolling standard deviation of log returns)
# ---------------------------------------------------------------------------

def historical_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """
    Compute rolling historical volatility.

    Parameters
    ----------
    returns : pd.Series
        Log return series.
    window : int
        Rolling window in trading days (default 21 ≈ 1 month).
    annualize : bool
        If True, multiply by √252 to annualize.

    Returns
    -------
    pd.Series
        Rolling HV series.
    """
    hv = returns.rolling(window=window).std()
    if annualize:
        hv = hv * np.sqrt(TRADING_DAYS_PER_YEAR)
    return hv


# ---------------------------------------------------------------------------
# Realized Volatility (sum of squared returns, annualized)
# ---------------------------------------------------------------------------

def realized_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """
    Compute rolling realized volatility using sum of squared returns.
    RV = sqrt(sum(r_i^2) * (252 / window))
    """
    squared = returns ** 2
    rv = squared.rolling(window=window).sum()
    if annualize:
        rv = np.sqrt(rv * (TRADING_DAYS_PER_YEAR / window))
    else:
        rv = np.sqrt(rv)
    return rv


# ---------------------------------------------------------------------------
# Volatility Percentile
# ---------------------------------------------------------------------------

def volatility_percentile(
    hv_series: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """
    Compute the percentile rank of current HV vs. trailing lookback window.

    A value of 90 means current HV is higher than 90% of values
    in the trailing window → volatility is unusually high.

    Parameters
    ----------
    hv_series : pd.Series
        Historical volatility series.
    lookback : int
        Number of trailing days to compare against (default 252 ≈ 1 year).

    Returns
    -------
    pd.Series
        Percentile rank (0–100) for each date.
    """
    result = pd.Series(index=hv_series.index, dtype=float)

    values = hv_series.dropna().values
    indices = hv_series.dropna().index

    for i in range(lookback, len(values)):
        window = values[i - lookback : i]
        current = values[i]
        result.loc[indices[i]] = percentileofscore(window, current, kind="rank")

    return result


def current_vol_percentile(
    hv_series: pd.Series,
    lookback: int = 252,
) -> float:
    """Return the most recent volatility percentile value."""
    hv_clean = hv_series.dropna()
    if len(hv_clean) < lookback + 1:
        return np.nan
    window = hv_clean.values[-(lookback + 1) : -1]
    current = hv_clean.values[-1]
    return percentileofscore(window, current, kind="rank")


# ---------------------------------------------------------------------------
# Volatility Term Structure
# ---------------------------------------------------------------------------

def volatility_term_structure(
    returns: pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute HV across multiple lookback windows.

    Returns a DataFrame with one column per window, useful for
    identifying term-structure inversions (short-term vol > long-term vol).
    """
    if windows is None:
        windows = [5, 10, 21, 63]

    result = pd.DataFrame(index=returns.index)
    for w in windows:
        result[f"HV_{w}d"] = historical_volatility(returns, window=w)
    return result


# ---------------------------------------------------------------------------
# Volatility Cone
# ---------------------------------------------------------------------------

def volatility_cone(
    returns: pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute min/max/mean/median/current HV across multiple windows.
    Used for the volatility cone visualization.

    Returns
    -------
    pd.DataFrame
        Columns: Window, Min, P25, Median, Mean, P75, Max, Current
    """
    if windows is None:
        windows = [5, 10, 21, 42, 63, 126, 252]

    rows = []
    for w in windows:
        hv = historical_volatility(returns, window=w).dropna()
        if hv.empty:
            continue
        rows.append({
            "Window": w,
            "Min": hv.min(),
            "P25": hv.quantile(0.25),
            "Median": hv.median(),
            "Mean": hv.mean(),
            "P75": hv.quantile(0.75),
            "Max": hv.max(),
            "Current": hv.iloc[-1],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GARCH(1,1) Volatility Forecasting
# ---------------------------------------------------------------------------

def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "StudentsT",
) -> object:
    """
    Fit a GARCH(p, q) model to the return series.

    Parameters
    ----------
    returns : pd.Series
        Log return series (raw, not scaled).
    p : int
        GARCH lag order.
    q : int
        ARCH lag order.
    dist : str
        Error distribution ('Normal', 'StudentsT', 'SkewStudent').

    Returns
    -------
    ARCHModelResult
        Fitted GARCH model result.
    """
    from arch import arch_model

    # Scale returns by 100 for better optimizer convergence
    scaled = returns.dropna() * 100

    model = arch_model(
        scaled,
        mean="Zero",
        vol="Garch",
        p=p,
        q=q,
        dist=dist,
        rescale=False,
    )
    result = model.fit(disp="off", show_warning=False)
    return result


def forecast_volatility(
    garch_result,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Generate forward-looking volatility forecasts from a fitted GARCH model.

    Returns
    -------
    pd.DataFrame
        Forecasted variance and annualized volatility for each horizon step.
    """
    forecast = garch_result.forecast(horizon=horizon)
    variance = forecast.variance.iloc[-1]  # Last row = forecast from today

    # Convert back from scaled (×100) to original scale
    vol_daily = np.sqrt(variance) / 100
    vol_annual = vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    result = pd.DataFrame({
        "Horizon": range(1, horizon + 1),
        "Forecast_Daily_Vol": vol_daily.values,
        "Forecast_Annual_Vol": vol_annual.values,
    })
    return result


def garch_conditional_volatility(
    garch_result,
) -> pd.Series:
    """
    Extract the fitted conditional volatility series from a GARCH model.
    Useful for plotting historical model-implied vol vs. realized vol.
    """
    # Conditional volatility is in scaled units (×100), convert back
    cond_vol = garch_result.conditional_volatility / 100
    annual_vol = cond_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return annual_vol


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from market_data import fetch_ohlcv, get_returns

    print("Fetching NIFTY data for volatility analysis...")
    nifty = fetch_ohlcv("NIFTY")
    returns = get_returns(nifty)

    hv = historical_volatility(returns, window=21)
    print(f"\n21-day HV (last 5):\n{hv.tail()}")

    rv = realized_volatility(returns, window=21)
    print(f"\n21-day RV (last 5):\n{rv.tail()}")

    pct = current_vol_percentile(hv)
    print(f"\nCurrent vol percentile: {pct:.1f}%")

    cone = volatility_cone(returns)
    print(f"\nVolatility cone:\n{cone}")

    print("\nFitting GARCH(1,1)...")
    try:
        garch = fit_garch(returns)
        print(garch.summary().tables[1])
        fcast = forecast_volatility(garch, horizon=5)
        print(f"\n5-day vol forecast:\n{fcast}")
    except Exception as e:
        print(f"GARCH fitting failed: {e}")
