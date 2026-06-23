# =============================================================================
# Module 1: Market Data Collection
# =============================================================================
# Downloads OHLCV data from yfinance, caches to CSV, computes returns.
# Handles yfinance API quirks: multi-index columns, timezone issues, gaps.
# =============================================================================

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TICKERS, START_DATE, END_DATE, DATA_DIR


def fetch_ohlcv(
    ticker_name: str,
    start: str = START_DATE,
    end: str = END_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance and cache to CSV.

    Parameters
    ----------
    ticker_name : str
        Human-readable ticker name (e.g. 'NIFTY', 'RELIANCE').
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    force_refresh : bool
        If True, re-download even if cached file exists.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date (index), Open, High, Low, Close, Volume.
    """
    yf_symbol = TICKERS.get(ticker_name, ticker_name)
    cache_file = os.path.join(DATA_DIR, f"{ticker_name}_{start}_{end}.csv")

    # Return cached data if available
    if os.path.exists(cache_file) and not force_refresh:
        df = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
        if not df.empty:
            return df

    # Download from yfinance
    try:
        raw = yf.download(yf_symbol, start=start, end=end, progress=False)
    except Exception as e:
        raise RuntimeError(f"Failed to download {ticker_name} ({yf_symbol}): {e}")

    if raw.empty:
        raise ValueError(
            f"No data returned for {ticker_name} ({yf_symbol}) "
            f"between {start} and {end}."
        )

    # Handle multi-index columns (yfinance sometimes returns MultiIndex)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only OHLCV columns
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    # yfinance may return 'Adj Close' — drop it, we use raw Close
    available = [c for c in expected_cols if c in raw.columns]
    df = raw[available].copy()

    # Remove timezone info if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "Date"

    # Drop rows with all NaN
    df.dropna(how="all", inplace=True)

    # Forward-fill small gaps (holidays across exchanges)
    df.ffill(inplace=True)

    # Cache to CSV
    df.to_csv(cache_file)

    return df


def load_cached_data(ticker_name: str) -> pd.DataFrame | None:
    """
    Load the most recent cached CSV for a ticker.
    Returns None if no cached file exists.
    """
    files = [
        f for f in os.listdir(DATA_DIR)
        if f.startswith(ticker_name) and f.endswith(".csv")
    ]
    if not files:
        return None

    # Pick the most recently modified file
    files.sort(
        key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)),
        reverse=True,
    )
    path = os.path.join(DATA_DIR, files[0])
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def fetch_all_tickers(
    start: str = START_DATE,
    end: str = END_DATE,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all configured tickers.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker_name → OHLCV DataFrame.
    """
    result = {}
    for name in TICKERS:
        try:
            result[name] = fetch_ohlcv(name, start, end, force_refresh)
        except (RuntimeError, ValueError) as e:
            print(f"[WARN] Skipping {name}: {e}")
    return result


def get_returns(df: pd.DataFrame, method: str = "log") -> pd.Series:
    """
    Compute returns from a price DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Close' column.
    method : str
        'log' for log returns, 'simple' for arithmetic returns.

    Returns
    -------
    pd.Series
        Return series (first value is NaN, dropped).
    """
    close = df["Close"]
    if method == "log":
        returns = np.log(close / close.shift(1))
    elif method == "simple":
        returns = close.pct_change()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'log' or 'simple'.")

    return returns.dropna()


def get_close_prices(df: pd.DataFrame) -> pd.Series:
    """Extract the Close price series."""
    return df["Close"].dropna()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Fetching NIFTY data...")
    nifty = fetch_ohlcv("NIFTY")
    print(f"Shape: {nifty.shape}")
    print(f"Date range: {nifty.index[0]} → {nifty.index[-1]}")
    print(nifty.tail())

    returns = get_returns(nifty)
    print(f"\nLog returns (last 5):\n{returns.tail()}")
