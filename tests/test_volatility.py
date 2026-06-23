# =============================================================================
# Tests: Volatility Module
# =============================================================================

import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.volatility import (
    historical_volatility,
    realized_volatility,
    volatility_percentile,
    current_vol_percentile,
    volatility_cone,
    volatility_term_structure,
)


@pytest.fixture
def sample_returns():
    """Generate synthetic log returns for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=504, freq="B")
    returns = pd.Series(np.random.normal(0.0005, 0.015, 504), index=dates)
    return returns


class TestHistoricalVolatility:
    def test_output_length(self, sample_returns):
        hv = historical_volatility(sample_returns, window=21)
        assert len(hv) == len(sample_returns)

    def test_first_values_nan(self, sample_returns):
        hv = historical_volatility(sample_returns, window=21)
        assert hv.iloc[:20].isna().all()

    def test_positive_values(self, sample_returns):
        hv = historical_volatility(sample_returns, window=21).dropna()
        assert (hv > 0).all()

    def test_annualization(self, sample_returns):
        hv_ann = historical_volatility(sample_returns, window=21, annualize=True)
        hv_raw = historical_volatility(sample_returns, window=21, annualize=False)
        ratio = (hv_ann / hv_raw).dropna()
        expected_ratio = np.sqrt(252)
        assert np.allclose(ratio, expected_ratio, atol=0.001)

    def test_known_value(self):
        """Test HV on constant returns (should be 0)."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        constant_returns = pd.Series(0.001, index=dates)
        hv = historical_volatility(constant_returns, window=21).dropna()
        assert np.allclose(hv, 0, atol=1e-10)


class TestRealizedVolatility:
    def test_positive_values(self, sample_returns):
        rv = realized_volatility(sample_returns, window=21).dropna()
        assert (rv > 0).all()

    def test_output_length(self, sample_returns):
        rv = realized_volatility(sample_returns, window=21)
        assert len(rv) == len(sample_returns)


class TestVolatilityPercentile:
    def test_range_0_100(self, sample_returns):
        hv = historical_volatility(sample_returns, window=21)
        pct = volatility_percentile(hv, lookback=252)
        valid = pct.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_current_percentile(self, sample_returns):
        hv = historical_volatility(sample_returns, window=21)
        pct = current_vol_percentile(hv)
        assert 0 <= pct <= 100

    def test_insufficient_data(self):
        """Should return NaN for insufficient data."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        short_hv = pd.Series(np.random.rand(10), index=dates)
        pct = current_vol_percentile(short_hv, lookback=252)
        assert np.isnan(pct)


class TestVolatilityCone:
    def test_columns(self, sample_returns):
        cone = volatility_cone(sample_returns)
        expected_cols = ["Window", "Min", "P25", "Median", "Mean", "P75", "Max", "Current"]
        assert list(cone.columns) == expected_cols

    def test_min_less_than_max(self, sample_returns):
        cone = volatility_cone(sample_returns)
        assert (cone["Min"] <= cone["Max"]).all()

    def test_current_within_range(self, sample_returns):
        cone = volatility_cone(sample_returns)
        assert (cone["Current"] >= cone["Min"]).all()
        assert (cone["Current"] <= cone["Max"]).all()


class TestVolatilityTermStructure:
    def test_columns(self, sample_returns):
        ts = volatility_term_structure(sample_returns, windows=[5, 21, 63])
        assert list(ts.columns) == ["HV_5d", "HV_21d", "HV_63d"]

    def test_output_length(self, sample_returns):
        ts = volatility_term_structure(sample_returns)
        assert len(ts) == len(sample_returns)
