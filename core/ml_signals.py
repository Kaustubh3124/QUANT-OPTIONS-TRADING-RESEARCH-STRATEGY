# =============================================================================
# Advanced: Machine Learning Volatility Signal
# =============================================================================
# XGBoost / Random Forest models that predict volatility expansion vs. crush.
# Uses walk-forward cross-validation to prevent look-ahead bias.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ML_LOOKBACK_WINDOWS, ML_FORWARD_WINDOW, ML_TRAIN_MIN_SAMPLES,
    TRADING_DAYS_PER_YEAR,
)
from core.volatility import historical_volatility, realized_volatility


# =============================================================================
# Feature Engineering
# =============================================================================

def build_features(
    returns: pd.Series,
    close: pd.Series,
    volume: pd.Series | None = None,
    regime: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build ML features from price data.

    Features
    --------
    - Lagged HV (5, 10, 21, 63 day)
    - HV momentum (rate of change of 21-day HV)
    - HV term structure slope (short vs long)
    - Volume ratio (current / 20-day avg) if available
    - Return skewness (21-day rolling)
    - Return kurtosis (21-day rolling)
    - Recent return (5-day cumulative)
    - Regime encoding (one-hot) if available

    Returns
    -------
    pd.DataFrame
        Feature matrix (NaN rows at the start due to lookback).
    """
    features = pd.DataFrame(index=returns.index)

    # Lagged historical volatilities
    for w in ML_LOOKBACK_WINDOWS:
        hv = historical_volatility(returns, window=w)
        features[f"hv_{w}d"] = hv

    # HV momentum (10-day change in 21-day HV)
    hv_21 = features.get("hv_21d")
    if hv_21 is not None:
        features["hv_momentum_10d"] = hv_21 - hv_21.shift(10)
        features["hv_momentum_5d"] = hv_21 - hv_21.shift(5)

    # HV term structure slope (short / long)
    hv_5 = features.get("hv_5d")
    hv_63 = features.get("hv_63d")
    if hv_5 is not None and hv_63 is not None:
        features["hv_term_slope"] = hv_5 / hv_63.replace(0, np.nan)

    # Volume ratio (if volume data available)
    if volume is not None:
        vol_ma = volume.rolling(20).mean()
        features["volume_ratio"] = volume / vol_ma.replace(0, np.nan)

    # Return statistics
    features["return_skew_21d"] = returns.rolling(21).skew()
    features["return_kurt_21d"] = returns.rolling(21).kurt()
    features["return_5d"] = returns.rolling(5).sum()
    features["return_10d"] = returns.rolling(10).sum()

    # Absolute return magnitude (volatility proxy)
    features["abs_return_avg_5d"] = returns.abs().rolling(5).mean()

    # Distance from 20-day high/low
    if close is not None:
        high_20 = close.rolling(20).max()
        low_20 = close.rolling(20).min()
        features["dist_from_high"] = (close - high_20) / high_20
        features["dist_from_low"] = (close - low_20) / low_20

    # Regime encoding (one-hot)
    if regime is not None:
        for r in ["BULL", "BEAR", "SIDEWAYS"]:
            features[f"regime_{r}"] = (regime == r).astype(int)

    return features


# =============================================================================
# Target Variable
# =============================================================================

def build_target(
    returns: pd.Series,
    forward_window: int = ML_FORWARD_WINDOW,
    hv_window: int = 21,
) -> pd.Series:
    """
    Build binary target: Volatility Expansion (1) vs. Crush (0).

    Logic: If realized volatility over the next `forward_window` days
    exceeds the current historical volatility, label = 1 (expansion).

    Parameters
    ----------
    returns : pd.Series
        Log return series.
    forward_window : int
        Number of forward days to compute future RV.
    hv_window : int
        Window for current HV comparison.

    Returns
    -------
    pd.Series
        Binary labels (1 = vol expansion, 0 = vol crush).
    """
    current_hv = historical_volatility(returns, window=hv_window)

    # Forward-looking realized vol (shift backwards to align with current date)
    future_rv = realized_volatility(returns, window=forward_window).shift(-forward_window)

    target = (future_rv > current_hv).astype(int)
    target.name = "vol_expansion"

    return target


# =============================================================================
# Walk-Forward Cross-Validation
# =============================================================================

def walk_forward_split(
    n_samples: int,
    min_train: int = ML_TRAIN_MIN_SAMPLES,
    test_size: int = 63,  # ~3 months
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward train/test splits.

    The training window expands; the test window slides forward.
    This prevents look-ahead bias.

    Returns
    -------
    list of (train_indices, test_indices) tuples.
    """
    splits = []
    start = min_train

    while start + test_size <= n_samples:
        train_idx = np.arange(0, start)
        test_idx = np.arange(start, min(start + test_size, n_samples))
        splits.append((train_idx, test_idx))
        start += test_size

    return splits


# =============================================================================
# Model Training
# =============================================================================

def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_type: str = "xgboost",
) -> dict:
    """
    Train a volatility prediction model with walk-forward validation.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    target : pd.Series
        Binary target (vol expansion = 1, crush = 0).
    model_type : str
        'xgboost' or 'random_forest'.

    Returns
    -------
    dict with keys:
        - model: fitted model (trained on all data)
        - accuracy: walk-forward OOS accuracy
        - auc: walk-forward OOS AUC
        - feature_importance: pd.Series
        - predictions: pd.Series (OOS predictions)
    """
    # Align features and target, drop NaN
    combined = pd.concat([features, target], axis=1).dropna()
    X = combined[features.columns]
    y = combined[target.name]

    if len(X) < ML_TRAIN_MIN_SAMPLES + 63:
        return {
            "model": None,
            "accuracy": 0,
            "auc": 0,
            "feature_importance": pd.Series(),
            "predictions": pd.Series(),
            "error": "Insufficient data for training",
        }

    # Scale features
    scaler = StandardScaler()

    # Walk-forward validation
    splits = walk_forward_split(len(X))
    all_preds = pd.Series(index=X.index, dtype=float)
    all_true = pd.Series(index=X.index, dtype=float)

    for train_idx, test_idx in splits:
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = _get_model(model_type)
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        all_preds.iloc[test_idx] = preds
        all_true.iloc[test_idx] = y_test.values

    # Compute OOS metrics
    valid = all_preds.dropna()
    valid_true = all_true.loc[valid.index]

    accuracy = accuracy_score(valid_true, valid)
    try:
        auc = roc_auc_score(valid_true, valid)
    except ValueError:
        auc = 0.5

    # Train final model on all data
    X_scaled = scaler.fit_transform(X)
    final_model = _get_model(model_type)
    final_model.fit(X_scaled, y)

    # Feature importance
    importance = pd.Series(
        final_model.feature_importances_,
        index=features.columns,
    ).sort_values(ascending=False)

    return {
        "model": final_model,
        "scaler": scaler,
        "accuracy": round(accuracy, 4),
        "auc": round(auc, 4),
        "feature_importance": importance,
        "predictions": all_preds,
        "feature_names": list(features.columns),
    }


def _get_model(model_type: str):
    """Instantiate the ML model."""
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        except ImportError:
            print("[WARN] XGBoost not installed, falling back to Random Forest")
            model_type = "random_forest"

    return RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )


# =============================================================================
# Prediction
# =============================================================================

def predict_signal(
    model_result: dict,
    current_features: pd.DataFrame,
) -> dict:
    """
    Predict volatility expansion probability for current market state.

    Returns
    -------
    dict with keys:
        - prediction: int (0 or 1)
        - probability: float (probability of vol expansion)
        - signal: str ('VOL_EXPANSION' or 'VOL_CRUSH')
    """
    if model_result.get("model") is None:
        return {
            "prediction": 0,
            "probability": 0.5,
            "signal": "NO_SIGNAL",
        }

    model = model_result["model"]
    scaler = model_result["scaler"]

    # Ensure feature order matches training
    feature_names = model_result.get("feature_names", current_features.columns)
    X = current_features[feature_names].iloc[-1:].fillna(0)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]

    return {
        "prediction": int(pred),
        "probability": round(float(prob[1]), 4),  # P(vol expansion)
        "signal": "VOL_EXPANSION" if pred == 1 else "VOL_CRUSH",
    }


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    from core.market_data import fetch_ohlcv, get_returns, get_close_prices
    from core.regime import detect_regime

    print("Building ML features for NIFTY...")
    nifty = fetch_ohlcv("NIFTY")
    returns = get_returns(nifty)
    close = get_close_prices(nifty)
    volume = nifty["Volume"] if "Volume" in nifty.columns else None
    regime = detect_regime(nifty)

    features = build_features(returns, close, volume, regime)
    target = build_target(returns)

    print(f"Features shape: {features.shape}")
    print(f"Target distribution:\n{target.value_counts()}")

    print("\nTraining XGBoost model...")
    result = train_model(features, target, model_type="xgboost")
    print(f"OOS Accuracy: {result['accuracy']:.4f}")
    print(f"OOS AUC: {result['auc']:.4f}")
    print(f"\nTop 10 features:\n{result['feature_importance'].head(10)}")

    signal = predict_signal(result, features)
    print(f"\nCurrent signal: {signal}")
