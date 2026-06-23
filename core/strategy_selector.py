# =============================================================================
# Module 5: Strategy Selection Logic (Decision Engine)
# =============================================================================
# Rules-based + scoring engine that maps market conditions
# (vol percentile, regime, TTE) to optimal option strategies.
# =============================================================================

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VOL_HIGH_PERCENTILE, VOL_LOW_PERCENTILE
from core.options import (
    OptionStrategy, LongStraddle, ShortStraddle, IronCondor,
    BullCallSpread, BearPutSpread, CoveredCall, CalendarSpread,
    STRATEGY_REGISTRY, get_strategy,
)


# =============================================================================
# Rules-Based Strategy Selection
# =============================================================================

def select_strategy(
    vol_percentile: float,
    regime: str,
    tte_days: int = 30,
) -> OptionStrategy:
    """
    Select the optimal strategy based on current market conditions.

    Decision Tree
    -------------
    1. High vol (percentile > 80):
       - Sideways → Iron Condor (sell premium, defined risk)
       - Trending → Short Straddle (sell premium, undefined risk but high edge)

    2. Low vol (percentile < 20):
       - Any regime → Long Straddle (buy cheap vol, expect expansion)

    3. Medium vol:
       - Bull → Bull Call Spread (directional, capped risk)
       - Bear → Bear Put Spread (directional, capped risk)
       - Sideways + short TTE → Iron Condor
       - Sideways + long TTE → Calendar Spread
       - Default → Covered Call (income generation)

    Parameters
    ----------
    vol_percentile : float
        Current volatility percentile (0–100).
    regime : str
        Market regime: 'BULL', 'BEAR', 'SIDEWAYS', 'UNKNOWN'.
    tte_days : int
        Days to expiry for the planned trade.

    Returns
    -------
    OptionStrategy
        Instantiated strategy object.
    """
    regime = regime.upper()

    # High volatility regime → sell premium
    if vol_percentile > VOL_HIGH_PERCENTILE:
        if regime == "SIDEWAYS":
            return IronCondor()
        else:
            return ShortStraddle()

    # Low volatility regime → buy cheap options
    if vol_percentile < VOL_LOW_PERCENTILE:
        return LongStraddle()

    # Medium volatility → directional or neutral strategies
    if regime == "BULL":
        return BullCallSpread()
    elif regime == "BEAR":
        return BearPutSpread()
    elif regime == "SIDEWAYS":
        if tte_days <= 30:
            return IronCondor()
        else:
            return CalendarSpread()
    else:
        return CoveredCall()


# =============================================================================
# Strategy Scoring
# =============================================================================

# Scoring weights for each condition
_SCORE_MATRIX = {
    # (regime, vol_bucket) → {strategy_name: base_score}
    ("BULL", "HIGH"): {
        "ShortStraddle": 60, "IronCondor": 70, "BullCallSpread": 50,
        "BearPutSpread": 10, "CoveredCall": 55, "LongStraddle": 20,
        "CalendarSpread": 40,
    },
    ("BULL", "MED"): {
        "ShortStraddle": 40, "IronCondor": 45, "BullCallSpread": 80,
        "BearPutSpread": 10, "CoveredCall": 65, "LongStraddle": 30,
        "CalendarSpread": 50,
    },
    ("BULL", "LOW"): {
        "ShortStraddle": 20, "IronCondor": 25, "BullCallSpread": 60,
        "BearPutSpread": 5, "CoveredCall": 40, "LongStraddle": 85,
        "CalendarSpread": 55,
    },
    ("BEAR", "HIGH"): {
        "ShortStraddle": 55, "IronCondor": 65, "BullCallSpread": 10,
        "BearPutSpread": 50, "CoveredCall": 20, "LongStraddle": 25,
        "CalendarSpread": 40,
    },
    ("BEAR", "MED"): {
        "ShortStraddle": 35, "IronCondor": 40, "BullCallSpread": 10,
        "BearPutSpread": 80, "CoveredCall": 25, "LongStraddle": 35,
        "CalendarSpread": 45,
    },
    ("BEAR", "LOW"): {
        "ShortStraddle": 15, "IronCondor": 20, "BullCallSpread": 5,
        "BearPutSpread": 55, "CoveredCall": 15, "LongStraddle": 85,
        "CalendarSpread": 50,
    },
    ("SIDEWAYS", "HIGH"): {
        "ShortStraddle": 70, "IronCondor": 90, "BullCallSpread": 15,
        "BearPutSpread": 15, "CoveredCall": 50, "LongStraddle": 15,
        "CalendarSpread": 55,
    },
    ("SIDEWAYS", "MED"): {
        "ShortStraddle": 50, "IronCondor": 70, "BullCallSpread": 20,
        "BearPutSpread": 20, "CoveredCall": 55, "LongStraddle": 30,
        "CalendarSpread": 65,
    },
    ("SIDEWAYS", "LOW"): {
        "ShortStraddle": 25, "IronCondor": 35, "BullCallSpread": 15,
        "BearPutSpread": 15, "CoveredCall": 30, "LongStraddle": 80,
        "CalendarSpread": 60,
    },
}


def _vol_bucket(vol_percentile: float) -> str:
    """Map vol percentile to a bucket."""
    if vol_percentile > VOL_HIGH_PERCENTILE:
        return "HIGH"
    elif vol_percentile < VOL_LOW_PERCENTILE:
        return "LOW"
    return "MED"


def score_strategy(
    strategy_name: str,
    vol_percentile: float,
    regime: str,
    tte_days: int = 30,
) -> float:
    """
    Score a strategy for the given market conditions (0–100).

    Higher score = better fit for current conditions.
    """
    regime = regime.upper()
    if regime not in ("BULL", "BEAR", "SIDEWAYS"):
        regime = "SIDEWAYS"  # Default for UNKNOWN

    vol_bucket = _vol_bucket(vol_percentile)
    key = (regime, vol_bucket)

    scores = _SCORE_MATRIX.get(key, {})
    base_score = scores.get(strategy_name, 30)  # Default 30 for unknown

    # TTE bonus: some strategies favor shorter/longer TTE
    tte_bonus = 0
    if strategy_name in ("IronCondor", "ShortStraddle") and tte_days <= 30:
        tte_bonus = 10  # Theta decay accelerates < 30 DTE
    elif strategy_name == "CalendarSpread" and tte_days >= 45:
        tte_bonus = 10
    elif strategy_name == "LongStraddle" and tte_days >= 30:
        tte_bonus = 5

    return min(base_score + tte_bonus, 100)


def recommend_top_n(
    vol_percentile: float,
    regime: str,
    tte_days: int = 30,
    n: int = 3,
) -> list[dict]:
    """
    Return top-N strategy recommendations with scores.

    Returns
    -------
    list[dict]
        Each dict: {name, strategy, score, reason}
    """
    regime = regime.upper()
    results = []

    for name in STRATEGY_REGISTRY:
        sc = score_strategy(name, vol_percentile, regime, tte_days)
        reason = _explain_recommendation(name, vol_percentile, regime, tte_days)
        results.append({
            "name": name,
            "strategy": get_strategy(name),
            "score": sc,
            "reason": reason,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:n]


def _explain_recommendation(
    strategy_name: str,
    vol_percentile: float,
    regime: str,
    tte_days: int,
) -> str:
    """Generate a human-readable explanation for the recommendation."""
    vol_desc = (
        "high" if vol_percentile > VOL_HIGH_PERCENTILE
        else "low" if vol_percentile < VOL_LOW_PERCENTILE
        else "moderate"
    )

    explanations = {
        "LongStraddle": (
            f"Volatility is {vol_desc} (P{vol_percentile:.0f}). "
            f"Options are cheap — buying a straddle captures potential vol expansion."
        ),
        "ShortStraddle": (
            f"Volatility is {vol_desc} (P{vol_percentile:.0f}) in a {regime.lower()} market. "
            f"Selling premium captures elevated theta with mean-reversion edge."
        ),
        "IronCondor": (
            f"Volatility is {vol_desc} (P{vol_percentile:.0f}) in a {regime.lower()} market. "
            f"Defined-risk premium collection with clear profit zone."
        ),
        "BullCallSpread": (
            f"Market is {regime.lower()} with {vol_desc} volatility. "
            f"Debit spread captures upside with capped risk."
        ),
        "BearPutSpread": (
            f"Market is {regime.lower()} with {vol_desc} volatility. "
            f"Debit spread captures downside with capped risk."
        ),
        "CoveredCall": (
            f"Moderate conditions — covered call generates income "
            f"with downside from the underlying position."
        ),
        "CalendarSpread": (
            f"Volatility is {vol_desc} with {tte_days} DTE. "
            f"Calendar spread profits from time decay differential between expiries."
        ),
    }

    return explanations.get(strategy_name, "Strategy suitable for current conditions.")


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    # Test different scenarios
    scenarios = [
        {"vol_percentile": 90, "regime": "SIDEWAYS", "tte_days": 30},
        {"vol_percentile": 15, "regime": "BULL", "tte_days": 30},
        {"vol_percentile": 50, "regime": "BEAR", "tte_days": 45},
        {"vol_percentile": 85, "regime": "BULL", "tte_days": 21},
    ]

    for s in scenarios:
        print(f"\n{'=' * 60}")
        print(f"Scenario: vol_pct={s['vol_percentile']}, "
              f"regime={s['regime']}, tte={s['tte_days']}d")

        selected = select_strategy(**s)
        print(f"Selected: {selected.name}")

        print(f"\nTop 3 recommendations:")
        for rec in recommend_top_n(**s):
            print(f"  {rec['name']} (score={rec['score']:.0f})")
            print(f"    → {rec['reason']}")
