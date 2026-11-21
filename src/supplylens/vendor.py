"""Vendor performance scorecards.

Each vendor is measured on delivery, quality, price compliance and lead-time
consistency. The four measures are converted to 0-100 scores and combined
into one weighted composite score, which sets the vendor's tier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import add_derived_columns

DEFAULT_WEIGHTS = {
    "delivery_score": 0.35,
    "quality_score": 0.30,
    "price_score": 0.20,
    "consistency_score": 0.15,
}

# A reject rate or price overrun at or above these caps scores zero.
DEFECT_RATE_CAP = 0.10
PRICE_OVERRUN_CAP = 0.10
# A lead-time coefficient of variation at or above this cap scores zero.
LEAD_TIME_CV_CAP = 1.0

TIER_A_MIN = 85.0
TIER_B_MIN = 70.0


def _normalise_weights(weights: dict[str, float] | None) -> dict[str, float]:
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    unknown = set(w) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise ValueError(f"Unknown weight(s): {', '.join(sorted(unknown))}")
    if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
        raise ValueError("Weights must be non-negative and sum to more than zero")
    total = sum(w.values())
    return {k: w.get(k, 0.0) / total for k in DEFAULT_WEIGHTS}


def vendor_scorecard(orders: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Return one row per vendor with raw metrics, 0-100 scores and a tier.

    Raw metrics
        on_time_rate        share of order lines delivered by the promised date
        avg_days_late       mean days late, over late lines only
        defect_rate         units rejected / units ordered
        price_variance_pct  spend above (+) or below (-) contract price, as a
                            share of spend at contract price
        avg_lead_time_days  mean days from order to delivery
        lead_time_cv        standard deviation of lead time / mean lead time

    ``weights`` may override ``DEFAULT_WEIGHTS``; they are rescaled to sum to 1.
    """
    w = _normalise_weights(weights)
    df = add_derived_columns(orders)
    df["contract_spend"] = df["quantity"] * df["contract_price"]
    df["late_days_only"] = df["days_late"].where(df["is_late"])

    g = df.groupby("vendor")
    card = pd.DataFrame(
        {
            "order_lines": g.size(),
            "units": g["quantity"].sum(),
            "spend": g["spend"].sum(),
            "on_time_rate": 1 - g["is_late"].mean(),
            "avg_days_late": g["late_days_only"].mean().fillna(0.0),
            "defect_rate": g["units_rejected"].sum() / g["quantity"].sum(),
            "price_variance_pct": (g["spend"].sum() - g["contract_spend"].sum()) / g["contract_spend"].sum(),
            "avg_lead_time_days": g["lead_time_days"].mean(),
            "lead_time_cv": (g["lead_time_days"].std(ddof=0) / g["lead_time_days"].mean()).fillna(0.0),
        }
    )

    card["delivery_score"] = 100 * card["on_time_rate"]
    card["quality_score"] = 100 * (1 - card["defect_rate"] / DEFECT_RATE_CAP).clip(0, 1)
    # Paying below contract is not penalised; only overruns reduce the score.
    card["price_score"] = 100 * (1 - card["price_variance_pct"].clip(lower=0) / PRICE_OVERRUN_CAP).clip(0, 1)
    card["consistency_score"] = 100 * (1 - card["lead_time_cv"] / LEAD_TIME_CV_CAP).clip(0, 1)
    card["composite_score"] = sum(card[k] * w[k] for k in w)

    card["tier"] = np.select(
        [card["composite_score"] >= TIER_A_MIN, card["composite_score"] >= TIER_B_MIN],
        ["A", "B"],
        default="C",
    )
    card = card.sort_values("composite_score", ascending=False)
    card.insert(0, "rank", range(1, len(card) + 1))
    return card.reset_index()


def vendor_trend(orders: pd.DataFrame, freq: str = "QS") -> pd.DataFrame:
    """On-time rate and defect rate per vendor per period (default: quarter)."""
    df = add_derived_columns(orders)
    g = df.groupby(["vendor", pd.Grouper(key="order_date", freq=freq)])
    out = pd.DataFrame(
        {
            "order_lines": g.size(),
            "on_time_rate": 1 - g["is_late"].mean(),
            "defect_rate": g["units_rejected"].sum() / g["quantity"].sum(),
        }
    )
    return out[out["order_lines"] > 0].reset_index().rename(columns={"order_date": "period"})
