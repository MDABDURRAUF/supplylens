"""Prescriptive analytics: what to change to avoid a predicted disruption.

For each order the model flags as likely to be late, every realistic
alternative (another vendor that has supplied the same SKU, another carrier)
is priced and re-scored, and the option with the lowest expected total cost
is recommended, provided it is also less likely to arrive late. Expected cost includes the cost of a late delivery weighted
by its predicted probability, so a cheaper but unreliable option does not win
by default.
"""

from __future__ import annotations

import pandas as pd

from .data import add_derived_columns
from .risk import LateDeliveryModel


def disruption_kpis(orders: pd.DataFrame) -> dict[str, float]:
    """Headline delivery-reliability and cost indicators for a set of orders."""
    df = add_derived_columns(orders)
    late = df[df["is_late"]]
    return {
        "order_lines": float(len(df)),
        "on_time_delivery_rate": float(1 - df["is_late"].mean()),
        "late_order_lines": float(len(late)),
        "avg_days_late_when_late": float(late["days_late"].mean()) if len(late) else 0.0,
        "p90_days_late_when_late": float(late["days_late"].quantile(0.9)) if len(late) else 0.0,
        "logistics_cost_per_shipment": float(df["freight_cost"].mean()),
        "logistics_cost_per_unit": float(df["freight_cost"].sum() / df["quantity"].sum()),
        "defect_rate": float(df["units_rejected"].sum() / df["quantity"].sum()),
    }


def recommend_alternatives(
    model: LateDeliveryModel,
    orders: pd.DataFrame,
    history: pd.DataFrame | None = None,
    min_risk: float = 0.30,
    late_cost_pct: float = 0.10,
    min_history_lines: int = 5,
) -> pd.DataFrame:
    """Recommend a vendor and/or carrier change for high-risk orders.

    Parameters
    ----------
    model
        A fitted ``LateDeliveryModel``.
    orders
        The orders to assess (for example, open or recently placed orders).
    history
        Past orders used to estimate each vendor's price for a SKU, each
        vendor's usual promised lead time and each lane's freight rate.
        Defaults to ``orders``.
    min_risk
        Only orders with a predicted late probability at or above this are
        considered.
    late_cost_pct
        Assumed cost of a late delivery as a share of the order's goods
        value (lost sales, expediting, penalties). This is a business
        assumption and the recommendations depend on it.
    min_history_lines
        A vendor is only offered as an alternative for a SKU it has supplied
        on at least this many past order lines.

    Returns
    -------
    One row per order where an alternative is both less likely to be late and
    has a lower expected cost than the current choice, with the current and recommended vendor/carrier, the late
    probability of each, and ``expected_saving``. The current choice is costed
    with the same historical estimates as the alternatives, so the comparison
    is like for like. Orders with no better alternative are left out.
    """
    if not 0 <= late_cost_pct:
        raise ValueError("late_cost_pct must be zero or positive")
    hist = add_derived_columns(orders if history is None else history)
    df = add_derived_columns(orders)
    df["current_risk"] = model.predict_proba(orders)
    risky = df[df["current_risk"] >= min_risk]
    empty_cols = [
        "order_id", "sku", "quantity", "current_vendor", "current_carrier", "current_risk",
        "recommended_vendor", "recommended_carrier", "recommended_risk", "change", "expected_saving",
    ]
    if risky.empty:
        return pd.DataFrame(columns=empty_cols)

    # What history says each option would cost and promise.
    g = hist.groupby(["sku", "vendor"])
    vendor_sku = pd.DataFrame(
        {"lines": g.size(), "est_price": g["spend"].sum() / g["quantity"].sum()}
    ).reset_index()
    vendor_sku = vendor_sku[vendor_sku["lines"] >= min_history_lines].drop(columns="lines")
    vendor_lead = hist.groupby("vendor")["promised_lead_days"].median().rename("est_lead_days").reset_index()
    lane_rate = (
        hist.groupby(["region", "carrier"])["freight_per_unit"].median().rename("est_freight_per_unit").reset_index()
    )

    base = risky[["order_id", "order_date", "sku", "category", "region", "quantity", "vendor", "carrier",
                  "current_risk"]].rename(columns={"vendor": "current_vendor", "carrier": "current_carrier"})
    cand = (
        base.merge(vendor_sku, on="sku")
        .merge(vendor_lead, on="vendor")
        .merge(lane_rate, on="region")
    )
    cand["promised_date"] = cand["order_date"] + pd.to_timedelta(cand["est_lead_days"].round(), unit="D")
    cand["late_probability"] = model.predict_proba(cand).to_numpy()

    goods = cand["quantity"] * cand["est_price"]
    cand["expected_cost"] = (
        goods + cand["quantity"] * cand["est_freight_per_unit"] + cand["late_probability"] * late_cost_pct * goods
    )
    is_current = (cand["vendor"] == cand["current_vendor"]) & (cand["carrier"] == cand["current_carrier"])
    # The aim is to prevent the disruption, so an alternative must be less
    # likely to arrive late than the current choice, not merely cheaper.
    safer = cand["late_probability"] < cand["current_risk"]
    cand, is_current = cand[safer | is_current], is_current[safer | is_current]
    current_cost = cand[is_current].set_index("order_id")["expected_cost"]

    best = cand.loc[cand.groupby("order_id")["expected_cost"].idxmin()].set_index("order_id")
    best = best[best.index.isin(current_cost.index)]
    best["expected_saving"] = current_cost.loc[best.index] - best["expected_cost"]
    best = best[best["expected_saving"] > 0]

    vendor_changed = best["vendor"] != best["current_vendor"]
    carrier_changed = best["carrier"] != best["current_carrier"]
    best["change"] = "vendor and carrier"
    best.loc[vendor_changed & ~carrier_changed, "change"] = "vendor"
    best.loc[~vendor_changed & carrier_changed, "change"] = "carrier"

    out = best.reset_index().rename(
        columns={"vendor": "recommended_vendor", "carrier": "recommended_carrier",
                 "late_probability": "recommended_risk"}
    )
    return out[empty_cols].sort_values("expected_saving", ascending=False).reset_index(drop=True)
