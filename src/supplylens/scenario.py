"""What-if analysis: price a planned order, and stress-test losing a supplier."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import add_derived_columns
from .risk import LateDeliveryModel


def evaluate_options(
    model: LateDeliveryModel,
    history: pd.DataFrame,
    sku: str,
    region: str,
    quantity: float,
    order_date,
    late_cost_pct: float = 0.10,
    min_history_lines: int = 5,
) -> pd.DataFrame:
    """Every vendor and carrier choice for a planned order, ranked by expected cost.

    Options are the vendors that have supplied ``sku`` on at least
    ``min_history_lines`` past order lines, combined with every carrier that
    has served ``region``. Price, promised lead time and freight rate are
    estimated from ``history``; the model supplies the late probability.

        expected_cost = goods_cost + freight_cost + expected_late_cost
        expected_late_cost = late_probability x late_cost_pct x goods_cost

    The first row is the lowest expected cost. ``lowest_risk`` marks the
    option least likely to be late, which is not always the same one.
    """
    hist = add_derived_columns(history)
    sku_hist = hist[hist["sku"] == sku]
    if sku_hist.empty:
        raise ValueError(f"SKU '{sku}' does not appear in the order history")
    if quantity <= 0:
        raise ValueError("quantity must be positive")

    g = sku_hist.groupby("vendor")
    vendors = pd.DataFrame({"lines": g.size(), "est_price": g["spend"].sum() / g["quantity"].sum()})
    vendors = vendors[vendors["lines"] >= min_history_lines].drop(columns="lines").reset_index()
    if vendors.empty:
        raise ValueError(f"No vendor has supplied '{sku}' on at least {min_history_lines} order lines")
    vendors = vendors.merge(
        hist.groupby("vendor")["promised_lead_days"].median().rename("est_lead_days").reset_index(), on="vendor"
    )
    lanes = hist[hist["region"] == region].groupby("carrier")["freight_per_unit"].median()
    if lanes.empty:
        raise ValueError(f"Region '{region}' does not appear in the order history")
    lanes = lanes.rename("est_freight_per_unit").reset_index()

    opts = vendors.merge(lanes, how="cross")
    opts["sku"], opts["region"], opts["quantity"] = sku, region, float(quantity)
    opts["category"] = sku_hist["category"].iloc[0]
    opts["order_date"] = pd.Timestamp(order_date)
    opts["promised_date"] = opts["order_date"] + pd.to_timedelta(opts["est_lead_days"].round(), unit="D")
    opts["late_probability"] = model.predict_proba(opts).to_numpy()

    opts["goods_cost"] = opts["quantity"] * opts["est_price"]
    opts["freight_cost"] = opts["quantity"] * opts["est_freight_per_unit"]
    opts["expected_late_cost"] = opts["late_probability"] * late_cost_pct * opts["goods_cost"]
    opts["expected_cost"] = opts["goods_cost"] + opts["freight_cost"] + opts["expected_late_cost"]
    opts = opts.sort_values("expected_cost").reset_index(drop=True)
    opts["extra_vs_best"] = opts["expected_cost"] - opts["expected_cost"].iloc[0]
    opts["lowest_risk"] = opts["late_probability"] == opts["late_probability"].min()
    return opts[
        ["vendor", "carrier", "est_lead_days", "est_price", "late_probability", "goods_cost", "freight_cost",
         "expected_late_cost", "expected_cost", "extra_vs_best", "lowest_risk"]
    ]


def simulate_vendor_disruption(
    orders: pd.DataFrame, vendor: str, weeks: float = 8, min_history_lines: int = 5
) -> pd.DataFrame:
    """What it would take to replace ``vendor`` for ``weeks`` weeks, SKU by SKU.

    For each SKU the vendor supplies, the volume affected is the vendor's
    average weekly volume of that SKU x ``weeks``. The replacement is the
    other vendor with the lowest average price for the SKU (among those with
    at least ``min_history_lines`` order lines).

    Columns: ``units_affected`` and ``spend_affected`` (at the lost vendor's
    price); ``alternative_vendor``; ``price_change_pct`` and ``extra_cost`` of
    buying from the alternative; ``lead_time_change_days`` and the
    alternative's ``alternative_on_time_rate``; and ``status``, which is
    ``covered`` or ``no alternative``. Rows with no alternative are supply
    that cannot be replaced from the existing vendor base.

    Alternatives are assumed able to absorb the volume; capacity is not
    modelled, so treat the result as the best case.
    """
    df = add_derived_columns(orders)
    if vendor not in set(df["vendor"]):
        raise ValueError(f"Vendor '{vendor}' does not appear in the orders")
    if weeks <= 0:
        raise ValueError("weeks must be positive")
    span_weeks = max((df["order_date"].max() - df["order_date"].min()).days / 7, 1.0)

    g = df.groupby(["sku", "vendor"])
    stats = pd.DataFrame(
        {
            "lines": g.size(),
            "units": g["quantity"].sum(),
            "price": g["spend"].sum() / g["quantity"].sum(),
            "lead": g["lead_time_days"].mean(),
            "on_time": 1 - g["is_late"].mean(),
        }
    ).reset_index()

    lost = stats[stats["vendor"] == vendor].set_index("sku")
    others = stats[(stats["vendor"] != vendor) & (stats["lines"] >= min_history_lines)]
    best = others.loc[others.groupby("sku")["price"].idxmin()].set_index("sku") if len(others) else others

    out = pd.DataFrame(index=lost.index)
    out["category"] = df.groupby("sku")["category"].first().reindex(lost.index)
    out["units_affected"] = lost["units"] / span_weeks * weeks
    out["spend_affected"] = out["units_affected"] * lost["price"]
    out["alternative_vendor"] = best["vendor"].reindex(lost.index) if len(best) else np.nan
    alt_price = best["price"].reindex(lost.index) if len(best) else np.nan
    out["price_change_pct"] = alt_price / lost["price"] - 1
    out["extra_cost"] = (alt_price - lost["price"]) * out["units_affected"]
    out["lead_time_change_days"] = (best["lead"].reindex(lost.index) if len(best) else np.nan) - lost["lead"]
    out["alternative_on_time_rate"] = best["on_time"].reindex(lost.index) if len(best) else np.nan
    out["status"] = np.where(out["alternative_vendor"].isna(), "no alternative", "covered")
    return out.sort_values("spend_affected", ascending=False).reset_index()


def vendor_exposure(orders: pd.DataFrame, weeks: float = 8, min_history_lines: int = 5) -> pd.DataFrame:
    """Rank every vendor by what losing it for ``weeks`` weeks would cost.

    ``uncovered_spend`` is spend on SKUs with no alternative supplier and
    ``extra_cost`` is the net price difference of moving the rest to the
    cheapest alternative (negative means the alternatives are cheaper).
    Vendors are sorted by uncovered spend, then by extra cost.
    """
    df = add_derived_columns(orders)
    total = df["spend"].sum()
    rows = []
    for vendor, spend in df.groupby("vendor")["spend"].sum().items():
        sim = simulate_vendor_disruption(orders, vendor, weeks, min_history_lines)
        uncovered = sim["status"] == "no alternative"
        rows.append(
            {
                "vendor": vendor,
                "share_of_spend": spend / total,
                "skus_supplied": len(sim),
                "skus_without_alternative": int(uncovered.sum()),
                "spend_affected": float(sim["spend_affected"].sum()),
                "uncovered_spend": float(sim.loc[uncovered, "spend_affected"].sum()),
                "extra_cost": float(sim["extra_cost"].sum()),
                "avg_lead_time_change_days": float(sim["lead_time_change_days"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["uncovered_spend", "extra_cost"], ascending=False).reset_index(drop=True)
