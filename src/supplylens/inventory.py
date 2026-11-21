"""Inventory buffers against demand and lead-time uncertainty, and stockout risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import add_derived_columns


def inventory_policy(
    orders: pd.DataFrame,
    service_level: float = 0.95,
    on_hand: dict[str, float] | pd.Series | None = None,
) -> pd.DataFrame:
    """Safety stock and reorder point per SKU, and stockout risk if stock is given.

    Uses the standard formula for uncertain demand and uncertain lead time:

        demand during lead time:  mean = d * L
                                  sd   = sqrt(L * sd_d^2 + d^2 * sd_L^2)
        safety stock  = z * sd          (z from the target service level)
        reorder point = d * L + safety stock

    where ``d`` and ``sd_d`` are the mean and standard deviation of daily
    demand (estimated from weekly totals) and ``L`` and ``sd_L`` are the mean
    and standard deviation of lead time in days. Demand during the lead time
    is treated as normally distributed and units ordered are used as the
    measure of demand.

    ``on_hand`` maps SKU to units currently in stock. When given, the result
    includes ``stockout_probability``: the chance that demand during one lead
    time exceeds the stock on hand, i.e. the risk of running out before a
    replenishment order placed today arrives.
    """
    if not 0.5 <= service_level < 1:
        raise ValueError("service_level must be at least 0.5 and below 1")
    df = add_derived_columns(orders)
    z = float(norm.ppf(service_level))

    weeks = pd.date_range(
        df["order_date"].min().to_period("W").start_time, df["order_date"].max(), freq="W-MON"
    )
    weekly = (
        df.groupby(["sku", pd.Grouper(key="order_date", freq="W-MON", label="left", closed="left")])["quantity"]
        .sum()
        .unstack("sku")
        .reindex(weeks)
        .fillna(0.0)
    )
    # The first and last weeks are usually partial; leave them out when there is enough data.
    if len(weekly) > 4:
        weekly = weekly.iloc[1:-1]

    lead = df.groupby("sku")["lead_time_days"].agg(["mean", "std"]).fillna(0.0)
    out = pd.DataFrame(
        {
            "category": df.groupby("sku")["category"].first(),
            "avg_daily_demand": weekly.mean() / 7,
            "sd_daily_demand": weekly.std(ddof=0) / np.sqrt(7),
            "avg_lead_time_days": lead["mean"],
            "sd_lead_time_days": lead["std"],
        }
    )
    out["lead_time_demand"] = out["avg_daily_demand"] * out["avg_lead_time_days"]
    out["lead_time_demand_sd"] = np.sqrt(
        out["avg_lead_time_days"] * out["sd_daily_demand"] ** 2
        + out["avg_daily_demand"] ** 2 * out["sd_lead_time_days"] ** 2
    )
    out["safety_stock"] = np.ceil(z * out["lead_time_demand_sd"])
    out["reorder_point"] = np.ceil(out["lead_time_demand"] + out["safety_stock"])

    if on_hand is not None:
        stock = pd.Series(on_hand, dtype=float).reindex(out.index)
        out["on_hand"] = stock
        sd = out["lead_time_demand_sd"].replace(0, np.nan)
        prob = pd.Series(norm.sf((stock - out["lead_time_demand"]) / sd), index=out.index)
        certain = (stock < out["lead_time_demand"]).astype(float)
        out["stockout_probability"] = prob.fillna(certain).where(stock.notna())
        out["below_reorder_point"] = (stock < out["reorder_point"]).where(stock.notna())
        out = out.sort_values("stockout_probability", ascending=False)

    return out.reset_index()
