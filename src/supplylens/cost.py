"""Distribution cost analysis: where freight money goes and where it leaks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import add_derived_columns


def cost_summary(orders: pd.DataFrame, by: str | list[str] = "carrier") -> pd.DataFrame:
    """Freight cost per group, with cost per unit and cost as a share of goods value."""
    df = add_derived_columns(orders)
    g = df.groupby(by)
    out = pd.DataFrame(
        {
            "shipments": g.size(),
            "units": g["quantity"].sum(),
            "goods_value": g["spend"].sum(),
            "freight_cost": g["freight_cost"].sum(),
        }
    )
    out["freight_per_unit"] = out["freight_cost"] / out["units"]
    out["freight_pct_of_goods"] = out["freight_cost"] / out["goods_value"]
    out["share_of_freight"] = out["freight_cost"] / out["freight_cost"].sum()
    return out.sort_values("freight_cost", ascending=False).reset_index()


def cost_outliers(orders: pd.DataFrame, threshold: float = 3.5, min_group_size: int = 20) -> pd.DataFrame:
    """Flag shipments whose freight cost per unit is abnormal for their lane.

    A lane is a (region, carrier) pair. Within each lane the modified z-score
    is computed on log freight-per-unit using the median and the median
    absolute deviation, which are not distorted by the outliers themselves.
    Shipments with a score above ``threshold`` are returned, along with an
    estimate of the excess cost over the lane's median rate.
    """
    df = add_derived_columns(orders)
    df["log_rate"] = np.log(df["freight_per_unit"])
    g = df.groupby(["region", "carrier"])["log_rate"]
    median = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median())
    size = g.transform("size")

    df["robust_z"] = np.where(mad > 0, 0.6745 * (df["log_rate"] - median) / mad, 0.0)
    df["lane_median_rate"] = np.exp(median)
    df["excess_cost"] = (df["freight_per_unit"] - df["lane_median_rate"]) * df["quantity"]

    flagged = df[(df["robust_z"] > threshold) & (size >= min_group_size)]
    cols = [
        "order_id",
        "order_date",
        "vendor",
        "region",
        "carrier",
        "quantity",
        "freight_cost",
        "freight_per_unit",
        "lane_median_rate",
        "robust_z",
        "excess_cost",
    ]
    return flagged[cols].sort_values("excess_cost", ascending=False).reset_index(drop=True)


def pareto(orders: pd.DataFrame, by: str = "vendor", value: str = "spend") -> pd.DataFrame:
    """Rank groups by ``value`` with cumulative share and an ABC class.

    Class A covers the groups making up the first 80% of the total, B the
    next 15%, and C the remainder.
    """
    df = add_derived_columns(orders)
    if value not in df.columns:
        raise ValueError(f"Unknown value column '{value}'")
    out = df.groupby(by)[value].sum().sort_values(ascending=False).reset_index()
    out["share"] = out[value] / out[value].sum()
    out["cumulative_share"] = out["share"].cumsum()
    before = out["cumulative_share"] - out["share"]
    out["abc_class"] = np.select([before < 0.80, before < 0.95], ["A", "B"], default="C")
    return out
