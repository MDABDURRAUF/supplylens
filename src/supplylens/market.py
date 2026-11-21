"""Market analysis: growth, seasonality, price movement and price benchmarks.

These functions answer the questions a market analyst asks of order data:
which segments are growing, when demand peaks, how prices are moving, and
where the same item is being bought at very different prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import add_derived_columns


def _full_years(df: pd.DataFrame) -> list[int]:
    """Calendar years that the data covers from January through December."""
    months = df.groupby(df["order_date"].dt.year)["order_date"].agg(lambda s: s.dt.month.nunique())
    return sorted(int(y) for y in months[months == 12].index)


def market_growth(orders: pd.DataFrame, by: str = "category", value: str = "spend") -> pd.DataFrame:
    """Size, share and growth of each segment.

    Growth is measured between the first and last *complete* calendar years
    in the data, so a partial year never shows up as a false decline.
    ``cagr`` is the compound annual growth rate between those two years and
    ``share_change_pts`` is the change in the segment's share of the total,
    in percentage points.
    """
    df = add_derived_columns(orders)
    if value not in df.columns:
        raise ValueError(f"Unknown value column '{value}'")
    years = _full_years(df)
    if len(years) < 2:
        raise ValueError("Growth analysis needs at least two complete calendar years of orders")
    first, last = years[0], years[-1]

    yearly = df[df["order_date"].dt.year.isin(years)].pivot_table(
        index=by, columns=df["order_date"].dt.year, values=value, aggfunc="sum", fill_value=0.0
    )
    out = pd.DataFrame(
        {
            f"{value}_{first}": yearly[first],
            f"{value}_{last}": yearly[last],
            "share_first": yearly[first] / yearly[first].sum(),
            "share_last": yearly[last] / yearly[last].sum(),
        }
    )
    ratio = (yearly[last] / yearly[first]).replace([np.inf, -np.inf], np.nan)
    out["total_growth"] = ratio - 1
    out["cagr"] = ratio ** (1 / (last - first)) - 1
    out["share_change_pts"] = 100 * (out["share_last"] - out["share_first"])
    return out.sort_values(f"{value}_{last}", ascending=False).reset_index()


def seasonality_index(orders: pd.DataFrame, by: str = "category") -> pd.DataFrame:
    """Seasonal index of units ordered, by calendar month, for each segment.

    An index of 1.30 for November means that segment's November demand runs
    30% above its own average month. Rows are segments, columns are months
    1-12. Only complete calendar years are used.
    """
    df = add_derived_columns(orders)
    years = _full_years(df)
    if not years:
        raise ValueError("Seasonality analysis needs at least one complete calendar year of orders")
    df = df[df["order_date"].dt.year.isin(years)]
    monthly = df.pivot_table(
        index=by, columns=df["order_date"].dt.month, values="quantity", aggfunc="sum", fill_value=0
    )
    index = monthly.div(monthly.mean(axis=1), axis=0)
    index.columns.name = "month"
    return index


def price_index(orders: pd.DataFrame, by: str = "category", freq: str = "QS") -> pd.DataFrame:
    """Price level over time for each segment, with the first period = 100.

    The price of each SKU is tracked separately and the SKU price changes are
    combined using fixed weights (each SKU's share of total spend), so a shift
    in what is being bought is not mistaken for a change in price.
    Rows are periods, columns are segments.
    """
    df = add_derived_columns(orders)
    g = df.groupby([by, "sku", pd.Grouper(key="order_date", freq=freq)])
    price = (g["spend"].sum() / g["quantity"].sum()).rename("price").reset_index()
    wide = price.pivot_table(index="order_date", columns=[by, "sku"], values="price").ffill()
    relative = wide / wide.bfill().iloc[0]

    weights = df.groupby([by, "sku"])["spend"].sum()
    result = {}
    for segment in relative.columns.get_level_values(0).unique():
        rel = relative[segment]
        w = weights[segment].reindex(rel.columns)
        result[segment] = 100 * rel.mul(w, axis=1).sum(axis=1) / rel.notna().mul(w, axis=1).sum(axis=1)
    out = pd.DataFrame(result)
    out.index.name = "period"
    return out


def price_benchmark(orders: pd.DataFrame, min_lines: int = 5) -> pd.DataFrame:
    """Compare what each vendor charges for the same SKU.

    For every SKU bought from two or more vendors, each vendor's average unit
    price is compared with the lowest vendor average for that SKU.
    ``potential_saving`` is what would have been saved had those units been
    bought at the lowest average price; it is an upper bound, since it ignores
    capacity, quality and delivery differences between vendors.
    Vendor/SKU pairs with fewer than ``min_lines`` order lines are ignored.
    """
    df = add_derived_columns(orders)
    g = df.groupby(["category", "sku", "vendor"])
    stats = pd.DataFrame(
        {"order_lines": g.size(), "units": g["quantity"].sum(), "spend": g["spend"].sum()}
    ).reset_index()
    stats = stats[stats["order_lines"] >= min_lines]
    stats["avg_price"] = stats["spend"] / stats["units"]

    by_sku = stats.groupby("sku")["avg_price"]
    stats["vendors_for_sku"] = by_sku.transform("size")
    stats["best_price"] = by_sku.transform("min")
    stats = stats[stats["vendors_for_sku"] >= 2]
    stats["premium_pct"] = stats["avg_price"] / stats["best_price"] - 1
    stats["potential_saving"] = (stats["avg_price"] - stats["best_price"]) * stats["units"]
    return stats.sort_values("potential_saving", ascending=False).reset_index(drop=True)


def segment_matrix(orders: pd.DataFrame, rows: str = "region", columns: str = "category") -> pd.DataFrame:
    """Share of total spend in each (rows x columns) cell; all cells sum to 1."""
    df = add_derived_columns(orders)
    table = df.pivot_table(index=rows, columns=columns, values="spend", aggfunc="sum", fill_value=0.0)
    return table / table.to_numpy().sum()
