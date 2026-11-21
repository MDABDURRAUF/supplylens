"""Build a single-file HTML report covering every analysis."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from .cost import cost_outliers, cost_summary
from .data import add_derived_columns
from .forecast import backtest, demand_series, forecast
from .market import market_growth, price_benchmark
from .prescribe import recommend_alternatives
from .risk import concentration, train_late_delivery_model
from .vendor import vendor_scorecard

_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#1c2430}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{margin:0 0 4px;font-size:28px} h2{margin:40px 0 8px;font-size:20px}
p.note{color:#56627a;margin:0 0 12px;font-size:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.kpi{background:#fff;border:1px solid #dfe3ea;border-radius:8px;padding:14px}
.kpi b{display:block;font-size:22px} .kpi span{font-size:13px;color:#56627a}
.scroll{overflow-x:auto;background:#fff;border:1px solid #dfe3ea;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid #edf0f4;white-space:nowrap}
th{background:#eef1f6;font-weight:600} th:first-child,td:first-child{text-align:left}
"""


def _table(df: pd.DataFrame, formats: dict[str, str] | None = None, max_rows: int = 25) -> str:
    formats = formats or {}
    df = df.head(max_rows)
    head = "".join(f"<th>{escape(str(c).replace('_', ' '))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for col, val in row.items():
            if col in formats and pd.notna(val):
                text = formats[col].format(val)
            elif isinstance(val, pd.Timestamp):
                text = val.strftime("%Y-%m-%d")
            elif isinstance(val, float):
                text = f"{val:,.2f}"
            else:
                text = str(val)
            cells.append(f"<td>{escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def build_report(orders: pd.DataFrame, horizon: int = 6) -> str:
    """Return the report as an HTML string."""
    df = add_derived_columns(orders)
    card = vendor_scorecard(orders)
    conc = concentration(orders)
    lanes = cost_summary(orders, by=["region", "carrier"])
    outliers = cost_outliers(orders)
    series = demand_series(orders)
    scores = backtest(series)
    fc = forecast(series, horizon=horizon)

    kpis = [
        (f"{len(df):,}", "order lines"),
        (f"${df['spend'].sum():,.0f}", "total spend"),
        (f"{1 - df['is_late'].mean():.1%}", "on-time delivery"),
        (f"{df['units_rejected'].sum() / df['quantity'].sum():.2%}", "defect rate"),
        (f"${df['freight_cost'].sum():,.0f}", "freight cost"),
        (f"${outliers['excess_cost'].sum():,.0f}", "excess freight flagged"),
    ]
    kpi_html = "".join(f'<div class="kpi"><b>{v}</b><span>{escape(k)}</span></div>' for v, k in kpis)

    pct = "{:.1%}"
    money = "${:,.0f}"
    sections = [
        "<h2>Vendor scorecard</h2>"
        '<p class="note">Composite score weights delivery, quality, price compliance and lead-time consistency. '
        "Tier A is 85 and above, B is 70 and above, C is below 70.</p>"
        + _table(
            card[
                [
                    "rank", "vendor", "tier", "composite_score", "on_time_rate", "defect_rate",
                    "price_variance_pct", "avg_lead_time_days", "spend",
                ]
            ],
            {"on_time_rate": pct, "defect_rate": "{:.2%}", "price_variance_pct": "{:+.1%}", "spend": money,
             "composite_score": "{:.1f}", "avg_lead_time_days": "{:.1f}"},
        ),
        "<h2>Supplier concentration</h2>"
        '<p class="note">HHI is the sum of squared supplier shares of spend (10,000 = one supplier).</p>'
        + _table(conc, {"top_vendor_share": pct, "total_spend": money, "hhi": "{:,.0f}"}),
        "<h2>Demand forecast</h2>"
        f'<p class="note">Model chosen by rolling-origin backtest: <b>{escape(str(fc["model"].iloc[0]))}</b>. '
        "Backtest accuracy for each candidate model is shown below the forecast.</p>"
        + _table(fc.drop(columns="model"), {c: "{:,.0f}" for c in ["forecast", "lower_80", "upper_80"]})
        + "<br>"
        + _table(scores, {"mae": "{:,.0f}", "rmse": "{:,.0f}", "wape": pct}),
        "<h2>Freight cost by lane</h2>"
        + _table(
            lanes,
            {"units": "{:,.0f}", "goods_value": money, "freight_cost": money, "freight_per_unit": "${:.3f}",
             "freight_pct_of_goods": "{:.2%}", "share_of_freight": pct},
        ),
        "<h2>Freight cost outliers</h2>"
        f'<p class="note">{len(outliers)} shipment(s) cost far more per unit than is normal for their lane. '
        "Largest excess cost first.</p>"
        + _table(
            outliers,
            {"freight_cost": money, "freight_per_unit": "${:.3f}", "lane_median_rate": "${:.3f}",
             "robust_z": "{:.1f}", "excess_cost": money},
        ),
    ]

    try:
        growth = market_growth(orders)
        bench = price_benchmark(orders)
        growth_fmt = {c: money for c in growth.columns if c.startswith("spend_")}
        growth_fmt.update({"share_first": pct, "share_last": pct, "total_growth": "{:+.1%}", "cagr": "{:+.1%}",
                           "share_change_pts": "{:+.1f}"})
        sections.insert(
            0,
            "<h2>Market growth by category</h2>"
            '<p class="note">Compares the first and last complete calendar years in the data.</p>'
            + _table(growth, growth_fmt)
            + "<h2>Price benchmark</h2>"
            f'<p class="note">Same SKU, different vendors. Buying every unit at the lowest vendor average would '
            f'have saved at most ${bench["potential_saving"].sum():,.0f}; this ignores capacity, quality and '
            "delivery differences.</p>"
            + _table(
                bench[["category", "sku", "vendor", "units", "avg_price", "best_price", "premium_pct",
                       "potential_saving"]],
                {"units": "{:,.0f}", "avg_price": "${:.2f}", "best_price": "${:.2f}", "premium_pct": "{:+.1%}",
                 "potential_saving": money},
                max_rows=12,
            ),
        )
    except ValueError as exc:
        sections.insert(0, f'<h2>Market growth</h2><p class="note">Skipped: {escape(str(exc))}</p>')

    try:
        model = train_late_delivery_model(orders)
        m = model.metrics
        sections.append(
            "<h2>Late-delivery prediction</h2>"
            f'<p class="note">Logistic regression tested on the most recent {m["test_rows"]:,} orders '
            f'(trained on the {m["train_rows"]:,} before them). ROC AUC {m["roc_auc"]:.3f}; '
            f'Brier score {m["brier_score"]:.4f} against {m["brier_baseline"]:.4f} for always predicting the '
            "average late rate (lower is better). Strongest drivers are listed below.</p>"
            + _table(model.drivers(), {"coefficient": "{:+.3f}", "odds_ratio": "{:.2f}"}, max_rows=12)
        )
        recent = orders.tail(200)
        recs = recommend_alternatives(model, recent, history=orders)
        sections.append(
            "<h2>Recommended changes for high-risk orders</h2>"
            f'<p class="note">Among the {len(recent)} most recent orders, {len(recs)} have a predicted late '
            "probability of 30% or more and a vendor or carrier alternative with a lower expected cost. "
            "Assumes a late delivery costs 10% of goods value. Total expected saving: "
            f'${recs["expected_saving"].sum():,.0f}.</p>'
            + _table(
                recs.drop(columns=["sku", "quantity"]),
                {"current_risk": pct, "recommended_risk": pct, "expected_saving": money},
                max_rows=15,
            )
        )
    except ValueError as exc:
        sections.append(f'<h2>Late-delivery prediction</h2><p class="note">Skipped: {escape(str(exc))}</p>')

    period = f"{df['order_date'].min():%b %Y} to {df['order_date'].max():%b %Y}"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>SupplyLens report</title><style>{_CSS}</style></head><body><main>"
        f'<h1>Supply chain performance report</h1><p class="note">Orders from {period}</p>'
        f'<div class="kpis">{kpi_html}</div>{"".join(sections)}</main></body></html>'
    )


def write_report(orders: pd.DataFrame, path: str | Path, horizon: int = 6) -> Path:
    """Write the HTML report to ``path`` and return the path."""
    path = Path(path)
    path.write_text(build_report(orders, horizon=horizon), encoding="utf-8")
    return path
