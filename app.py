"""SupplyLens dashboard. Run with:  streamlit run app.py"""

from __future__ import annotations

import sys
from pathlib import Path

# Let the app run straight from a checkout (for example on Streamlit Community
# Cloud) without the package being installed first.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import supplylens as sl
from supplylens.data import REQUIRED_COLUMNS, add_derived_columns
from supplylens.report import build_report
from supplylens.vendor import DEFAULT_WEIGHTS

st.set_page_config(page_title="SupplyLens", layout="wide")

RED, GREEN, BLUE, GREY = "#d1495b", "#2a9d8f", "#3a6ea5", "#9aa5b1"
TIER_COLORS = {"A": GREEN, "B": "#e9a23b", "C": RED}


# ------------------------------------------------------------------- helpers
def fingerprint(df: pd.DataFrame) -> tuple:
    """Cheap identity for a filtered orders table, used as a cache key."""
    return (len(df), str(df["order_date"].min()), str(df["order_date"].max()),
            round(float(df["freight_cost"].sum()), 2), int(df["quantity"].sum()))


@st.cache_data
def sample_data() -> pd.DataFrame:
    return sl.generate_sample_data()


@st.cache_resource(max_entries=8)
def get_model(fp: tuple, _orders: pd.DataFrame):
    return sl.train_late_delivery_model(_orders)


@st.cache_data(max_entries=8)
def get_comparison(fp: tuple, _orders: pd.DataFrame) -> pd.DataFrame:
    return sl.compare_models(_orders)


@st.cache_data(max_entries=16)
def get_exposure(fp: tuple, _orders: pd.DataFrame, weeks: int) -> pd.DataFrame:
    return sl.vendor_exposure(_orders, weeks=weeks)


def show(fig: go.Figure, height: int = 380, **kwargs):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return st.plotly_chart(fig, width="stretch", **kwargs)


def download(df: pd.DataFrame, name: str):
    st.download_button(f"Download {name}.csv", df.to_csv(index=False), file_name=f"{name}.csv", mime="text/csv")


def need_model():
    try:
        return get_model(fingerprint(orders), orders)
    except ValueError as exc:
        st.warning(f"The late-delivery model cannot be trained on this selection: {exc}")
        st.stop()


PCT = st.column_config.NumberColumn(format="percent")
USD = st.column_config.NumberColumn(format="dollar")


# ------------------------------------------------------------------- sidebar
st.sidebar.title("SupplyLens")
st.sidebar.caption("Predict, explain and prevent supply chain disruptions")

PAGES = [
    "Overview",
    "Predict and explain",
    "Plan an order (what-if)",
    "Supplier failure scenario",
    "Inventory stress test",
    "Model quality",
    "Market analysis",
    "Vendor scorecard",
    "Demand forecast",
    "Distribution cost",
]
page = st.sidebar.radio("View", PAGES)

st.sidebar.divider()
source = st.sidebar.radio("Data", ["Sample data (synthetic)", "Upload a CSV"])
if source == "Upload a CSV":
    upload = st.sidebar.file_uploader("Orders CSV", type="csv")
    st.sidebar.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))
    if upload is None:
        st.info("Upload an orders CSV to begin, or switch to the sample data.")
        st.stop()
    try:
        all_orders = sl.validate_orders(pd.read_csv(upload))
    except sl.SchemaError as exc:
        st.error(str(exc))
        st.stop()
else:
    all_orders = sample_data()
    st.sidebar.caption("The sample data is simulated. It is not real company data.")

lo, hi = all_orders["order_date"].min().date(), all_orders["order_date"].max().date()
start, end = st.sidebar.slider("Order date range", lo, hi, (lo, hi))
f_cat = st.sidebar.multiselect("Categories", sorted(all_orders["category"].unique()), placeholder="All categories")
f_reg = st.sidebar.multiselect("Regions", sorted(all_orders["region"].unique()), placeholder="All regions")

orders = all_orders[(all_orders["order_date"].dt.date >= start) & (all_orders["order_date"].dt.date <= end)]
if f_cat:
    orders = orders[orders["category"].isin(f_cat)]
if f_reg:
    orders = orders[orders["region"].isin(f_reg)]
orders = orders.reset_index(drop=True)
if orders.empty:
    st.warning("No orders match the filters.")
    st.stop()
df = add_derived_columns(orders)
st.sidebar.caption(f"{len(orders):,} of {len(all_orders):,} order lines selected")
st.sidebar.download_button("Download HTML report", build_report(orders) if len(orders) > 200 else "",
                           file_name="supplylens_report.html", mime="text/html", disabled=len(orders) <= 200)


# ================================================================== OVERVIEW
if page == "Overview":
    st.title("Supply chain disruption analytics")
    k = sl.disruption_kpis(orders)
    c = st.columns(5)
    c[0].metric("Order lines", f"{len(df):,}")
    c[1].metric("Total spend", f"${df['spend'].sum() / 1e6:,.2f}M")
    c[2].metric("On-time delivery", f"{k['on_time_delivery_rate']:.1%}")
    c[3].metric("Avg days late (when late)", f"{k['avg_days_late_when_late']:.1f}")
    c[4].metric("Logistics cost / shipment", f"${k['logistics_cost_per_shipment']:,.0f}")

    monthly = df.groupby(pd.Grouper(key="order_date", freq="MS")).agg(
        on_time=("is_late", lambda s: 1 - s.mean()), orders=("order_id", "size"), spend=("spend", "sum")
    ).reset_index()
    left, right = st.columns(2)
    with left:
        st.subheader("On-time delivery by month")
        fig = px.line(monthly, x="order_date", y="on_time", markers=True, labels={"order_date": "", "on_time": "On-time rate"})
        fig.update_yaxes(tickformat=".0%")
        fig.update_traces(line_color=BLUE)
        show(fig, 320)
    with right:
        st.subheader("Spend by month and category")
        by_cat = df.groupby([pd.Grouper(key="order_date", freq="MS"), "category"])["spend"].sum().reset_index()
        show(px.area(by_cat, x="order_date", y="spend", color="category", labels={"order_date": "", "spend": "Spend ($)"}), 320)

    st.subheader("Where late deliveries concentrate")
    dims = st.columns(2)
    row_dim = dims[0].selectbox("Rows", ["vendor", "category", "region", "carrier"], index=0)
    col_dim = dims[1].selectbox("Columns", ["carrier", "region", "category", "vendor"], index=0)
    if row_dim == col_dim:
        st.info("Choose two different dimensions.")
    else:
        heat = df.pivot_table(index=row_dim, columns=col_dim, values="is_late", aggfunc="mean")
        counts = df.pivot_table(index=row_dim, columns=col_dim, values="is_late", aggfunc="size")
        fig = px.imshow(heat, color_continuous_scale="Reds", aspect="auto", text_auto=".0%",
                        labels={"color": "Late rate"})
        fig.update_traces(customdata=counts.to_numpy(),
                          hovertemplate="%{y} / %{x}<br>late rate %{z:.1%}<br>%{customdata:,.0f} order lines<extra></extra>")
        show(fig, max(320, 30 * len(heat) + 80))
        st.caption("Each cell is the share of order lines delivered after the promised date. Hover for the number of lines behind it.")

    st.subheader("What needs attention")
    card = sl.vendor_scorecard(orders)
    conc = sl.concentration(orders)
    outliers = sl.cost_outliers(orders)
    worst = card.iloc[-1]
    single = conc[conc["concentration_risk"] == "single source"]["category"].tolist()
    notes = [
        f"**{worst['vendor']}** has the lowest vendor score ({worst['composite_score']:.0f}/100, "
        f"{worst['on_time_rate']:.0%} on time).",
        (f"**{', '.join(single)}** {'is' if len(single) == 1 else 'are'} single-sourced: there is no alternative supplier."
         if single else "No category is single-sourced."),
        f"**{len(outliers)}** shipment(s) have abnormal freight charges, about **${outliers['excess_cost'].sum():,.0f}** above normal for their lanes.",
    ]
    for note in notes:
        st.markdown("- " + note)

# ======================================================= PREDICT AND EXPLAIN
elif page == "Predict and explain":
    st.title("Predict and explain late deliveries")
    model = need_model()
    c1, c2, c3 = st.columns(3)
    n_recent = c1.slider("Assess the most recent orders", 50, min(2000, len(orders)), min(300, len(orders)), 50)
    min_risk = c2.slider("High-risk threshold (late probability)", 0.10, 0.90, 0.30, 0.05)
    late_cost = c3.slider("Cost of a late delivery (share of goods value)", 0.0, 0.50, 0.10, 0.01)

    recent = orders.tail(n_recent)
    scored = add_derived_columns(recent).assign(late_probability=model.predict_proba(recent))
    risky = scored[scored["late_probability"] >= min_risk].sort_values("late_probability", ascending=False)
    recs = sl.recommend_alternatives(model, recent, history=orders, min_risk=min_risk, late_cost_pct=late_cost)

    m = st.columns(4)
    m[0].metric("High-risk orders", f"{len(risky)} of {len(recent)}")
    m[1].metric("Spend at risk", f"${risky['spend'].sum():,.0f}")
    m[2].metric("Orders with a better option", f"{len(recs)}")
    m[3].metric("Expected saving", f"${recs['expected_saving'].sum():,.0f}")

    fig = px.histogram(scored, x="late_probability", nbins=40, labels={"late_probability": "Predicted late probability"},
                       color_discrete_sequence=[BLUE])
    fig.add_vline(x=min_risk, line_dash="dash", line_color=RED, annotation_text="threshold")
    fig.update_xaxes(tickformat=".0%")
    show(fig, 240)

    st.subheader("High-risk orders")
    st.caption("Click a row to see why that order is at risk and what the alternatives are.")
    table = risky[["order_id", "order_date", "vendor", "carrier", "region", "sku", "quantity", "spend", "late_probability"]]
    event = st.dataframe(
        table, hide_index=True, on_select="rerun", selection_mode="single-row", key="risky_table",
        column_config={"late_probability": st.column_config.ProgressColumn("Late probability", format="percent", min_value=0, max_value=1),
                       "spend": USD, "order_date": st.column_config.DateColumn("Order date")},
    )
    if risky.empty:
        st.info("No orders at or above the threshold. Lower the threshold to see more.")
        st.stop()
    rows = event.selection.rows
    chosen = risky.iloc[rows[0] if rows else 0]
    one = recent[recent["order_id"] == chosen["order_id"]]

    st.subheader(f"Why {chosen['order_id']} is at risk ({chosen['late_probability']:.0%} late probability)")
    left, right = st.columns(2)
    with left:
        contrib = model.contributions(one).iloc[0].sort_values()
        fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h",
                               marker_color=[RED if v > 0 else GREEN for v in contrib.values],
                               hovertemplate="%{y}: odds x%{customdata:.2f}<extra></extra>", customdata=np.exp(contrib.values)))
        fig.update_layout(xaxis_title="Effect on risk (log-odds vs. an average order)")
        show(fig, 320)
        exp = model.explain(one, top_n=3)
        st.markdown("Main reasons: " + "; ".join(
            f"**{r.factor}** ({r.value}) multiplies the odds of lateness by {r.odds_multiplier:.2f}" for r in exp.itertuples()) + ".")
        st.caption("Red bars raise the risk above an average order, green bars lower it. The model is linear in "
                   "log-odds, so these effects are exact and add up to the prediction.")
    with right:
        try:
            opts = sl.evaluate_options(model, orders, chosen["sku"], chosen["region"], chosen["quantity"],
                                       chosen["order_date"], late_cost_pct=late_cost)
        except ValueError as exc:
            st.info(f"No alternatives can be priced: {exc}")
        else:
            opts["option"] = opts["vendor"] + " / " + opts["carrier"]
            opts["kind"] = np.where((opts["vendor"] == chosen["vendor"]) & (opts["carrier"] == chosen["carrier"]),
                                    "current choice", "alternative")
            fig = px.scatter(opts, x="late_probability", y="expected_cost", color="kind", hover_name="option",
                             color_discrete_map={"current choice": RED, "alternative": BLUE},
                             labels={"late_probability": "Late probability", "expected_cost": "Expected cost ($)"})
            fig.update_traces(marker_size=11)
            fig.update_xaxes(tickformat=".0%")
            show(fig, 320)
            best = opts.iloc[0]
            st.markdown(f"Lowest expected cost: **{best['option']}** at ${best['expected_cost']:,.0f} with "
                        f"{best['late_probability']:.0%} late probability. Options towards the bottom-left are better.")

    st.subheader("Recommended changes")
    st.caption("Orders where an alternative is both less likely to be late and cheaper in expected cost.")
    st.dataframe(recs, hide_index=True, column_config={"current_risk": PCT, "recommended_risk": PCT, "expected_saving": USD})
    download(recs, "recommendations")

# ============================================================= PLAN AN ORDER
elif page == "Plan an order (what-if)":
    st.title("Plan an order")
    st.caption("Describe an order you are about to place. Every vendor and carrier that could fill it is priced and scored.")
    model = need_model()
    c = st.columns(3)
    category = c[0].selectbox("Category", sorted(orders["category"].unique()))
    sku = c[1].selectbox("SKU", sorted(orders.loc[orders["category"] == category, "sku"].unique()))
    region = c[2].selectbox("Destination region", sorted(orders["region"].unique()))
    c = st.columns(3)
    quantity = c[0].number_input("Quantity (units)", 1, 100000, 500, 50)
    order_date = c[1].date_input("Order date", hi)
    late_cost = c[2].slider("Cost of a late delivery (share of goods value)", 0.0, 0.50, 0.10, 0.01)
    try:
        opts = sl.evaluate_options(model, orders, sku, region, quantity, order_date, late_cost_pct=late_cost)
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()
    opts["option"] = opts["vendor"] + " / " + opts["carrier"]
    best, safest = opts.iloc[0], opts.loc[opts["late_probability"].idxmin()]
    m = st.columns(3)
    m[0].metric("Lowest expected cost", best["option"], f"${best['expected_cost']:,.0f}", delta_color="off")
    m[1].metric("Lowest risk", safest["option"], f"{safest['late_probability']:.0%} late probability", delta_color="off")
    m[2].metric("Price of choosing the safest", f"${safest['extra_vs_best']:,.0f}",
                f"{safest['late_probability'] - best['late_probability']:+.0%} risk", delta_color="inverse")

    fig = px.scatter(opts, x="late_probability", y="expected_cost", color="vendor", symbol="carrier", hover_name="option",
                     hover_data={"est_price": ":.2f", "est_lead_days": True, "expected_late_cost": ":,.0f"},
                     labels={"late_probability": "Late probability", "expected_cost": "Expected total cost ($)"})
    fig.update_traces(marker_size=12)
    fig.update_xaxes(tickformat=".0%")
    show(fig, 420)
    st.caption("Each point is one vendor and carrier combination. Bottom-left is cheaper and more reliable. "
               "Raise the cost of a late delivery to see reliable options overtake cheap ones.")

    stacked = opts.head(8).melt(id_vars="option", value_vars=["freight_cost", "expected_late_cost"], var_name="part", value_name="cost")
    fig = px.bar(stacked, x="cost", y="option", color="part", orientation="h",
                 labels={"cost": "Cost on top of goods ($)", "option": ""},
                 color_discrete_map={"freight_cost": BLUE, "expected_late_cost": RED})
    fig.update_yaxes(autorange="reversed")
    show(fig, 320)
    st.dataframe(opts.drop(columns="option"), hide_index=True,
                 column_config={"late_probability": PCT, "est_price": USD, "goods_cost": USD, "freight_cost": USD,
                                "expected_late_cost": USD, "expected_cost": USD, "extra_vs_best": USD})

# ================================================== SUPPLIER FAILURE SCENARIO
elif page == "Supplier failure scenario":
    st.title("Supplier failure scenario")
    st.caption("What would it take to replace a vendor that stops supplying? Alternatives are assumed able to absorb "
               "the volume, so this is the best case.")
    weeks = st.slider("Length of the disruption (weeks)", 1, 26, 8)
    exposure = get_exposure(fingerprint(orders), orders, weeks)

    st.subheader("Exposure to each vendor")
    fig = go.Figure()
    fig.add_bar(x=exposure["vendor"], y=exposure["uncovered_spend"], name="Spend with no alternative supplier", marker_color=RED)
    fig.add_bar(x=exposure["vendor"], y=exposure["spend_affected"] - exposure["uncovered_spend"],
                name="Spend that can move to another vendor", marker_color=GREY)
    fig.update_layout(barmode="stack", yaxis_title=f"Spend over {weeks} weeks ($)")
    show(fig, 340)

    vendor = st.selectbox("Vendor to remove", exposure["vendor"].tolist())
    sim = sl.simulate_vendor_disruption(orders, vendor, weeks=weeks)
    uncovered = sim[sim["status"] == "no alternative"]
    m = st.columns(4)
    m[0].metric("Spend affected", f"${sim['spend_affected'].sum():,.0f}")
    m[1].metric("SKUs with no alternative", f"{len(uncovered)} of {len(sim)}")
    m[2].metric("Spend that cannot be replaced", f"${uncovered['spend_affected'].sum():,.0f}")
    extra = sim["extra_cost"].sum()
    m[3].metric("Cost of switching the rest", f"${extra:+,.0f}", help="Negative means the alternatives are cheaper on average")

    covered = sim[sim["status"] == "covered"]
    if len(covered):
        fig = px.scatter(covered, x="lead_time_change_days", y="price_change_pct", size="spend_affected", color="alternative_vendor",
                         hover_name="sku", labels={"lead_time_change_days": "Change in lead time (days)", "price_change_pct": "Change in price"})
        fig.update_yaxes(tickformat="+.0%")
        fig.add_hline(y=0, line_color=GREY)
        fig.add_vline(x=0, line_color=GREY)
        show(fig, 360)
        st.caption("Each bubble is a SKU moved to its cheapest alternative vendor; size is the spend affected. "
                   "Top-right means both dearer and slower.")
    st.dataframe(sim, hide_index=True, column_config={"spend_affected": USD, "extra_cost": USD, "price_change_pct": PCT,
                                                      "alternative_on_time_rate": PCT,
                                                      "units_affected": st.column_config.NumberColumn(format="%.0f"),
                                                      "lead_time_change_days": st.column_config.NumberColumn(format="%.1f")})
    download(sim, f"disruption_{vendor.replace(' ', '_')}")

# ====================================================== INVENTORY STRESS TEST
elif page == "Inventory stress test":
    st.title("Inventory stress test")
    st.caption("Simulate a year of daily demand and replenishment for one item, then apply a disruption and compare.")
    c = st.columns(3)
    sku = c[0].selectbox("SKU", sorted(orders["sku"].unique()))
    service = c[1].slider("Target service level", 0.80, 0.99, 0.95, 0.01)
    cover = c[2].slider("Each replenishment covers (days of demand)", 3, 60, 14)
    st.markdown("**Disruption**")
    c = st.columns(4)
    surge = c[0].slider("Demand change", -50, 100, 30, 5, format="%d%%")
    delay = c[1].slider("Extra supplier lead time (days)", 0, 45, 10)
    shock_start = c[2].slider("Starts on day", 0, 300, 120, 10)
    shock_len = c[3].slider("Lasts (days)", 7, 365, 60, 7)

    base = sl.simulate_sku(orders, sku, service_level=service, order_cover_days=cover)
    shocked = sl.simulate_sku(orders, sku, service_level=service, order_cover_days=cover, demand_multiplier=1 + surge / 100,
                              extra_lead_time_days=delay, shock_start_day=shock_start, shock_days=shock_len)
    p = base["policy"]
    st.markdown(f"Policy for **{sku}**: reorder at **{p['reorder_point']:,.0f}** units (of which safety stock "
                f"**{p['safety_stock']:,.0f}**), order **{p['avg_daily_demand'] * cover:,.0f}** units each time. "
                f"Average demand {p['avg_daily_demand']:.1f} units/day, average lead time {p['avg_lead_time_days']:.1f} days.")

    m = st.columns(4)
    m[0].metric("Fill rate", f"{shocked['fill_rate']:.1%}", f"{(shocked['fill_rate'] - base['fill_rate']) * 100:+.1f} pts vs normal")
    m[1].metric("Stockouts per year", f"{shocked['stockouts_per_year']:.1f}",
                f"{shocked['stockouts_per_year'] - base['stockouts_per_year']:+.1f} vs normal", delta_color="inverse")
    m[2].metric("Recovery time (days out of stock)", f"{shocked['avg_recovery_days']:.1f}",
                f"{shocked['avg_recovery_days'] - base['avg_recovery_days']:+.1f} vs normal", delta_color="inverse")
    m[3].metric("Days out of stock", f"{shocked['stockout_day_share']:.1%}",
                f"{(shocked['stockout_day_share'] - base['stockout_day_share']) * 100:+.1f} pts vs normal", delta_color="inverse")

    fig = go.Figure()
    for res, name, color in [(base, "Normal", GREY), (shocked, "With disruption", RED)]:
        path = res["sample_path"]
        fig.add_scatter(x=path["day"], y=path["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=path["day"], y=path["p05"], line=dict(width=0), fill="tonexty", showlegend=False, hoverinfo="skip",
                        fillcolor="rgba(209,73,91,0.15)" if color == RED else "rgba(154,165,177,0.20)")
        fig.add_scatter(x=path["day"], y=path["median"], name=f"{name} (median)", line=dict(color=color, width=2))
    fig.add_vrect(x0=shock_start, x1=min(365, shock_start + shock_len), fillcolor=RED, opacity=0.07, line_width=0,
                  annotation_text="disruption", annotation_position="top left")
    fig.add_hline(y=p["reorder_point"], line_dash="dot", line_color=BLUE, annotation_text="reorder point")
    fig.update_layout(xaxis_title="Day", yaxis_title="Units on hand")
    show(fig, 420)
    st.caption("Lines are the median of 300 simulated years; bands cover the 5th to 95th percentile. "
               "Raise the service level or the replenishment size to see how much buffer the disruption needs.")

    with st.expander("Safety stock and reorder point for every SKU"):
        policy = sl.inventory_policy(orders, service_level=service)
        st.dataframe(policy, hide_index=True)
        download(policy, "inventory_policy")

# ============================================================== MODEL QUALITY
elif page == "Model quality":
    st.title("How far the predictions can be trusted")
    model = need_model()
    mt = model.metrics
    st.caption(f"Every number here comes from the most recent {mt['test_rows']:,} orders, which the model did not see "
               f"while it was trained on the {mt['train_rows']:,} before them.")

    st.subheader("Does the explainable model give up accuracy?")
    comp = get_comparison(fingerprint(orders), orders)
    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(comp, x="roc_auc", y="model", orientation="h", text_auto=".3f", color="model",
                     color_discrete_sequence=[GREY, BLUE, GREEN], labels={"roc_auc": "ROC AUC (higher is better)", "model": ""})
        fig.update_layout(showlegend=False, xaxis_range=[0.4, 1])
        show(fig, 240)
    with right:
        st.dataframe(comp, hide_index=True, column_config={k: st.column_config.NumberColumn(format="%.3f") for k in comp.columns[1:]})
    gap = comp.loc[2, "roc_auc"] - comp.loc[1, "roc_auc"]
    st.markdown(f"Gradient boosting scores **{gap:+.3f}** ROC AUC against the logistic regression. "
                + ("The explainable model is at least as accurate here." if gap <= 0.005 else
                   "The flexible model finds some signal the linear one misses; that is the price of exact explanations here."))

    left, right = st.columns(2)
    with left:
        st.subheader("Calibration")
        bins = st.slider("Number of bands", 4, 15, 10)
        cal = sl.calibration_table(model, bins=bins)
        top = max(cal["predicted"].max(), cal["observed"].max()) * 1.1
        fig = px.scatter(cal, x="predicted", y="observed", size="orders", color_discrete_sequence=[BLUE],
                         labels={"predicted": "Predicted late rate", "observed": "Observed late rate"})
        fig.add_scatter(x=[0, top], y=[0, top], mode="lines", line=dict(dash="dash", color=GREY), name="perfect")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False)
        show(fig, 340)
        st.caption("Points on the dashed line mean a predicted 30% really is about 30%.")
    with right:
        st.subheader("Choosing an alert threshold")
        tt = sl.threshold_table(model, np.round(np.arange(0.05, 0.91, 0.01), 2))
        t = st.slider("Threshold", 0.05, 0.90, 0.30, 0.01)
        row = tt.iloc[(tt["threshold"] - t).abs().argmin()]
        fig = go.Figure()
        fig.add_scatter(x=tt["threshold"], y=tt["recall"], name="Recall: late orders caught", line_color=BLUE)
        fig.add_scatter(x=tt["threshold"], y=tt["precision"], name="Precision: flags that were right", line_color=GREEN)
        fig.add_scatter(x=tt["threshold"], y=tt["share_flagged"], name="Share of orders flagged", line=dict(color=GREY, dash="dot"))
        fig.add_vline(x=t, line_dash="dash", line_color=RED)
        fig.update_layout(legend=dict(orientation="h", y=-0.25), xaxis_title="Threshold")
        fig.update_yaxes(tickformat=".0%")
        show(fig, 340)
        st.markdown(f"At **{t:.0%}**: {row['orders_flagged']:.0f} orders flagged ({row['share_flagged']:.0%}), catching "
                    f"**{row['recall']:.0%}** of late orders; **{row['precision']:.0%}** of flags are right.")

    st.subheader("What drives late delivery overall")
    drivers = model.drivers().head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=drivers["odds_ratio"] - 1, y=drivers["feature"], orientation="h", base=1,
                           marker_color=[RED if v > 1 else GREEN for v in drivers["odds_ratio"]]))
    fig.update_layout(xaxis_title="Odds ratio (above 1 = more late deliveries)")
    show(fig, 420)

# ============================================================ MARKET ANALYSIS
elif page == "Market analysis":
    st.title("Market analysis")
    segment = st.selectbox("Segment by", ["category", "region", "vendor"])
    try:
        growth = sl.market_growth(orders, by=segment)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Growth against size")
            last = [c for c in growth.columns if c.startswith("spend_")][-1]
            fig = px.scatter(growth, x="share_last", y="cagr", size=last, color=segment, text=segment,
                             labels={"share_last": "Share of spend (last full year)", "cagr": "Compound annual growth"})
            fig.update_traces(textposition="top center")
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickformat="+.0%")
            fig.update_layout(showlegend=False)
            show(fig, 360)
        with right:
            st.subheader("Seasonality")
            season = sl.seasonality_index(orders, by=segment)
            season.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            show(px.imshow(season, color_continuous_scale="RdBu_r", color_continuous_midpoint=1, aspect="auto", text_auto=".2f",
                           labels={"color": "Index"}), 360)
        st.caption("Growth compares the first and last complete calendar years. A seasonal index of 1.30 means demand "
                   "in that month runs 30% above the segment's average month.")
        st.dataframe(growth, hide_index=True, column_config={
            **{c: USD for c in growth.columns if c.startswith("spend_")}, "share_first": PCT, "share_last": PCT,
            "total_growth": PCT, "cagr": PCT, "share_change_pts": st.column_config.NumberColumn(format="%.1f")})

    st.subheader("Price level over time (first quarter = 100)")
    pi = sl.price_index(orders, by="category").reset_index().melt(id_vars="period", var_name="category", value_name="index")
    show(px.line(pi, x="period", y="index", color="category", markers=True, labels={"period": "", "index": "Price index"}), 320)

    st.subheader("Price benchmark: same SKU, different vendors")
    bench = sl.price_benchmark(orders)
    st.caption(f"Buying every unit at the lowest vendor average would have saved at most ${bench['potential_saving'].sum():,.0f}. "
               "This is an upper bound: it ignores capacity, quality and delivery.")
    if len(bench):
        top = bench.groupby("vendor")["potential_saving"].sum().sort_values(ascending=False).reset_index()
        show(px.bar(top, x="vendor", y="potential_saving", color_discrete_sequence=[BLUE],
                    labels={"potential_saving": "Paid above the lowest vendor price ($)", "vendor": ""}), 300)
    st.dataframe(bench[["category", "sku", "vendor", "units", "avg_price", "best_price", "premium_pct", "potential_saving"]],
                 hide_index=True, column_config={"avg_price": USD, "best_price": USD, "premium_pct": PCT, "potential_saving": USD})

# =========================================================== VENDOR SCORECARD
elif page == "Vendor scorecard":
    st.title("Vendor scorecard")
    labels = {"delivery_score": "On-time delivery", "quality_score": "Quality", "price_score": "Price compliance",
              "consistency_score": "Lead-time consistency"}
    with st.expander("Score weights", expanded=True):
        cols = st.columns(4)
        weights = {key: cols[i].slider(labels[key], 0.0, 1.0, DEFAULT_WEIGHTS[key], 0.05) for i, key in enumerate(DEFAULT_WEIGHTS)}
    if sum(weights.values()) == 0:
        st.warning("Set at least one weight above zero.")
        st.stop()
    card = sl.vendor_scorecard(orders, weights)
    left, right = st.columns(2)
    with left:
        fig = px.bar(card, x="composite_score", y="vendor", color="tier", orientation="h", text_auto=".0f",
                     color_discrete_map=TIER_COLORS, labels={"composite_score": "Composite score (0-100)", "vendor": ""})
        fig.update_yaxes(categoryorder="total ascending")
        show(fig, 400)
    with right:
        fig = px.scatter(card, x="on_time_rate", y="defect_rate", size="spend", color="tier", hover_name="vendor", text="vendor",
                         color_discrete_map=TIER_COLORS, labels={"on_time_rate": "On-time rate", "defect_rate": "Defect rate"})
        fig.update_traces(textposition="top center")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".1%")
        show(fig, 400)
    st.caption("Right: bubbles are sized by spend. Bottom-right is reliable and good quality. Change the weights to see ranks move.")

    st.subheader("Vendor profile")
    picked = st.multiselect("Compare vendors", card["vendor"].tolist(), default=card["vendor"].tolist()[:1] + card["vendor"].tolist()[-1:])
    if picked:
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            for v in picked:
                r = card[card["vendor"] == v].iloc[0]
                vals = [r[k] for k in labels]
                fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=list(labels.values()) + [list(labels.values())[0]], name=v, fill="toself", opacity=0.6))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
            show(fig, 380)
        with right:
            trend = sl.vendor_trend(orders)
            fig = px.line(trend[trend["vendor"].isin(picked)], x="period", y="on_time_rate", color="vendor", markers=True,
                          labels={"period": "", "on_time_rate": "On-time rate by quarter"})
            fig.update_yaxes(tickformat=".0%")
            show(fig, 380)
    st.dataframe(card, hide_index=True, column_config={
        "composite_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
        "on_time_rate": PCT, "defect_rate": PCT, "price_variance_pct": PCT, "spend": USD})
    download(card, "vendor_scorecard")

# ============================================================ DEMAND FORECAST
elif page == "Demand forecast":
    st.title("Demand forecast")
    c1, c2, c3 = st.columns(3)
    category = c1.selectbox("Category", ["All categories"] + sorted(orders["category"].unique()))
    horizon = c2.slider("Months to forecast", 1, 12, 6)
    try:
        series = sl.demand_series(orders, category=None if category == "All categories" else category)
        scores = sl.backtest(series)
    except ValueError as exc:
        st.warning(f"Cannot forecast this selection: {exc}")
        st.stop()
    choice = c3.selectbox("Model", ["auto (best in backtest)"] + scores["model"].tolist())
    fc = sl.forecast(series, horizon=horizon, model="auto" if choice.startswith("auto") else choice)

    fig = go.Figure()
    fig.add_scatter(x=series.index, y=series.values, name="Actual", line_color=BLUE)
    if "upper_80" in fc:
        fig.add_scatter(x=fc["period"], y=fc["upper_80"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fc["period"], y=fc["lower_80"], line=dict(width=0), fill="tonexty", fillcolor="rgba(209,73,91,0.18)", name="80% interval")
    fig.add_scatter(x=fc["period"], y=fc["forecast"], name=f"Forecast ({fc['model'].iloc[0]})", line=dict(color=RED, dash="dash"))
    fig.update_layout(yaxis_title="Units per month")
    show(fig, 420)

    left, right = st.columns(2)
    with left:
        st.subheader("Backtest accuracy")
        fig = px.bar(scores, x="wape", y="model", orientation="h", text_auto=".1%", color_discrete_sequence=[BLUE],
                     labels={"wape": "Error as a share of actual demand (lower is better)", "model": ""})
        fig.update_yaxes(categoryorder="total descending")
        fig.update_xaxes(tickformat=".0%")
        show(fig, 260)
        st.caption("Each model is scored only on months it did not see during fitting.")
    with right:
        st.subheader("Forecast")
        st.dataframe(fc.drop(columns="model").set_index("period").round(0).reset_index(), hide_index=True,
                     column_config={"period": st.column_config.DateColumn("Month", format="MMM YYYY")})

# ========================================================== DISTRIBUTION COST
elif page == "Distribution cost":
    st.title("Distribution cost")
    by = st.selectbox("Group freight cost by", ["carrier", "region", "vendor", "category"])
    summary = sl.cost_summary(orders, by=by)
    left, right = st.columns(2)
    with left:
        show(px.bar(summary, x=by, y="freight_per_unit", color_discrete_sequence=[BLUE], text_auto=".2f",
                    labels={"freight_per_unit": "Freight cost per unit ($)", by: ""}), 320)
    with right:
        lanes = df.pivot_table(index="region", columns="carrier", values="freight_per_unit", aggfunc="median")
        show(px.imshow(lanes, color_continuous_scale="Blues", text_auto=".2f", aspect="auto",
                       labels={"color": "Median $ / unit"}), 320)
    st.dataframe(summary, hide_index=True, column_config={"goods_value": USD, "freight_cost": USD, "freight_pct_of_goods": PCT, "share_of_freight": PCT})

    st.subheader("Shipments with abnormal freight cost")
    threshold = st.slider("Sensitivity (lower flags more shipments)", 2.0, 6.0, 3.5, 0.5)
    outliers = sl.cost_outliers(orders, threshold=threshold)
    plot = df.assign(flag=np.where(df["order_id"].isin(outliers["order_id"]), "abnormal", "normal"))
    fig = px.scatter(plot, x="quantity", y="freight_cost", color="flag", log_x=True, log_y=True, hover_name="order_id",
                     hover_data=["vendor", "region", "carrier"], opacity=0.6,
                     color_discrete_map={"normal": GREY, "abnormal": RED}, labels={"quantity": "Units", "freight_cost": "Freight cost ($)"})
    show(fig, 400)
    st.caption(f"{len(outliers)} shipment(s) cost far more per unit than is normal for their region and carrier. "
               f"Estimated excess over the lane median: ${outliers['excess_cost'].sum():,.0f}.")
    st.dataframe(outliers, hide_index=True, column_config={"freight_cost": USD, "excess_cost": USD})
    download(outliers, "freight_outliers")

    st.subheader("Spend concentration (Pareto)")
    par = sl.pareto(orders, by="vendor")
    fig = go.Figure()
    fig.add_bar(x=par["vendor"], y=par["spend"], name="Spend", marker_color=BLUE)
    fig.add_scatter(x=par["vendor"], y=par["cumulative_share"], name="Cumulative share", yaxis="y2", line_color=RED)
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 1.05]), yaxis_title="Spend ($)")
    show(fig, 340)
