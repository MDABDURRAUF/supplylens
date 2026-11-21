import numpy as np
import pandas as pd
import pytest

import supplylens as sl
from supplylens.cli import main
from supplylens.data import REQUIRED_COLUMNS
from supplylens.forecast import _predict
from supplylens.report import build_report


@pytest.fixture(scope="module")
def orders():
    return sl.generate_sample_data(n_orders=3000, seed=7)


def tiny_orders(**overrides):
    """Four hand-checkable order lines from two vendors."""
    base = pd.DataFrame(
        {
            "order_id": ["1", "2", "3", "4"],
            "order_date": ["2024-01-01"] * 4,
            "vendor": ["Good", "Good", "Bad", "Bad"],
            "category": ["Cat"] * 4,
            "sku": ["S1"] * 4,
            "quantity": [100, 100, 100, 100],
            "unit_price": [10.0, 10.0, 11.0, 11.0],
            "contract_price": [10.0] * 4,
            "promised_date": ["2024-01-11"] * 4,
            "delivery_date": ["2024-01-10", "2024-01-11", "2024-01-15", "2024-01-11"],
            "units_rejected": [0, 0, 5, 5],
            "region": ["East"] * 4,
            "carrier": ["C1"] * 4,
            "freight_cost": [50.0] * 4,
        }
    )
    for key, value in overrides.items():
        base[key] = value
    return sl.validate_orders(base)


# ----------------------------------------------------------------------- data
def test_sample_data_is_reproducible_and_valid():
    a = sl.generate_sample_data(n_orders=200, seed=1)
    b = sl.generate_sample_data(n_orders=200, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == REQUIRED_COLUMNS
    assert len(a) == 200


def test_validation_reports_missing_columns():
    with pytest.raises(sl.SchemaError, match="freight_cost"):
        sl.validate_orders(tiny_orders().drop(columns="freight_cost"))


def test_validation_rejects_bad_rows():
    with pytest.raises(sl.SchemaError, match="quantity <= 0"):
        tiny_orders(quantity=[100, 0, 100, 100])
    with pytest.raises(sl.SchemaError, match="unreadable 'order_date'"):
        tiny_orders(order_date=["2024-01-01", "not a date", "2024-01-01", "2024-01-01"])


# --------------------------------------------------------------------- vendor
def test_scorecard_metrics_match_hand_calculation():
    card = sl.vendor_scorecard(tiny_orders()).set_index("vendor")
    assert card.loc["Good", "on_time_rate"] == 1.0
    assert card.loc["Bad", "on_time_rate"] == 0.5
    assert card.loc["Bad", "avg_days_late"] == 4.0
    assert card.loc["Bad", "defect_rate"] == pytest.approx(0.05)
    assert card.loc["Bad", "price_variance_pct"] == pytest.approx(0.10)
    # 5% defects against a 10% cap -> 50; a 10% overrun hits the cap -> 0.
    assert card.loc["Bad", "quality_score"] == pytest.approx(50.0)
    assert card.loc["Bad", "price_score"] == pytest.approx(0.0)
    assert card.loc["Good", "rank"] == 1 and card.loc["Good", "tier"] == "A"
    assert card.loc["Bad", "tier"] == "C"


def test_scorecard_weights_are_rescaled_and_checked(orders):
    doubled = {k: 2 * v for k, v in sl.vendor.DEFAULT_WEIGHTS.items()}
    pd.testing.assert_frame_equal(sl.vendor_scorecard(orders), sl.vendor_scorecard(orders, doubled))
    with pytest.raises(ValueError, match="Unknown weight"):
        sl.vendor_scorecard(orders, {"speed": 1.0})


# ------------------------------------------------------------------- forecast
def test_seasonal_models_recover_a_seasonal_pattern():
    pattern = np.array([10, 12, 15, 20, 30, 45, 60, 45, 30, 20, 15, 12], dtype=float)
    y = np.concatenate([pattern + 5 * year for year in range(4)])
    actual_next = pattern + 5 * 4
    hw = _predict(y, 12, "holt_winters", 12)
    naive = _predict(y, 12, "naive", 12)
    assert np.abs(hw - actual_next).mean() < 5.0
    assert np.abs(hw - actual_next).mean() < np.abs(naive - actual_next).mean()
    np.testing.assert_array_equal(_predict(y, 12, "seasonal_naive", 12), y[-12:])


def test_backtest_and_forecast(orders):
    series = sl.demand_series(orders)
    assert series.index.is_monotonic_increasing and (series >= 0).all()
    scores = sl.backtest(series)
    assert scores["mae"].is_monotonic_increasing
    fc = sl.forecast(series, horizon=6)
    assert len(fc) == 6
    assert fc["period"].iloc[0] > series.index[-1]
    assert fc["model"].iloc[0] == scores["model"].iloc[0]
    assert (fc["lower_80"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["upper_80"]).all()


def test_demand_series_drops_partial_last_month():
    df = tiny_orders(order_date=["2024-01-05", "2024-01-20", "2024-02-10", "2024-03-02"],
                     promised_date=["2024-04-01"] * 4, delivery_date=["2024-04-01"] * 4)
    series = sl.demand_series(df)
    assert list(series.index.strftime("%Y-%m")) == ["2024-01", "2024-02"]
    assert list(series) == [200.0, 100.0]


def test_short_series_falls_back_gracefully():
    series = pd.Series([5.0, 6.0, 7.0], index=pd.date_range("2024-01-01", periods=3, freq="MS"))
    fc = sl.forecast(series, horizon=2)
    assert len(fc) == 2 and fc["forecast"].notna().all()


# ----------------------------------------------------------------------- cost
def test_cost_summary_totals(orders):
    summary = sl.cost_summary(orders, by="carrier")
    assert summary["freight_cost"].sum() == pytest.approx(orders["freight_cost"].sum())
    assert summary["share_of_freight"].sum() == pytest.approx(1.0)


def test_cost_outliers_finds_an_injected_spike(orders):
    spiked = orders.copy()
    spiked.loc[10, "freight_cost"] *= 40
    found = sl.cost_outliers(spiked)
    assert spiked.loc[10, "order_id"] in set(found["order_id"])
    assert (found["robust_z"] > 3.5).all() and (found["excess_cost"] > 0).all()


def test_pareto_classes(orders):
    p = sl.pareto(orders, by="vendor")
    assert p["cumulative_share"].iloc[-1] == pytest.approx(1.0)
    assert p["abc_class"].iloc[0] == "A"
    assert p["spend"].is_monotonic_decreasing


# --------------------------------------------------------------------- market
def test_market_growth_hand_calculation():
    df = tiny_orders()
    rows = []
    for year, qty in [(2022, 100), (2023, 150), (2024, 400)]:
        for month in range(1, 13):
            row = df.iloc[0].copy()
            row["order_date"] = pd.Timestamp(year, month, 1)
            row["promised_date"] = row["delivery_date"] = pd.Timestamp(year, month, 5)
            row["quantity"] = qty
            rows.append(row)
    growth = sl.market_growth(sl.validate_orders(pd.DataFrame(rows)))
    assert growth.loc[0, "total_growth"] == pytest.approx(3.0)  # 100 -> 400 units at the same price
    assert growth.loc[0, "cagr"] == pytest.approx(1.0)  # doubling each year for two years


def test_market_growth_needs_two_full_years():
    with pytest.raises(ValueError, match="two complete calendar years"):
        sl.market_growth(tiny_orders())


def test_seasonality_and_price_index(orders):
    season = sl.seasonality_index(orders)
    np.testing.assert_allclose(season.mean(axis=1), 1.0)
    # Electronics is generated with a November peak.
    assert season.loc["Electronics"].idxmax() in (10, 11, 12)
    index = sl.price_index(orders)
    np.testing.assert_allclose(index.iloc[0], 100.0)


def test_price_benchmark_hand_calculation():
    df = pd.concat([tiny_orders()] * 3, ignore_index=True)
    bench = sl.price_benchmark(df, min_lines=5).set_index("vendor")
    assert bench.loc["Bad", "premium_pct"] == pytest.approx(0.10)
    assert bench.loc["Bad", "potential_saving"] == pytest.approx(600.0)  # 600 units x $1
    assert bench.loc["Good", "potential_saving"] == 0.0


def test_segment_matrix_sums_to_one(orders):
    assert sl.segment_matrix(orders).to_numpy().sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------- risk
def test_concentration_labels():
    conc = sl.concentration(tiny_orders()).iloc[0]
    # Shares are 2000/4200 and 2200/4200.
    assert conc["hhi"] == pytest.approx(10000 * ((2000 / 4200) ** 2 + (2200 / 4200) ** 2))
    assert conc["concentration_risk"] == "high"
    single = sl.concentration(tiny_orders(vendor=["Only"] * 4)).iloc[0]
    assert single["hhi"] == pytest.approx(10000) and single["concentration_risk"] == "single source"


def test_sample_data_has_a_single_sourced_category(orders):
    conc = sl.concentration(orders).set_index("category")
    assert conc.loc["Electronics", "concentration_risk"] == "single source"


def test_late_delivery_model_beats_baseline(orders):
    model = sl.train_late_delivery_model(orders)
    m = model.metrics
    assert m["train_rows"] + m["test_rows"] == len(orders)
    assert m["roc_auc"] > 0.6
    assert m["brier_score"] < m["brier_baseline"]
    proba = model.predict_proba(orders)
    assert len(proba) == len(orders) and proba.between(0, 1).all()
    assert {"feature", "odds_ratio"} <= set(model.drivers().columns)


def test_model_requires_fit_and_enough_data():
    with pytest.raises(RuntimeError):
        sl.LateDeliveryModel().predict_proba(tiny_orders())
    with pytest.raises(ValueError, match="Not enough data"):
        sl.train_late_delivery_model(tiny_orders())


# -------------------------------------------------------------- report and CLI
def test_report_contains_every_section(orders):
    html = build_report(orders)
    for heading in ["Market growth", "Price benchmark", "Vendor scorecard", "Supplier concentration",
                    "Demand forecast", "Freight cost by lane", "Late-delivery prediction",
                    "Recommended changes"]:
        assert heading in html


def test_cli_end_to_end(tmp_path, capsys):
    csv = tmp_path / "orders.csv"
    assert main(["sample", "--out", str(csv), "--orders", "1500"]) == 0
    assert main(["scorecard", "--input", str(csv), "--out", str(tmp_path / "card.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "card.csv")) == 12
    for command in (["forecast"], ["cost"], ["cost", "--outliers"], ["risk"], ["market"], ["market", "--benchmark"],
                    ["disruptions"], ["inventory"], ["evaluate"], ["scenario"],
                    ["scenario", "--vendor", "Vendor A"],
                    ["whatif", "--sku", "APP-002", "--region", "West", "--quantity", "500"]):
        assert main([command[0], "--input", str(csv), *command[1:]]) == 0
    assert main(["report", "--input", str(csv), "--out", str(tmp_path / "r.html")]) == 0
    assert (tmp_path / "r.html").stat().st_size > 5000
    capsys.readouterr()


def test_cli_reports_bad_input(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    assert main(["scorecard", "--input", str(bad)]) == 1
    assert "Missing required column" in capsys.readouterr().err


# ------------------------------------------------- explain, prescribe, inventory
@pytest.fixture(scope="module")
def model(orders):
    return sl.train_late_delivery_model(orders)


def test_contributions_sum_exactly_to_the_prediction(orders, model):
    contrib = model.contributions(orders)
    p = model.predict_proba(orders)
    log_odds = np.log(p / (1 - p))
    # Relative contributions + the average log-odds reproduce every prediction.
    offset = (log_odds - contrib.sum(axis=1))
    np.testing.assert_allclose(offset, offset.iloc[0], atol=1e-9)
    assert abs(contrib.mean()).max() < 1e-9


def test_explain_lists_top_risk_factors(orders, model):
    exp = model.explain(orders.tail(50), top_n=2)
    assert exp.groupby("order_id").size().max() <= 2
    assert (exp["odds_multiplier"] > 1).all()
    assert set(exp["factor"]) <= {"vendor", "carrier", "region", "category", "order size",
                                  "promised lead time", "time of year"}


def test_extra_numeric_feature_is_used(orders):
    df = orders.copy()
    late = df["delivery_date"] > df["promised_date"]
    rng = np.random.default_rng(0)
    df["storm_index"] = late * 2.0 + rng.normal(0, 1, len(df))  # informative by construction
    with_extra = sl.train_late_delivery_model(df, extra_numeric=("storm_index",))
    plain = sl.train_late_delivery_model(df)
    assert with_extra.metrics["roc_auc"] > plain.metrics["roc_auc"]
    assert "storm_index" in with_extra.contributions(df).columns
    with pytest.raises(ValueError, match="not in the data"):
        sl.train_late_delivery_model(orders, extra_numeric=("storm_index",))


def test_recommendations_lower_expected_cost(orders, model):
    recent = orders.tail(300)
    recs = sl.recommend_alternatives(model, recent, history=orders, min_risk=0.25)
    assert len(recs) > 0
    assert (recs["expected_saving"] > 0).all()
    assert (recs["current_risk"] >= 0.25).all()
    assert (recs["recommended_risk"] < recs["current_risk"]).all()
    changed = (recs["recommended_vendor"] != recs["current_vendor"]) | (
        recs["recommended_carrier"] != recs["current_carrier"])
    assert changed.all()
    assert set(recs["order_id"]) <= set(recent["order_id"])
    # With no cost attached to lateness, only genuinely cheaper options remain.
    free = sl.recommend_alternatives(model, recent, history=orders, min_risk=0.25, late_cost_pct=0.0)
    assert (free["expected_saving"] > 0).all()
    # A higher cost of lateness makes switching to a safer option worth more.
    costly = sl.recommend_alternatives(model, recent, history=orders, min_risk=0.25, late_cost_pct=0.5)
    assert costly["expected_saving"].sum() >= recs["expected_saving"].sum()


def test_recommendations_empty_when_nothing_is_risky(orders, model):
    recs = sl.recommend_alternatives(model, orders.tail(50), history=orders, min_risk=0.999)
    assert recs.empty and "expected_saving" in recs.columns


def test_disruption_kpis(orders):
    k = sl.disruption_kpis(tiny_orders())
    assert k["on_time_delivery_rate"] == 0.75
    assert k["avg_days_late_when_late"] == 4.0
    assert k["logistics_cost_per_shipment"] == 50.0
    assert 0 < sl.disruption_kpis(orders)["on_time_delivery_rate"] < 1


def test_inventory_policy_formula(orders):
    pol = sl.inventory_policy(orders, service_level=0.95).set_index("sku")
    row = pol.iloc[0]
    sd = np.sqrt(row["avg_lead_time_days"] * row["sd_daily_demand"] ** 2
                 + row["avg_daily_demand"] ** 2 * row["sd_lead_time_days"] ** 2)
    assert row["safety_stock"] == np.ceil(1.6448536 * sd)
    assert row["reorder_point"] == np.ceil(row["lead_time_demand"] + row["safety_stock"])
    higher = sl.inventory_policy(orders, service_level=0.99).set_index("sku")
    assert (higher["safety_stock"] >= pol["safety_stock"]).all()
    with pytest.raises(ValueError):
        sl.inventory_policy(orders, service_level=1.0)


def test_stockout_probability(orders):
    base = sl.inventory_policy(orders).set_index("sku")
    sku = base.index[0]
    at_mean = sl.inventory_policy(orders, on_hand={sku: base.loc[sku, "lead_time_demand"]}).set_index("sku")
    assert at_mean.loc[sku, "stockout_probability"] == pytest.approx(0.5)
    plenty = sl.inventory_policy(orders, on_hand={sku: base.loc[sku, "reorder_point"] * 10}).set_index("sku")
    assert plenty.loc[sku, "stockout_probability"] < 0.001
    assert plenty.drop(index=sku)["stockout_probability"].isna().all()


# ------------------------------------------- evaluation, scenarios, simulation
def test_compare_models_and_holdout_tables(orders, model):
    comp = sl.compare_models(orders).set_index("model")
    assert len(comp) == 3
    constant = comp.loc["constant (average late rate)"]
    logistic = comp.loc["logistic regression (explainable)"]
    assert constant["roc_auc"] == 0.5
    assert logistic["roc_auc"] == pytest.approx(model.metrics["roc_auc"])
    assert logistic["brier_score"] < constant["brier_score"]

    cal = sl.calibration_table(model, bins=5)
    assert cal["orders"].sum() == model.metrics["test_rows"]
    assert cal["predicted"].is_monotonic_increasing
    assert (cal["gap"].abs() < 0.15).all()

    tt = sl.threshold_table(model)
    assert tt["orders_flagged"].is_monotonic_decreasing and tt["recall"].is_monotonic_decreasing
    everything = sl.threshold_table(model, [0.0]).iloc[0]
    assert everything["recall"] == 1.0 and everything["share_flagged"] == 1.0
    assert everything["precision"] == pytest.approx(model.metrics["test_late_rate"])


def test_evaluate_options_costs_add_up(orders, model):
    sku = orders["sku"].value_counts().index[0]
    opts = sl.evaluate_options(model, orders, sku, "West", 500, "2025-11-15", late_cost_pct=0.2)
    assert opts["expected_cost"].is_monotonic_increasing and opts["extra_vs_best"].iloc[0] == 0
    np.testing.assert_allclose(opts["expected_late_cost"], opts["late_probability"] * 0.2 * opts["goods_cost"])
    np.testing.assert_allclose(opts["expected_cost"], opts[["goods_cost", "freight_cost", "expected_late_cost"]].sum(axis=1))
    assert opts["lowest_risk"].sum() >= 1
    assert len(opts) == opts["vendor"].nunique() * orders["carrier"].nunique()
    # With no cost of lateness the ranking is purely goods + freight.
    free = sl.evaluate_options(model, orders, sku, "West", 500, "2025-11-15", late_cost_pct=0.0)
    assert (free["expected_late_cost"] == 0).all()
    with pytest.raises(ValueError, match="does not appear"):
        sl.evaluate_options(model, orders, "NOPE-1", "West", 500, "2025-11-15")


def test_vendor_disruption_hand_calculation():
    # Two vendors on one SKU over ~10 weeks; "Bad" sells 200 units at $11, "Good" at $10.
    df = pd.concat([tiny_orders()] * 3, ignore_index=True)
    df["order_date"] = pd.date_range("2024-01-01", periods=len(df), freq="6D")
    df["promised_date"] = df["order_date"] + pd.Timedelta(days=10)
    df["delivery_date"] = df["order_date"] + pd.Timedelta(days=10)
    df = sl.validate_orders(df)
    span_weeks = (df["order_date"].max() - df["order_date"].min()).days / 7
    sim = sl.simulate_vendor_disruption(df, "Bad", weeks=span_weeks).iloc[0]
    assert sim["status"] == "covered" and sim["alternative_vendor"] == "Good"
    assert sim["units_affected"] == pytest.approx(600)
    assert sim["spend_affected"] == pytest.approx(6600)
    assert sim["price_change_pct"] == pytest.approx(10 / 11 - 1)
    assert sim["extra_cost"] == pytest.approx(-600)
    only = sl.simulate_vendor_disruption(df[df["vendor"] == "Bad"], "Bad").iloc[0]
    assert only["status"] == "no alternative" and np.isnan(only["extra_cost"])


def test_vendor_exposure_flags_the_single_source_vendor(orders):
    exp = sl.vendor_exposure(orders)
    assert set(exp["vendor"]) == set(orders["vendor"])
    assert exp["share_of_spend"].sum() == pytest.approx(1.0)
    top = exp.iloc[0]
    assert top["vendor"] == "Vendor A" and top["skus_without_alternative"] > 0 and top["uncovered_spend"] > 0
    with pytest.raises(ValueError, match="does not appear"):
        sl.simulate_vendor_disruption(orders, "Nobody")


def test_inventory_simulation_behaves_sensibly():
    common = dict(avg_daily_demand=100, sd_daily_demand=30, avg_lead_time_days=10, sd_lead_time_days=2,
                  order_quantity=1400, n_runs=200)
    generous = sl.simulate_inventory(reorder_point=3000, **common)
    assert generous["fill_rate"] == 1.0 and generous["stockouts_per_year"] == 0 and generous["avg_recovery_days"] == 0
    tight = sl.simulate_inventory(reorder_point=900, **common)
    assert tight["fill_rate"] < generous["fill_rate"] and tight["stockouts_per_year"] > 0
    shocked = sl.simulate_inventory(reorder_point=1400, demand_multiplier=1.5, extra_lead_time_days=10, **common)
    normal = sl.simulate_inventory(reorder_point=1400, **common)
    assert shocked["fill_rate"] < normal["fill_rate"]
    assert shocked["stockout_day_share"] > normal["stockout_day_share"]
    # Same seed, same answer; the path covers every day.
    again = sl.simulate_inventory(reorder_point=1400, **common)
    assert again["fill_rate"] == normal["fill_rate"]
    assert len(normal["sample_path"]) == 365 and (normal["sample_path"]["p05"] <= normal["sample_path"]["p95"]).all()


def test_simulated_demand_keeps_its_mean_when_lumpy():
    # sd three times the mean: a clipped normal would inflate demand, gamma must not.
    res = sl.simulate_inventory(avg_daily_demand=10, sd_daily_demand=30, avg_lead_time_days=5, sd_lead_time_days=0,
                                reorder_point=10_000, order_quantity=10_000, n_runs=400)
    used = 20_000 - res["sample_path"]["median"].iloc[-1]  # no reorder is triggered within the year
    assert used == pytest.approx(10 * 365, rel=0.1)


def test_simulate_sku_uses_the_derived_policy(orders):
    sku = orders["sku"].value_counts().index[0]
    res = sl.simulate_sku(orders, sku, service_level=0.95, n_runs=100)
    policy = sl.inventory_policy(orders, service_level=0.95).set_index("sku").loc[sku]
    assert res["policy"]["reorder_point"] == policy["reorder_point"]
    assert 0.8 < res["fill_rate"] <= 1.0
    better = sl.simulate_sku(orders, sku, service_level=0.99, n_runs=100)
    assert better["fill_rate"] >= res["fill_rate"]
