"""Command-line interface: ``supplylens <command>``."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__
from .cost import cost_outliers, cost_summary
from .data import SchemaError, generate_sample_data, load_orders
from .forecast import backtest, demand_series, forecast
from .market import market_growth, price_benchmark
from .evaluate import calibration_table, compare_models, threshold_table
from .inventory import inventory_policy
from .prescribe import disruption_kpis, recommend_alternatives
from .report import write_report
from .scenario import evaluate_options, simulate_vendor_disruption, vendor_exposure
from .risk import concentration, train_late_delivery_model
from .vendor import vendor_scorecard


def _emit(df: pd.DataFrame, out: str | None) -> None:
    if out:
        df.to_csv(out, index=False)
        print(f"Wrote {len(df)} row(s) to {out}")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:,.3f}".format):
            print(df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="supplylens", description="Predict, explain and prevent supply chain disruptions.")
    p.add_argument("--version", action="version", version=f"supplylens {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="write a synthetic sample orders CSV")
    s.add_argument("--out", default="sample_orders.csv")
    s.add_argument("--orders", type=int, default=6000)
    s.add_argument("--seed", type=int, default=42)

    def with_input(name: str, help_text: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--input", required=True, help="orders CSV file")
        sp.add_argument("--out", help="write the result to this file instead of printing it")
        return sp

    with_input("scorecard", "vendor performance scorecard")

    f = with_input("forecast", "demand forecast with backtested model selection")
    f.add_argument("--horizon", type=int, default=6)
    f.add_argument("--category")
    f.add_argument("--model", default="auto")

    c = with_input("cost", "freight cost summary or outliers")
    c.add_argument("--by", default="carrier", help="column to group by (default: carrier)")
    c.add_argument("--outliers", action="store_true", help="list abnormal shipments instead")

    with_input("risk", "supplier concentration and late-delivery model")

    d = with_input("disruptions", "predict late orders, explain why, and recommend alternatives")
    d.add_argument("--recent", type=int, default=200, help="assess this many most recent orders (default: 200)")
    d.add_argument("--min-risk", type=float, default=0.30, help="late probability that counts as high risk")
    d.add_argument("--late-cost-pct", type=float, default=0.10,
                   help="assumed cost of a late delivery as a share of goods value (default: 0.10)")

    e = with_input("evaluate", "model comparison, calibration and alert thresholds on held-out orders")

    w = with_input("whatif", "rank every vendor and carrier for a planned order")
    w.add_argument("--sku", required=True)
    w.add_argument("--region", required=True)
    w.add_argument("--quantity", type=float, required=True)
    w.add_argument("--date", help="order date, YYYY-MM-DD (default: day after the last order)")
    w.add_argument("--late-cost-pct", type=float, default=0.10)

    sc = with_input("scenario", "cost of losing a vendor (omit --vendor to rank all vendors)")
    sc.add_argument("--vendor")
    sc.add_argument("--weeks", type=float, default=8)

    i = with_input("inventory", "safety stock and reorder point per SKU")
    i.add_argument("--service-level", type=float, default=0.95)

    m = with_input("market", "segment growth, or price benchmark across vendors")
    m.add_argument("--by", default="category", help="segment column (default: category)")
    m.add_argument("--benchmark", action="store_true", help="compare vendor prices for the same SKU instead")

    r = sub.add_parser("report", help="write the full HTML report")
    r.add_argument("--input", required=True)
    r.add_argument("--out", default="supplylens_report.html")
    r.add_argument("--horizon", type=int, default=6)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "sample":
            df = generate_sample_data(n_orders=args.orders, seed=args.seed)
            df.to_csv(args.out, index=False)
            print(f"Wrote {len(df)} synthetic order lines to {args.out}")
            return 0

        orders = load_orders(args.input)
        if args.command == "scorecard":
            _emit(vendor_scorecard(orders), args.out)
        elif args.command == "forecast":
            series = demand_series(orders, category=args.category)
            print("Backtest (rolling origin, best model first):")
            _emit(backtest(series), None)
            print("\nForecast:")
            _emit(forecast(series, horizon=args.horizon, model=args.model), args.out)
        elif args.command == "cost":
            _emit(cost_outliers(orders) if args.outliers else cost_summary(orders, by=args.by), args.out)
        elif args.command == "risk":
            print("Supplier concentration:")
            _emit(concentration(orders), args.out)
            model = train_late_delivery_model(orders)
            print("\nLate-delivery model (tested on the most recent orders):")
            for key, value in model.metrics.items():
                print(f"  {key}: {value:,.4f}" if isinstance(value, float) else f"  {key}: {value:,}")
            print("\nStrongest drivers:")
            _emit(model.drivers().head(10), None)
        elif args.command == "disruptions":
            print("Reliability and cost indicators:")
            for key, value in disruption_kpis(orders).items():
                print(f"  {key}: {value:,.4f}")
            model = train_late_delivery_model(orders)
            recent = orders.tail(args.recent)
            risky = model.explain(recent)
            risky = risky[risky["late_probability"] >= args.min_risk]
            print(f"\nWhy the highest-risk orders among the last {len(recent)} are at risk:")
            _emit(risky.sort_values("late_probability", ascending=False).head(15), None)
            print("\nRecommended changes (largest expected saving first):")
            _emit(
                recommend_alternatives(model, recent, history=orders, min_risk=args.min_risk,
                                       late_cost_pct=args.late_cost_pct),
                args.out,
            )
        elif args.command == "evaluate":
            print("Model comparison on held-out orders:")
            _emit(compare_models(orders), args.out)
            model = train_late_delivery_model(orders)
            print("\nCalibration (predicted vs observed late rate):")
            _emit(calibration_table(model), None)
            print("\nAlert thresholds:")
            _emit(threshold_table(model), None)
        elif args.command == "whatif":
            model = train_late_delivery_model(orders)
            date = args.date or (orders["order_date"].max() + pd.Timedelta(days=1))
            _emit(evaluate_options(model, orders, args.sku, args.region, args.quantity, date,
                                   late_cost_pct=args.late_cost_pct), args.out)
        elif args.command == "scenario":
            if args.vendor:
                _emit(simulate_vendor_disruption(orders, args.vendor, weeks=args.weeks), args.out)
            else:
                _emit(vendor_exposure(orders, weeks=args.weeks), args.out)
        elif args.command == "inventory":
            _emit(inventory_policy(orders, service_level=args.service_level), args.out)
        elif args.command == "market":
            _emit(price_benchmark(orders) if args.benchmark else market_growth(orders, by=args.by), args.out)
        elif args.command == "report":
            print(f"Wrote report to {write_report(orders, args.out, horizon=args.horizon)}")
    except (SchemaError, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
