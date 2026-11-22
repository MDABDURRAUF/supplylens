# SupplyLens

**Predict, explain and prevent supply chain disruptions.**

**[Try the live dashboard](https://supplylens.streamlit.app)** — runs in your browser on simulated sample data, nothing to install.

SupplyLens is an open-source Python toolkit and dashboard for supply chain analysts. It takes a
table of purchase orders and answers three questions in sequence:

1. **Predict** – which orders are likely to arrive late, and which items are at risk of stocking out?
2. **Explain** – what is driving the risk for each individual order?
3. **Prescribe** – which alternative vendor or carrier lowers both the risk and the expected cost?

Most analytics tools stop at the first question. SupplyLens is built around the step from
prediction to an explained, cost-justified corrective action.

It also covers the market and performance analysis that surrounds those decisions: segment growth
and seasonality, price benchmarking, vendor scorecards, demand forecasting and freight cost control.

## What it does

| Area | Function | What you get |
|---|---|---|
| Disruption prediction | `train_late_delivery_model` | Probability that each order arrives late, tested on the most recent orders rather than a random sample |
| Explanation | `model.explain`, `model.contributions` | The factors raising each order's risk (vendor, carrier, region, order size, season), as exact contributions to the prediction |
| Prescription | `recommend_alternatives` | For each high-risk order, the vendor/carrier change that is less likely to be late and has the lowest expected cost |
| Order planning | `evaluate_options` | Every vendor and carrier that could fill a planned order, ranked by expected cost with its late probability |
| Supplier failure | `simulate_vendor_disruption`, `vendor_exposure` | What losing a vendor would cost, SKU by SKU: replacement vendor, price and lead-time change, and supply with no alternative |
| Inventory stress test | `simulate_inventory`, `simulate_sku` | Monte Carlo simulation of a reorder-point policy under a demand surge or supplier delay: fill rate, stockouts per year, recovery time |
| Model trust | `compare_models`, `calibration_table`, `threshold_table` | Explainable model against gradient boosting, predicted against observed late rates, and what each alert threshold catches |
| Stockout risk | `inventory_policy` | Safety stock, reorder point and stockout probability per SKU under uncertain demand and lead time |
| KPIs | `disruption_kpis` | On-time delivery rate, days late, logistics cost per shipment and per unit, defect rate |
| Supplier risk | `concentration` | Single-source and high-concentration categories (HHI) |
| Market analysis | `market_growth`, `seasonality_index`, `price_index`, `price_benchmark` | Growth and share by segment, seasonal demand pattern, price level over time, price gaps between vendors for the same SKU |
| Vendor performance | `vendor_scorecard`, `vendor_trend` | Weighted 0–100 score and A/B/C tier from delivery, quality, price compliance and lead-time consistency |
| Demand forecasting | `backtest`, `forecast` | Four models compared by rolling-origin backtest; the best one produces the forecast |
| Distribution cost | `cost_summary`, `cost_outliers`, `pareto` | Freight cost by lane, abnormal shipments and their excess cost, ABC classification |

## Quick start

```bash
git clone https://github.com/MDABDURRAUF/supplylens.git
cd supplylens
pip install -e ".[app,dev]"
```

Launch the dashboard (it opens with built-in sample data, or upload your own CSV):

```bash
streamlit run app.py
```

## Run with Docker
No local Python setup needed:
```bash
docker build -t supplylens .
docker run --rm -p 8501:8501 supplylens
```
Then open http://localhost:8501 in your browser. The dashboard opens with built-in sample data.

The dashboard has ten views, all driven by the date, category and region filters in the sidebar:

| View | What you can do |
|---|---|
| Overview | Headline KPIs, monthly trends, and a late-rate heatmap across any two dimensions |
| Predict and explain | Set the risk threshold, click any high-risk order to see why it is at risk and how its alternatives compare |
| Plan an order (what-if) | Describe an order before placing it and see every vendor/carrier option on a cost-against-risk chart |
| Supplier failure scenario | Remove a vendor for a chosen number of weeks and see what can and cannot be replaced |
| Inventory stress test | Apply a demand surge or supplier delay and watch simulated stock levels, fill rate and recovery time respond |
| Model quality | Model comparison, calibration, and a threshold slider showing precision and recall |
| Market analysis | Growth against share, seasonality heatmap, price index, vendor price benchmark |
| Vendor scorecard | Adjustable score weights, vendor comparison radar and trend |
| Demand forecast | Choose the model or let the backtest choose; forecast with interval |
| Distribution cost | Cost by lane, abnormal shipments with adjustable sensitivity, Pareto chart |

Or use the command line:

```bash
supplylens sample --out orders.csv                 # write a synthetic sample file
supplylens disruptions --input orders.csv          # predict, explain, recommend
supplylens evaluate --input orders.csv             # model comparison, calibration, thresholds
supplylens whatif --input orders.csv --sku APP-002 --region West --quantity 500
supplylens scenario --input orders.csv --vendor "Vendor A" --weeks 8
supplylens inventory --input orders.csv            # safety stock and reorder points
supplylens market --input orders.csv --benchmark   # vendor price gaps for the same SKU
supplylens scorecard --input orders.csv            # vendor scorecard
supplylens forecast --input orders.csv --horizon 6 # demand forecast
supplylens cost --input orders.csv --outliers      # abnormal freight charges
supplylens report --input orders.csv --out report.html   # everything in one HTML file
```

Or from Python:

```python
import supplylens as sl

orders = sl.load_orders("orders.csv")

model = sl.train_late_delivery_model(orders)
print(model.metrics)                        # held-out ROC AUC and Brier score

open_orders = orders.tail(200)
print(model.explain(open_orders))           # why each order is at risk
print(sl.recommend_alternatives(model, open_orders, history=orders))

print(sl.inventory_policy(orders, service_level=0.95, on_hand={"APP-002": 1200}))
```

## Input data

One CSV file with one row per purchase-order line:

| Column | Meaning |
|---|---|
| `order_id` | Order identifier |
| `order_date`, `promised_date`, `delivery_date` | Dates the order was placed, promised and received |
| `vendor`, `category`, `sku` | Supplier, product category and item |
| `quantity`, `units_rejected` | Units ordered, and units rejected on receipt |
| `unit_price`, `contract_price` | Price paid and price agreed, per unit |
| `region`, `carrier`, `freight_cost` | Destination region, carrier and freight charge for the line |

Files are validated on load. Missing columns, unreadable dates and impossible values are reported
with row counts instead of being silently dropped.

**Adding external data.** Any numeric signal known when an order is placed – a weather severity
index, port congestion, a fuel price – can be joined to the orders table as an extra column and
passed to the model, where it is used for prediction and appears in the explanations:

```python
model = sl.train_late_delivery_model(orders, extra_numeric=("storm_index",))
```

## Methods

- **Late-delivery model.** Logistic regression on vendor, carrier, region, category, order size,
  promised lead time and time of year. The most recent 25% of orders are held out and the model is
  scored on them (ROC AUC and Brier score, with the Brier score of a constant prediction as the
  baseline), so reported accuracy reflects orders placed after the training period.
- **Explanations.** The model is linear in log-odds, so each factor's contribution to an order's
  risk is computed exactly and the contributions sum to the prediction. No approximation method is
  involved.
- **Recommendations.** Candidate alternatives are vendors that have previously supplied the same
  SKU and any carrier on the same lane. Each is priced from historical averages and re-scored by
  the model. Expected cost = goods + freight + (late probability × cost of a late delivery). The
  cost of a late delivery is a business assumption set by the user (default 10% of goods value).
  An alternative is only recommended if it is also less likely to be late than the current choice.
- **Inventory.** Safety stock = z × √(L·σ_d² + d²·σ_L²), the standard formula for uncertain demand
  and uncertain lead time, with demand during the lead time treated as normal.
- **Inventory simulation.** A reorder-point policy is simulated day by day over many runs. Daily
  demand is gamma-distributed (never negative, and correct in mean and variance even for lumpy
  demand), lead times are normal, unmet demand is lost. A disruption is a demand multiplier and/or
  extra lead time applied over a chosen window.
- **Supplier failure.** Each SKU the lost vendor supplies is moved to the cheapest other vendor
  with a history on that SKU. Alternatives are assumed to have capacity, so results are a best case.
- **Model checks.** The explainable model is compared with gradient boosting on the same held-out
  orders; calibration and alert-threshold tables are computed from held-out predictions only.
- **Forecasting.** Naive, seasonal naive, moving average and additive Holt-Winters, implemented in
  NumPy. Models are compared on periods they were not fitted to.
- **Freight outliers.** Modified z-score (median and MAD) of log cost per unit within each
  region–carrier lane.

## Sample data

`supplylens sample` and the dashboard's default data set are **synthetic**. They are generated by
a seeded simulation (`generate_sample_data`) and do not describe any real company. Results shown
on the sample data demonstrate that the methods work; they are not findings about a real supply
chain.

## Limitations and roadmap

- Alternatives are limited to vendors and carriers already present in the order history; capacity
  limits and minimum order quantities are not modelled.
- Recommendations depend on the assumed cost of a late delivery. Vary it to see how sensitive they are.
- The stockout estimate needs current stock levels, which the user supplies; there is no inventory
  ledger in the input schema yet.
- Planned: connectors for public weather and freight data, supplier financial-health signals,
  non-linear models with matching explanation methods, and tracking of whether recommended changes
  were adopted and what they saved.

## Tests

```bash
pytest
```

The suite checks each calculation against hand-worked examples and runs the command line end to end.

## Background

SupplyLens implements, as working software, methods the author has studied in published research:

- Rauf, M. A., Jim, M. M. I., Rahman, M. M., & Tariquzzaman, M. (2024). AI-powered predictive
  analytics for intellectual property risk management in supply chain operations: A big data
  approach. *International Journal of Science and Engineering, 1*(04), 32–46.
- Rauf, M. A., & Shish, Z. H. (2022). Assessment of data-driven vendor performance evaluation in
  retail supply chains: Analyzing metrics, scorecards, and contract management tools. *Journal of
  Sustainable Development and Policy, 1*(04), 71–116.
- Rauf, M. A., & Dhanekula, A. (2026). Meta-analysis of distribution cost control in retail supply
  chains: Insights from data modeling. *Journal of Sustainable Development and Policy, 5*(01), 73–109.

## Author

Md Abdur Rauf — ORCID [0000-0002-5105-9892](https://orcid.org/0000-0002-5105-9892)

## License

MIT. See [LICENSE](LICENSE).
