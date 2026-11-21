"""Loading, validating and generating purchase-order data.

Every analysis in SupplyLens works from one table of purchase-order lines.
The expected columns are listed in ``REQUIRED_COLUMNS``; ``validate_orders``
checks a DataFrame against that schema and normalises the types.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "vendor",
    "category",
    "sku",
    "quantity",
    "unit_price",
    "contract_price",
    "promised_date",
    "delivery_date",
    "units_rejected",
    "region",
    "carrier",
    "freight_cost",
]

DATE_COLUMNS = ["order_date", "promised_date", "delivery_date"]
NUMERIC_COLUMNS = ["quantity", "unit_price", "contract_price", "units_rejected", "freight_cost"]


class SchemaError(ValueError):
    """Raised when an orders table does not match the expected schema."""


def validate_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Check ``df`` against the orders schema and return a cleaned copy.

    Dates are parsed, numeric columns are coerced, and rows that cannot be
    used (missing dates, non-positive quantity) raise a ``SchemaError`` that
    says how many rows are affected, so bad data is never silently dropped.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {', '.join(missing)}")

    out = df.copy()
    for col in DATE_COLUMNS:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    problems = []
    for col in DATE_COLUMNS + NUMERIC_COLUMNS:
        n_bad = int(out[col].isna().sum())
        if n_bad:
            problems.append(f"{n_bad} row(s) with missing or unreadable '{col}'")
    if (out["quantity"] <= 0).any():
        problems.append(f"{int((out['quantity'] <= 0).sum())} row(s) with quantity <= 0")
    if (out["units_rejected"] < 0).any() or (out["units_rejected"] > out["quantity"]).any():
        problems.append("units_rejected must be between 0 and quantity")
    if (out["delivery_date"] < out["order_date"]).any():
        problems.append("delivery_date is earlier than order_date in some rows")
    if problems:
        raise SchemaError("Invalid orders data: " + "; ".join(problems))

    for col in ["vendor", "category", "sku", "region", "carrier"]:
        out[col] = out[col].astype(str).str.strip()
    return out.sort_values("order_date").reset_index(drop=True)


def load_orders(path: str | Path) -> pd.DataFrame:
    """Read an orders CSV file and validate it."""
    return validate_orders(pd.read_csv(path))


def add_derived_columns(orders: pd.DataFrame) -> pd.DataFrame:
    """Add the per-line measures the analyses share."""
    out = orders.copy()
    out["spend"] = out["quantity"] * out["unit_price"]
    out["lead_time_days"] = (out["delivery_date"] - out["order_date"]).dt.days
    out["promised_lead_days"] = (out["promised_date"] - out["order_date"]).dt.days
    out["days_late"] = (out["delivery_date"] - out["promised_date"]).dt.days.clip(lower=0)
    out["is_late"] = out["delivery_date"] > out["promised_date"]
    out["freight_per_unit"] = out["freight_cost"] / out["quantity"]
    return out


def generate_sample_data(
    n_orders: int = 6000,
    n_vendors: int = 12,
    start: str = "2023-01-01",
    months: int = 36,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic, reproducible orders table for demos and tests.

    The data is simulated, not real. Vendors are given different underlying
    reliability, quality and pricing behaviour, demand has trend and yearly
    seasonality, and late delivery depends on vendor, carrier, region, order
    size and season, so every analysis has a real signal to find.
    """
    rng = np.random.default_rng(seed)

    categories = {
        # name: (base unit price, seasonal amplitude, peak month, yearly growth)
        "Electronics": (85.0, 0.45, 11, 0.10),
        "Packaging": (2.4, 0.15, 10, 0.04),
        "Raw Materials": (14.0, 0.10, 3, 0.06),
        "Apparel": (22.0, 0.35, 8, 0.02),
        "Home Goods": (31.0, 0.25, 12, 0.08),
    }
    regions = {"Northeast": 1.15, "Southeast": 1.00, "Midwest": 0.95, "West": 1.25, "Southwest": 1.05}
    carriers = {"RoadLink": (1.00, 0.00), "BlueFreight": (0.90, 0.35), "Summit Logistics": (1.20, -0.40)}

    cat_names = list(categories)
    vendors = [f"Vendor {chr(65 + i)}" if i < 26 else f"Vendor {i + 1}" for i in range(n_vendors)]
    # Latent vendor behaviour.
    v_late_logit = rng.normal(-1.9, 0.7, n_vendors)
    v_defect = rng.beta(1.5, 60, n_vendors)
    v_price_drift = rng.normal(0.01, 0.025, n_vendors)
    v_lead = rng.uniform(7, 24, n_vendors)
    # Each vendor serves two or three categories; the first category is
    # served by only one vendor so the concentration analysis has a finding.
    v_categories = [list(rng.choice(cat_names[1:], size=rng.integers(2, 4), replace=False)) for _ in vendors]
    v_categories[0].append(cat_names[0])
    v_share = rng.dirichlet(np.full(n_vendors, 1.2))

    month_starts = pd.date_range(start, periods=months, freq="MS")
    rows = []
    for i in range(n_orders):
        v = int(rng.choice(n_vendors, p=v_share))
        cat = str(rng.choice(v_categories[v]))
        base_price, amp, peak, growth = categories[cat]

        # Pick the order month in proportion to trend x seasonality.
        m_idx = np.arange(months)
        season = 1 + amp * np.cos(2 * np.pi * (month_starts.month.to_numpy() - peak) / 12)
        weight = season * (1 + growth) ** (m_idx / 12)
        m = int(rng.choice(months, p=weight / weight.sum()))
        order_date = month_starts[m] + pd.Timedelta(days=int(rng.integers(0, month_starts[m].days_in_month)))

        region = str(rng.choice(list(regions)))
        carrier = str(rng.choice(list(carriers)))
        quantity = int(np.clip(rng.lognormal(4.6, 0.8), 5, 5000))
        sku = f"{cat[:3].upper()}-{int(rng.integers(1, 9)):03d}"

        contract_price = round(base_price * (1 + 0.03 * (int(sku[-3:]) - 4)), 2)
        unit_price = round(contract_price * (1 + v_price_drift[v] * (1 + m / months) + rng.normal(0, 0.01)), 2)

        promised_lead = int(max(3, round(v_lead[v] + rng.normal(0, 1.5))))
        logit = (
            v_late_logit[v]
            + carriers[carrier][1]
            + (0.35 if region == "West" else 0.0)
            + 0.30 * (np.log(quantity) - 4.6)
            + (0.55 if order_date.month in (10, 11, 12) else 0.0)
        )
        late = rng.random() < 1 / (1 + np.exp(-logit))
        delay = int(rng.geometric(0.22)) if late else -int(rng.integers(0, 3))
        actual_lead = max(1, promised_lead + delay)

        rejected = int(rng.binomial(quantity, min(0.5, v_defect[v] * rng.lognormal(0, 0.4))))
        freight = quantity * 0.38 * regions[region] * carriers[carrier][0] * rng.lognormal(0, 0.18) + 35
        if rng.random() < 0.012:  # occasional expedited shipment
            freight *= rng.uniform(2.5, 5.0)

        rows.append(
            {
                "order_id": f"PO-{100000 + i}",
                "order_date": order_date,
                "vendor": vendors[v],
                "category": cat,
                "sku": sku,
                "quantity": quantity,
                "unit_price": unit_price,
                "contract_price": contract_price,
                "promised_date": order_date + pd.Timedelta(days=promised_lead),
                "delivery_date": order_date + pd.Timedelta(days=actual_lead),
                "units_rejected": rejected,
                "region": region,
                "carrier": carrier,
                "freight_cost": round(float(freight), 2),
            }
        )
    return validate_orders(pd.DataFrame(rows))
