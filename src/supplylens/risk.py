"""Supply risk: supplier concentration and late-delivery prediction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import add_derived_columns

# Herfindahl-Hirschman Index bands (0-10,000 scale) used to label concentration.
HHI_MODERATE = 1500
HHI_HIGH = 2500

CATEGORICAL_FEATURES = ["vendor", "carrier", "region", "category"]
NUMERIC_FEATURES = ["log_quantity", "promised_lead_days", "month_sin", "month_cos"]


def concentration(orders: pd.DataFrame, by: str = "category") -> pd.DataFrame:
    """Supplier concentration of spend within each group.

    Returns the number of vendors, the top vendor and its share of spend, the
    HHI (sum of squared percentage shares; 10,000 means a single supplier)
    and a risk label: ``single source``, ``high``, ``moderate`` or ``low``.
    """
    df = add_derived_columns(orders)
    spend = df.groupby([by, "vendor"])["spend"].sum().reset_index()
    spend["share"] = spend["spend"] / spend.groupby(by)["spend"].transform("sum")

    rows = []
    for key, grp in spend.groupby(by):
        top = grp.loc[grp["share"].idxmax()]
        hhi = float((grp["share"] * 100).pow(2).sum())
        if len(grp) == 1:
            label = "single source"
        elif hhi >= HHI_HIGH:
            label = "high"
        elif hhi >= HHI_MODERATE:
            label = "moderate"
        else:
            label = "low"
        rows.append(
            {
                by: key,
                "vendors": len(grp),
                "total_spend": float(grp["spend"].sum()),
                "top_vendor": top["vendor"],
                "top_vendor_share": float(top["share"]),
                "hhi": hhi,
                "concentration_risk": label,
            }
        )
    return pd.DataFrame(rows).sort_values("hhi", ascending=False).reset_index(drop=True)


FACTOR_LABELS = {
    "log_quantity": "order size",
    "promised_lead_days": "promised lead time",
    "month_sin": "time of year",
    "month_cos": "time of year",
}


def _features(orders: pd.DataFrame, extra_numeric: tuple[str, ...] = ()) -> pd.DataFrame:
    """Model inputs. Only information known when the order is placed is used."""
    df = orders
    month = df["order_date"].dt.month
    missing = [c for c in extra_numeric if c not in df.columns]
    if missing:
        raise ValueError(f"Extra feature column(s) not in the data: {', '.join(missing)}")
    out = pd.DataFrame(
        {
            "vendor": df["vendor"],
            "carrier": df["carrier"],
            "region": df["region"],
            "category": df["category"],
            "log_quantity": np.log(df["quantity"]),
            "promised_lead_days": (df["promised_date"] - df["order_date"]).dt.days,
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=df.index,
    )
    for col in extra_numeric:
        out[col] = pd.to_numeric(df[col])
    return out


@dataclass
class LateDeliveryModel:
    """Logistic regression that estimates the probability an order arrives late.

    ``fit`` holds out the most recent ``test_fraction`` of orders (by order
    date) and reports metrics on them, so the scores describe performance on
    orders placed after the training period rather than on shuffled data.

    ``extra_numeric`` names additional numeric columns in the orders table to
    use as predictors, for example a weather severity index or a port
    congestion measure joined on by the user. They must be known when the
    order is placed.
    """

    test_fraction: float = 0.25
    extra_numeric: tuple[str, ...] = ()
    pipeline: Pipeline | None = None
    metrics: dict = field(default_factory=dict)
    holdout: pd.DataFrame | None = None
    _baseline: pd.Series | None = None

    def _build(self) -> Pipeline:
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                ("num", StandardScaler(), NUMERIC_FEATURES + list(self.extra_numeric)),
            ]
        )
        return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])

    def fit(self, orders: pd.DataFrame) -> "LateDeliveryModel":
        df = add_derived_columns(orders).sort_values("order_date")
        X, y = _features(df, self.extra_numeric), df["is_late"].astype(int)
        n_test = int(len(df) * self.test_fraction)
        if n_test < 30 or y.iloc[:-n_test].nunique() < 2:
            raise ValueError("Not enough data (or no late/on-time variation) to train the model")

        X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
        y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
        holdout = self._build().fit(X_train, y_train)
        p = holdout.predict_proba(X_test)[:, 1]
        self.holdout = pd.DataFrame({"actual_late": y_test, "late_probability": p}, index=y_test.index)
        base = float(y_train.mean())
        self.metrics = {
            "train_rows": len(y_train),
            "test_rows": len(y_test),
            "test_late_rate": float(y_test.mean()),
            "roc_auc": float(roc_auc_score(y_test, p)) if y_test.nunique() == 2 else float("nan"),
            "brier_score": float(brier_score_loss(y_test, p)),
            # Brier score of always predicting the training late rate.
            "brier_baseline": float(brier_score_loss(y_test, np.full(len(y_test), base))),
        }
        # Refit on all data for scoring new orders.
        self.pipeline = self._build().fit(X, y)
        self._baseline = self._raw_contributions(X).mean()
        return self

    def _raw_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Log-odds contributed by each factor (coefficient x encoded value)."""
        pre, clf = self.pipeline.named_steps["pre"], self.pipeline.named_steps["clf"]
        Z = pre.transform(X)
        Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
        factors = []
        for name in pre.get_feature_names_out():
            kind, rest = name.split("__", 1)
            if kind == "cat":
                factors.append(next(c for c in CATEGORICAL_FEATURES if rest.startswith(c + "_")))
            else:
                factors.append(FACTOR_LABELS.get(rest, rest))
        contrib = pd.DataFrame(Z * clf.coef_[0], index=X.index, columns=factors)
        return contrib.T.groupby(level=0).sum().T

    def contributions(self, orders: pd.DataFrame) -> pd.DataFrame:
        """How much each factor moves each order's risk, in log-odds.

        Values are relative to the average order in the training data: a
        positive number means the factor makes this order more likely to be
        late than average, a negative number less likely. Because the model is
        linear in log-odds these contributions are exact, not approximations:
        for every order they sum to its log-odds minus the average log-odds.
        """
        if self.pipeline is None:
            raise RuntimeError("Call fit() before contributions()")
        return self._raw_contributions(_features(orders, self.extra_numeric)) - self._baseline

    def explain(self, orders: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        """The ``top_n`` factors raising each order's risk the most.

        Returns one row per (order, factor) with the factor's value for that
        order and ``odds_multiplier``: how many times higher the odds of late
        delivery are because of that factor, compared with an average order.
        """
        df = add_derived_columns(orders)
        contrib = self.contributions(orders)
        proba = self.predict_proba(orders)
        values = {
            "vendor": df["vendor"],
            "carrier": df["carrier"],
            "region": df["region"],
            "category": df["category"],
            "order size": df["quantity"].map("{:,.0f} units".format),
            "promised lead time": df["promised_lead_days"].map("{:.0f} days".format),
            "time of year": df["order_date"].dt.strftime("%B"),
        }
        long = contrib.stack().rename("effect").reset_index()
        long.columns = ["row", "factor", "effect"]
        long = long[long["effect"] > 0].sort_values(["row", "effect"], ascending=[True, False])
        long = long.groupby("row").head(top_n)
        long["value"] = [
            values[f].loc[r] if f in values else f"{df.loc[r, f]:,.2f}" for r, f in zip(long["row"], long["factor"])
        ]
        long["odds_multiplier"] = np.exp(long["effect"])
        long["order_id"] = df["order_id"].loc[long["row"]].to_numpy()
        long["late_probability"] = proba.loc[long["row"]].to_numpy()
        return long[["order_id", "late_probability", "factor", "value", "odds_multiplier"]].reset_index(drop=True)

    def predict_proba(self, orders: pd.DataFrame) -> pd.Series:
        """Probability of late delivery for each order line."""
        if self.pipeline is None:
            raise RuntimeError("Call fit() before predict_proba()")
        X = _features(orders, self.extra_numeric)
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=X.index, name="late_probability")

    def drivers(self) -> pd.DataFrame:
        """Model coefficients as odds ratios, largest effect first.

        For categorical features an odds ratio above 1 means that level is
        associated with more late deliveries than average; for numeric
        features it is the change in odds per one standard deviation.
        """
        if self.pipeline is None:
            raise RuntimeError("Call fit() before drivers()")
        names = self.pipeline.named_steps["pre"].get_feature_names_out()
        coef = self.pipeline.named_steps["clf"].coef_[0]
        out = pd.DataFrame(
            {
                "feature": [n.split("__", 1)[1] for n in names],
                "coefficient": coef,
                "odds_ratio": np.exp(coef),
            }
        )
        return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def train_late_delivery_model(
    orders: pd.DataFrame, test_fraction: float = 0.25, extra_numeric: tuple[str, ...] = ()
) -> LateDeliveryModel:
    """Fit and return a ``LateDeliveryModel``; metrics are in ``model.metrics``."""
    return LateDeliveryModel(test_fraction=test_fraction, extra_numeric=tuple(extra_numeric)).fit(orders)
