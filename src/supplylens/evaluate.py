"""How far the late-delivery predictions can be trusted.

Three checks, all on orders the model did not see during training:

* ``compare_models``     is the explainable model giving up accuracy against
                         a more flexible one?
* ``calibration_table``  when the model says 30%, are about 30% late?
* ``threshold_table``    what does each alert threshold catch and miss?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .risk import CATEGORICAL_FEATURES, LateDeliveryModel, _features


def compare_models(
    orders: pd.DataFrame, test_fraction: float = 0.25, extra_numeric: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Score three predictors on the most recent ``test_fraction`` of orders.

    * a constant prediction (the training late rate), as the floor;
    * the logistic regression that SupplyLens uses and explains;
    * a gradient-boosted tree model, which can learn interactions and
      non-linear effects but cannot be explained exactly.

    If gradient boosting is not clearly better, the explainable model is
    costing little or nothing in accuracy. Higher is better for ``roc_auc``
    and ``average_precision``; lower is better for ``brier_score`` and
    ``log_loss``.
    """
    df = orders.sort_values("order_date")
    X = _features(df, tuple(extra_numeric))
    y = (df["delivery_date"] > df["promised_date"]).astype(int)
    n_test = int(len(df) * test_fraction)
    if n_test < 30 or y.iloc[:-n_test].nunique() < 2 or y.iloc[-n_test:].nunique() < 2:
        raise ValueError("Not enough data (or no late/on-time variation) to compare models")
    X_train, X_test, y_train, y_test = X.iloc[:-n_test], X.iloc[-n_test:], y.iloc[:-n_test], y.iloc[-n_test:]

    logistic = LateDeliveryModel(extra_numeric=tuple(extra_numeric))._build().fit(X_train, y_train)
    boosted = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
                    remainder="passthrough",
                ),
            ),
            ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)),
        ]
    ).fit(X_train, y_train)

    predictions = {
        "constant (average late rate)": np.full(len(y_test), y_train.mean()),
        "logistic regression (explainable)": logistic.predict_proba(X_test)[:, 1],
        "gradient boosting": boosted.predict_proba(X_test)[:, 1],
    }
    rows = [
        {
            "model": name,
            "roc_auc": roc_auc_score(y_test, p),
            "average_precision": average_precision_score(y_test, p),
            "brier_score": brier_score_loss(y_test, p),
            "log_loss": log_loss(y_test, p, labels=[0, 1]),
        }
        for name, p in predictions.items()
    ]
    return pd.DataFrame(rows)


def _holdout(model: LateDeliveryModel) -> pd.DataFrame:
    if model.holdout is None:
        raise RuntimeError("Call fit() on the model first")
    return model.holdout


def calibration_table(model: LateDeliveryModel, bins: int = 10) -> pd.DataFrame:
    """Predicted against observed late rate, by band of predicted probability.

    Held-out orders are split into ``bins`` equal-sized groups from lowest to
    highest predicted risk. In a well-calibrated model ``predicted`` and
    ``observed`` are close in every row.
    """
    h = _holdout(model)
    band = pd.qcut(h["late_probability"], q=bins, duplicates="drop")
    g = h.groupby(band, observed=True)
    out = pd.DataFrame(
        {
            "orders": g.size(),
            "predicted": g["late_probability"].mean(),
            "observed": g["actual_late"].mean(),
        }
    ).reset_index(drop=True)
    out["gap"] = out["observed"] - out["predicted"]
    return out


def threshold_table(model: LateDeliveryModel, thresholds=None) -> pd.DataFrame:
    """What each alert threshold catches on held-out orders.

    For each threshold: how many orders would be flagged, ``precision`` (the
    share of flagged orders that really were late) and ``recall`` (the share
    of all late orders that were flagged). Lowering the threshold catches more
    late orders at the price of more false alarms.
    """
    h = _holdout(model)
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2) if thresholds is None else thresholds
    total_late = int(h["actual_late"].sum())
    rows = []
    for t in thresholds:
        flagged = h[h["late_probability"] >= t]
        caught = int(flagged["actual_late"].sum())
        rows.append(
            {
                "threshold": float(t),
                "orders_flagged": len(flagged),
                "share_flagged": len(flagged) / len(h),
                "late_orders_caught": caught,
                "precision": caught / len(flagged) if len(flagged) else np.nan,
                "recall": caught / total_late if total_late else np.nan,
            }
        )
    return pd.DataFrame(rows)
