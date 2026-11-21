"""Demand forecasting with honest, out-of-sample model selection.

Four models are implemented directly in NumPy so their behaviour is easy to
inspect: naive, seasonal naive, moving average and additive Holt-Winters.
``backtest`` scores them with rolling-origin evaluation (each fold forecasts
periods the model has not seen) and ``forecast`` uses the best one.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

MODELS = ("naive", "seasonal_naive", "moving_average", "holt_winters")
_HW_GRID = (0.1, 0.3, 0.5, 0.7)


def demand_series(
    orders: pd.DataFrame,
    freq: str = "MS",
    category: str | None = None,
    sku: str | None = None,
) -> pd.Series:
    """Units ordered per period, with empty periods filled with zero.

    The last period is dropped if the data ends before that period does,
    because a partial month would look like a collapse in demand.
    """
    df = orders
    if category is not None:
        df = df[df["category"] == category]
    if sku is not None:
        df = df[df["sku"] == sku]
    if df.empty:
        raise ValueError("No orders match the requested category / sku")

    series = df.set_index("order_date")["quantity"].resample(freq).sum().astype(float)
    last_period_end = series.index[-1] + pd.tseries.frequencies.to_offset(freq)
    if orders["order_date"].max() < last_period_end - pd.Timedelta(days=1):
        series = series.iloc[:-1]
    series.name = "demand"
    return series


def _fit_holt_winters(y: np.ndarray, m: int, alpha: float, beta: float, gamma: float):
    """Run additive Holt-Winters over ``y``; return final state and in-sample SSE."""
    level = y[:m].mean()
    trend = (y[m : 2 * m].mean() - y[:m].mean()) / m
    season = list(y[:m] - level)
    sse = 0.0
    for t in range(len(y)):
        s = season[t % m]
        sse += (y[t] - (level + trend + s)) ** 2
        new_level = alpha * (y[t] - s) + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        season[t % m] = gamma * (y[t] - new_level) + (1 - gamma) * s
        level = new_level
    return level, trend, season, sse


def _predict(y: np.ndarray, horizon: int, model: str, m: int) -> np.ndarray:
    n = len(y)
    if model == "naive":
        return np.full(horizon, y[-1])
    if model == "seasonal_naive":
        if n < m:
            raise ValueError("seasonal_naive needs at least one full season of history")
        return np.array([y[n - m + (h % m)] for h in range(horizon)])
    if model == "moving_average":
        return np.full(horizon, y[-min(3, n) :].mean())
    if model == "holt_winters":
        if n < 2 * m:
            raise ValueError("holt_winters needs at least two full seasons of history")
        best = min(
            (_fit_holt_winters(y, m, a, b, g) for a, b, g in product(_HW_GRID, repeat=3)),
            key=lambda fit: fit[3],
        )
        level, trend, season, _ = best
        return np.array([level + (h + 1) * trend + season[(n + h) % m] for h in range(horizon)])
    raise ValueError(f"Unknown model '{model}'. Choose from: {', '.join(MODELS)}")


def _available_models(n: int, m: int) -> list[str]:
    models = ["naive", "moving_average"]
    if n >= m:
        models.append("seasonal_naive")
    if n >= 2 * m:
        models.append("holt_winters")
    return models


def backtest(series: pd.Series, horizon: int = 3, n_folds: int = 4, season_length: int = 12) -> pd.DataFrame:
    """Rolling-origin evaluation of every model that the history can support.

    Fold k trains on everything before the cut-off and forecasts the next
    ``horizon`` periods; cut-offs move back one horizon at a time. Returns one
    row per model with MAE, RMSE and WAPE (sum of absolute errors / sum of
    actuals), sorted best first by MAE.
    """
    y = series.to_numpy(dtype=float)
    n = len(y)
    rows = []
    for model in MODELS:
        errors, actuals = [], []
        for k in range(n_folds):
            cut = n - horizon * (k + 1)
            if cut < 2 or model not in _available_models(cut, season_length):
                continue
            pred = _predict(y[:cut], horizon, model, season_length)
            errors.extend(y[cut : cut + horizon] - pred)
            actuals.extend(y[cut : cut + horizon])
        if not errors:
            continue
        e, a = np.array(errors), np.array(actuals)
        rows.append(
            {
                "model": model,
                "mae": float(np.abs(e).mean()),
                "rmse": float(np.sqrt((e**2).mean())),
                "wape": float(np.abs(e).sum() / np.abs(a).sum()) if np.abs(a).sum() else np.nan,
                "n_forecasts": len(e),
            }
        )
    if not rows:
        raise ValueError("Series is too short to backtest; reduce horizon or n_folds")
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def forecast(
    series: pd.Series,
    horizon: int = 6,
    model: str = "auto",
    season_length: int = 12,
) -> pd.DataFrame:
    """Forecast the next ``horizon`` periods.

    With ``model="auto"`` the model with the lowest backtest MAE is used. The
    80% interval is the point forecast +/- 1.28 x the backtest RMSE of the
    chosen model, a simple approximation that does not widen with horizon.
    """
    y = series.to_numpy(dtype=float)
    scores = None
    try:
        scores = backtest(series, horizon=min(horizon, 3), season_length=season_length)
    except ValueError:
        pass
    if model == "auto":
        model = scores.iloc[0]["model"] if scores is not None else "naive"

    pred = np.clip(_predict(y, horizon, model, season_length), 0, None)
    freq = series.index.freq or pd.infer_freq(series.index)
    index = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]
    out = pd.DataFrame({"period": index, "forecast": pred, "model": model})

    if scores is not None and model in set(scores["model"]):
        rmse = float(scores.loc[scores["model"] == model, "rmse"].iloc[0])
        out["lower_80"] = np.clip(pred - 1.28 * rmse, 0, None)
        out["upper_80"] = pred + 1.28 * rmse
    return out
