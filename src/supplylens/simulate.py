"""Monte Carlo simulation of an inventory policy under uncertainty and shocks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .inventory import inventory_policy


def simulate_inventory(
    avg_daily_demand: float,
    sd_daily_demand: float,
    avg_lead_time_days: float,
    sd_lead_time_days: float,
    reorder_point: float,
    order_quantity: float,
    days: int = 365,
    n_runs: int = 300,
    demand_multiplier: float = 1.0,
    extra_lead_time_days: float = 0.0,
    shock_start_day: int = 0,
    shock_days: int | None = None,
    seed: int = 0,
) -> dict:
    """Simulate a reorder-point policy day by day and measure stockouts.

    Policy: whenever stock on hand plus stock on order falls to
    ``reorder_point`` or below, order ``order_quantity``. Daily demand is
    drawn from a gamma distribution with the given mean and standard
    deviation, and each order's lead time from a normal distribution floored
    at one day. Demand that cannot be met from
    stock is lost. Each run starts with ``reorder_point + order_quantity``
    units on hand.

    A disruption is modelled with ``demand_multiplier`` (e.g. 1.3 for a 30%
    demand surge) and ``extra_lead_time_days`` (delay added to every order
    placed), applied from ``shock_start_day`` for ``shock_days`` days, or for
    the rest of the horizon when ``shock_days`` is None.

    Returns a dict with:

    * ``fill_rate``              units supplied / units demanded
    * ``stockout_day_share``     share of days that ended with no stock
    * ``stockouts_per_year``     separate stockout episodes per 365 days
    * ``avg_recovery_days``      mean length of a stockout episode
    * ``runs_with_stockout``     share of runs with at least one stockout
    * ``avg_on_hand``            mean units on hand
    * ``sample_path``            DataFrame of on-hand stock by day for the
                                 first run plus the 5th-95th percentile band
    """
    if order_quantity <= 0 or days <= 0 or n_runs <= 0:
        raise ValueError("order_quantity, days and n_runs must be positive")
    rng = np.random.default_rng(seed)
    # Gamma demand: never negative, right-skewed, and keeps the stated mean and
    # standard deviation even when demand is lumpy (sd larger than the mean).
    if sd_daily_demand > 0 and avg_daily_demand > 0:
        gamma_shape = (avg_daily_demand / sd_daily_demand) ** 2
        gamma_scale = sd_daily_demand**2 / avg_daily_demand
    else:
        gamma_shape = gamma_scale = None
    shock_end = days if shock_days is None else min(days, shock_start_day + shock_days)

    on_hand = np.full(n_runs, float(reorder_point + order_quantity))
    horizon = days + int(avg_lead_time_days + 6 * sd_lead_time_days + extra_lead_time_days) + 5
    arrivals = np.zeros((n_runs, horizon))
    on_order = np.zeros(n_runs)

    demanded = np.zeros(n_runs)
    supplied = np.zeros(n_runs)
    stockout_days = np.zeros(n_runs)
    episodes = np.zeros(n_runs)
    was_out = np.zeros(n_runs, dtype=bool)
    path = np.zeros((days, n_runs))
    rows = np.arange(n_runs)

    for day in range(days):
        shocked = shock_start_day <= day < shock_end
        on_hand += arrivals[:, day]
        on_order -= arrivals[:, day]

        if gamma_shape is None:
            demand = np.full(n_runs, float(avg_daily_demand))
        else:
            demand = rng.gamma(gamma_shape, gamma_scale, n_runs)
        if shocked:
            demand *= demand_multiplier
        served = np.minimum(demand, on_hand)
        on_hand -= served
        demanded += demand
        supplied += served

        out = on_hand <= 1e-9
        stockout_days += out
        episodes += out & ~was_out
        was_out = out

        need = (on_hand + on_order) <= reorder_point
        lead = np.rint(rng.normal(avg_lead_time_days, sd_lead_time_days, n_runs)).clip(min=1)
        if shocked:
            lead += round(extra_lead_time_days)
        due = np.minimum(day + lead.astype(int), horizon - 1)
        arrivals[rows[need], due[need]] += order_quantity
        on_order[need] += order_quantity
        path[day] = on_hand

    total_episodes = episodes.sum()
    return {
        "fill_rate": float(supplied.sum() / demanded.sum()) if demanded.sum() else 1.0,
        "stockout_day_share": float(stockout_days.sum() / (days * n_runs)),
        "stockouts_per_year": float(episodes.mean() * 365 / days),
        "avg_recovery_days": float(stockout_days.sum() / total_episodes) if total_episodes else 0.0,
        "runs_with_stockout": float((episodes > 0).mean()),
        "avg_on_hand": float(path.mean()),
        "sample_path": pd.DataFrame(
            {
                "day": np.arange(1, days + 1),
                "one_run": path[:, 0],
                "p05": np.percentile(path, 5, axis=1),
                "median": np.percentile(path, 50, axis=1),
                "p95": np.percentile(path, 95, axis=1),
            }
        ),
    }


def simulate_sku(
    orders: pd.DataFrame,
    sku: str,
    service_level: float = 0.95,
    order_cover_days: float = 14,
    **shock,
) -> dict:
    """Simulate one SKU using the policy ``inventory_policy`` derives for it.

    The reorder point comes from ``inventory_policy`` at ``service_level``
    and each replenishment order covers ``order_cover_days`` of average
    demand. Extra keyword arguments (``demand_multiplier``,
    ``extra_lead_time_days``, ``shock_start_day``, ``shock_days``, ``days``,
    ``n_runs``, ``seed``) are passed to ``simulate_inventory``. The policy row
    used is returned under ``policy``.
    """
    policy = inventory_policy(orders, service_level=service_level).set_index("sku")
    if sku not in policy.index:
        raise ValueError(f"SKU '{sku}' does not appear in the orders")
    p = policy.loc[sku]
    result = simulate_inventory(
        avg_daily_demand=p["avg_daily_demand"],
        sd_daily_demand=p["sd_daily_demand"],
        avg_lead_time_days=p["avg_lead_time_days"],
        sd_lead_time_days=p["sd_lead_time_days"],
        reorder_point=p["reorder_point"],
        order_quantity=max(1.0, p["avg_daily_demand"] * order_cover_days),
        **shock,
    )
    result["policy"] = p
    return result
