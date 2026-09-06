"""
Deterministic inventory core. Every number a seller eventually sees traces to a
function in this file, not to an LLM. Pure functions only: dataframe/values in,
values out, no file I/O — that's what makes them trivially testable and safe to
call as LangGraph tools later without worrying what else they touch.
"""

from typing import Literal

import numpy as np
import pandas as pd

Risk = Literal["stockout_risk", "overstock", "healthy"]


def _croston_sba(values: np.ndarray, alpha: float = 0.2) -> tuple[float, float]:
    """
    Croston's method with the Syntetos-Boylan bias correction — for series where
    demand is mostly zero, where a trailing median reads 0 regardless of the true
    rate because zeros are the majority. Croston separates the series into how big
    a sale is when one happens, and how many periods pass between sales, and
    exponentially smooths each independently instead of taking a single median
    over a series dominated by zeros.
    """
    nonzero_idx = np.flatnonzero(values)
    if len(nonzero_idx) == 0:
        return 0.0, 0.0  # no sale anywhere in this window — nothing to estimate from

    sizes = values[nonzero_idx]
    intervals = np.diff(np.concatenate([[-1], nonzero_idx]))  # periods since the previous sale

    z_hat, p_hat = float(sizes[0]), float(intervals[0])
    for z, p in zip(sizes[1:], intervals[1:]):
        z_hat = alpha * z + (1 - alpha) * z_hat
        p_hat = alpha * p + (1 - alpha) * p_hat

    mu = (1 - alpha / 2) * z_hat / p_hat  # SBA correction for Croston's known upward bias

    # per-period variance of a compound process: demand is 0 with prob (1-rate), or a
    # draw from the observed sale-size distribution with prob `rate` (law of total variance)
    rate = 1.0 / p_hat
    size_var = float(np.var(sizes)) if len(sizes) > 1 else 0.0
    per_period_var = rate * size_var + rate * (1 - rate) * z_hat**2
    sigma = float(np.sqrt(max(0.0, per_period_var)))

    # z_hat/p_hat pick up numpy.float64 after the first smoothing step (mixed arithmetic
    # with the numpy array elements in sizes/intervals promotes past the initial float()
    # cast) -- explicit here so callers reliably get plain Python floats, not just "close
    # enough to a float" that breaks strict serializers like a LangGraph checkpointer.
    return float(mu), float(sigma)


def _winsorized_mean(values: np.ndarray, p: float = 0.05) -> float:
    """
    Central tendency for the dense-series case. A mean, not a median, because the
    quantity the reorder point needs is expected demand summed over the lead time:
    E[sum over L days] = L * E[daily]. Means add that way; medians do not. On
    right-skewed count data a median sits systematically below the mean, so
    mu * L came out low for essentially every SKU (measured: median ratio 0.87
    against the true trailing mean, nine of twenty-five SKUs low by more than 20%).

    Winsorizing preserves the outlier resistance the median was originally chosen
    for -- a bulk buyer or a data-entry error is clipped to the 5th/95th percentile
    rather than chased -- at a cost of about one point of accuracy (median ratio
    0.98 winsorized vs 0.99 for a raw mean).
    """
    if len(values) == 0:
        return 0.0
    lo, hi = np.quantile(values, [p, 1 - p])
    return float(np.clip(values, lo, hi).mean())


def demand_rate(
    sales_history: pd.DataFrame,
    window: int = 60,
    exclude_stockouts: bool = True,
    intermittent_threshold: float = 0.5,
) -> tuple[float, float]:
    """
    Demand rate (mu) and its std (sigma) over the last `window` days.

    A stockout day's units_sold is censored (capped by whatever stock was left),
    not a true zero-demand signal, so those days are excluded by default rather
    than letting them drag the estimate down — falls back to the full window if
    exclusion leaves too little data.

    Be clear about what that exclusion does and does not buy. Dropping censored
    days is a partial mitigation, not a fix: stockouts correlate with demand
    spikes, so deleting them also deletes the high days. That is the classic
    lost-sales selection bias, and it leaves mu biased low on exactly the SKUs
    that stock out most (Umbrella and Raincoat through the monsoon are the worst
    cases here). The statistically correct treatment is to keep those days as
    right-censored observations — demand >= on_hand_start — and fit a censored
    likelihood. A Poisson censored MLE was measured at roughly one further point
    of median accuracy over what this function now does; it is a worthwhile
    follow-up, not a prerequisite. See PROJECT_REPORT.md section 9.

    Estimator is chosen by how sparse the (post-exclusion) series is: once at least
    `intermittent_threshold` of the days are zero, a mean over the raw series is
    dominated by zeros and a median reads 0 outright, so this switches to
    Croston/SBA instead. Same (mu, sigma) shape either way, so callers never need
    to know which estimator ran.
    """
    recent = sales_history.sort_values("date").tail(window)
    if exclude_stockouts:
        clean = recent[recent["stockout_flag"] == 0]
        if len(clean) >= max(7, window // 4):
            recent = clean

    values = recent["units_sold"].to_numpy(dtype=float)
    zero_fraction = float((values == 0).mean()) if len(values) else 1.0

    if zero_fraction >= intermittent_threshold:
        return _croston_sba(values)

    mu = _winsorized_mean(values)
    # sigma stays on MAD: it is a spread estimate, where robustness costs nothing
    # and there is no additivity argument pulling toward the moment estimator.
    # Deviations are taken about the median, which is the MAD's own definition --
    # using mu here would inflate sigma whenever the series is skewed.
    sigma = float(np.median(np.abs(values - np.median(values)))) * 1.4826  # normal-consistent scale
    return mu, sigma


def reorder_point(mu: float, sigma: float, lead_time_days: float, z: float = 1.65) -> int:
    return int(np.ceil(mu * lead_time_days + z * sigma * np.sqrt(lead_time_days)))


def safety_stock(sigma: float, lead_time_days: float, z: float = 1.65) -> int:
    return int(np.ceil(z * sigma * np.sqrt(lead_time_days)))


def order_quantity(rop: int, on_hand: int, on_order: int = 0) -> int:
    return max(0, rop - (on_hand + on_order))


def days_of_cover(on_hand: int, mu: float) -> float:
    if mu <= 0:
        return float("inf") if on_hand > 0 else 0.0
    return on_hand / mu


def classify_risk(on_hand: int, rop: int, days_cover: float, overstock_days: float = 45) -> Risk:
    if on_hand <= rop:
        return "stockout_risk"
    if days_cover >= overstock_days:
        return "overstock"
    return "healthy"


def recent_trend_ratio(
    sales_history: pd.DataFrame,
    recent_window: int = 14,
    baseline_window: int = 180,
    cap: float = 20.0,
) -> float | None:
    """
    Ratio of the recent demand rate to a much longer-run baseline rate -- a cheap
    proxy for "is something happening right now that a single trailing window
    wouldn't show" (a seasonal ramp, a trend), without fitting a real forecasting
    model. Reuses demand_rate() for both windows, so it inherits the same stockout
    exclusion and winsorized-mean/Croston switching, no separate estimator to validate.

    >1 means recent demand is running above its usual rate, <1 means below.
    Capped rather than allowed to hit true infinity (when baseline is ~0 but
    recent isn't) -- a finite "20x normal" is exactly as informative to a human
    as "infinite" and doesn't risk being the next float that breaks somewhere
    downstream expecting an ordinary number. None means there's nothing to
    compare -- no meaningful demand in either window.
    """
    recent_mu, _ = demand_rate(sales_history, window=recent_window)
    baseline_mu, _ = demand_rate(sales_history, window=baseline_window)
    if baseline_mu < 0.01 and recent_mu < 0.01:
        return None
    if baseline_mu < 0.01:
        return cap
    return min(cap, recent_mu / baseline_mu)


def on_order_qty(item_id: str, as_of_date: pd.Timestamp, purchase_orders: pd.DataFrame) -> int:
    """
    Units already bought and not yet arrived: ordered on or before today, expected
    after today. Kept separate from assess_item's own lookups because it reads a
    different table, and pure like everything else here so it can be tested without
    a filesystem.

    Without this the two backtest policies guarded against double-ordering
    (`if pending: return 0`) but the live product did not, so the dashboard
    re-recommended the same order every morning until the goods physically landed.
    On the default business date that affected 15 of 18 flagged items.
    """
    if purchase_orders is None or purchase_orders.empty:
        return 0
    open_pos = purchase_orders[
        (purchase_orders.item_id == item_id)
        & (purchase_orders.date_ordered <= as_of_date)
        & (purchase_orders.date_expected > as_of_date)
    ]
    return int(open_pos.qty_ordered.sum())


def days_to_next_arrival(item_id: str, as_of_date: pd.Timestamp,
                         purchase_orders: pd.DataFrame) -> int | None:
    """Days until the soonest outstanding delivery, or None if nothing is in transit.
    A flagged item with stock already on the way still needs to be shown -- a late PO
    is precisely the case a seller must see -- so the UI needs the arrival date, not
    just the quantity."""
    if purchase_orders is None or purchase_orders.empty:
        return None
    open_pos = purchase_orders[
        (purchase_orders.item_id == item_id)
        & (purchase_orders.date_ordered <= as_of_date)
        & (purchase_orders.date_expected > as_of_date)
    ]
    if open_pos.empty:
        return None
    return int((open_pos.date_expected.min() - as_of_date).days)


def assess_item(
    item_id: str,
    as_of_date: pd.Timestamp,
    sales: pd.DataFrame,
    items: pd.DataFrame,
    suppliers: pd.DataFrame,
    z: float = 1.65,
    lead_time_buffer_days: float = 0.0,
    on_order: int = 0,
    arriving_in_days: int | None = None,
) -> dict:
    """
    The per-SKU report: pulls current on-hand and lead time, runs the formulas above,
    and returns one flat dict. This is the shape the graph registers as a tool.

    z, lead_time_buffer_days and on_order all default to the values every call has
    always used, so every existing caller is unaffected -- but z and
    lead_time_buffer_days are real parameters, not constants, so the Feedback Agent
    has something to actually move, and on_order lets a caller net off stock that is
    already in transit (see on_order_qty). This function still reads no tables it
    was not handed, so it stays exactly as pure as it has always been.
    A rejection that adjusts a seller's per-item z or buffer only changes behaviour
    for calls that pass the adjusted value back in; nothing here reads from a
    parameter store itself, keeping this function exactly as pure as it's always been.
    """
    item = items.set_index("item_id").loc[item_id]
    supplier = suppliers.set_index("supplier_id").loc[item.supplier_id]

    history = sales[(sales.item_id == item_id) & (sales.date <= as_of_date)]
    if history.empty:
        raise ValueError(f"no sales history for {item_id} as of {as_of_date}")

    mu, sigma = demand_rate(history)
    lead_time = float(supplier.mean_lead_time_days) + lead_time_buffer_days
    rop = reorder_point(mu, sigma, lead_time, z=z)
    ss = safety_stock(sigma, lead_time, z=z)

    on_hand = int(history.sort_values("date").iloc[-1].on_hand_end)
    cover = days_of_cover(on_hand, mu)
    # risk is judged on physical stock: an item with nothing on the shelf IS at
    # risk of stocking out today, whether or not a delivery is in transit. Only
    # the order quantity nets off what is already coming, so the seller still sees
    # the item -- with a quantity of zero and an arrival date -- rather than having
    # a late PO silently suppressed.
    risk = classify_risk(on_hand, rop, cover)
    qty = order_quantity(rop, on_hand, on_order) if risk == "stockout_risk" else 0
    trend_ratio = recent_trend_ratio(history)

    return dict(
        item_id=item_id,
        item_name=item.item_name,
        category=item.category,
        mu=round(mu, 2),
        sigma=round(sigma, 2),
        lead_time_days=lead_time,
        on_hand=on_hand,
        on_order=int(on_order),
        arriving_in_days=arriving_in_days,
        reorder_point=rop,
        safety_stock=ss,
        days_of_cover=round(cover, 1) if cover != float("inf") else None,
        risk=risk,
        suggested_order_qty=qty,
        trend_ratio=round(trend_ratio, 2) if trend_ratio is not None else None,
    )
