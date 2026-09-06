import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast import (
    make_seller_heuristic_policy,
    make_actual_policy,
    make_context_aware_policy,
    make_plain_rop_policy,
    simulate_policy,
)

_ITEMS = pd.DataFrame([dict(item_id="I999", item_name="Test Item", category="Festive")])
_NO_FESTIVALS = pd.DataFrame(columns=["date", "festival_name", "affected_categories", "uplift_multiplier", "ramp_days_before"])
_NO_OVERRIDES = pd.DataFrame(columns=["festival_name", "item_id", "uplift_multiplier"])
_NO_PROMOS = pd.DataFrame(columns=["date", "item_id", "promo_uplift"])


def _dates(start, n):
    return pd.date_range(start, periods=n)


# ---------------------------------------------------------------- make_actual_policy

def test_actual_policy_replays_exact_recorded_orders():
    purchase_orders = pd.DataFrame([
        dict(item_id="I001", date_ordered=pd.Timestamp("2026-08-01"), qty_ordered=40),
        dict(item_id="I001", date_ordered=pd.Timestamp("2026-08-10"), qty_ordered=25),
        dict(item_id="I002", date_ordered=pd.Timestamp("2026-08-01"), qty_ordered=99),  # different item
    ])
    policy = make_actual_policy("I001", purchase_orders)
    assert policy(pd.Timestamp("2026-08-01"), None, 0, []) == 40
    assert policy(pd.Timestamp("2026-08-10"), None, 0, []) == 25
    assert policy(pd.Timestamp("2026-08-02"), None, 0, []) == 0  # no order that day


# ---------------------------------------------------------------- make_plain_rop_policy

def test_plain_rop_orders_nothing_while_a_pending_order_exists():
    policy = make_plain_rop_policy(lead_time_days=5)
    history = pd.DataFrame({"date": _dates("2026-06-01", 30), "units_sold": [10] * 30, "stockout_flag": [0] * 30})
    qty = policy(pd.Timestamp("2026-07-01"), history, on_hand=0, pending=[(pd.Timestamp("2026-07-05"), 50)])
    assert qty == 0


def test_plain_rop_orders_when_below_reorder_point():
    policy = make_plain_rop_policy(lead_time_days=5)
    history = pd.DataFrame({"date": _dates("2026-06-01", 30), "units_sold": [10] * 30, "stockout_flag": [0] * 30})
    qty_low_stock = policy(pd.Timestamp("2026-07-01"), history, on_hand=2, pending=[])
    qty_high_stock = policy(pd.Timestamp("2026-07-01"), history, on_hand=500, pending=[])
    assert qty_low_stock > 0
    assert qty_high_stock == 0


# ---------------------------------------------------------------- make_context_aware_policy

def test_context_aware_orders_more_ahead_of_a_prophet_forecasted_spike():
    # no festival calendar entries here -- isolates the Prophet-forecast signal
    flat_table = pd.Series(2.0, index=_dates("2026-08-01", 20))
    spike_table = flat_table.copy()
    spike_table.loc["2026-08-10":"2026-08-15"] = 40.0

    history = pd.DataFrame({"date": _dates("2026-07-01", 30), "units_sold": [2] * 30, "stockout_flag": [0] * 30})
    kwargs = dict(item_id="I999", lead_time_days=5, baseline_mu=2.0, items=_ITEMS,
                  festival_calendar=_NO_FESTIVALS, festival_overrides=_NO_OVERRIDES, promotions=_NO_PROMOS)

    qty_flat = make_context_aware_policy(forecast_table=flat_table, **kwargs)(
        pd.Timestamp("2026-08-08"), history, on_hand=10, pending=[])
    qty_spike = make_context_aware_policy(forecast_table=spike_table, **kwargs)(
        pd.Timestamp("2026-08-08"), history, on_hand=10, pending=[])
    assert qty_spike > qty_flat


def test_context_aware_falls_back_to_domain_uplift_when_prophet_has_no_precedent():
    # Prophet's forecast is flat -- as it would be for a festival with zero occurrences
    # before the training cutoff -- but the Context Agent's hand-set uplift multiplier
    # should still drive a bigger order. This is the actual cold-start bug this fix closes.
    flat_table = pd.Series(0.1, index=_dates("2026-08-01", 20))  # Prophet has learned ~nothing
    festival_calendar = pd.DataFrame([dict(
        date=pd.Timestamp("2026-08-12"), festival_name="Founders Day",
        affected_categories="Festive", uplift_multiplier=10.0, ramp_days_before=3,
    )])
    history = pd.DataFrame({"date": _dates("2026-07-01", 30), "units_sold": [1] * 30, "stockout_flag": [0] * 30})

    policy = make_context_aware_policy(
        item_id="I999", lead_time_days=5, forecast_table=flat_table, baseline_mu=1.0,
        items=_ITEMS, festival_calendar=festival_calendar,
        festival_overrides=_NO_OVERRIDES, promotions=_NO_PROMOS,
    )
    qty_before_window = policy(pd.Timestamp("2026-08-01"), history, on_hand=10, pending=[])
    qty_near_festival = policy(pd.Timestamp("2026-08-09"), history, on_hand=10, pending=[])
    assert qty_near_festival > qty_before_window


def test_context_aware_orders_nothing_while_a_pending_order_exists():
    table = pd.Series(50.0, index=_dates("2026-08-01", 20))
    policy = make_context_aware_policy(
        item_id="I999", lead_time_days=5, forecast_table=table, baseline_mu=5.0,
        items=_ITEMS, festival_calendar=_NO_FESTIVALS,
        festival_overrides=_NO_OVERRIDES, promotions=_NO_PROMOS,
    )
    history = pd.DataFrame({"date": _dates("2026-07-01", 5), "units_sold": [2] * 5, "stockout_flag": [0] * 5})
    qty = policy(pd.Timestamp("2026-08-05"), history, on_hand=0, pending=[(pd.Timestamp("2026-08-06"), 10)])
    assert qty == 0


# ---------------------------------------------------------------- simulate_policy

def test_simulate_policy_perfect_fill_when_stock_never_runs_out():
    demand = pd.Series(5, index=_dates("2026-08-01", 10))
    never_order = lambda date, history, on_hand, pending: 0
    result = simulate_policy("X", never_order, demand, lead_time_mean=4, lead_time_std=1,
                              start_date=demand.index[0], end_date=demand.index[-1],
                              initial_on_hand=10_000, unit_cost=10.0, seed=1)
    assert result["stockout_days"] == 0
    assert result["fill_rate"] == 1.0


def test_simulate_policy_total_stockout_when_never_restocked_and_no_initial_stock():
    demand = pd.Series(5, index=_dates("2026-08-01", 10))
    never_order = lambda date, history, on_hand, pending: 0
    result = simulate_policy("X", never_order, demand, lead_time_mean=4, lead_time_std=1,
                              start_date=demand.index[0], end_date=demand.index[-1],
                              initial_on_hand=0, unit_cost=10.0, seed=1)
    assert result["stockout_days"] == 10
    assert result["fill_rate"] == 0.0


def test_simulate_policy_order_arrives_and_ends_the_stockout():
    demand = pd.Series(5, index=_dates("2026-08-01", 15))
    ordered_once = {"done": False}

    def order_on_day_one_only(date, history, on_hand, pending):
        if not ordered_once["done"] and date == demand.index[0]:
            ordered_once["done"] = True
            return 200
        return 0

    result = simulate_policy("X", order_on_day_one_only, demand, lead_time_mean=3, lead_time_std=0,
                              start_date=demand.index[0], end_date=demand.index[-1],
                              initial_on_hand=0, unit_cost=10.0, seed=1)
    # stocked out days 1-3 (order in transit), then covered once the order lands on day 4
    assert result["stockout_days"] == 3
    assert result["orders_placed"] == 1


def test_seller_heuristic_reorders_only_below_threshold_and_with_nothing_pending():
    # avg_rate 10 -> rop = 50, qty = 140 (the constants generate_dataset.py uses)
    policy = make_seller_heuristic_policy(10.0)
    d, h = pd.Timestamp("2026-01-01"), pd.DataFrame()
    assert policy(d, h, on_hand=60, pending=[]) == 0     # above threshold
    assert policy(d, h, on_hand=50, pending=[]) == 140   # at threshold
    assert policy(d, h, on_hand=10, pending=[]) == 140   # well below
    assert policy(d, h, on_hand=10, pending=[(d, 140)]) == 0  # already on the way


def test_seller_heuristic_is_closed_loop_unlike_the_replay():
    # the point of replacing make_actual_policy as the headline baseline: this one
    # reacts to the stock level it is actually handed, so it stays a fair
    # comparator once the simulator draws its own lead times
    policy = make_seller_heuristic_policy(1.0)
    d, h = pd.Timestamp("2026-01-01"), pd.DataFrame()
    assert policy(d, h, on_hand=0, pending=[]) > 0
    assert policy(d, h, on_hand=999, pending=[]) == 0


def test_simulate_policy_reports_raw_counts_for_a_weighted_fill_rate():
    # fill_rate alone cannot be aggregated across SKUs without the denominators
    demand = pd.Series([2] * 10, index=pd.date_range("2026-01-01", periods=10))
    res = simulate_policy("I001", lambda *a: 0, demand, 4, 0.5,
                          pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10"),
                          initial_on_hand=6, unit_cost=10.0, seed=1)
    assert res["total_demand"] == 20
    assert res["units_short"] == 14
    assert res["fill_rate"] == round(1 - 14 / 20, 3)
