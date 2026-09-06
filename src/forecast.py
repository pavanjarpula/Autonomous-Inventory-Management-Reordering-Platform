"""
Forecasting and backtest. Fits a per-SKU Prophet model on the festival
calendar and promo regressor, then runs four ordering policies through the same
held-out window against the same true demand, so their outcomes are directly
comparable:

  seller         -- the seller's own gut-feel rule, run closed-loop (the headline baseline)
  actual_replay  -- open-loop replay of the recorded purchase orders, for reference
  plain_rop      -- reorder-point math only: reorder point off a trailing rate, no calendar
  context_aware  -- the same safety-stock term, but sized off the Prophet forecast
                     over the lead-time window instead of a flat trailing rate

All four see only information available as of each simulated day -- the Prophet
model is trained strictly before the backtest window starts, and plain_rop /
context_aware compute their stats from each policy's own observed (censored) sales
so far, never from true_demand_uncensored. That column is the evaluator's oracle,
not a model input.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_agent import get_context, promo_flag_series, to_prophet_holidays
from engine import demand_rate, order_quantity, reorder_point, safety_stock  # noqa: F401


def fit_item_forecast(item_id: str, sales: pd.DataFrame, festival_calendar: pd.DataFrame,
                       promotions: pd.DataFrame, train_end_date: pd.Timestamp) -> Prophet:
    """Fits on data strictly before train_end_date -- the backtest window must never
    leak into what the forecaster was trained on."""
    hist = sales[(sales.item_id == item_id) & (sales.date < train_end_date)][["date", "units_sold"]]
    hist = hist.rename(columns={"date": "ds", "units_sold": "y"}).sort_values("ds")
    hist["promo_active"] = promo_flag_series(item_id, hist["ds"], promotions).values

    model = Prophet(holidays=to_prophet_holidays(festival_calendar), seasonality_mode="multiplicative")
    model.add_regressor("promo_active")
    model.fit(hist)
    return model


def precompute_forecast_table(model: Prophet, item_id: str, dates: pd.DatetimeIndex,
                               promotions: pd.DataFrame) -> pd.Series:
    """One predict() call over the whole backtest+lookahead range, so a policy's
    per-day decision is a lookup into this, not a fresh model call every day."""
    future = pd.DataFrame({"ds": dates})
    future["promo_active"] = promo_flag_series(item_id, dates, promotions).values
    forecast = model.predict(future)
    return forecast.set_index("ds")["yhat"].clip(lower=0)


def make_seller_heuristic_policy(avg_rate: float):
    """
    The seller's actual rule, run CLOSED-LOOP against its own stock trajectory:
    reorder a fixed quantity whenever stock falls to a gut-feel threshold and
    nothing is already on the way. These are exactly the constants
    generate_dataset.py used to produce purchase_orders.csv.

    This replaces make_actual_policy() as the headline baseline. The replay was
    not a fair comparator: it fires fixed order dates derived from a trajectory
    that no longer applies once the simulator draws its own lead times, so an
    open-loop schedule was competing against two closed-loop policies and lost
    partly for that reason. Beating a properly-run version of the seller's own
    rule is the claim worth making.
    """
    rop = max(1, int(np.ceil(avg_rate * 5)))
    qty = max(3, int(np.ceil(avg_rate * 14)))

    def policy(date, history, on_hand, pending):
        if pending or on_hand > rop:
            return 0
        return qty
    return policy


def make_actual_policy(item_id: str, purchase_orders: pd.DataFrame):
    """
    Open-loop replay of the recorded purchase orders. Kept for reference and
    reported alongside the closed-loop version, but it is NOT the headline
    baseline -- see make_seller_heuristic_policy() for why.
    """
    orders = purchase_orders[purchase_orders.item_id == item_id].set_index("date_ordered")["qty_ordered"]
    orders.index = pd.to_datetime(orders.index)

    def policy(date, history, on_hand, pending):
        return int(orders.get(date, 0))
    return policy


def make_plain_rop_policy(lead_time_days: float):
    def policy(date, history, on_hand, pending):
        if pending:
            return 0
        mu, sigma = demand_rate(history)
        rop = reorder_point(mu, sigma, lead_time_days)
        return order_quantity(rop, on_hand) if on_hand <= rop else 0
    return policy


def make_context_aware_policy(
    item_id: str,
    lead_time_days: float,
    forecast_table: pd.Series,
    baseline_mu: float,
    items: pd.DataFrame,
    festival_calendar: pd.DataFrame,
    festival_overrides: pd.DataFrame,
    promotions: pd.DataFrame,
    review_buffer: int = 3,
):
    """
    Sizes the order off the greater of two signals for each day in the lookahead
    horizon: Prophet's learned forecast (which can only reflect a festival if that
    festival actually occurred at least once before the training cutoff -- useless
    for a once-a-year event with zero precedent in a single year of history), and
    the item's known long-run baseline rate scaled by the Context Agent's hand-set
    uplift multiplier -- domain knowledge, not learned, so it still has *something*
    to say about a festival this SKU has never sold through before. Whichever signal
    is larger wins, so items with real precedent lean on Prophet's learned shape and
    items without any still get a real, if rougher, signal from the calendar.
    """
    horizon_days = int(lead_time_days) + review_buffer

    def policy(date, history, on_hand, pending):
        if pending:
            return 0
        _, sigma = demand_rate(history)  # today's observed variability, for the safety-stock term

        expected_demand = 0.0
        for offset in range(horizon_days):
            d = date + pd.Timedelta(days=offset)
            prophet_yhat = float(forecast_table.get(d, baseline_mu))
            ctx = get_context(item_id, d, items, festival_calendar, festival_overrides, promotions)
            domain_estimate = baseline_mu * ctx["uplift_multiplier"]
            expected_demand += max(prophet_yhat, domain_estimate)

        target = expected_demand + safety_stock(sigma, lead_time_days)
        gap = target - on_hand
        return max(0, int(np.ceil(gap))) if gap > 0 else 0
    return policy


def simulate_policy(item_id: str, policy_fn, true_demand: pd.Series, lead_time_mean: float,
                     lead_time_std: float, start_date: pd.Timestamp, end_date: pd.Timestamp,
                     initial_on_hand: int, unit_cost: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    on_hand = initial_on_hand
    pending: list[tuple[pd.Timestamp, int]] = []
    sold_so_far_rows = []

    stockout_days = total_days = total_demand = units_short = orders_placed = 0
    capital_snapshots = []

    for date in pd.date_range(start_date, end_date):
        arrived = sum(q for arr, q in pending if arr == date)
        if arrived:
            on_hand += arrived
            pending = [(a, q) for a, q in pending if a != date]

        demand_today = int(true_demand.loc[date])
        sold = min(demand_today, on_hand)
        stockout = demand_today > on_hand
        on_hand -= sold

        total_days += 1
        total_demand += demand_today
        units_short += max(0, demand_today - sold)
        stockout_days += int(stockout)
        capital_snapshots.append(on_hand * unit_cost)

        sold_so_far_rows.append(dict(date=date, units_sold=sold, stockout_flag=int(stockout)))
        order_qty = policy_fn(date, pd.DataFrame(sold_so_far_rows), on_hand, pending)

        if order_qty > 0:
            lead = max(1, int(round(rng.normal(lead_time_mean, lead_time_std))))
            pending.append((date + pd.Timedelta(days=lead), order_qty))
            orders_placed += 1

    fill_rate = 1 - (units_short / total_demand) if total_demand > 0 else 1.0
    return dict(
        item_id=item_id,
        stockout_days=stockout_days,
        total_days=total_days,
        # raw counts travel alongside the ratio so a demand-weighted fill rate can
        # be computed exactly, rather than reconstructed from a rounded per-SKU
        # figure. An unweighted mean of fill_rate gives an 8-unit gift hamper the
        # same say as 486 units of sugar, which is not the number a shop owner cares
        # about -- report both, and lead with the weighted one.
        total_demand=int(total_demand),
        units_short=int(units_short),
        fill_rate=round(fill_rate, 3),
        avg_capital_tied_up=round(float(np.mean(capital_snapshots)), 1),
        orders_placed=orders_placed,
    )
