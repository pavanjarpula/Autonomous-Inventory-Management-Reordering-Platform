import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engine import (
    assess_item,
    classify_risk,
    days_of_cover,
    days_to_next_arrival,
    on_order_qty,
    demand_rate,
    order_quantity,
    recent_trend_ratio,
    reorder_point,
    safety_stock,
)


def _history(units, stockouts=None, start="2026-01-01"):
    n = len(units)
    stockouts = stockouts or [0] * n
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n),
        "units_sold": units,
        "stockout_flag": stockouts,
    })


def test_demand_rate_constant_series_has_zero_sigma():
    mu, sigma = demand_rate(_history([10] * 30))
    assert mu == 10.0
    assert sigma == 0.0


def test_demand_rate_excludes_stockout_days_by_default():
    # true demand is ~10/day; a majority of days (18/30) are censored down to 1 unit sold —
    # enough to drag a naive median down, which is exactly what exclusion should prevent.
    # (a minority of censored days wouldn't move the median at all, since it's robust by
    # design — that's the point of using median over mean here.)
    units = [1] * 18 + [10] * 12
    stockouts = [1] * 18 + [0] * 12
    mu_excluded, _ = demand_rate(_history(units, stockouts), window=30, exclude_stockouts=True)
    mu_included, _ = demand_rate(_history(units, stockouts), window=30, exclude_stockouts=False)
    assert mu_excluded == 10.0
    assert mu_included < mu_excluded


def test_demand_rate_switches_to_croston_for_mostly_zero_series():
    # sells 6 units every 3rd day -> true long-run rate is 2/day, but 20/30 days are
    # zero, so a plain median would read 0. Croston should land nowhere near that.
    values = [0, 0, 6] * 10
    mu, sigma = demand_rate(_history(values), window=30)
    assert mu > 1.0
    assert sigma > 0.0


def test_croston_path_returns_plain_python_floats_not_numpy_scalars():
    # z_hat/p_hat pick up numpy.float64 after the first smoothing step,
    # which silently breaks strict serializers (a LangGraph checkpointer's msgpack
    # encoder, for one) even though the value looks fine in every comparison.
    # Needs >= 2 non-zero events to exercise the smoothing loop at all.
    values = [0, 0, 6] * 10
    mu, sigma = demand_rate(_history(values), window=30)
    assert type(mu) is float
    assert type(sigma) is float


# ---------------------------------------------------------------- recent_trend_ratio

def test_recent_trend_ratio_detects_a_genuine_ramp():
    # 46 days at rate 2, then the last 14 days at rate 10 -- both windows stay
    # dense enough to use the winsorized-mean path, so the ratio is hand-verifiable:
    # recent mean 10 / baseline mean (46*2 + 14*10)/60 = 3.867  ->  2.586.
    # Lower than the 5.0 a median baseline gave, because the baseline window
    # CONTAINS the ramp and a mean feels it while a median did not. That damping
    # is correct: the comparison is against the item's own recent-inclusive
    # average, and it still clears the 1.5 threshold _trend_note() acts on.
    values = [2] * 46 + [10] * 14
    ratio = recent_trend_ratio(_history(values), recent_window=14, baseline_window=60)
    assert ratio == pytest.approx(2.586, abs=0.001)


def test_recent_trend_ratio_is_one_for_flat_demand():
    ratio = recent_trend_ratio(_history([5] * 60), recent_window=14, baseline_window=60)
    assert ratio == 1.0


def test_recent_trend_ratio_drops_toward_zero_when_sales_have_stopped():
    values = [8] * 46 + [0] * 14  # sold steadily, then nothing at all recently
    ratio = recent_trend_ratio(_history(values), recent_window=14, baseline_window=60)
    assert ratio == 0.0


def test_recent_trend_ratio_is_none_when_theres_no_signal_anywhere():
    ratio = recent_trend_ratio(_history([0] * 60), recent_window=14, baseline_window=60)
    assert ratio is None


def test_recent_trend_ratio_is_capped_not_infinite_for_a_cold_start():
    # 300 days of nothing, then a real burst -- exercises the Croston path on the
    # baseline side; exact value isn't the point, just that it reads as strongly
    # elevated and never exceeds the cap (never literal infinity)
    values = [0] * 300 + [8] * 5
    ratio = recent_trend_ratio(_history(values), recent_window=5, baseline_window=305)
    assert ratio is not None
    assert 3.0 < ratio <= 20.0


def test_demand_rate_uses_winsorized_mean_for_mostly_nonzero_series():
    # twelve 9s, twelve 10s, six 11s -> mean 9.8. Nothing is extreme enough for
    # the 5/95 clip to bite, so this is the plain mean. It is deliberately NOT
    # the median (10.0): the reorder point needs E[demand over L days] = L * E[daily],
    # and medians do not add that way.
    mu, _ = demand_rate(_history([9, 10, 11, 9, 10] * 6), window=30)
    assert mu == pytest.approx(9.8)


def test_demand_rate_does_not_underestimate_a_right_skewed_series():
    # regression guard for the bias this replaced. Mostly 2s with a handful of
    # busy days: the median reads 2 and the reorder point comes out ~30% low.
    values = [2] * 24 + [8, 9, 10, 11, 12, 13]
    mu, _ = demand_rate(_history(values), window=30)
    assert mu > 2.0
    assert mu == pytest.approx(float(np.mean(values)), abs=0.5)


def test_demand_rate_still_clips_a_single_absurd_outlier():
    # the robustness the median was originally chosen for has to survive the swap:
    # one bulk buyer or one data-entry error must not drag mu far off the level
    clean = [5] * 39 + [5]
    spiked = [5] * 39 + [900]
    mu_clean, _ = demand_rate(_history(clean), window=40)
    mu_spiked, _ = demand_rate(_history(spiked), window=40)
    assert mu_spiked == pytest.approx(mu_clean, abs=0.01)


def test_demand_rate_returns_zero_for_series_with_no_sales_at_all():
    mu, sigma = demand_rate(_history([0] * 30), window=30)
    assert mu == 0.0
    assert sigma == 0.0


def test_demand_rate_falls_back_to_full_window_when_exclusion_leaves_too_little():
    # almost every day is a stockout -> excluding them would leave too few points
    units = [5] * 3 + [1] * 27
    stockouts = [0] * 3 + [1] * 27
    mu, _ = demand_rate(_history(units, stockouts), window=30, exclude_stockouts=True)
    # falls back to the full (mostly-censored) window rather than erroring.
    # (5*3 + 1*27)/30 = 1.4; the three high days are 10% of the window so the
    # 5/95 winsorization does not clip them away.
    assert mu == pytest.approx(1.4)


def test_reorder_point_with_zero_variance_equals_mean_demand_over_lead_time():
    assert reorder_point(mu=5.0, sigma=0.0, lead_time_days=4) == 20


def test_reorder_point_grows_with_lead_time_variance():
    tight = reorder_point(mu=5.0, sigma=1.0, lead_time_days=6)
    volatile = reorder_point(mu=5.0, sigma=4.0, lead_time_days=6)
    assert volatile > tight


def test_safety_stock_is_zero_when_sigma_is_zero():
    assert safety_stock(sigma=0.0, lead_time_days=10) == 0


def test_order_quantity_never_negative_when_overstocked():
    assert order_quantity(rop=20, on_hand=50) == 0


def test_order_quantity_covers_the_gap_including_open_orders():
    assert order_quantity(rop=20, on_hand=5, on_order=10) == 5


def test_days_of_cover_zero_demand_with_stock_is_infinite():
    assert days_of_cover(on_hand=12, mu=0.0) == float("inf")


def test_days_of_cover_zero_demand_zero_stock_is_zero():
    assert days_of_cover(on_hand=0, mu=0.0) == 0.0


@pytest.mark.parametrize("on_hand,rop,cover,expected", [
    (5, 20, 2.0, "stockout_risk"),   # at/under the reorder point
    (20, 20, 5.0, "stockout_risk"),  # boundary: on_hand == rop counts as at-risk
    (200, 20, 90.0, "overstock"),    # comfortably above rop, far too much cover
    (30, 20, 15.0, "healthy"),
])
def test_classify_risk_boundaries(on_hand, rop, cover, expected):
    assert classify_risk(on_hand, rop, cover) == expected


# ---------------------------------------------------------------- assess_item overrides

@pytest.fixture
def item_fixture():
    items = pd.DataFrame([dict(item_id="X", item_name="Test Item", category="Staples", supplier_id="S1")])
    suppliers = pd.DataFrame([dict(supplier_id="S1", mean_lead_time_days=5, lead_time_std_days=1)])
    sales = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=60),
        "item_id": "X", "units_sold": 6, "stockout_flag": 0, "on_hand_end": 20,
    })
    return dict(sales=sales, items=items, suppliers=suppliers, as_of=sales["date"].iloc[-1])


def test_assess_item_default_z_matches_prior_hardcoded_behaviour(item_fixture):
    # z's default (1.65) must match what every earlier phase already assumed --
    # existing callers that don't pass z shouldn't see any behaviour change
    result = assess_item("X", item_fixture["as_of"], item_fixture["sales"],
                          item_fixture["items"], item_fixture["suppliers"])
    assert result["reorder_point"] == reorder_point(6.0, 0.0, 5.0, z=1.65)


def test_assess_item_lower_z_shrinks_the_reorder_point(item_fixture):
    default = assess_item("X", item_fixture["as_of"], item_fixture["sales"],
                           item_fixture["items"], item_fixture["suppliers"])
    lowered = assess_item("X", item_fixture["as_of"], item_fixture["sales"],
                           item_fixture["items"], item_fixture["suppliers"], z=1.0)
    assert lowered["reorder_point"] <= default["reorder_point"]
    assert lowered["safety_stock"] <= default["safety_stock"]


def test_assess_item_lead_time_buffer_raises_the_reorder_point(item_fixture):
    default = assess_item("X", item_fixture["as_of"], item_fixture["sales"],
                           item_fixture["items"], item_fixture["suppliers"])
    buffered = assess_item("X", item_fixture["as_of"], item_fixture["sales"],
                            item_fixture["items"], item_fixture["suppliers"], lead_time_buffer_days=5.0)
    assert buffered["reorder_point"] > default["reorder_point"]
    assert buffered["lead_time_days"] == default["lead_time_days"] + 5.0


# ---------------------------------------------------------------- in-transit stock

def _po(rows):
    return pd.DataFrame(rows, columns=["item_id", "date_ordered", "date_expected", "qty_ordered"])


def test_on_order_counts_only_orders_placed_and_not_yet_arrived():
    po = _po([
        ("I001", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-05"), 40),  # in transit
        ("I001", pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-06"), 10),  # in transit
        ("I001", pd.Timestamp("2025-12-20"), pd.Timestamp("2025-12-24"), 99),  # already arrived
        ("I002", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-05"), 77),  # another item
    ])
    assert on_order_qty("I001", pd.Timestamp("2026-01-03"), po) == 50


def test_on_order_is_zero_with_no_purchase_order_table():
    assert on_order_qty("I001", pd.Timestamp("2026-01-03"), None) == 0
    assert on_order_qty("I001", pd.Timestamp("2026-01-03"), _po([])) == 0


def test_days_to_next_arrival_picks_the_soonest_outstanding_delivery():
    po = _po([
        ("I001", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-09"), 40),
        ("I001", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-05"), 10),
    ])
    assert days_to_next_arrival("I001", pd.Timestamp("2026-01-03"), po) == 2
    assert days_to_next_arrival("I001", pd.Timestamp("2026-01-03"), None) is None


def test_assess_item_nets_off_stock_already_in_transit():
    # the double-ordering bug: without on_order the same quantity is recommended
    # every morning until the goods physically land
    sales = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=60),
        "item_id": ["I001"] * 60,
        "units_sold": [5] * 60,
        "stockout_flag": [0] * 60,
        "on_hand_end": [4] * 60,
    })
    items = pd.DataFrame([dict(item_id="I001", item_name="Rice", category="Staples", supplier_id="S1")])
    suppliers = pd.DataFrame([dict(supplier_id="S1", mean_lead_time_days=4, lead_time_std_days=1.0)])
    as_of = pd.Timestamp("2026-03-01")

    bare = assess_item("I001", as_of, sales, items, suppliers)
    assert bare["risk"] == "stockout_risk"
    assert bare["suggested_order_qty"] > 0

    covered = assess_item("I001", as_of, sales, items, suppliers,
                          on_order=bare["suggested_order_qty"], arriving_in_days=2)
    assert covered["suggested_order_qty"] == 0
    assert covered["on_order"] == bare["suggested_order_qty"]
    assert covered["arriving_in_days"] == 2
    # still flagged: a late delivery is exactly what the seller must be shown
    assert covered["risk"] == "stockout_risk"
