"""
Backtest -- the evidence slide. Four policies, same 45-day held-out window
(2026-07-18 to 2026-08-31 -- the monsoon uplift on Raincoat/Umbrella lands right
inside it, deliberately), same true demand, scored on stockout days, fill rate,
and capital tied up.

Two reporting decisions worth knowing before reading the output:

  - Fill rate is reported BOTH ways. weighted_fill_rate is units served divided
    by units demanded across the whole store; sku_mean_fill_rate is the unweighted
    average across SKUs. The weighted figure is the headline, because the
    unweighted one gives an 8-unit gift hamper the same say as 486 units of sugar
    and flatters any policy that wins on the long tail.
  - The headline baseline is `seller`, the shop owner's own rule run closed-loop.
    `actual_replay` -- the open-loop replay of recorded purchase orders -- is kept
    for reference but is not a fair comparator; see make_seller_heuristic_policy().
"""

import sys
import time
import zlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import demand_rate
from forecast import (
    fit_item_forecast,
    make_actual_policy,
    make_context_aware_policy,
    make_plain_rop_policy,
    make_seller_heuristic_policy,
    precompute_forecast_table,
    simulate_policy,
)

POLICY_ORDER = ["seller", "actual_replay", "plain_rop", "context_aware"]

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

sales = pd.read_csv(DATA / "daily_sales.csv", parse_dates=["date"])
items = pd.read_csv(DATA / "items.csv")
suppliers = pd.read_csv(DATA / "suppliers.csv")
festival_calendar = pd.read_csv(DATA / "festival_calendar.csv", parse_dates=["date"])
festival_overrides = pd.read_csv(DATA / "festival_item_overrides.csv")
promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["date"])
purchase_orders = pd.read_csv(DATA / "purchase_orders.csv", parse_dates=["date_ordered", "date_expected"])

HOLDOUT_DAYS = 45
end_date = sales["date"].max()
start_date = end_date - pd.Timedelta(days=HOLDOUT_DAYS - 1)
print(f"backtest window: {start_date.date()} to {end_date.date()} ({HOLDOUT_DAYS} days)\n")

results = []
t0 = time.time()

for _, item in items.iterrows():
    item_id = item.item_id
    supplier = suppliers.set_index("supplier_id").loc[item.supplier_id]
    lead_mean, lead_std = float(supplier.mean_lead_time_days), float(supplier.lead_time_std_days)

    true_demand = (
        sales[sales.item_id == item_id].set_index("date")["true_demand_uncensored"]
    )
    initial_on_hand = int(
        sales[(sales.item_id == item_id) & (sales.date < start_date)]
        .sort_values("date").iloc[-1].on_hand_end
    )

    model = fit_item_forecast(item_id, sales, festival_calendar, promotions, start_date)
    forecast_dates = pd.date_range(start_date, end_date + pd.Timedelta(days=int(lead_mean) + 10))
    forecast_table = precompute_forecast_table(model, item_id, forecast_dates, promotions)

    # long-run baseline rate from the same pre-holdout window Prophet trains on -- the
    # domain-knowledge fallback the context-aware policy uses when Prophet has no
    # historical precedent for an upcoming festival to have learned from
    pre_holdout = sales[(sales.item_id == item_id) & (sales.date < start_date)]
    baseline_mu, _ = demand_rate(pre_holdout, window=len(pre_holdout))

    policies = {
        "seller": make_seller_heuristic_policy(baseline_mu),
        "actual_replay": make_actual_policy(item_id, purchase_orders),
        "plain_rop": make_plain_rop_policy(lead_mean),
        "context_aware": make_context_aware_policy(
            item_id, lead_mean, forecast_table, baseline_mu, items,
            festival_calendar, festival_overrides, promotions,
        ),
    }
    for policy_name, policy_fn in policies.items():
        # crc32, not hash(): Python randomises string hashing per process, which
        # would give this backtest different lead-time draws — and different
        # headline numbers — on every run
        seed = zlib.crc32(f"{item_id}:{policy_name}".encode())
        res = simulate_policy(item_id, policy_fn, true_demand, lead_mean, lead_std,
                               start_date, end_date, initial_on_hand, float(item.unit_cost_inr), seed)
        res["policy"] = policy_name
        res["item_name"] = item.item_name
        res["category"] = item.category
        results.append(res)

print(f"fit + simulated 25 items x {len(POLICY_ORDER)} policies in {time.time() - t0:.1f}s\n")

df = pd.DataFrame(results)

print("=== headline: totals across all 25 SKUs ===")
summary = df.groupby("policy").agg(
    total_stockout_days=("stockout_days", "sum"),
    sku_mean_fill_rate=("fill_rate", "mean"),
    total_capital_tied_up=("avg_capital_tied_up", "sum"),
    total_orders_placed=("orders_placed", "sum"),
)
# units served / units demanded across the whole store -- the number a shop owner
# actually feels. Computed from raw counts, not from the rounded per-SKU ratio.
summary["weighted_fill_rate"] = (
    1 - df.groupby("policy").units_short.sum() / df.groupby("policy").total_demand.sum()
)
summary = summary.round(3).loc[POLICY_ORDER, [
    "total_stockout_days", "weighted_fill_rate", "sku_mean_fill_rate",
    "total_capital_tied_up", "total_orders_placed",
]]
print(summary.to_string())
print()
print("weighted_fill_rate is the headline. sku_mean_fill_rate is the unweighted")
print("average across SKUs, shown so the gap between the two stays visible.")
print()

print("=== the SKUs already flagged as the interesting cases ===")
watch_list = ["I021", "I020", "I008", "I009", "I017", "I016"]  # Raincoat, Umbrella, Biscuits, Namkeen, Holi Colors, Rakhi Set
focus = df[df.item_id.isin(watch_list)].pivot_table(
    index=["item_id", "item_name"], columns="policy",
    values=["stockout_days", "fill_rate"],
)
print(focus.to_string())
print()

print("=== where the wins actually are: fill-rate delta vs demand volume ===")
# guards against a headline driven by long-tail SKUs: a +0.5 fill rate on an item
# with 8 units of annual demand is not the same win as +0.05 on 500 units
piv = df.pivot_table(index=["item_id", "item_name"], columns="policy", values="fill_rate")
piv["demand"] = df.groupby(["item_id", "item_name"]).total_demand.first()
piv["ctx_vs_seller"] = (piv["context_aware"] - piv["seller"]).round(3)
print(piv[["demand", "seller", "plain_rop", "context_aware", "ctx_vs_seller"]]
      .sort_values("demand", ascending=False).to_string())

df.to_csv(DATA / "backtest_results.csv", index=False)
print(f"\nfull results written to {DATA / 'backtest_results.csv'}")
