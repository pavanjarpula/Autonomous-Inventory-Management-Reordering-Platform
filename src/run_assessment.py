"""
Sanity check: run every SKU through assess_item() as of the last date in the
dataset and print the result. This is the sanity check — does a steady staple get a
sane reorder point, does a near-zero festive SKU get treated differently from a
steady FMCG item, does the naive on-hand for Raincoat/Umbrella actually show up
flagged after the monsoon run? Not a backtest — just: are the
numbers themselves sane before anything gets built on top of them.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import assess_item, days_to_next_arrival, on_order_qty

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

sales = pd.read_csv(DATA / "daily_sales.csv", parse_dates=["date"])
items = pd.read_csv(DATA / "items.csv")
suppliers = pd.read_csv(DATA / "suppliers.csv")
purchase_orders = pd.read_csv(DATA / "purchase_orders.csv", parse_dates=["date_ordered", "date_expected"])

as_of = sales["date"].max()
rows = [assess_item(item_id, as_of, sales, items, suppliers,
                     on_order=on_order_qty(item_id, as_of, purchase_orders),
                     arriving_in_days=days_to_next_arrival(item_id, as_of, purchase_orders))
        for item_id in items["item_id"]]
report = pd.DataFrame(rows).sort_values(["risk", "item_name"])

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 30)

print(f"assessment as of {as_of.date()}\n")
for risk in ["stockout_risk", "overstock", "healthy"]:
    subset = report[report.risk == risk]
    print(f"--- {risk} ({len(subset)}) ---")
    print(subset[["item_id", "item_name", "category", "mu", "sigma", "lead_time_days",
                  "on_hand", "reorder_point", "days_of_cover", "suggested_order_qty"]]
          .to_string(index=False))
    print()
