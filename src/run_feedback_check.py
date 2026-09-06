"""
Sanity check: reject the same item's recommendation with the same reason
three times in a row and confirm two things -- the parameter doesn't move on the
first two rejections, and it does on the third, with assess_item() actually
producing a smaller recommendation afterward. Not just "a number changed
somewhere" -- the full loop, traced end to end.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import assess_item
from feedback_agent import default_parameters, submit_feedback

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

sales = pd.read_csv(DATA / "daily_sales.csv", parse_dates=["date"])
items = pd.read_csv(DATA / "items.csv")
suppliers = pd.read_csv(DATA / "suppliers.csv")

ITEM_ID = "I008"  # Biscuits Pack -- flagged stockout_risk with a large suggested_order_qty
AS_OF = pd.Timestamp("2026-08-24")

feedback_log = pd.DataFrame(columns=[
    "feedback_id", "recommendation_id", "item_id", "seller_decision",
    "reason_code", "timestamp", "parameter_adjusted", "old_value", "new_value",
])
params_by_item = {}

before = assess_item(ITEM_ID, AS_OF, sales, items, suppliers, **{
    k: v for k, v in default_parameters(ITEM_ID).items() if k != "item_id"
})
print(f"BEFORE any feedback: z={default_parameters(ITEM_ID)['z']}  "
      f"reorder_point={before['reorder_point']}  suggested_order_qty={before['suggested_order_qty']}\n")

for i in range(1, 4):
    feedback_log, params_by_item, audit = submit_feedback(
        feedback_log, params_by_item, recommendation_id=f"REC-{ITEM_ID}-{i}",
        item_id=ITEM_ID, decision="reject", reason_code="qty_too_high",
        timestamp=AS_OF + pd.Timedelta(days=i),
    )
    params = params_by_item[ITEM_ID]
    moved = f"-> z adjusted to {audit['new_value']}" if audit else "-> no change yet"
    print(f"Rejection #{i} (qty_too_high)  {moved}")

print()
current_params = {k: v for k, v in params_by_item[ITEM_ID].items() if k != "item_id"}
after = assess_item(ITEM_ID, AS_OF, sales, items, suppliers, **current_params)
print(f"AFTER 3 rejections: z={current_params['z']}  "
      f"reorder_point={after['reorder_point']}  suggested_order_qty={after['suggested_order_qty']}\n")

print(f"suggested_order_qty: {before['suggested_order_qty']} -> {after['suggested_order_qty']}\n")

print("=== audit trail (feedback_log) ===")
print(feedback_log[["feedback_id", "seller_decision", "reason_code",
                     "parameter_adjusted", "old_value", "new_value"]].to_string(index=False))
