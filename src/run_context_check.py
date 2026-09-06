"""
Sanity check: confirm the context layer resolves the two SKUs the plain
engine got wrong — Holi Colors Pack and Rakhi Set — and spot-check the Prophet-prep
helpers produce the right shape for the forecaster.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_agent import get_context, promo_flag_series, to_prophet_holidays

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

items = pd.read_csv(DATA / "items.csv")
festival_calendar = pd.read_csv(DATA / "festival_calendar.csv", parse_dates=["date"])
festival_overrides = pd.read_csv(DATA / "festival_item_overrides.csv")
promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["date"])


def show(label, item_id, date):
    ctx = get_context(item_id, pd.Timestamp(date), items, festival_calendar, festival_overrides, promotions)
    print(f"{label}")
    for k, v in ctx.items():
        print(f"  {k}: {v}")
    print()


print("=== the two SKUs the plain engine misread ===\n")

show("Holi Colors Pack, 8 days before its window opens (2026-02-17)", "I017", "2026-02-17")
show("Holi Colors Pack, on Holi itself (2026-03-04)", "I017", "2026-03-04")

show("Rakhi Set, 8 days out — window not open yet (2026-08-20)", "I016", "2026-08-20")
show("Rakhi Set, inside the window (2026-08-24)", "I016", "2026-08-24")

show("For contrast — Rice 5kg (Staples) on the same date, not a relevant category", "I001", "2026-08-24")

print("=== to_prophet_holidays() — the holidays= dataframe passed straight to Prophet ===")
print(to_prophet_holidays(festival_calendar).to_string(index=False))
print()

print("=== promo_flag_series for Rice 5kg — active days only ===")
dates = pd.date_range("2025-09-01", "2026-08-31")
flags = promo_flag_series("I001", dates, promotions)
print(flags[flags == 1])
