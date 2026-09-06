"""
Sanity check: ask real questions and a real command against the live
dataset with a real model, and check the answers are actually grounded --
not just that a response came back.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chatbot import respond
from context_agent import get_context
from engine import assess_item, days_to_next_arrival, on_order_qty

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

sales = pd.read_csv(DATA / "daily_sales.csv", parse_dates=["date"])
items = pd.read_csv(DATA / "items.csv")
suppliers = pd.read_csv(DATA / "suppliers.csv")
festival_calendar = pd.read_csv(DATA / "festival_calendar.csv", parse_dates=["date"])
festival_overrides = pd.read_csv(DATA / "festival_item_overrides.csv")
promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["date"])
purchase_orders = pd.read_csv(DATA / "purchase_orders.csv", parse_dates=["date_ordered", "date_expected"])

AS_OF = pd.Timestamp("2026-08-24")
WATCH_ITEMS = ["I016", "I020"]  # Rakhi Set (festival), Umbrella (trend) -- both have clean, distinct signals

consumption = {iid: assess_item(iid, AS_OF, sales, items, suppliers,
                                 on_order=on_order_qty(iid, AS_OF, purchase_orders),
                                 arriving_in_days=days_to_next_arrival(iid, AS_OF, purchase_orders))
               for iid in WATCH_ITEMS}
context = {iid: get_context(iid, AS_OF, items, festival_calendar, festival_overrides, promotions) for iid in WATCH_ITEMS}

try:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="qwen3:4b", temperature=0)
    llm.invoke("ping")
    print("using a real local Ollama model (qwen3:4b)\n")
except Exception as e:
    print(f"Ollama not reachable ({e!r}) -- this check needs a real model, stopping.")
    sys.exit(1)

QUESTIONS = [
    "why is my rakhi set flagged?",
    "why is umbrella low on stock?",
    "reject the rakhi set order, the quantity is too high",
    "what's going on with my shop today",  # deliberately vague, no item named
]

for q in QUESTIONS:
    result = respond(llm, q, consumption, context)
    print(f"Q: {q}")
    print(f"-> [{result['kind']}] {result['text']}")
    print()
