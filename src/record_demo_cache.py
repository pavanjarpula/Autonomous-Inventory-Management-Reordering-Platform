"""
Records real qwen3:4b responses for the fixed demo script into
data/demo_llm_cache.json -- run this once during rehearsal (or whenever the
demo script/dataset changes), then the dashboard's default "cached" mode
replays these instead of depending on a live model call working on stage.

Covers exactly the walkthrough: the graph's reasoning step for the demo date,
and the chatbot questions from run_chatbot_check.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chatbot import respond
from context_agent import get_context
from engine import assess_item, days_to_next_arrival, on_order_qty
from graph import build_graph
from llm_cache import CachedLLM

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE_PATH = DATA / "demo_llm_cache.json"

AS_OF = "2026-08-24"
CHAT_QUESTIONS = [
    "why is my rakhi set flagged?",
    "why is umbrella low on stock?",
    "why toothpaste is recommended",
    "do I need to order more toothpaste?",
    "reject the rakhi set order, the quantity is too high",
]

sales = pd.read_csv(DATA / "daily_sales.csv", parse_dates=["date"])
items = pd.read_csv(DATA / "items.csv")
suppliers = pd.read_csv(DATA / "suppliers.csv")
festival_calendar = pd.read_csv(DATA / "festival_calendar.csv", parse_dates=["date"])
festival_overrides = pd.read_csv(DATA / "festival_item_overrides.csv")
promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["date"])
purchase_orders = pd.read_csv(DATA / "purchase_orders.csv", parse_dates=["date_ordered", "date_expected"])

from llm_provider import make_llm, resolve_provider

provider = resolve_provider()
real_llm = make_llm(provider)
real_llm.invoke("ping")  # fail fast if the provider isn't reachable
print(f"{provider} reachable, recording...\n")

cache = CachedLLM(CACHE_PATH, real_llm=real_llm, record=True)

print(f"1. Reasoning graph for {AS_OF}")
# purchase_orders MUST be passed here for the same reason the dashboard passes it:
# cache entries are keyed by the exact prompt, and in-transit stock changes the
# facts block. Recording without it produces a cache that misses on every entry.
graph = build_graph(sales, items, suppliers, festival_calendar, festival_overrides, promotions, cache,
                     purchase_orders=purchase_orders)
result = graph.invoke({"as_of_date": AS_OF}, {"configurable": {"thread_id": f"record-{AS_OF}"}})
n = len(result["ranked_recommendations"])
print(f"   recorded reasoning for {n} recommendations\n")

print("2. Chatbot questions")
# every flagged item, matching exactly what the dashboard's chat panel passes in --
# a narrower list here would silently fail to resolve items the seller can actually
# ask about, and record an "unclear" answer for a question that works live
as_of_ts = pd.Timestamp(AS_OF)
consumption, context = {}, {}
for item_id in items["item_id"]:
    assessment = assess_item(item_id, as_of_ts, sales, items, suppliers,
                              on_order=on_order_qty(item_id, as_of_ts, purchase_orders),
                              arriving_in_days=days_to_next_arrival(item_id, as_of_ts, purchase_orders))
    if assessment["risk"] == "healthy":
        continue
    consumption[item_id] = assessment
    context[item_id] = get_context(item_id, as_of_ts, items, festival_calendar, festival_overrides, promotions)

for q in CHAT_QUESTIONS:
    r = respond(cache, q, consumption, context)
    print(f"   \"{q}\" -> [{r['kind']}]")

cache.save()
print(f"\nsaved {len(cache)} cached responses to {CACHE_PATH}")
