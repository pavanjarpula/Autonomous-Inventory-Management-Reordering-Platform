"""
Sanity check. Picks a date with real, varied at-risk SKUs -- a monsoon
stockout, a festival window, a growth-trend FMCG item -- and runs the full graph
against the actual dataset.

Tries a real local Ollama model first, since that's the only thing that can
validate what actually matters here: does the rationale correctly cite the SPECIFIC
reason (the festival, the trend, the low stock) rather than a generic restatement
of the numbers. If Ollama isn't reachable, falls back to a canned stub just to
prove the real-data wiring completes end to end -- clearly labeled as mechanical
proof only, not a substitute for the real check.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import RankedItem, RankedRecommendations, build_graph

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

sales = pd.read_csv(DATA / "daily_sales.csv", parse_dates=["date"])
items = pd.read_csv(DATA / "items.csv")
suppliers = pd.read_csv(DATA / "suppliers.csv")
festival_calendar = pd.read_csv(DATA / "festival_calendar.csv", parse_dates=["date"])
festival_overrides = pd.read_csv(DATA / "festival_item_overrides.csv")
promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["date"])
purchase_orders = pd.read_csv(DATA / "purchase_orders.csv", parse_dates=["date_ordered", "date_expected"])

AS_OF = "2026-08-24"  # mid-monsoon (Raincoat/Umbrella) + inside the Raksha Bandhan window (Rakhi Set)

try:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="qwen3:4b", temperature=0)
    llm.invoke("ping")  # fail fast here if Ollama isn't actually reachable
    print("using a real local Ollama model (qwen3:4b)\n")
except Exception as e:
    print(f"Ollama not reachable ({e!r}) -- falling back to a stub LLM.")
    print("This proves the graph wiring runs end to end on real data, NOT that the")
    print("rationale text is any good -- that check needs a real model. Install Ollama")
    print("and `ollama pull qwen3:4b`, then re-run this script.\n")

    class _StubStructured:
        def invoke(self, prompt):
            # crude but deterministic: high urgency for stockout_risk, medium otherwise
            import re
            items_in_prompt = re.findall(r"- (\w+) .*?risk=(\w+)", prompt)
            return RankedRecommendations(recommendations=[
                RankedItem(item_id=iid, urgency="high" if risk == "stockout_risk" else "medium",
                           rationale="(stub LLM -- Ollama not connected, see message above)")
                for iid, risk in items_in_prompt
            ])

    class _StubLLM:
        def with_structured_output(self, schema):
            return _StubStructured()

    llm = _StubLLM()

graph = build_graph(sales, items, suppliers, festival_calendar, festival_overrides, promotions, llm,
                     purchase_orders=purchase_orders)
config = {"configurable": {"thread_id": f"check-{AS_OF}"}}
result = graph.invoke({"as_of_date": AS_OF}, config)

recs = result["ranked_recommendations"]
total_flagged = result["total_flagged_count"]
print(f"as of {AS_OF}: {total_flagged} items flagged (out of {len(items)} total) -- showing top {len(recs)}\n")
for r in recs:
    print(f"[{r['urgency'].upper():6s}] {r['item_id']} {r['item_name']} ({r['category']})")
    print(f"         qty={r['suggested_order_qty']}  days_of_cover={r['days_of_cover']}  risk={r['risk']}")
    print(f"         \"{r['rationale']}\"")
    print()
