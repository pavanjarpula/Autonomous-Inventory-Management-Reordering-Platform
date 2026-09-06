"""
The Reasoning Agent and Orchestrator. The only place an LLM makes a judgement:
engine.py, context_agent.py and forecast.py are tools the graph calls, never
decision-makers.

Graph shape:
  START -> gather_signals -> reasoning -> human_approval (interrupt) -> END

The graph halts at human_approval and hands the ranked list to whatever is driving
it. Resume works -- Command(resume=...) returns the decision into state, and
test_graph.py exercises it -- but the DASHBOARD does not use it, deliberately.
interrupt() here is a single batch-level pause covering the whole list, while the
seller decides per item; approving one recommendation and rejecting another is not
one resume value. The dashboard therefore reads the interrupt payload and applies
each decision through feedback_agent.submit_feedback() directly.

An earlier version carried an apply_decision node after human_approval that
returned {} and did nothing at all. It is gone. If per-item resume is ever wanted,
the honest shape is one interrupt per item, not a node bolted onto this one.

Two rules carried over from every earlier phase, enforced structurally here:
  - the LLM never computes a quantity. assess_item() already computed
    suggested_order_qty; the reasoning node's job is to rank and explain, and the
    merge step below pulls the quantity back from the deterministic facts by
    item_id lookup, never from the model's own output.
  - a hallucinated item_id (one the model invents that wasn't in the input) is
    silently dropped, not trusted.

LangSmith integration: all graph nodes are traceable. Set LANGCHAIN_TRACING_V2=true
and LANGCHAIN_API_KEY to enable tracing.
"""

import re
import sys
from pathlib import Path
from typing import Literal, Optional, TypedDict

import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type="chain"):
        def decorator(func):
            return func
        return decorator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_agent import get_context
from engine import assess_item, days_to_next_arrival, on_order_qty
from logger import get_logger

logger = get_logger(__name__, extra_data={"module": "graph"})


class SellerSenseState(TypedDict):
    as_of_date: str
    consumption_signals: dict
    context_signals: dict
    total_flagged_count: int
    ranked_recommendations: list[dict]
    seller_decision: Optional[dict]
    run_metadata: Optional[dict]  # per-run context for tracing


class RankedItem(BaseModel):
    item_id: str = Field(description="the item_id exactly as given -- never invent a new one")
    urgency: Literal["high", "medium", "low"]
    rationale: str = Field(
        description="one or two plain-language sentences a busy shop owner can read in five "
        "seconds, citing the SPECIFIC reason from the facts given (a festival, a growth trend, "
        "low stock, a slow-moving supplier) -- not a generic restatement of the numbers"
    )


class RankedRecommendations(BaseModel):
    recommendations: list[RankedItem]


# ---------------------------------------------------------------- gather_signals

def _severity_key(c: dict) -> tuple:
    """Sorts stockout risk ahead of overstock, then by days of cover ascending --
    None means infinite cover (nothing on hand to run out of, or no demand at all),
    which must sort LAST, not first, so it doesn't look falsely urgent."""
    cover = c["days_of_cover"]
    return (c["risk"] != "stockout_risk", cover if cover is not None else float("inf"))


def make_gather_signals_node(sales, items, suppliers, festival_calendar, festival_overrides,
                              promotions, max_items: int = 8, params_by_item: dict | None = None,
                              purchase_orders=None):
    """Deterministic, no LLM. Filters to items actually worth the Reasoning node's
    attention -- healthy items never reach the model at all -- then caps to the
    max_items most severe. Two independent reasons for the cap: a
    seller reviewing 18 alerts at once defeats the "top few, not a flood" premise
    of the product, and a live model takes noticeably longer (and has more room to
    silently omit something) the more items it has to handle in one batch. The
    full flagged count still travels in state so nothing pretends fewer items
    needed attention than actually did.

    params_by_item threads the Feedback Agent's per-item adjustments (z, lead_time_buffer_days)
    back into the assessment -- without this, a seller's feedback would change
    what the Inventory tab shows but never actually reach a future recommendation,
    which defeats the entire point of the Feedback Agent existing.

    purchase_orders nets off stock already in transit. Optional so existing callers
    and tests keep working, but the dashboard passes it: without it the same order
    is recommended every morning until the goods physically arrive."""
    params_by_item = params_by_item or {}

    @traceable(name="gather_signals", run_type="chain")
    def gather_signals(state: SellerSenseState) -> dict:
        as_of = pd.Timestamp(state["as_of_date"])
        flagged = {}
        for item_id in items["item_id"]:
            params = params_by_item.get(item_id, {})
            assessment = assess_item(item_id, as_of, sales, items, suppliers,
                                      z=params.get("z", 1.65),
                                      lead_time_buffer_days=params.get("lead_time_buffer_days", 0.0),
                                      on_order=on_order_qty(item_id, as_of, purchase_orders),
                                      arriving_in_days=days_to_next_arrival(item_id, as_of, purchase_orders))
            if assessment["risk"] == "healthy":
                continue
            flagged[item_id] = assessment

        top_ids = sorted(flagged, key=lambda iid: _severity_key(flagged[iid]))[:max_items]
        consumption = {iid: flagged[iid] for iid in top_ids}
        context = {
            iid: get_context(iid, as_of, items, festival_calendar, festival_overrides, promotions)
            for iid in top_ids
        }
        return {
            "consumption_signals": consumption,
            "context_signals": context,
            "total_flagged_count": len(flagged),
        }
    return gather_signals


# ---------------------------------------------------------------- reasoning

def _trend_note(c: dict) -> str | None:
    """Surfaces engine.py's recent_trend_ratio() as a plain-language note -- this
    is the signal that catches a seasonal ramp like the Raincoat/Umbrella monsoon
    case, which the festival calendar alone has no way to see (no festival is
    involved, it's pure seasonality). Only worth mentioning when it's a genuine
    departure from the item's own usual rate, not routine day-to-day noise."""
    ratio = c.get("trend_ratio")
    if ratio is None:
        return None
    if ratio >= 1.5:
        return f"recent demand is running {ratio}x its usual rate"
    if ratio <= 0.5:
        return f"recent demand is running at only {ratio}x its usual rate"
    return None


def _format_item_facts(item_id: str, consumption: dict, context: dict) -> str:
    c = consumption[item_id]
    ctx = context.get(item_id, {})
    line = (
        f"- {item_id} {c['item_name']} ({c['category']}): risk={c['risk']}, on_hand={c['on_hand']}, "
        f"days_of_cover={c['days_of_cover']}, suggested_order_qty={c['suggested_order_qty']}"
    )
    # stock already in transit is the difference between "order this now" and
    # "it is handled, watch the date" -- the model has no way to tell those apart
    # from a quantity of 0 alone
    if c.get("on_order"):
        arriving = c.get("arriving_in_days")
        when = f", arriving in {arriving} day(s)" if arriving is not None else ""
        line += f"\n  {c['on_order']} unit(s) already on order{when}"
    has_festival_signal = False
    if ctx.get("is_festival_window"):
        line += f"\n  currently inside the {ctx['active_festival']} window (uplift x{ctx['uplift_multiplier']})"
        has_festival_signal = True
    elif ctx.get("days_to_next_festival") is not None:
        line += f"\n  {ctx['next_festival_name']} is {ctx['days_to_next_festival']} days away"
        has_festival_signal = True
    trend = _trend_note(c)
    if trend:
        # only claim "no festival involved" when that's actually true for this item --
        # a festival-active item can also have a real trend signal on top of it
        qualifier = "" if has_festival_signal else " (no festival involved -- likely a seasonal or demand shift)"
        line += f"\n  {trend}{qualifier}"
    return line


def _plain_stock_sentence(c: dict) -> str:
    """Stock position in one sentence, no context signals -- shared by the
    model-outage fallback so it never contradicts the grounded rationale used
    everywhere else."""
    s = f"On hand: {c['on_hand']} units, {c['days_of_cover']} days of cover."
    if c.get("on_order"):
        arriving = c.get("arriving_in_days")
        when = f", arriving in {arriving} day(s)" if arriving is not None else ""
        s += f" {c['on_order']} unit(s) already on order{when}."
    return s


def _fallback_ranking(consumption: dict) -> list[dict]:
    """Used only if the LLM fails on every retry -- deterministic, no rationale
    beyond saying so plainly, but never blocks the graph on a model outage."""
    ranked = sorted(consumption.values(), key=_severity_key)
    return [
        dict(
            item_id=c["item_id"], item_name=c["item_name"], category=c["category"],
            urgency="high" if c["risk"] == "stockout_risk" else "medium",
            rationale=_plain_stock_sentence(c),
            suggested_order_qty=c["suggested_order_qty"], days_of_cover=c["days_of_cover"], risk=c["risk"],
            on_order=c.get("on_order", 0), arriving_in_days=c.get("arriving_in_days"),
            source="fallback",
        )
        for c in ranked
    ]


def _safe_fallback_rationale(item_id: str, consumption: dict, context: dict) -> str:
    """Built entirely from verified facts, no LLM involved -- used whenever the
    model's own rationale fails the grounding check below, so a seller never sees
    a fabricated cause even when the ranking/urgency itself is fine. Covers both
    signal types available: the festival calendar and the recent-trend
    proxy -- the latter is what catches a Raincoat/Umbrella-style
    seasonal ramp that has no festival behind it at all."""
    c = consumption[item_id]
    ctx = context.get(item_id, {})
    parts = [f"On hand: {c['on_hand']} units, {c['days_of_cover']} days of cover."]
    if c.get("on_order"):
        arriving = c.get("arriving_in_days")
        when = f", arriving in {arriving} day(s)" if arriving is not None else ""
        parts.append(f"{c['on_order']} unit(s) already on order{when}.")
    if ctx.get("is_festival_window"):
        parts.append(f"Currently inside the {ctx['active_festival']} window (demand uplift x{ctx['uplift_multiplier']}).")
    elif ctx.get("days_to_next_festival") is not None:
        parts.append(f"{ctx['next_festival_name']} is {ctx['days_to_next_festival']} days away.")
    trend = _trend_note(c)
    if trend:
        parts.append(trend.capitalize() + ".")
    return " ".join(parts)


_TREND_WORDS = re.compile(
    r"\b(trend(ing)?|surg\w+|spik\w+|rising|risen|climb\w+|soar\w+|"
    r"falling|fallen|declin\w+|slow\w+ down|picking up)\b", re.IGNORECASE)
_SUPPLIER_WORDS = re.compile(
    r"\b(supplier|vendor|lead[\s-]?time|restock\w* delay|delivery delay|"
    r"unreliable|shipment delay)\b", re.IGNORECASE)


def _is_rationale_grounded(item_id: str, rationale: str, consumption: dict, context: dict,
                            all_festival_names: set[str]) -> bool:
    """
    Three checks, one per class of claim the facts block can actually support.

    1. Borrowed festival. The original failure: a smaller model batching many items
       in one prompt pattern-matches a festival name from a DIFFERENT item onto this
       one. True names for this item are allowed; any other real festival name in the
       text means it borrowed a reason that isn't this item's own.
    2. Invented trend. _format_item_facts only emits a trend line when
       _trend_note() fires. A rationale claiming a surge or a decline for an item
       that has no trend signal is asserting something it was never told.
    3. Invented supplier problem. The facts block never mentions supplier
       reliability or lead time at all, so ANY supplier claim is necessarily
       fabricated -- no allow-list needed, the whole class is out of bounds.

    Deliberately not checked: the numbers. Those are merged back from the
    deterministic dict by item_id lookup and never read from the model's text, so
    a number in a rationale can only be right or ignorable, never load-bearing.
    """
    ctx = context.get(item_id, {})
    true_names = {ctx.get("active_festival"), ctx.get("next_festival_name")} - {None}
    if any(name in rationale for name in (all_festival_names - true_names) if name):
        return False

    if _trend_note(consumption[item_id]) is None and _TREND_WORDS.search(rationale):
        return False

    if _SUPPLIER_WORDS.search(rationale):
        return False

    return True


def make_reasoning_node(llm, festival_calendar: pd.DataFrame, max_attempts: int = 3):
    structured_llm = llm.with_structured_output(RankedRecommendations)
    all_festival_names = set(festival_calendar["festival_name"])

    @traceable(name="reasoning", run_type="llm")
    def reasoning(state: SellerSenseState) -> dict:
        consumption, context = state["consumption_signals"], state["context_signals"]
        if not consumption:
            return {"ranked_recommendations": []}

        facts = "\n".join(_format_item_facts(iid, consumption, context) for iid in consumption)
        prompt = (
            "You are a retail inventory analyst reviewing today's flagged items for a small "
            "store owner. The facts below are already computed -- do not recompute or second-"
            "guess the numbers, and do not state a quantity yourself. For each item, assign an "
            "urgency (high/medium/low) and write a one-to-two sentence rationale citing the "
            "specific reason from THAT item's OWN facts only -- never mention a festival, trend, "
            "or supplier issue that isn't explicitly listed for this exact item, even if it was "
            "mentioned for another item above. When an item's facts do name an active festival, "
            "lead with it and give its name: an approaching event is the most useful thing a shop "
            "owner can be told. Rank most urgent first.\n\n" + facts
        )

        result = None
        for _ in range(max_attempts):
            try:
                result = structured_llm.invoke(prompt)
                break
            except Exception:
                continue

        if result is None:
            return {"ranked_recommendations": _fallback_ranking(consumption)}

        merged = []
        seen_ids = set()
        for r in result.recommendations:
            if r.item_id not in consumption:
                continue  # hallucinated id -- drop it, never trust an id we didn't hand in
            c = consumption[r.item_id]
            rationale = r.rationale
            if not _is_rationale_grounded(r.item_id, rationale, consumption, context, all_festival_names):
                rationale = _safe_fallback_rationale(r.item_id, consumption, context)
            merged.append(dict(
                item_id=r.item_id, item_name=c["item_name"], category=c["category"],
                urgency=r.urgency, rationale=rationale,
                suggested_order_qty=c["suggested_order_qty"], days_of_cover=c["days_of_cover"], risk=c["risk"], on_order=c.get("on_order", 0), arriving_in_days=c.get("arriving_in_days"),
                source="model",
            ))
            seen_ids.add(r.item_id)

        # a model handling a long list can silently omit items rather than erroring --
        # a long list makes this more likely. Never let a flagged item
        # vanish just because the model's response didn't mention it.
        for item_id, c in consumption.items():
            if item_id in seen_ids:
                continue
            merged.append(dict(
                item_id=item_id, item_name=c["item_name"], category=c["category"],
                urgency="high" if c["risk"] == "stockout_risk" else "medium",
                rationale=_safe_fallback_rationale(item_id, consumption, context),
                suggested_order_qty=c["suggested_order_qty"], days_of_cover=c["days_of_cover"], risk=c["risk"], on_order=c.get("on_order", 0), arriving_in_days=c.get("arriving_in_days"),
                source="backfill",
            ))
        return {"ranked_recommendations": merged}
    return reasoning


# ---------------------------------------------------------------- human approval

@traceable(name="human_approval", run_type="chain")
def human_approval(state: SellerSenseState) -> dict:
    decision = interrupt({
        "recommendations": state["ranked_recommendations"],
        "prompt": "Approve, reject (with a reason), or snooze each recommendation.",
    })
    return {"seller_decision": decision}


# ---------------------------------------------------------------- graph assembly

@traceable(name="build_graph", run_type="chain")
def build_graph(
    sales, items, suppliers, festival_calendar, festival_overrides, 
    promotions, llm, max_items: int = 8, params_by_item: dict | None = None, 
    purchase_orders=None
):
    """Build the LangGraph orchestrator.
    
    Tracing metadata can be passed at invoke time via the config parameter:
        graph.invoke(state, config=RunnableConfig(
            tags=["dashboard", "date-2026-08-24"],
            metadata={"provider": "groq", "n_items": 25}
        ))
    """
    logger.info("Building graph", extra={"extra_data": {
        "max_items": max_items,
    }})
    
    graph = StateGraph(SellerSenseState)
    graph.add_node("gather_signals", make_gather_signals_node(
        sales, items, suppliers, festival_calendar, festival_overrides, promotions, max_items,
        params_by_item, purchase_orders))
    graph.add_node("reasoning", make_reasoning_node(llm, festival_calendar))
    graph.add_node("human_approval", human_approval)

    graph.add_edge(START, "gather_signals")
    graph.add_edge("gather_signals", "reasoning")
    graph.add_edge("reasoning", "human_approval")
    graph.add_edge("human_approval", END)

    # A checkpointer is required for interrupt() to work at all, but nothing
    # resumes from it -- see the module docstring. InMemorySaver is the right
    # choice precisely because that state is not meant to outlive the call.
    return graph.compile(checkpointer=InMemorySaver())
