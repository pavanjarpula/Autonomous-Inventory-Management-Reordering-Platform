"""
The Chatbot. Answers seller questions from what the rest of the system already
computed -- not a new reasoning surface, and explicitly not a second place
inventory decisions get made.

Which item a message is about is resolved deterministically (resolve_item_id),
not asked of the model -- that's a lookup, not a judgment call, the same reason
quantities and item IDs have stayed out of the LLM's hands in every phase so
far. Once the item is known, one structured LLM call (classify_intent) decides
whether the message is a question or an explicit command, and for a question,
answer_question() answers it grounded in the SAME _format_item_facts() the
Reasoning Agent uses, so the chatbot and the Reasoning Agent can never quietly
disagree about the same item.

A detected command is returned, never executed. It comes back shaped exactly
like feedback_agent.submit_feedback()'s arguments, so a caller who decides to
act on it can pass it straight through -- but that decision, and the actual
side effect, stays outside this module. Same rule as every earlier phase: the
LLM proposes, something else decides.

No retrieval, no embeddings: the whole flagged set plus recent feedback history
comfortably fits in a prompt at this scale -- a vector index here would be
complexity with no payoff.

LangSmith integration: all LLM calls are traceable via @traceable decorators.
Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable tracing.
"""

import re
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type="chain"):
        def decorator(func):
            return func
        return decorator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feedback_agent import REASON_CODES
from graph import _trend_note
from logger import get_logger

logger = get_logger(__name__, extra_data={"module": "chatbot"})


class ChatIntent(BaseModel):
    kind: Literal["question", "command", "unclear"]
    command_action: Optional[Literal["approve", "reject", "snooze"]] = Field(
        default=None, description="set only when kind == 'command'"
    )
    reject_reason: Optional[Literal[*REASON_CODES]] = Field(
        default=None, description="set only for a reject command, if a reason is stated or clearly implied"
    )


class ChatAnswer(BaseModel):
    answer: str = Field(
        description="a short, plain-language answer grounded ONLY in the facts given -- "
        "if the facts don't actually answer the question, say so plainly instead of guessing"
    )


_MEASUREMENT = re.compile(r"^\d+(\.\d+)?(g|kg|ml|l|gm|ltr)$")
_GENERIC_TOKENS = {"pack", "set", "box", "small", "large", "of", "and", "assorted"}


def _stem(word: str) -> str:
    """Minimal stemmer: strip trailing 's', 'es', 'ies' so 'biscuit' matches
    'biscuits', 'battery' matches 'batteries'. Not a real stemmer -- just
    enough for the 25 item names in this dataset."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3]
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _content_tokens(name: str) -> set[str]:
    """The words in an item name that actually identify it. Drops sizes ("150g",
    "5kg") and generic packaging words ("pack", "set", "box") -- nobody asks
    about their "Toothpaste 150g", they ask about toothpaste, and "box" on its
    own points at three different items."""
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    return {
        _stem(t) for t in tokens
        if t not in _GENERIC_TOKENS and not _MEASUREMENT.match(t) and not t.isdigit()
    }


def resolve_item_id(message: str, consumption: dict) -> Optional[str]:
    """
    Deterministic name matching, no LLM. Scores each item by how many of its
    identifying words appear in the message, and takes the clear winner -- so
    "why is toothpaste low" finds "Toothpaste 150g" without the seller typing
    the size. Whole-word matching only, so "tea" doesn't match "steal". The
    bare item_id matches too, for a UI that already knows it. Returns None --
    never a guess -- when nothing matches, or when two items tie (a bare "gift"
    could mean either the Gift Hamper or the Sweets Gift Box).
    """
    words = {_stem(w) for w in re.findall(r"[a-z0-9]+", message.lower())}

    for item_id in consumption:
        if item_id.lower() in words:
            return item_id

    scores = {iid: len(_content_tokens(c["item_name"]) & words) for iid, c in consumption.items()}
    best = max(scores.values(), default=0)
    if best == 0:
        return None
    winners = [iid for iid, s in scores.items() if s == best]
    return winners[0] if len(winners) == 1 else None


_ACTION_CUES = {
    "reject": ("reject", "decline", "cancel", "skip", "don't order", "dont order", "no thanks"),
    "approve": ("approve", "confirm", "go ahead", "place the order", "order it", "yes do it"),
    "snooze": ("snooze", "later", "remind me", "not now", "tomorrow"),
}

_REASON_CUES = {
    "qty_too_high": ("too high", "too many", "too much", "fewer", "less than", "reduce", "lower the qty"),
    "qty_too_low": ("too low", "too few", "not enough", "more than that", "increase"),
    "supplier_unreliable": ("supplier", "late", "delay", "unreliable", "never delivers"),
    "not_needed_now": ("not needed", "don't need", "dont need", "no cash", "no money", "next week"),
}


def _cue_match(message: str, cues: dict) -> Optional[str]:
    lowered = message.lower()
    hits = [key for key, phrases in cues.items() if any(p in lowered for p in phrases)]
    return hits[0] if len(hits) == 1 else None


def extract_reject_reason(message: str) -> Optional[str]:
    """
    Read the reason out of what the seller actually wrote. Deliberately never
    taken from the model: a small model fills in optional enum fields on nearly
    every call, and a fabricated reason here would land in the audit log as the
    seller's own words and drive a real parameter change. No cue in the message
    means no reason -- ask them, don't guess.
    """
    return _cue_match(message, _REASON_CUES)


def extract_command_action(message: str) -> Optional[str]:
    """Deterministic first pass at what the seller is asking for. Unlike the
    reason, a wrong guess here only produces a confirmation prompt, so the
    model's own answer is still usable as a fallback when no cue matches."""
    return _cue_match(message, _ACTION_CUES)


@traceable(name="classify_intent", run_type="llm")
def classify_intent(llm, message: str, item_name: str) -> ChatIntent:
    prompt = (
        f"A shop owner sent this message about \"{item_name}\", an item currently flagged "
        "by their inventory copilot. Classify it.\n"
        "'command' means they're telling the system to DO something right now (approve an "
        "order, reject a recommendation, snooze it). 'question' means they're asking about "
        "it. 'unclear' if it's neither.\n\n"
        f"Message: \"{message}\""
    )
    try:
        return llm.with_structured_output(ChatIntent).invoke(prompt)
    except Exception:
        return ChatIntent(kind="unclear")


def plain_facts(item_id: str, consumption: dict, context: dict) -> str:
    """
    The same underlying numbers the Reasoning Agent gets, written as sentences
    instead of key=value pairs. The terse form is fine for ranking, where the
    model is sorting known fields; it is not fine for answering a person's
    question, where the model has to know that "flagged" means "reorder this"
    before it can say anything useful. Spelling that out here is what turns
    "the facts do not specify whether this is required" into a real answer.
    """
    c = consumption[item_id]
    ctx = context.get(item_id, {})
    lines = [f"Item: {c['item_name']} ({c['category']} category)",
             f"In stock right now: {c['on_hand']} units"]

    if c["days_of_cover"] is not None:
        lines.append(f"At the current selling rate that is roughly {c['days_of_cover']} days of stock left.")

    if c["risk"] == "stockout_risk":
        lines.append(
            f"This item is flagged as at risk of running out, which is why the system is "
            f"recommending reordering {c['suggested_order_qty']} units now. "
            f"It reorders once stock falls to {c['reorder_point']} units."
        )
    elif c["risk"] == "overstock":
        lines.append("This item is flagged as overstocked: there is more stock on hand than "
                     "the current selling rate justifies, so money is tied up in it.")
    else:
        lines.append(f"This item is healthy: stock is above its reorder point of "
                     f"{c['reorder_point']} units, so no order is needed right now.")

    if ctx.get("is_festival_window"):
        lines.append(f"{ctx['active_festival']} is on right now, which typically multiplies demand "
                     f"for this item by about {ctx['uplift_multiplier']}x.")
    elif ctx.get("days_to_next_festival") is not None:
        lines.append(f"{ctx['next_festival_name']} is {ctx['days_to_next_festival']} days away.")

    trend = _trend_note(c)
    if trend:
        lines.append(trend.capitalize() + ", compared with how it normally sells.")

    return "\n".join(lines)


@traceable(name="answer_question", run_type="llm")
def answer_question(llm, message: str, item_id: str, consumption: dict, context: dict) -> str:
    facts = plain_facts(item_id, consumption, context)
    prompt = (
        "You are an inventory assistant answering a shop owner's question about one item "
        "in their shop. Answer in one or two short, plain sentences they'd find useful -- "
        "no jargon, no field names.\n\n"
        "Use only the facts below and never invent a number. If they're asking whether they "
        "need to order something, the line about what the system is recommending is the answer. "
        "If they're asking why something was flagged, explain the reason in the facts (running "
        "low, a festival driving demand, or selling faster than usual).\n\n"
        f"Facts:\n{facts}\n\nTheir question: \"{message}\""
    )
    try:
        return llm.with_structured_output(ChatAnswer).invoke(prompt).answer
    except Exception:
        return f"Here's what I have on {c_name(consumption, item_id)}:\n{facts}"


def c_name(consumption: dict, item_id: str) -> str:
    return consumption[item_id]["item_name"]


@traceable(name="respond", run_type="chain")
def respond(llm, message: str, consumption: dict, context: dict) -> dict:
    """
    The single entry point. Returns one of:
      {"kind": "answer", "text": "..."}
      {"kind": "command", "item_id": ..., "action": ..., "reject_reason": ...,
       "text": "..."}                                   -- detected, NOT executed
      {"kind": "unclear", "text": "..."}
    """
    item_id = resolve_item_id(message, consumption)
    if item_id is None:
        return dict(kind="unclear", text="I'm not sure which item that's about -- try naming it directly.")

    item_name = consumption[item_id]["item_name"]
    intent = classify_intent(llm, message, item_name)

    if intent.kind == "command":
        # the seller's own words win over the model's for both fields; the reason
        # is taken ONLY from the message, never from the model (see extract_reject_reason)
        action = extract_command_action(message) or intent.command_action
        reason = extract_reject_reason(message)

        if action is None:
            return dict(kind="unclear",
                        text=f"I can tell this is about {item_name}, but not what you want done with it -- approve, reject, or snooze?")
        if action == "reject" and reason is None:
            return dict(kind="unclear",
                        text=f"Got that you want to reject {item_name} -- what's the reason? "
                             "(quantity too high, quantity too low, supplier unreliable, or not needed right now)")
        return dict(
            kind="command", item_id=item_id, action=action, reject_reason=reason,
            text=f"Got it -- {action} on {item_name}"
                 + (f" ({reason.replace('_', ' ')})" if reason else "") + ". Confirm to apply this.",
        )

    if intent.kind == "question":
        return dict(kind="answer", text=answer_question(llm, message, item_id, consumption, context))

    return dict(kind="unclear", text=f"I found {item_name} but I'm not sure what you're asking -- could you rephrase?")
