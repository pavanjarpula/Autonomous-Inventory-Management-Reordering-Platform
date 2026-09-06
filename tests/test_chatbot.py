import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import (
    ChatAnswer,
    ChatIntent,
    answer_question,
    classify_intent,
    plain_facts,
    resolve_item_id,
    respond,
)


class _Structured:
    def __init__(self, response=None, raises=False):
        self.response = response
        self.raises = raises

    def invoke(self, prompt):
        if self.raises:
            raise ValueError("simulated LLM failure")
        return self.response


class _FakeLLM:
    """Hands back a different canned response depending on which schema was
    requested, since respond() can chain classify_intent() then answer_question()
    -- each asks with_structured_output() for a different schema in sequence."""
    def __init__(self, intent=None, answer=None, intent_raises=False, answer_raises=False):
        self.intent = intent
        self.answer = answer
        self.intent_raises = intent_raises
        self.answer_raises = answer_raises

    def with_structured_output(self, schema):
        if schema is ChatIntent:
            return _Structured(self.intent, raises=self.intent_raises)
        if schema is ChatAnswer:
            return _Structured(self.answer, raises=self.answer_raises)
        raise AssertionError(f"unexpected schema requested: {schema}")


CONSUMPTION = {
    "I021": dict(item_id="I021", item_name="Raincoat", category="Durables",
                 risk="stockout_risk", on_hand=0, days_of_cover=0.0, reorder_point=39,
                 suggested_order_qty=39, trend_ratio=None),
    "I016": dict(item_id="I016", item_name="Rakhi Set", category="Festive",
                 risk="stockout_risk", on_hand=0, days_of_cover=0.0, reorder_point=2,
                 suggested_order_qty=2, trend_ratio=15.6),
    "I013": dict(item_id="I013", item_name="Toothpaste 150g", category="FMCG",
                 risk="stockout_risk", on_hand=1, days_of_cover=0.3, reorder_point=39,
                 suggested_order_qty=38, trend_ratio=None),
}
CONTEXT = {
    "I021": dict(is_festival_window=False, days_to_next_festival=None),
    "I016": dict(is_festival_window=True, active_festival="Raksha Bandhan", uplift_multiplier=14.0),
    "I013": dict(is_festival_window=False, days_to_next_festival=None),
}


# ---------------------------------------------------------------- resolve_item_id (deterministic, no LLM)

def test_resolve_item_id_matches_by_name_case_insensitively():
    assert resolve_item_id("why is my Rakhi Set low?", CONSUMPTION) == "I016"
    assert resolve_item_id("RAINCOAT stock?", CONSUMPTION) == "I021"


def test_resolve_item_id_matches_without_the_size_suffix():
    # a seller says "toothpaste", never "Toothpaste 150g" -- requiring the full
    # name including the size made every realistic question fail to resolve
    assert resolve_item_id("why toothpaste is recommended", CONSUMPTION) == "I013"
    assert resolve_item_id("do I need more toothpaste?", CONSUMPTION) == "I013"


def test_resolve_item_id_still_matches_the_full_name_with_size():
    assert resolve_item_id("Toothpaste 150g is required?", CONSUMPTION) == "I013"


def test_resolve_item_id_matches_by_bare_id():
    assert resolve_item_id("what about I021", CONSUMPTION) == "I021"


def test_resolve_item_id_returns_none_when_nothing_matches():
    assert resolve_item_id("what about my umbrellas", CONSUMPTION) is None


def test_resolve_item_id_matches_whole_words_only():
    # "tea" must not match inside "steal"/"instead"
    assert resolve_item_id("instead of that, what should I steal", CONSUMPTION) is None


def test_resolve_item_id_returns_none_when_ambiguous():
    assert resolve_item_id("raincoat and rakhi set both need restocking", CONSUMPTION) is None


# ---------------------------------------------------------------- classify_intent

def test_classify_intent_passes_through_the_structured_result():
    canned = ChatIntent(kind="question")
    result = classify_intent(_FakeLLM(intent=canned), "why is it low", "Raincoat")
    assert result.kind == "question"


def test_classify_intent_falls_back_to_unclear_on_llm_failure():
    result = classify_intent(_FakeLLM(intent_raises=True), "anything", "Raincoat")
    assert result.kind == "unclear"


# ---------------------------------------------------------------- answer_question

def test_answer_question_falls_back_to_facts_when_llm_fails():
    text = answer_question(_FakeLLM(answer_raises=True), "why is this low?", "I021", CONSUMPTION, CONTEXT)
    assert "Raincoat" in text
    assert "In stock right now: 0 units" in text  # fact-only fallback, in plain language


# ---------------------------------------------------------------- plain_facts

def test_plain_facts_spells_out_what_flagged_actually_means():
    # the model can't answer "do I need to order this?" from `risk=stockout_risk`
    # alone -- the facts have to say what that flag implies
    facts = plain_facts("I013", CONSUMPTION, CONTEXT)
    assert "recommending reordering 38 units" in facts
    assert "0.3 days of stock left" in facts


def test_plain_facts_includes_an_active_festival_and_its_uplift():
    facts = plain_facts("I016", CONSUMPTION, CONTEXT)
    assert "Raksha Bandhan is on right now" in facts
    assert "14.0x" in facts


def test_plain_facts_has_no_raw_field_names():
    facts = plain_facts("I021", CONSUMPTION, CONTEXT)
    for jargon in ("risk=", "on_hand=", "days_of_cover=", "suggested_order_qty="):
        assert jargon not in facts


# ---------------------------------------------------------------- respond (the full loop)

def test_respond_resolves_the_item_by_name_before_ever_calling_the_llm():
    llm = _FakeLLM(intent=ChatIntent(kind="question"), answer=ChatAnswer(answer="Zero stock, no cover."))
    result = respond(llm, "why is rakhi set flagged?", CONSUMPTION, CONTEXT)
    assert result["kind"] == "answer"
    assert "Zero stock" in result["text"]


def test_respond_is_unclear_immediately_when_no_item_name_matches_no_llm_call_needed():
    # the LLM is never given a response here -- if respond() tried to call it,
    # the fake would raise AssertionError from the unexpected-schema branch
    result = respond(_FakeLLM(), "how's business today", CONSUMPTION, CONTEXT)
    assert result["kind"] == "unclear"


def test_respond_detects_a_command_without_executing_it():
    llm = _FakeLLM(intent=ChatIntent(kind="command", command_action="reject", reject_reason="qty_too_high"))
    result = respond(llm, "reject the rakhi set order, the quantity is too high", CONSUMPTION, CONTEXT)
    assert result["kind"] == "command"
    assert result["item_id"] == "I016"
    assert result["action"] == "reject"
    assert result["reject_reason"] == "qty_too_high"
    # respond() never imports or calls anything from feedback_agent -- detection only


def test_respond_ignores_a_reject_reason_the_seller_never_gave():
    # a small model fills in optional enum fields on nearly every call. A reason
    # invented here would land in the audit log as the seller's own words and
    # drive a real parameter change, so it is only ever read from the message.
    llm = _FakeLLM(intent=ChatIntent(kind="command", command_action="reject", reject_reason="not_needed_now"))
    result = respond(llm, "reject the raincoat", CONSUMPTION, CONTEXT)  # no reason stated
    assert result["kind"] == "unclear"
    assert "reason" in result["text"].lower()


def test_respond_reads_the_action_from_the_sellers_words():
    # the model returning no action at all must not block a plainly worded command
    llm = _FakeLLM(intent=ChatIntent(kind="command", command_action=None, reject_reason=None))
    result = respond(llm, "reject the raincoat, supplier is always late", CONSUMPTION, CONTEXT)
    assert result["kind"] == "command"
    assert result["action"] == "reject"
    assert result["reject_reason"] == "supplier_unreliable"


def test_respond_does_not_render_a_bogus_none_action():
    # kind=="command" with command_action left unset by the model must
    # not silently render as "Got it -- None on <item>" -- ask instead of guessing
    llm = _FakeLLM(intent=ChatIntent(kind="command", command_action=None))
    result = respond(llm, "do something about the raincoat", CONSUMPTION, CONTEXT)
    assert result["kind"] == "unclear"
    assert "None" not in result["text"]


def test_respond_asks_for_a_reason_when_reject_has_none():
    llm = _FakeLLM(intent=ChatIntent(kind="command", command_action="reject", reject_reason=None))
    result = respond(llm, "reject the raincoat", CONSUMPTION, CONTEXT)
    assert result["kind"] == "unclear"
    assert "reason" in result["text"].lower()


def test_respond_handles_unclear_intent_gracefully():
    llm = _FakeLLM(intent=ChatIntent(kind="unclear"))
    result = respond(llm, "raincoat, hmm", CONSUMPTION, CONTEXT)
    assert result["kind"] == "unclear"
