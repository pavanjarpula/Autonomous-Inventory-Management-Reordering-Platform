import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph import (
    RankedItem,
    RankedRecommendations,
    _format_item_facts,
    _trend_note,
    build_graph,
    make_gather_signals_node,
)


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def small_dataset():
    dates = pd.date_range("2026-06-01", periods=70)
    items = pd.DataFrame([
        dict(item_id="LOW", item_name="Low Stock Item", category="Staples", supplier_id="S1",
             unit_cost_inr=10, unit_price_inr=15),
        dict(item_id="OK", item_name="Healthy Item", category="Staples", supplier_id="S1",
             unit_cost_inr=10, unit_price_inr=15),
    ])
    suppliers = pd.DataFrame([dict(supplier_id="S1", supplier_name="Test Supplier",
                                    mean_lead_time_days=4, lead_time_std_days=1)])
    # LOW: steady demand, almost nothing on hand -> stockout_risk
    # OK: steady demand, plenty on hand -> healthy, should never reach the LLM
    rows = []
    for d in dates:
        rows.append(dict(date=d, item_id="LOW", units_sold=8, stockout_flag=0, on_hand_end=3))
        rows.append(dict(date=d, item_id="OK", units_sold=8, stockout_flag=0, on_hand_end=200))
    sales = pd.DataFrame(rows)
    festival_calendar = pd.DataFrame(columns=["date", "festival_name", "affected_categories", "uplift_multiplier", "ramp_days_before"])
    festival_overrides = pd.DataFrame(columns=["festival_name", "item_id", "uplift_multiplier"])
    promotions = pd.DataFrame(columns=["date", "item_id", "promo_uplift"])
    return dict(sales=sales, items=items, suppliers=suppliers, festival_calendar=festival_calendar,
                festival_overrides=festival_overrides, promotions=promotions, as_of=dates[-1])


class _FakeStructuredLLM:
    def __init__(self, response=None, raise_n_times=0):
        self.response = response
        self.raise_n_times = raise_n_times
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self.calls <= self.raise_n_times:
            raise ValueError("simulated structured-output parse failure")
        return self.response


class _FakeLLM:
    def __init__(self, response=None, raise_n_times=0):
        self._structured = _FakeStructuredLLM(response, raise_n_times)

    def with_structured_output(self, schema):
        return self._structured


# ---------------------------------------------------------------- trend note

def test_trend_note_fires_for_a_genuine_ramp():
    assert "3.2x" in _trend_note(dict(trend_ratio=3.2))


def test_trend_note_fires_for_a_genuine_drop():
    note = _trend_note(dict(trend_ratio=0.2))
    assert note is not None and "0.2x" in note


def test_trend_note_silent_for_ordinary_day_to_day_variation():
    assert _trend_note(dict(trend_ratio=1.1)) is None
    assert _trend_note(dict(trend_ratio=0.9)) is None


def test_trend_note_silent_when_no_signal_at_all():
    assert _trend_note(dict(trend_ratio=None)) is None


def test_format_item_facts_surfaces_a_notable_trend_with_no_festival_involved():
    # this is the Raincoat/Umbrella case: a real seasonal ramp with no festival
    # behind it at all -- the trend note is the ONLY way this reaches the model
    consumption = {"X": dict(item_id="X", item_name="Raincoat", category="Durables",
                              risk="stockout_risk", on_hand=0, days_of_cover=0.0,
                              suggested_order_qty=39, trend_ratio=3.6)}
    context = {"X": dict(is_festival_window=False, days_to_next_festival=None)}
    facts = _format_item_facts("X", consumption, context)
    assert "3.6x" in facts
    assert "no festival involved" in facts


def test_format_item_facts_does_not_claim_no_festival_when_one_is_active():
    # an item can be BOTH inside a festival window AND showing a real
    # trend signal at once (e.g. Rakhi Set, actively selling during its own
    # festival) -- the "no festival involved" caveat must not fire for it
    consumption = {"X": dict(item_id="X", item_name="Rakhi Set", category="Festive",
                              risk="stockout_risk", on_hand=0, days_of_cover=0.0,
                              suggested_order_qty=2, trend_ratio=15.6)}
    context = {"X": dict(is_festival_window=True, active_festival="Raksha Bandhan", uplift_multiplier=14.0)}
    facts = _format_item_facts("X", consumption, context)
    assert "15.6x" in facts
    assert "no festival involved" not in facts
    assert "Raksha Bandhan" in facts


def test_format_item_facts_stays_quiet_when_trend_is_ordinary(small_dataset):
    node = make_gather_signals_node(
        small_dataset["sales"], small_dataset["items"], small_dataset["suppliers"],
        small_dataset["festival_calendar"], small_dataset["festival_overrides"], small_dataset["promotions"],
    )
    out = node({"as_of_date": str(small_dataset["as_of"].date())})
    facts = _format_item_facts("LOW", out["consumption_signals"], out["context_signals"])
    assert "usual rate" not in facts  # LOW's demand is flat in the fixture -- nothing notable to say


# ---------------------------------------------------------------- gather_signals

def test_gather_signals_excludes_healthy_items(small_dataset):
    node = make_gather_signals_node(
        small_dataset["sales"], small_dataset["items"], small_dataset["suppliers"],
        small_dataset["festival_calendar"], small_dataset["festival_overrides"], small_dataset["promotions"],
    )
    out = node({"as_of_date": str(small_dataset["as_of"].date())})
    assert "LOW" in out["consumption_signals"]
    assert "OK" not in out["consumption_signals"]  # healthy items never reach the LLM


def test_gather_signals_applies_per_item_parameter_overrides(small_dataset):
    # a seller's feedback adjusts z/lead_time_buffer_days,
    # but that's worthless if a future gather_signals call ignores it -- the whole
    # point of the Feedback Agent is that it reaches the NEXT recommendation.
    # LOW's demand needs real variance for z to matter at all: reorder_point's
    # z*sigma*sqrt(L) term is 0 regardless of z when sigma is 0, which is exactly
    # LOW's default fixture shape (constant 8/day) -- so this uses its own
    # alternating-demand sales series instead of the shared fixture's.
    dates = small_dataset["sales"]["date"].unique()
    variable_low_sales = pd.DataFrame({
        "date": dates, "item_id": "LOW",
        "units_sold": [4, 12] * (len(dates) // 2), "stockout_flag": 0, "on_hand_end": 3,
    })
    other_sales = small_dataset["sales"][small_dataset["sales"].item_id != "LOW"]
    dataset = {**small_dataset, "sales": pd.concat([other_sales, variable_low_sales], ignore_index=True)}

    default_node = make_gather_signals_node(
        dataset["sales"], dataset["items"], dataset["suppliers"],
        dataset["festival_calendar"], dataset["festival_overrides"], dataset["promotions"],
    )
    adjusted_node = make_gather_signals_node(
        dataset["sales"], dataset["items"], dataset["suppliers"],
        dataset["festival_calendar"], dataset["festival_overrides"], dataset["promotions"],
        params_by_item={"LOW": {"z": 1.0, "lead_time_buffer_days": 0.0}},  # z well below the 1.65 default
    )
    default_out = default_node({"as_of_date": str(dataset["as_of"].date())})
    adjusted_out = adjusted_node({"as_of_date": str(dataset["as_of"].date())})

    assert adjusted_out["consumption_signals"]["LOW"]["reorder_point"] < default_out["consumption_signals"]["LOW"]["reorder_point"]


def test_gather_signals_caps_to_the_most_severe_items(small_dataset):
    # LOW already has on_hand_end=3 (days_of_cover 3/8=0.375). Add four more at-risk
    # items at distinct, non-tying severities, then cap to 3 and confirm only the
    # three lowest days_of_cover survive -- while total_flagged_count still reports
    # the true count of all five.
    on_hand_by_item = {"RISK1": 1, "RISK2": 2, "RISK3": 5, "RISK4": 6}  # -> cover 0.125, 0.25, 0.625, 0.75
    extra_items = pd.DataFrame([
        dict(item_id=iid, item_name=iid, category="Staples", supplier_id="S1",
             unit_cost_inr=10, unit_price_inr=15)
        for iid in on_hand_by_item
    ])
    items = pd.concat([small_dataset["items"], extra_items], ignore_index=True)
    extra_rows = [
        dict(date=d, item_id=iid, units_sold=8, stockout_flag=0, on_hand_end=on_hand)
        for iid, on_hand in on_hand_by_item.items()
        for d in small_dataset["sales"].date.unique()
    ]
    sales = pd.concat([small_dataset["sales"], pd.DataFrame(extra_rows)], ignore_index=True)

    node = make_gather_signals_node(
        sales, items, small_dataset["suppliers"], small_dataset["festival_calendar"],
        small_dataset["festival_overrides"], small_dataset["promotions"], max_items=3,
    )
    out = node({"as_of_date": str(small_dataset["as_of"].date())})

    assert out["total_flagged_count"] == 5  # LOW + all 4 RISK items, uncapped count
    assert len(out["consumption_signals"]) == 3
    assert set(out["consumption_signals"]) == {"RISK1", "RISK2", "LOW"}  # the 3 most severe


def test_severity_key_sorts_infinite_cover_last_not_first():
    from graph import _severity_key
    no_demand_but_stocked = dict(risk="overstock", days_of_cover=None)
    genuinely_urgent = dict(risk="stockout_risk", days_of_cover=0.5)
    ranked = sorted([no_demand_but_stocked, genuinely_urgent], key=_severity_key)
    assert ranked[0] is genuinely_urgent  # None (infinite cover) must not look more urgent than 0.5 days


# ---------------------------------------------------------------- reasoning + merge safety

def test_reasoning_merge_pulls_quantity_from_engine_not_the_llm(small_dataset):
    # even if the LLM's structured output had a quantity field, the merge only
    # ever takes item_id/urgency/rationale from it -- qty always comes from consumption_signals
    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Nearly out of stock."),
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-merge"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    rec = result["ranked_recommendations"][0]
    assert rec["item_id"] == "LOW"
    assert rec["urgency"] == "high"
    # the quantity must equal what assess_item computed, not something the LLM invented
    from engine import assess_item
    expected_qty = assess_item("LOW", small_dataset["as_of"], small_dataset["sales"],
                                small_dataset["items"], small_dataset["suppliers"])["suggested_order_qty"]
    assert rec["suggested_order_qty"] == expected_qty


def test_reasoning_replaces_ungrounded_rationale_with_safe_fallback(small_dataset):
    # a model batching many items can borrow a festival name from a DIFFERENT
    # item in the same prompt and paste it onto one that festival doesn't affect (LOW is Staples; this festival affects FMCG)
    festival_calendar = pd.DataFrame([dict(
        date=pd.Timestamp("2026-08-28"), festival_name="Raksha Bandhan",
        affected_categories="FMCG", uplift_multiplier=1.4, ramp_days_before=7,
    )])
    dataset = {**small_dataset, "festival_calendar": festival_calendar}
    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high",
                   rationale="Low stock during the Raksha Bandhan window uplift period."),
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-grounding-catch"}}
    result = graph.invoke({"as_of_date": str(dataset["as_of"].date())}, config)

    rationale = result["ranked_recommendations"][0]["rationale"]
    assert "Raksha Bandhan" not in rationale  # fabricated for this item -- must not survive
    assert "days of cover" in rationale       # replaced with the fact-only fallback


def test_reasoning_keeps_a_correctly_grounded_rationale(small_dataset):
    festival_calendar = pd.DataFrame([dict(
        date=pd.Timestamp("2026-08-28"), festival_name="Raksha Bandhan",
        affected_categories="Staples", uplift_multiplier=1.4, ramp_days_before=7,  # LOW *is* Staples here
    )])
    dataset = {**small_dataset, "festival_calendar": festival_calendar}
    original = "Low stock ahead of the Raksha Bandhan window uplift period."
    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale=original),
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-grounding-keep"}}
    result = graph.invoke({"as_of_date": str(dataset["as_of"].date())}, config)

    assert result["ranked_recommendations"][0]["rationale"] == original  # true claim, kept as-is


def test_reasoning_never_drops_an_item_the_model_omitted(small_dataset):
    # a model handling a long list can silently omit items rather than erroring.
    # Add a third flagged item here and have the fake response cover only one of
    # the two flagged items -- the omitted one must still appear, not vanish.
    items = pd.concat([small_dataset["items"], pd.DataFrame([
        dict(item_id="LOW2", item_name="Also Low Stock", category="Staples", supplier_id="S1",
             unit_cost_inr=10, unit_price_inr=15),
    ])], ignore_index=True)
    sales = pd.concat([small_dataset["sales"], pd.DataFrame([
        dict(date=d, item_id="LOW2", units_sold=8, stockout_flag=0, on_hand_end=3)
        for d in small_dataset["sales"].date.unique()
    ])], ignore_index=True)
    dataset = {**small_dataset, "items": items, "sales": sales}

    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Nearly out of stock."),
        # LOW2 is flagged too but the model's response never mentions it
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-omission"}}
    result = graph.invoke({"as_of_date": str(dataset["as_of"].date())}, config)

    ids = {r["item_id"] for r in result["ranked_recommendations"]}
    assert ids == {"LOW", "LOW2"}  # LOW2 must still show up despite the model omitting it
    low2 = next(r for r in result["ranked_recommendations"] if r["item_id"] == "LOW2")
    assert low2["source"] == "backfill"


def test_reasoning_drops_a_hallucinated_item_id(small_dataset):
    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Real item."),
        RankedItem(item_id="DOES_NOT_EXIST", urgency="high", rationale="Invented by the model."),
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-hallucination"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    ids = [r["item_id"] for r in result["ranked_recommendations"]]
    assert ids == ["LOW"]  # the invented id never made it through


def test_reasoning_falls_back_when_llm_fails_every_attempt(small_dataset):
    llm = _FakeLLM(response=None, raise_n_times=99)  # always raises
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-fallback"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    assert len(result["ranked_recommendations"]) == 1
    assert result["ranked_recommendations"][0]["source"] == "fallback"


def test_reasoning_retries_before_succeeding(small_dataset):
    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="medium", rationale="Recovered after retry."),
    ])
    llm = _FakeLLM(response=fake_response, raise_n_times=2)  # fails twice, succeeds on the 3rd
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-retry"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    assert result["ranked_recommendations"][0]["rationale"] == "Recovered after retry."


# ---------------------------------------------------------------- interrupt / resume

def test_graph_pauses_for_approval_then_resumes_with_the_decision(small_dataset):
    from langgraph.types import Command

    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Nearly out of stock."),
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-interrupt"}}
    paused = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["recommendations"][0]["item_id"] == "LOW"

    resumed = graph.invoke(Command(resume={"decision": "approve"}), config)
    assert resumed["seller_decision"] == {"decision": "approve"}
    assert "__interrupt__" not in resumed


# ---------------------------------------------------------------- grounding checks

def _ctx(active=None, next_name=None):
    return {"I1": {"active_festival": active, "next_festival_name": next_name}}


def _cons(trend_ratio=None):
    return {"I1": {"item_id": "I1", "trend_ratio": trend_ratio, "on_hand": 3, "days_of_cover": 1.0}}


ALL_FESTIVALS = {"Diwali", "Holi", "Raksha Bandhan"}


def test_grounding_allows_the_items_own_festival():
    from graph import _is_rationale_grounded
    assert _is_rationale_grounded(
        "I1", "Diwali is 6 days away and stock is low.",
        _cons(), _ctx(next_name="Diwali"), ALL_FESTIVALS)


def test_grounding_rejects_a_festival_borrowed_from_another_item():
    from graph import _is_rationale_grounded
    assert not _is_rationale_grounded(
        "I1", "Stock up ahead of Raksha Bandhan.",
        _cons(), _ctx(next_name="Diwali"), ALL_FESTIVALS)


def test_grounding_rejects_a_trend_claim_when_the_item_has_no_trend_signal():
    from graph import _is_rationale_grounded
    # trend_ratio None -> _trend_note() emits nothing -> the facts block never
    # mentioned a trend, so the model invented it
    assert not _is_rationale_grounded(
        "I1", "Demand is surging for this item.", _cons(None), _ctx(), ALL_FESTIVALS)


def test_grounding_allows_a_trend_claim_when_the_signal_is_really_there():
    from graph import _is_rationale_grounded
    assert _is_rationale_grounded(
        "I1", "Demand is surging well above its usual rate.",
        _cons(trend_ratio=2.4), _ctx(), ALL_FESTIVALS)


def test_grounding_rejects_any_supplier_claim():
    from graph import _is_rationale_grounded
    # _format_item_facts never emits supplier information at all, so the whole
    # class is out of bounds -- no allow-list needed
    for text in ["The supplier is unreliable.",
                 "Long lead time on this one.",
                 "Expect a delivery delay."]:
        assert not _is_rationale_grounded("I1", text, _cons(2.4), _ctx(), ALL_FESTIVALS)
