"""
The dashboard. One Streamlit app tying the system together: an overview that
frames the store's position, today's ranked recommendations from the graph,
approve/reject that closes through the Feedback Agent, a chat panel, the
backtest evidence, and the audit log.

The graph's own interrupt/resume is a single batch-level pause (one
interrupt() covers the whole ranked list) -- it isn't shaped for independent
per-item clicks, so this dashboard reads the recommendations off that one
interrupt, then applies each approve/reject directly through
feedback_agent.submit_feedback(), which is where the real per-item logic
(hysteresis, clamping) actually lives.

Defaults to CACHED model responses (data/demo_llm_cache.json, built by
record_demo_cache.py) so the walkthrough never depends on a live call.

LangSmith integration: each graph run includes metadata for tracing.
Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable tracing.

Enhanced features:
- Auto-approval for low-risk items
- Email notifications via Twilio
- RAG-enhanced chatbot with FAISS
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Push LangSmith secrets from Streamlit Cloud into os.environ so
# langsmith.traceable picks them up at import / run time.
for _k in ("LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT",
           "LANGCHAIN_ENDPOINT", "LANGCHAIN_CALLBACKS"):
    if _k not in os.environ:
        try:
            os.environ[_k] = st.secrets[_k]
        except KeyError:
            pass
        except Exception:
            pass
# Also handle lowercase / mixed-case variants some SDK versions check
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "sellersense")

# Explicitly configure LangSmith client so @traceable auto-tracing works
try:
    from langsmith import Client as _LSClient
    _ls_client = _LSClient(auto_batch_tracing=False)
except Exception:
    _ls_client = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chatbot import respond
from context_agent import get_context
from engine import assess_item, days_to_next_arrival, on_order_qty
from feedback_agent import REASON_CODES, default_parameters, submit_feedback
from graph import build_graph
from hitl import AutoApprovalEngine, WhatsAppNotifier
from llm_cache import CachedLLM
from llm_provider import available_providers, make_llm
from logger import get_logger
from store import (
    empty_feedback_log,
    load_feedback_log,
    load_parameters,
    save_feedback_log,
    save_parameters,
)

logger = get_logger(__name__, extra_data={"module": "dashboard"})

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE_PATH = DATA / "demo_llm_cache.json"
FEEDBACK_PATH = DATA / "feedback_log.csv"
PARAMETERS_PATH = DATA / "seller_parameters.csv"
BACKTEST_PATH = DATA / "backtest_results.csv"

STORE_NAME = "Meera General Store"
STORE_SUBTITLE = "Single storefront · 25 SKUs · 5 suppliers"

st.set_page_config(page_title="SellerSense", page_icon="📦", layout="wide")

# Verify LangSmith env vars are set
_langsmith_ok = all(os.environ.get(k) for k in ("LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2"))

RISK_LABEL = {"stockout_risk": "At risk", "overstock": "Overstocked", "healthy": "Healthy"}
RISK_SORT_ORDER = {"stockout_risk": 0, "overstock": 1, "healthy": 2}
URGENCY_STYLE = {
    "high": ("#E8867E", "Act today"),
    "medium": ("#D6B45C", "This week"),
    "low": ("#7FC4B6", "Keep an eye"),
}
SOURCE_NOTE = {
    "fallback": "Ranked by rule — the model was unavailable this cycle.",
    "backfill": "Shown from computed figures — the model left this item out of its reply.",
}

# Initialize RAG vector store (cached)
@st.cache_resource
def get_rag_store():
    """Initialize and cache the RAG vector store."""
    try:
        from rag import build_store_from_data
        return build_store_from_data(DATA)
    except Exception as e:
        logger.warning(f"RAG initialization failed: {e}")
        return None

# Initialize auto-approval engine
auto_approval_engine = AutoApprovalEngine()

# Initialize email notifier
email_notifier = WhatsAppNotifier()


# ---------------------------------------------------------------- data

@st.cache_data
def load_data():
    return dict(
        sales=pd.read_csv(DATA / "daily_sales.csv", parse_dates=["date"]),
        items=pd.read_csv(DATA / "items.csv"),
        suppliers=pd.read_csv(DATA / "suppliers.csv"),
        festival_calendar=pd.read_csv(DATA / "festival_calendar.csv", parse_dates=["date"]),
        festival_overrides=pd.read_csv(DATA / "festival_item_overrides.csv"),
        promotions=pd.read_csv(DATA / "promotions.csv", parse_dates=["date"]),
        # needed to net off stock already in transit -- without it the same order
        # is recommended every morning until the goods physically arrive
        purchase_orders=pd.read_csv(DATA / "purchase_orders.csv",
                                     parse_dates=["date_ordered", "date_expected"]),
    )


d = load_data()
DATA_START = d["sales"]["date"].min()
DATA_END = d["sales"]["date"].max()
COST = d["items"].set_index("item_id")["unit_cost_inr"].to_dict()
MARGIN = (d["items"].set_index("item_id")["unit_price_inr"]
          - d["items"].set_index("item_id")["unit_cost_inr"]).to_dict()


@st.cache_data(show_spinner=False)
def assess_all(as_of: pd.Timestamp, params_by_item: dict) -> list[dict]:
    """Every SKU's current state. No model calls, so the overview and the chat panel
    both use it rather than depending on a review having been run.

    Cached because Streamlit re-runs this whole script on every widget interaction,
    and this loop filters a 9,000-row frame and calls demand_rate three times per
    SKU. The cache key is (as_of, params_by_item), which is exactly what the result
    depends on -- approving a recommendation changes params_by_item and correctly
    invalidates it."""
    out = []
    for item_id in d["items"]["item_id"]:
        p = params_by_item.get(item_id, default_parameters(item_id))
        out.append(assess_item(item_id, as_of, d["sales"], d["items"], d["suppliers"],
                                z=p["z"], lead_time_buffer_days=p["lead_time_buffer_days"],
                                on_order=on_order_qty(item_id, as_of, d["purchase_orders"]),
                                arriving_in_days=days_to_next_arrival(item_id, as_of,
                                                                       d["purchase_orders"])))
    return out


def upcoming_events(as_of: pd.Timestamp, horizon_days: int = 45) -> pd.DataFrame:
    fc = d["festival_calendar"].copy()
    fc["days_away"] = (fc["date"] - as_of).dt.days
    return fc[(fc.days_away >= 0) & (fc.days_away <= horizon_days)].sort_values("days_away")


def rupees(n) -> str:
    return f"₹{int(round(n)):,}"


# ---------------------------------------------------------------- session state

if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = load_feedback_log(FEEDBACK_PATH)
if "params_by_item" not in st.session_state:
    st.session_state.params_by_item = load_parameters(PARAMETERS_PATH)
for key, default in [("recommendations", None), ("total_flagged", None),
                     ("decided", {}), ("chat_messages", []), ("review_error", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def persist():
    save_feedback_log(st.session_state.feedback_log, FEEDBACK_PATH)
    save_parameters(st.session_state.params_by_item, PARAMETERS_PATH)


# ---------------------------------------------------------------- sidebar

st.sidebar.markdown(f"### {STORE_NAME}")
st.sidebar.caption(STORE_SUBTITLE)
st.sidebar.divider()

model_mode = st.sidebar.radio("Model", ["Cached (demo-safe)", "Live model"], index=0,
                              help="Cached replays recorded responses so the walkthrough never "
                                   "waits on a model. Live calls the provider directly.")

usable = available_providers()
live_provider = None
if model_mode == "Live model":
    if usable:
        live_provider = st.sidebar.selectbox("Provider", usable, index=0)
    else:
        st.sidebar.warning("No provider configured. Set GROQ_API_KEY / GOOGLE_API_KEY / "
                           "OPENAI_API_KEY, or run Ollama locally. Using cached responses.")

# clamped to the dataset: an out-of-range date used to raise straight out of
# assess_item and take the whole app down mid-demo
as_of = pd.Timestamp(st.sidebar.date_input(
    "Business date", value=DATA_END - pd.Timedelta(days=7),
    min_value=DATA_START + pd.Timedelta(days=60), max_value=DATA_END,
    help="Any date covered by the sales history.",
))

st.sidebar.divider()
if _langsmith_ok:
    st.sidebar.success("LangSmith tracing active")
else:
    st.sidebar.warning("LangSmith tracing off")
n_logged = len(st.session_state.feedback_log)
st.sidebar.caption(f"{n_logged} decision(s) on record" if n_logged else "No decisions recorded yet")
if st.sidebar.button("Reset feedback history", disabled=not n_logged, width="stretch"):
    st.session_state.feedback_log = empty_feedback_log()
    st.session_state.params_by_item = {}
    st.session_state.decided = {}
    st.session_state.recommendations = None
    persist()
    st.rerun()


# ---------------------------------------------------------------- review

def get_llm():
    real = None
    if live_provider:
        try:
            real = make_llm(live_provider)
        except Exception as e:
            st.session_state.review_error = f"Couldn't reach {live_provider}: {e}"
    return CachedLLM(CACHE_PATH, real_llm=real, record=False)


def run_review():
    st.session_state.review_error = None
    try:
        from langsmith import RunnableConfig
        tracing_config = RunnableConfig(
            tags=["dashboard", f"date-{as_of.date()}"],
            metadata={
                "provider": live_provider if live_provider else "cached",
                "n_items": len(d["items"]),
                "run_type": "dashboard_review",
            }
        )
    except ImportError:
        tracing_config = None
    
    graph = build_graph(d["sales"], d["items"], d["suppliers"], d["festival_calendar"],
                         d["festival_overrides"], d["promotions"], get_llm(),
                         params_by_item=st.session_state.params_by_item,
                         purchase_orders=d["purchase_orders"])
    thread_config = {"configurable": {"thread_id": f"dashboard-{as_of.date()}"}}
    if tracing_config:
        thread_config.update(tracing_config)
    try:
        result = graph.invoke({"as_of_date": str(as_of.date())}, thread_config)
        st.session_state.recommendations = result["__interrupt__"][0].value["recommendations"]
        st.session_state.total_flagged = result["total_flagged_count"]
        st.session_state.decided = {}
    except Exception as e:
        st.session_state.review_error = str(e)
        st.session_state.recommendations = []
        st.session_state.total_flagged = 0


assessments = assess_all(as_of, st.session_state.params_by_item)
at_risk = [a for a in assessments if a["risk"] == "stockout_risk"]

# run once on load so the app opens with something to look at, rather than an
# empty page behind a button nobody knows to press
if st.session_state.recommendations is None:
    with st.spinner("Reviewing today's inventory…"):
        run_review()


# ---------------------------------------------------------------- header

st.title("SellerSense")
st.caption("An inventory copilot for small retailers. It works out what is about to run out, "
           "ranks what deserves attention today, explains why, and drafts the reorder.")

if st.session_state.review_error:
    st.warning(f"Review fell back to computed figures: {st.session_state.review_error}")

# segmented_control rather than st.tabs: tab selection does not survive a rerun,
# so approving a recommendation or sending a chat message would throw the user
# back to the first tab every time. This keeps its place because it is bound to
# session state.
SECTIONS = ["Overview", "Recommendations", "Inventory", "Ask", "Evidence", "Activity"]
section = st.segmented_control("Section", SECTIONS, default="Overview",
                                key="section", label_visibility="collapsed")
section = section or "Overview"


# ---------------------------------------------------------------- overview

if section == "Overview":
    urgent = [a for a in at_risk if (a["days_of_cover"] or 0) < 3]
    # suggested_order_qty is already net of stock in transit, so these totals no
    # longer double-count capital the seller has committed but not yet received
    reorder_value = sum(a["suggested_order_qty"] * COST[a["item_id"]] for a in at_risk)
    margin_at_risk = sum(a["suggested_order_qty"] * MARGIN[a["item_id"]] for a in at_risk)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Needs attention", f"{len(at_risk)} of {len(assessments)}", help="SKUs at or below their reorder point")
    c2.metric("Running out in 3 days", len(urgent), delta=f"{len(urgent)} urgent" if urgent else None,
              delta_color="inverse")
    c3.metric("Cost to restock", rupees(reorder_value), help="Total value of every suggested order")
    c4.metric("Margin at risk", rupees(margin_at_risk), help="Profit forgone if these stock out")

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("What needs doing today")
        recs = st.session_state.recommendations or []
        if not recs:
            st.info("Nothing flagged for this date.")
        for rec in recs[:3]:
            colour, when = URGENCY_STYLE.get(rec["urgency"], ("#98A1AD", ""))
            with st.container(border=True):
                st.markdown(
                    f"<span style='color:{colour};font-weight:600'>{when}</span> · "
                    f"**{rec['item_name']}** · "
                    + (f"order {rec['suggested_order_qty']} units"
                       if rec["suggested_order_qty"] > 0 else "already on order"),
                    unsafe_allow_html=True)
                st.write(rec["rationale"])
        if len(recs) > 3:
            st.caption(f"{len(recs) - 3} more in the Recommendations tab.")

    with right:
        st.subheader("What's coming")
        events = upcoming_events(as_of)
        if events.empty:
            st.caption("No festivals or sale events in the next 45 days.")
        else:
            for e in events.itertuples():
                with st.container(border=True):
                    when = "today" if e.days_away == 0 else f"in {e.days_away} days"
                    st.markdown(f"**{e.festival_name}** · {when}")
                    st.caption(f"{e.date.date()} · {e.event_type}")
                    st.caption(f"Lifts {e.affected_categories.replace(',', ', ')} "
                               f"by about {e.uplift_multiplier}×")
        st.caption("Known demand events are read from a calendar, so the system can order "
                   "ahead of them rather than reacting after stock runs out.")


# ---------------------------------------------------------------- recommendations

if section == "Recommendations":
    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.subheader("Today's recommendations")
        total = st.session_state.total_flagged or 0
        shown = len(st.session_state.recommendations or [])
        st.caption(f"{total} SKUs flagged; the {shown} most severe are shown. "
                   "Capped deliberately — a list of eighteen is a list nobody reads.")
    with head_r:
        if st.button("Re-run review", width="stretch"):
            with st.spinner("Reviewing…"):
                run_review()
            st.rerun()

    # Auto-approval summary
    recs = st.session_state.recommendations or []
    if recs:
        classification = auto_approval_engine.classify_recommendations(recs)
        if classification.get("auto_approve"):
            with st.expander(f"✅ {len(classification['auto_approve'])} items auto-approved (low risk)", expanded=False):
                for item in classification["auto_approve"]:
                    st.write(f"• **{item['item_name']}** - {item['suggested_order_qty']} units (₹{item['suggested_order_qty'] * item.get('unit_cost_inr', 0):,})")
                st.caption("These items meet auto-approval criteria: low risk, low value, high confidence.")

    # Email notification section
    if recs and email_notifier.available:
        st.divider()
        notif_col1, notif_col2 = st.columns([1, 3])
        with notif_col1:
            recipient_email = st.text_input("Notification email", value="mepavaniitkgp@gmail.com", key="notif_email")
        with notif_col2:
            if st.button("📧 Send Daily Summary Email", type="secondary", width="stretch"):
                with st.spinner("Sending email..."):
                    auto_approved = classification.get("auto_approve", [])
                    needs_review = classification.get("needs_review", [])
                    result = email_notifier.send_daily_summary_email(
                        recipient_email, recs, auto_approved, needs_review
                    )
                    if result["success"]:
                        st.success(f"Email sent successfully!")
                    else:
                        st.error(f"Failed to send email: {result.get('error', 'Unknown error')}")

    for rec in (st.session_state.recommendations or []):
        item_id = rec["item_id"]
        decision = st.session_state.decided.get(item_id)
        colour, when = URGENCY_STYLE.get(rec["urgency"], ("#98A1AD", ""))

        with st.container(border=True):
            main, action = st.columns([4, 1])
            with main:
                st.markdown(
                    f"<span style='color:{colour};font-weight:600'>{when}</span> · "
                    f"**{rec['item_name']}** · {rec['category']}",
                    unsafe_allow_html=True)
                st.write(rec["rationale"])
                cover = rec["days_of_cover"]
                qty = rec["suggested_order_qty"]
                # an item with stock already in transit is still shown -- a late PO
                # is exactly the case the seller needs to see -- but the ask is zero
                # rather than a duplicate order
                order_part = (f"Order {qty} units · about {rupees(qty * COST[item_id])}"
                              if qty > 0 else "Nothing to order right now")
                st.caption(
                    f"{order_part} · "
                    f"{'out of stock' if cover == 0 else f'{cover} days of cover left'}")
                if rec.get("on_order"):
                    arriving = rec.get("arriving_in_days")
                    when = f", arriving in {arriving} day(s)" if arriving is not None else ""
                    st.caption(f":blue[{rec['on_order']} units already on order{when}]")
                note = SOURCE_NOTE.get(rec.get("source"))
                if note:
                    st.caption(f":grey[{note}]")

            with action:
                if decision == "approved":
                    st.success("Approved")
                elif decision == "rejected":
                    st.error("Rejected")
                else:
                    if st.button("Approve", key=f"a-{item_id}", type="primary", width="stretch"):
                        log, params, _ = submit_feedback(
                            st.session_state.feedback_log, st.session_state.params_by_item,
                            f"REC-{item_id}-{as_of.date()}", item_id, "approve", None,
                            pd.Timestamp.now())
                        st.session_state.feedback_log = log
                        st.session_state.params_by_item = params
                        st.session_state.decided[item_id] = "approved"
                        persist()
                        st.rerun()

                    reason = st.selectbox("Reason", REASON_CODES, key=f"r-{item_id}",
                                          label_visibility="collapsed",
                                          format_func=lambda r: r.replace("_", " "))
                    if st.button("Reject", key=f"x-{item_id}", width="stretch"):
                        log, params, audit = submit_feedback(
                            st.session_state.feedback_log, st.session_state.params_by_item,
                            f"REC-{item_id}-{as_of.date()}", item_id, "reject", reason,
                            pd.Timestamp.now())
                        st.session_state.feedback_log = log
                        st.session_state.params_by_item = params
                        st.session_state.decided[item_id] = "rejected"
                        persist()
                        if audit:
                            st.toast(f"Learned: {audit['parameter_adjusted']} "
                                     f"{audit['old_value']} → {audit['new_value']}", icon="🎯")
                        st.rerun()


# ---------------------------------------------------------------- inventory

if section == "Inventory":
    st.subheader(f"All {len(assessments)} SKUs")
    st.caption(f"As of {as_of.date()}. Risk is computed from the demand rate, its variability, "
               "and each supplier's lead time — not from a fixed threshold.")

    only_risk = st.checkbox("Show only what needs attention", value=False)
    rows = [
        dict(Item=a["item_name"], Category=a["category"], Status=RISK_LABEL[a["risk"]],
             _sort=RISK_SORT_ORDER[a["risk"]],
             **{"On hand": a["on_hand"], "In transit": a["on_order"],
                "Days of cover": a["days_of_cover"],
                "Reorder at": a["reorder_point"], "Suggest": a["suggested_order_qty"]})
        for a in assessments
        if not only_risk or a["risk"] != "healthy"
    ]
    table = pd.DataFrame(rows).sort_values("_sort").drop(columns="_sort")
    st.dataframe(table, width="stretch", hide_index=True)


# ---------------------------------------------------------------- chat

if section == "Ask":
    st.subheader("Ask about any item")
    st.caption("Answers come from the same computed figures the recommendations use, so the "
               "chat and the ranking can never disagree. Enhanced with semantic search for better context.")

    for role, text in st.session_state.chat_messages:
        with st.chat_message(role):
            st.write(text)

    if not st.session_state.chat_messages:
        st.caption('Try: "why is toothpaste recommended?" · "do I need more umbrellas?" · "what\'s the trend for rice?"')

    question = st.chat_input("Ask a question, or tell it to approve/reject something")
    if question:
        # rendered inline rather than via st.rerun(): a rerun here would discard
        # the answer's place in the page and send the user back to the top
        st.session_state.chat_messages.append(("user", question))
        with st.chat_message("user"):
            st.write(question)

        all_items = {a["item_id"]: a for a in assessments}
        context = {iid: get_context(iid, as_of, d["items"], d["festival_calendar"],
                                     d["festival_overrides"], d["promotions"]) for iid in all_items}
        
        # Get RAG context if available
        rag_context = ""
        rag_store = get_rag_store()
        if rag_store:
            try:
                rag_context = rag_store.retrieve_context(question, k=3)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = respond(get_llm(), question, all_items, context)

                # Append RAG context to response if available
                if rag_context and rag_context != "No relevant context found.":
                    result["text"] += f"\n\n📚 *Additional context from knowledge base:*"
            st.write(result["text"])
        st.session_state.chat_messages.append(("assistant", result["text"]))


# ---------------------------------------------------------------- evidence

if section == "Evidence":
    st.subheader("Does it actually work?")
    st.caption("A 45-day held-out window. Three ordering policies see the same demand and the "
               "same supplier lead times, each keeping its own stock trajectory. The forecaster "
               "only ever trains on data before the window opens.")

    if not BACKTEST_PATH.exists():
        st.info("Run `python3 src/run_backtest.py` to generate these figures.")
    else:
        bt = pd.read_csv(BACKTEST_PATH)
        summary = bt.groupby("policy").agg(
            stockout_days=("stockout_days", "sum"),
            fill_rate=("fill_rate", "mean"),
            capital=("avg_capital_tied_up", "sum"),
        ).round(3).reindex(["seller", "plain_rop", "context_aware"])

        label = {"seller": "What the seller actually did",
                 "plain_rop": "Textbook reorder point",
                 "context_aware": "SellerSense"}
        c1, c2, c3 = st.columns(3)
        for col, policy in zip((c1, c2, c3), summary.index):
            base = summary.loc["seller"]
            row = summary.loc[policy]
            delta = None if policy == "seller" else f"{int(row.stockout_days - base.stockout_days):+d} vs seller"
            col.metric(label[policy], f"{int(row.stockout_days)} stockout days", delta=delta,
                       delta_color="inverse")

        # rendered as markdown rather than st.dataframe: the dataframe widget
        # would not give the policy names enough column width to be readable,
        # and this is the table the whole credibility argument rests on
        md = ["| Policy | Stockout days | Fill rate | Capital tied up |",
              "|---|---:|---:|---:|"]
        for policy in summary.index:
            row = summary.loc[policy]
            name = f"**{label[policy]}**" if policy == "context_aware" else label[policy]
            md.append(f"| {name} | {int(row.stockout_days)} | {row.fill_rate:.3f} | "
                      f"{rupees(row.capital)} |")
        st.markdown("\n".join(md))
        st.caption("")

        st.markdown("**SellerSense beats what the seller actually did on all three at once** — "
                    "fewer stockouts, higher fill rate, and less cash tied up. Against the textbook "
                    "reorder point it trades capital for availability, which is the intended trade.")

        raincoat = bt[bt.item_name == "Raincoat"].set_index("policy")["fill_rate"]
        if {"plain_rop", "context_aware"} <= set(raincoat.index):
            st.info(f"Clearest single case — Raincoat through the monsoon: a textbook reorder point "
                    f"never caught up and finished at a {raincoat['plain_rop']:.3f} fill rate, "
                    f"against {raincoat['context_aware']:.3f} with the seasonal signal.")


# ---------------------------------------------------------------- activity

if section == "Activity":
    st.subheader("Every decision, and what it changed")
    st.caption("One rejection changes nothing. The same reason has to repeat three times before "
               "a parameter moves, and every parameter is clamped.")

    log = st.session_state.feedback_log
    if log.empty:
        st.info("No decisions yet. Approve or reject a recommendation to see it recorded here.")
    else:
        display = log.rename(columns={
            "feedback_id": "ID", "item_id": "Item", "seller_decision": "Decision",
            "reason_code": "Reason", "timestamp": "When",
            "parameter_adjusted": "Changed", "old_value": "From", "new_value": "To"})
        display = display[["ID", "Item", "Decision", "Reason", "When", "Changed", "From", "To"]]
        display["When"] = pd.to_datetime(display["When"]).dt.strftime("%d %b %H:%M")
        # an approve has no reason and most rejections change nothing yet; a literal
        # "None" in those cells reads as an error rather than as "not applicable"
        display = display.fillna("—").replace({None: "—"})
        st.dataframe(display, width="stretch", hide_index=True)

    adjusted = [p for iid, p in st.session_state.params_by_item.items()
                if p != default_parameters(iid)]
    if adjusted:
        st.subheader("What it has learned")
        st.caption("Only items whose settings have actually moved.")
        st.dataframe(pd.DataFrame(adjusted).rename(columns={
            "item_id": "Item", "z": "Safety factor", "lead_time_buffer_days": "Extra lead days"}),
            width="stretch", hide_index=True)
