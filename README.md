# SellerSense — Autonomous Inventory Reordering Platform

### [Live Demo](https://autonomous-inventory-management-reordering-platform-mx4aqvegc3.streamlit.app/)

---

## The Problem

A small shop owner with **25 items** reorders every morning based on gut feeling.

| What goes wrong | How it costs money |
|---|---|
| Runs out of fast-selling items | Lost sales — customer walks away |
| Overorders slow-moving items | Cash stuck on shelves for months |
| Misses festival demand spikes | Loses 40% of Diwali sales |
| No time to do the math daily | Nobody checks reorder points by hand |

The math (reorder point formula) has existed since 1913. **Nobody uses it because gathering inputs daily is the hard part, not the formula.**

---

## What SellerSense Does

**An AI assistant that watches your shop and tells you what to order, how much, and why.**

```
You: "What do I need to order today?"

SellerSense: 
  1. Rice 5kg — Order 23 units (running low, 2.5 days left)
  2. Biscuits Pack — Order 80 units (Diwali in 10 days, demand spikes)
  3. Umbrella — 24 units arriving tomorrow (already on order)
```

---

## How It Works

### System Architecture (Enhanced with Agentic AI)

```mermaid
flowchart TB
    subgraph INPUTS["Data Sources"]
        SALES[daily_sales.csv<br/>365 days × 25 items]
        ITEMS[items.csv<br/>cost, supplier, category]
        SUPPLIERS[suppliers.csv<br/>lead times]
        FESTIVALS[festival_calendar.csv<br/>6 events]
        PROMOS[promotions.csv<br/>random sales]
        GOOGLE_CAL[Google Calendar<br/>live events]
    end

    subgraph ENGINE["Deterministic Engine — engine.py"]
        DEMAND[demand_rate<br/>μ, σ per SKU]
        ROP[reorder_point<br/>ROP = μ·L + z·σ·√L]
        RISK[classify_risk<br/>stockout / overstock / healthy]
        ON_ORDER[on_order_qty<br/>net off in-transit stock]
    end

    subgraph CONTEXT["Context Layer — context_agent.py + google_calendar.py"]
        FEST[festival detection<br/>window + uplift multiplier]
        PROD[promo detection<br/>active sales]
        UPCOMING[upcoming events<br/>days until next festival]
    end

    subgraph ORCHESTRATOR["LangGraph Orchestrator — graph.py"]
        GATHER[gather_signals<br/>assess all SKUs, filter flagged]
        REASON[reasoning node<br/>LLM ranks + explains]
        HUMAN[human_approval<br/>interrupt for decision]
    end

    subgraph LLM["LLM Layer — llm_provider.py + llm_cache.py"]
        PROVIDER[Groq / Gemini / OpenAI]
        CACHE[demo cache<br/>replay without API]
    end

    subgraph TOOLS["Tool Calling — tools.py"]
        ASSESS_TOOL[assess_inventory_item<br/>per-SKU assessment]
        CONTEXT_TOOL[get_item_context<br/>festival/promo signals]
        TREND_TOOL[get_demand_trend<br/>recent vs baseline]
        BATCH_TOOL[batch_assess_items<br/>multi-SKU scan]
        FORECAST_TOOL[get_forecast_summary<br/>Prophet predictions]
    end

    subgraph AGENTS["Multi-Agent Supervisor — agents.py"]
        SUPERVISOR[supervisor_agent<br/>ReAct tool-calling loop]
        ANALYST[inventory_analyst_agent<br/>structured analysis]
        CHAT_AGENT[chat_agent<br/>NL Q&A with tools]
    end

    subgraph RAG["RAG — rag.py"]
        FAISS[FAISS Vector Store<br/>semantic search]
        EMBED[sentence-transformers<br/>all-MiniLM-L6-v2]
    end

    subgraph FEEDBACK["Feedback Agent — feedback_agent.py"]
        LOG[record_feedback<br/>approve / reject]
        ADJUST[apply_feedback<br/>adjust z or buffer<br/>after 3 same rejections]
    end

    subgraph CHATBOT["Chatbot — chatbot.py"]
        RESOLVE[resolve_item_id<br/>deterministic matching]
        CLASSIFY[classify_intent<br/>question or command?]
        ANSWER[answer_question<br/>grounded in facts]
    end

    subgraph HITL["Enhanced HITL — hitl.py"]
        AUTO[AutoApprovalEngine<br/>low-risk auto-approve]
        PER_ITEM[per_item_interrupt<br/>independent decisions]
        NOTIFY[WhatsAppNotifier<br/>email alerts]
    end

    subgraph ENHANCED_FORECAST["Enhanced Forecast — enhanced_forecast.py"]
        HIERARCHICAL[hierarchical forecasting<br/>global + category + item]
        ENSEMBLE[ensemble forecasting<br/>multiple models]
        CROSS_VAL[cross-validation<br/>model diagnostics]
    end

    subgraph UI["Dashboard — dashboard.py — Streamlit"]
        OVERVIEW[Overview<br/>at a glance]
        RECS[Recommendations<br/>ranked list + approve/reject]
        INV[Inventory<br/>all 25 SKUs table]
        ASK[Ask<br/>natural language chat]
        EVIDENCE[Evidence<br/>4-policy backtest]
        ACTIVITY[Activity<br/>audit log]
    end

    subgraph STORE["Persistence — store.py"]
        FEEDBACK_CSV[feedback_log.csv]
        PARAMS_CSV[seller_parameters.csv]
    end

    subgraph OBSERVE["Observability — logger.py + LangSmith"]
        STRUCTURED["Structured Logging<br/>JSON format"]
        TRACING["LangSmith Tracing<br/>trace_run nested spans"]
    end

    SALES --> ENGINE
    ITEMS --> ENGINE
    SUPPLIERS --> ENGINE
    FESTIVALS --> CONTEXT
    PROMOS --> CONTEXT
    ITEMS --> CONTEXT
    GOOGLE_CAL --> CONTEXT

    ENGINE --> GATHER
    CONTEXT --> GATHER

    GATHER --> REASON
    LLM --> REASON
    REASON --> HUMAN

    HUMAN --> HITL
    HITL --> FEEDBACK
    FEEDBACK --> STORE
    STORE --> ENGINE

    TOOLS --> AGENTS
    AGENTS --> SUPERVISOR
    RAG --> AGENTS

    SALES --> CHATBOT
    CONTEXT --> CHATBOT
    LLM --> CHATBOT
    FAISS --> CHATBOT

    UI --> ENGINE
    UI --> CONTEXT
    UI --> ORCHESTRATOR
    UI --> FEEDBACK
    UI --> CHATBOT
    UI --> HITL

    ENGINE --> OBSERVE
    LLM --> OBSERVE
    ORCHESTRATOR --> OBSERVE
```

### Daily Flow — What Happens Each Morning

```mermaid
flowchart LR
    START([Shop owner opens app]) --> ASSESS[Assess all 25 SKUs<br/>demand rate, stock, lead time]
    ASSESS --> FILTER[Filter to flagged items<br/>stockout risk or overstock]
    FILTER --> CONTEXT[Add context<br/>festivals, promos, trends]
    CONTEXT --> RANK[LLM ranks top 8<br/>urgency + rationale]
    RANK --> AUTO_CHECK{Auto-approve?}
    AUTO_CHECK -->|Low risk + low value| AUTO_ORDER[Auto-approved<br/>order placed]
    AUTO_CHECK -->|Needs review| SHOW[Show to seller<br/>with approve/reject buttons]
    SHOW --> DECIDE{Seller decides}
    DECIDE -->|Approve| ORDER[System drafts PO]
    DECIDE -->|Reject: qty too high| LOG_REJECT[Log rejection<br/>streak count +1]
    DECIDE -->|Reject: not needed| LOG_CIRCUMSTANTIAL[Log reason<br/>no parameter change]
    LOG_REJECT --> STREAK{3 same rejections<br/>in a row?}
    STREAK -->|Yes| ADJUST[Adjust parameter<br/>z or lead_time_buffer]
    STREAK -->|No| WAIT[Wait for next cycle]
    ADJUST --> WAIT
    ORDER --> WAIT
    AUTO_ORDER --> NOTIFY_SEND[Send email notification<br/>summary of orders]
    NOTIFY_SEND --> WAIT
    WAIT --> TOMORROW([Tomorrow morning<br/>repeat])
```

### Feedback Learning Loop

```mermaid
flowchart TB
    REJECT[Reject: qty too high] --> LOG1[Rejection #1<br/>streak = 1<br/>no change]
    LOG1 --> REJECT2[Reject: qty too high]
    REJECT2 --> LOG2[Rejection #2<br/>streak = 2<br/>no change]
    LOG2 --> REJECT3[Reject: qty too high]
    REJECT3 --> LOG3[Rejection #3<br/>streak = 3]
    LOG3 --> ADJUST[z: 1.65 to 1.50<br/>lower safety factor]
    ADJUST --> NEXT[Next recommendation<br/>uses new z]
    
    style LOG3 fill:#ff9800
    style ADJUST fill:#4caf50
```

### Chatbot Flow (Enhanced with RAG + Tools)

```mermaid
flowchart TB
    MSG[User message] --> MATCH[Resolve item<br/>deterministic word matching]
    MATCH -->|No match| UNCLEAR["I'm not sure which item"]
    MATCH -->|Match found| RAG_SEARCH[FAISS semantic search<br/>retrieve relevant context]
    RAG_SEARCH --> INTENT{Classify intent<br/>question or command?}
    INTENT -->|Question| TOOLS_CHECK{Need tools?}
    TOOLS_CHECK -->|Yes| CALL_TOOLS[Call inventory tools<br/>assess, context, trend]
    TOOLS_CHECK -->|No| ANSWER[Answer grounded<br/>in facts + RAG context]
    CALL_TOOLS --> ANSWER
    INTENT -->|Command: approve| DETECT[Detect action<br/>from user words]
    INTENT -->|Command: reject| DETECT
    DETECT -->|Reject without reason| ASK_REASON["What's the reason?"]
    DETECT -->|Complete command| READY[Ready to execute<br/>caller decides]
    ANSWER --> REPLY[Reply to user]

    style UNCLEAR fill:#ffcdd2
    style READY fill:#c8e6c9
```

### Multi-Agent Architecture

```mermaid
flowchart TB
    USER[User Query] --> SUPERVISOR[supervisor_agent<br/>ReAct tool-calling loop]
    
    SUPERVISOR --> TOOLS{Tool calls?}
    TOOLS -->|assess_inventory_item| ASSESS[Assessment Tool]
    TOOLS -->|get_item_context| CONTEXT_T[Context Tool]
    TOOLS -->|get_demand_trend| TREND[Trend Tool]
    TOOLS -->|batch_assess_items| BATCH[Batch Tool]
    TOOLS -->|get_forecast_summary| FORECAST[Forecast Tool]
    TOOLS -->|No tools needed| FINAL[Final Answer]
    
    ASSESS --> SUPERVISOR
    CONTEXT_T --> SUPERVISOR
    TREND --> SUPERVISOR
    BATCH --> SUPERVISOR
    FORECAST --> SUPERVISOR
    
    SUPERVISOR --> HANDOFF{Handoff?}
    HANDOFF -->|Inventory analysis| ANALYST[inventory_analyst_agent]
    HANDOFF -->|Chat question| CHAT[chat_agent]
    HANDOFF -->|Direct answer| FINAL
    
    ANALYST --> FINAL
    CHAT --> FINAL
    
    style SUPERVISOR fill:#2196f3
    style FINAL fill:#4caf50
```

---

## Dashboard — 6 Tabs

| Tab | What It Shows |
|---|---|
| **Overview** | How many items need attention, cost to restock, top 3 actions, upcoming festivals |
| **Recommendations** | Ranked list with urgency + rationale, approve/reject buttons |
| **Inventory** | Full table of all 25 SKUs with risk status, stock levels, reorder points |
| **Ask** | Natural language chat — ask questions or give commands |
| **Evidence** | Backtest results: 4 ordering policies compared over 45 days |
| **Activity** | Audit log of every approve/reject decision and parameter change |

---

## Key Design Rule

> **The LLM never computes numbers. Python does all the math. The LLM only ranks, explains, and classifies.**

| What Python handles | What the LLM handles |
|---|---|
| Demand rate (μ, σ) | "This is urgent because..." |
| Reorder point calculation | Ranking items by urgency |
| Risk classification | Classifying user intent (question vs command) |
| Festival/promo detection | Writing plain-language rationales |
| In-transit stock netting | Answering natural language questions |
| Auto-approval decisions | Tool calling orchestration |

This boundary ensures:
- Numbers are always correct (deterministic functions)
- The LLM's text is checked for 3 classes of hallucination before display
- A model outage degrades gracefully to computed fallbacks

---

## Results — Does It Work?

45-day holdout backtest, all 25 SKUs:

| Policy | Stockout Days | Fill Rate | Capital Tied Up |
|---|---|---|---|
| Seller's own rule | 150 | 82.8% | ₹63,672 |
| **SellerSense** | **171** | **80.2%** | **₹30,837** |
| Plain reorder point | 302 | 64.3% | ₹20,235 |

**SellerSense reaches 97% of the seller's fill rate at 48% of the working capital.**

The shop owner frees ₹33,000 in cash for just 2.5% fill rate — a trade most would take.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Orchestration | LangGraph |
| LLM | Groq (llama-3.3-70b), Gemini, OpenAI, or Ollama |
| Forecasting | Prophet (hierarchical + ensemble) |
| Vector Store | FAISS + sentence-transformers |
| Data | Pandas, NumPy |
| Observability | LangSmith + Structured Logging |
| Notifications | Twilio (WhatsApp + Email) |
| Testing | 186 tests (pytest) |
| Hosting | Streamlit Community Cloud |

---

## Local Setup

```bash
# Clone
git clone https://github.com/pavanjarpula/Autonomous-Inventory-Management-Reordering-Platform.git
cd Autonomous-Inventory-Management-Reordering-Platform

# Install
pip install -r requirements.txt

# Run dashboard
streamlit run src/dashboard.py

# Run tests
pytest

# Run backtest
python src/run_backtest.py
```

### Environment Variables (Optional)

```bash
# LLM Provider
export SELLERSENSE_LLM_PROVIDER=groq
export GROQ_API_KEY=your_key_here

# LangSmith Observability (free tier)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_langsmith_key
export LANGCHAIN_PROJECT=sellersense

# Twilio Notifications (optional)
export TWILIO_ACCOUNT_SID=your_sid
export TWILIO_AUTH_TOKEN=your_token
export TWILIO_WHATSAPP_FROM=+1234567890
export TWILIO_EMAIL_FROM=your_email@twilio.email
```

---

## Project Structure

```
├── src/
│   ├── engine.py              # Deterministic inventory math
│   ├── context_agent.py       # Festival + promo context
│   ├── forecast.py            # Prophet models + backtest
│   ├── enhanced_forecast.py   # Hierarchical + ensemble forecasting
│   ├── graph.py               # LangGraph orchestrator
│   ├── feedback_agent.py      # Learning from seller decisions
│   ├── chatbot.py             # Natural language interface
│   ├── dashboard.py           # Streamlit UI (6 tabs)
│   ├── llm_provider.py        # Multi-provider LLM support
│   ├── llm_cache.py           # Demo-safety caching
│   ├── store.py               # CSV persistence
│   ├── tools.py               # @tool-decorated functions (Phase 2)
│   ├── agents.py              # Multi-agent supervisor (Phase 2)
│   ├── rag.py                 # FAISS vector store + RAG (Phase 3)
│   ├── google_calendar.py     # Google Calendar integration (Phase 4)
│   ├── hitl.py                # Auto-approval + notifications (Phase 5)
│   ├── logger.py              # Structured JSON logging (Phase 1)
│   ├── tracing.py             # LangSmith nested trace helper
│   └── run_backtest.py        # Backtest runner
├── tests/                     # 186 tests
│   ├── test_engine.py
│   ├── test_context_agent.py
│   ├── test_forecast.py
│   ├── test_graph.py
│   ├── test_feedback_agent.py
│   ├── test_chatbot.py
│   ├── test_llm_cache.py
│   ├── test_llm_provider.py
│   ├── test_store.py
│   ├── test_tools_agents.py   # Tool + agent tests (Phase 2)
│   ├── test_rag.py            # RAG tests (Phase 3)
│   ├── test_google_calendar.py # Calendar tests (Phase 4)
│   ├── test_hitl.py           # HITL tests (Phase 5)
│   └── test_enhanced_forecast.py # Enhanced forecast tests (Phase 6)
├── data/
│   ├── items.csv              # 25 SKUs
│   ├── suppliers.csv          # 5 suppliers
│   ├── daily_sales.csv        # 365 days of sales
│   ├── festival_calendar.csv  # 6 festivals
│   ├── promotions.csv         # Random promos
│   ├── purchase_orders.csv    # In-transit stock
│   ├── festival_item_overrides.csv
│   ├── generate_dataset.py    # Dataset generator
│   └── backtest_results.csv   # 4-policy comparison
└── requirements.txt
```

---

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | LangSmith nested tracing (trace_run) + structured logging | ✅ Complete |
| 2 | Tool calling with @tool decorators + multi-agent supervisor | ✅ Complete |
| 3 | FAISS vector store + RAG-enhanced chatbot | ✅ Complete |
| 4 | Google Calendar integration for dynamic festival detection | ✅ Complete |
| 5 | Enhanced HITL - per-item interrupts + auto-approval + Email | ✅ Complete |
| 6 | Prophet enhancement - hierarchical priors + external regressors | ✅ Complete |

---

## License

MIT
