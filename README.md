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

### System Architecture

```mermaid
flowchart TB
    subgraph INPUTS["Data Sources"]
        SALES[daily_sales.csv<br/>365 days × 25 items]
        ITEMS[items.csv<br/>cost, supplier, category]
        SUPPLIERS[suppliers.csv<br/>lead times]
        FESTIVALS[festival_calendar.csv<br/>6 events]
        PROMOS[promotions.csv<br/>random sales]
    end

    subgraph ENGINE["Deterministic Engine — engine.py"]
        DEMAND[demand_rate<br/>μ, σ per SKU]
        ROP[reorder_point<br/>ROP = μ·L + z·σ·√L]
        RISK[classify_risk<br/>stockout / overstock / healthy]
        ON_ORDER[on_order_qty<br/>net off in-transit stock]
    end

    subgraph CONTEXT["Context Layer — context_agent.py"]
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

    subgraph FEEDBACK["Feedback Agent — feedback_agent.py"]
        LOG[record_feedback<br/>approve / reject]
        ADJUST[apply_feedback<br/>adjust z or buffer<br/>after 3 same rejections]
    end

    subgraph CHATBOT["Chatbot — chatbot.py"]
        RESOLVE[resolve_item_id<br/>deterministic matching]
        CLASSIFY[classify_intent<br/>question or command?]
        ANSWER[answer_question<br/>grounded in facts]
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

    SALES --> ENGINE
    ITEMS --> ENGINE
    SUPPLIERS --> ENGINE
    FESTIVALS --> CONTEXT
    PROMOS --> CONTEXT
    ITEMS --> CONTEXT

    ENGINE --> GATHER
    CONTEXT --> GATHER

    GATHER --> REASON
    LLM --> REASON
    REASON --> HUMAN

    HUMAN --> FEEDBACK
    FEEDBACK --> STORE
    STORE --> ENGINE

    SALES --> CHATBOT
    CONTEXT --> CHATBOT
    LLM --> CHATBOT

    UI --> ENGINE
    UI --> CONTEXT
    UI --> ORCHESTRATOR
    UI --> FEEDBACK
    UI --> CHATBOT
```

### Daily Flow — What Happens Each Morning

```mermaid
flowchart LR
    START([Shop owner opens app]) --> ASSESS[Assess all 25 SKUs<br/>demand rate, stock, lead time]
    ASSESS --> FILTER[Filter to flagged items<br/>stockout risk or overstock]
    FILTER --> CONTEXT[Add context<br/>festivals, promos, trends]
    CONTEXT --> RANK[LLM ranks top 8<br/>urgency + rationale]
    RANK --> SHOW[Show to seller<br/>with approve/reject buttons]
    SHOW --> DECIDE{Seller decides}
    DECIDE -->|Approve| ORDER[System drafts PO]
    DECIDE -->|Reject: qty too high| LOG_REJECT[Log rejection<br/>streak count +1]
    DECIDE -->|Reject: not needed| LOG_CIRCUMSTANTIAL[Log reason<br/>no parameter change]
    LOG_REJECT --> STREAK{3 same rejections<br/>in a row?}
    STREAK -->|Yes| ADJUST[Adjust parameter<br/>z or lead_time_buffer]
    STREAK -->|No| WAIT[Wait for next cycle]
    ADJUST --> WAIT
    ORDER --> WAIT
    WAIT --> TOMORROW([Tomorrow morning<br/>repeat])
```

### Feedback Learning Loop

```mermaid
flowchart TB
    REJECT[Reject: "qty too high"] --> LOG1[Rejection #1<br/>streak = 1<br/>no change]
    LOG1 --> REJECT2[Reject: "qty too high"]
    REJECT2 --> LOG2[Rejection #2<br/>streak = 2<br/>no change]
    LOG2 --> REJECT3[Reject: "qty too high"]
    REJECT3 --> LOG3[Rejection #3<br/>streak = 3]
    LOG3 --> ADJUST[z: 1.65 → 1.50<br/>lower safety factor]
    ADJUST --> NEXT[Next recommendation<br/>uses new z]
    
    style LOG3 fill:#ff9800
    style ADJUST fill:#4caf50
```

### Chatbot Flow

```mermaid
flowchart TB
    MSG[User message] --> MATCH[Resolve item<br/>deterministic word matching]
    MATCH -->|No match| UNCLEAR["I'm not sure which item"]
    MATCH -->|Match found| INTENT{Classify intent<br/>question or command?}
    INTENT -->|Question| ANSWER[Answer grounded<br/>in computed facts]
    INTENT -->|Command: approve| DETECT[Detect action<br/>from user words]
    INTENT -->|Command: reject| DETECT
    DETECT -->|Reject without reason| ASK_REASON["What's the reason?"]
    DETECT -->|Complete command| READY[Ready to execute<br/>caller decides]
    ANSWER --> REPLY[Reply to user]

    style UNCLEAR fill:#ffcdd2
    style READY fill:#c8e6c9
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
| Forecasting | Prophet |
| Data | Pandas, NumPy |
| Testing | 139 tests (pytest) |
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

---

## Project Structure

```
├── src/
│   ├── engine.py              # Deterministic inventory math
│   ├── context_agent.py       # Festival + promo context
│   ├── forecast.py            # Prophet models + backtest
│   ├── graph.py               # LangGraph orchestrator
│   ├── feedback_agent.py      # Learning from seller decisions
│   ├── chatbot.py             # Natural language interface
│   ├── dashboard.py           # Streamlit UI (6 tabs)
│   ├── llm_provider.py        # Multi-provider LLM support
│   ├── llm_cache.py           # Demo-safety caching
│   └── store.py               # CSV persistence
├── tests/                     # 139 tests
├── data/
│   ├── items.csv              # 25 SKUs
│   ├── suppliers.csv          # 5 suppliers
│   ├── daily_sales.csv        # 365 days of sales
│   ├── festival_calendar.csv  # 6 festivals
│   ├── promotions.csv         # Random promos
│   ├── generate_dataset.py    # Dataset generator
│   └── backtest_results.csv   # 4-policy comparison
└── requirements.txt
```

---

## License

MIT
