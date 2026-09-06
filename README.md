# SellerSense

An inventory copilot for a single small retailer. It reads sales history, works out
what is about to run out or is sitting dead on the shelf, ranks what actually needs
attention today, drafts a reorder, and adjusts its own thresholds when the seller
disagrees with it.

The problem it targets: a shop owner with 25 SKUs cannot run reorder-point
calculations by hand every morning, so they reorder on instinct. The maths for this
has existed since 1913. Nobody uses it because gathering the inputs and checking
them daily is the hard part, not the formula.

For the reasoning behind each design decision, what has already been tried, and
the failure modes worth knowing before changing anything, see
[PROJECT_REPORT.md](PROJECT_REPORT.md).

## Design rule

All arithmetic is deterministic Python. The language model ranks, explains, and
classifies intent. It never computes a quantity and never decides what counts as
at-risk.

Every number a seller sees traces to a function call. Where the model produces
text, three specific classes of claim are checked against the item's own facts
before display, and a rationale that fails any of them is replaced with a
fact-only sentence: a festival that is not this item's, a trend for an item with
no trend signal, and any supplier claim at all (the facts block never mentions
suppliers, so the whole class is fabricated by construction). An item the model
omits from its response is added back from the computed data rather than silently
dropped.

The numbers themselves are deliberately not checked, because they are never read
from the model's text — quantities are merged back from the deterministic dict by
`item_id` lookup, so a number in a rationale can only be right or ignorable.

## Components

| Module | Responsibility |
|---|---|
| `src/engine.py` | Demand rate, reorder point, safety stock, days of cover, in-transit netting, risk classification |
| `src/context_agent.py` | Festival calendar, per-item overrides, promotions |
| `src/forecast.py` | Per-SKU Prophet models and the backtest harness |
| `src/graph.py` | LangGraph orchestrator: gather signals, rank, pause for approval |
| `src/feedback_agent.py` | Applies approve/reject to per-item parameters, with guardrails |
| `src/chatbot.py` | Natural-language questions and commands against current state |
| `src/store.py` | Disk persistence for the feedback log and learned parameters |
| `src/llm_provider.py` | Selects the chat model: Groq, Gemini, OpenAI or local Ollama |
| `src/llm_cache.py` | Records and replays model responses so demos do not depend on a live call |
| `src/dashboard.py` | Streamlit interface |

### Demand estimation

Two estimators, chosen automatically by how sparse the series is. Dense series use
a 5% winsorized mean with MAD for spread — a mean rather than a median because the
reorder point needs demand *summed* over the lead time and medians do not add,
winsorized because the outlier resistance the median was chosen for is still worth
having. Above roughly half zero-days both a mean and a median collapse, so those
SKUs use Croston's method with the Syntetos-Boylan bias correction instead.

Stockout days are excluded from the trailing window, since sales on those days are
capped by available stock rather than by demand. That is a partial mitigation, not
a fix — stockouts correlate with demand spikes, so dropping them also drops the
high days. See PROJECT_REPORT.md section 9.

### In-transit stock

An item with a purchase order already on the way is still flagged — a late delivery
is exactly what a seller needs to see — but the suggested quantity nets off what is
coming, and the card shows when it arrives. Without this the same order is
recommended every morning until the goods physically land.

### Feedback guardrails

A single rejection changes nothing. The same reason has to appear in the item's
last three rejections, with no other reason among them, before a parameter moves,
and every parameter is clamped. Approvals do not reset the streak. A reason that
reflects the seller's circumstances rather than a miscalculation (`not_needed_now`)
is logged but never adjusts anything.

## Setup

Python 3.11 or later (developed on 3.13).

```bash
pip install -r requirements.txt
```

Prophet needs its CmdStan backend compiled once. This is a separate step and takes
several minutes:

```bash
python3 -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

On macOS this requires the Xcode command line tools (`xcode-select --install`).
Without this step the backtest and the cache recorder will fail; the dashboard and
the test suite will still run.

## Choosing a model

Install at least one provider. A hosted API is required for anything beyond running
on your own machine.

```bash
pip install langchain-groq          # then export GROQ_API_KEY
pip install langchain-google-genai  # then export GOOGLE_API_KEY
pip install langchain-openai        # then export OPENAI_API_KEY
```

For local development, install Ollama from ollama.com and pull a model:

```bash
ollama pull qwen3:4b
```

The provider is picked automatically from whatever is usable, preferring a hosted
API. To pin it:

```bash
export SELLERSENSE_LLM_PROVIDER=groq
export SELLERSENSE_LLM_MODEL=llama-3.3-70b-versatile
export OLLAMA_BASE_URL=http://host:11434   # only for a remote Ollama
```

## The interface

Six sections, all reachable without leaving the page:

| Section | What it shows |
|---|---|
| Overview | Position at a glance: what needs attention, cost to restock, margin at risk, the top three actions, and any demand event coming up |
| Recommendations | The ranked list with the reason for each, and approve/reject |
| Inventory | Every SKU with its risk state, cover, and reorder point |
| Ask | Natural-language questions about any flagged item |
| Evidence | The backtest: four policies over the same held-out window |
| Activity | Every decision recorded, and which setting it moved |

## Running

Run everything from the project root.

```bash
streamlit run src/dashboard.py     # the interface, on :8501
pytest                             # 139 tests
python3 src/run_backtest.py        # four-policy comparison, writes backtest_results.csv
python3 data/generate_dataset.py   # regenerate the dataset (seeded, reproducible)
python3 src/record_demo_cache.py   # record model responses for the demo script
```

Individual component checks:

```bash
python3 src/run_assessment.py      # every SKU's current risk state
python3 src/run_context_check.py   # festival resolution
python3 src/run_graph_check.py     # the full reasoning graph
python3 src/run_feedback_check.py  # three rejections moving a parameter
python3 src/run_chatbot_check.py   # questions and commands
```

## Demo mode

The dashboard defaults to cached responses read from `data/demo_llm_cache.json`,
so the rehearsed path is instant and does not depend on a model being reachable.
Switching to "Live model" in the sidebar calls the provider directly; the cache
still serves anything it already holds, and only a miss reaches the model.

**The cache shipped in this repo is stale.** The dataset was regenerated and the
prompts gained an in-transit line, so every entry now misses. Re-record before any
demo. Re-record the cache after changing prompts, the dataset, or the provider:

```bash
python3 src/record_demo_cache.py
```

Cache entries are keyed by the exact prompt, not by the model, so a cache recorded
against one provider will still be served when another is selected. Re-record when
switching providers.

Feedback and learned parameters persist to `data/feedback_log.csv` and
`data/seller_parameters.csv`. Use "Reset feedback history" in the sidebar between
rehearsals, otherwise a previous run's rejections still count toward a streak.

## Deployment

Ollama runs as a separate local server holding the model in memory. It is not
reachable from a deployed container, and bundling it is impractical on free tiers:
a 4B model needs roughly 2.6 GB of RAM before inference, and CPU-only inference is
slow enough to be unusable interactively.

Deploy with a hosted provider instead. On Streamlit Community Cloud, put the key in
the app's secrets rather than in the environment. Set `SELLERSENSE_LLM_PROVIDER`
explicitly so selection does not depend on what happens to be installed.

## Data

Synthetic, one store, 25 SKUs, 365 days, generated with a fixed seed. Patterns are
deliberate rather than incidental:

- Fast-moving staples, two FMCG lines on a growth trend
- Festival-driven demand across six events, including two SKUs that sell almost
  exclusively during one festival
- Monsoon seasonality on raincoats and umbrellas, landing inside the backtest window
- Slow movers in decline, accumulating dead stock
- Intermittent demand on accessories
- Three supplier profiles: reliable and fast, reliably slow, and high variance

`daily_sales.csv` carries `true_demand_uncensored` alongside `units_sold`. Observed
sales are capped by stock on hand, so comparing policies fairly needs the uncensored
figure. It is used only to score the backtest and is never a model input.

## Results

45-day holdout, all 25 SKUs. Fill rate is units served divided by units demanded
across the whole store — not the average of per-SKU fill rates, which gives an
8-unit gift hamper the same say as 486 units of sugar.

| Policy | Stockout days | Fill rate | Capital tied up |
|---|---|---|---|
| Seller's own rule, closed-loop | 150 | 0.828 | ₹63,672 |
| Recorded orders, replayed | 178 | 0.807 | ₹55,037 |
| Plain reorder point | 302 | 0.643 | ₹20,235 |
| Context-aware | 171 | 0.802 | ₹30,837 |

**The context-aware policy does not beat the seller on availability, and the claim
is not that it does.** It reaches 97% of the seller's fill rate on 48% of the
working capital. Against a plain reorder point it is a straight win: 43% fewer
stockout days and 16 more points of fill rate, for about 1.5x the capital.

That is the honest shape of the result, and it is the one worth selling. A shop
owner's constraint is usually cash, not shelf space: freeing ₹33,000 of working
capital for two and a half points of fill rate is a trade most would take, and it is
a trade they cannot currently evaluate at all.

Raincoat is the clearest single case. A plain reorder point never caught up with the
monsoon ramp inside the window and finished at a 0.000 fill rate, against 0.295 for
the context-aware policy and 0.216 for the seller.

An earlier version of this table reported the context-aware policy winning on all
three measures at once. That was an artefact of two things, both since fixed: the
baseline was an open-loop replay of recorded purchase orders rather than the
seller's rule run properly, and fill rate was averaged across SKUs rather than
weighted by demand. See PROJECT_REPORT.md section 8.

Lead times are drawn from each supplier's distribution during the simulation, seeded
per item and policy so the figures reproduce exactly across runs. Seed sensitivity
was measured directly: across ten alternative seed sets the plain reorder point
moved by at most 24 stockout days and 0.022 fill rate, roughly an order of magnitude
below the gaps above.

## Known limitations

Raksha Bandhan falls entirely inside the holdout window and occurs once in a year of
data, so no amount of fitting can learn its effect from history. A once-yearly event
with no prior occurrence is not recoverable from a single year of data; it needs
either more history or a category-level prior.

Demand estimation drops censored days rather than modelling them. Excluding
stockout days is correct in direction and incomplete in method: stockouts
correlate with demand spikes, so removing them removes the high days too. `mu`
remains biased low on the SKUs that stock out most (Umbrella and Raincoat sit at
0.71 and 0.81 of the truth). The fix is a censored likelihood, not deletion.

The festival uplift multipliers the context-aware policy reads are the same
numbers the generator used to synthesise demand, so the festival portion of the
backtest margin is oracle access to the simulation rather than transferable
evidence. The monsoon result does not have this problem and is the one to trust.

The intent classifier occasionally populates optional fields it was not asked to
fill. Reject reasons are therefore read from the seller's own words rather than
taken from the model.
