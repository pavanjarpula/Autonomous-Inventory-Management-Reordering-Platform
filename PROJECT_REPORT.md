# SellerSense — Engineering Report

Everything needed to pick this project up cold: what it does, why each piece is
built the way it is, what has already been tried and rejected, and where the real
edges are.

The README covers setup and commands. This document covers reasoning. Where a
decision looks arbitrary, the reason it isn't is written down here.

**Contents**

1. [Problem and thesis](#1-problem-and-thesis)
2. [System shape](#2-system-shape)
3. [The dataset](#3-the-dataset)
4. [Algorithms and why each was chosen](#4-algorithms-and-why-each-was-chosen)
5. [The reasoning layer](#5-the-reasoning-layer)
6. [Feedback and learning](#6-feedback-and-learning)
7. [Validation](#7-validation)
8. [Bugs found, and what they taught](#8-bugs-found-and-what-they-taught)
9. [Known limitations](#9-known-limitations)
10. [Extending it](#10-extending-it)
11. [File reference](#11-file-reference)

---

## 1. Problem and thesis

A shop owner running 25 SKUs reorders on instinct. They stock out of fast movers
and tie up cash in things that will not move. Both cost money; neither is visible
without daily arithmetic nobody has time for.

The mathematics is not the hard part. The reorder point has existed since 1913:

```
ROP = µd · L + z · σd · √L
```

It needs mean daily demand, its variability, the supplier's lead time, and a
service-level choice. Almost no small retailer tracks any of those, and none of
them check daily.

**The thesis: the binding constraint was never the mathematics, it was that
nobody could operate it.** SellerSense gathers the inputs, watches the thresholds,
decides what is worth interrupting someone about, explains it in their terms,
drafts the order, and adjusts when they disagree.

This matters for how the project should be pitched and extended. It is not a
forecasting contest entry. A well-tuned reorder point is a strong baseline and
the system does not reliably beat it on raw prediction. It wins on *operability*
— see [Validation](#7-validation) for what it does and does not demonstrate.

### The rule that governs everything

**All arithmetic is deterministic Python. The model ranks, explains, and
classifies intent.** It never computes a quantity and never decides what counts
as at-risk.

Every number a user sees traces to a function call, not to generated text. This
is not stylistic caution. Several of the bugs in section 8 were caught precisely
because the boundary existed, and would have shipped silently without it. When
extending this project, keep the boundary: if you find yourself asking a model to
compute, sort, look up, or match something, that work belongs in Python.

---

## 2. System shape

```
sales history ──► engine.py ──────┐
                  (demand rate,   │
                   ROP, risk)     │
                                  ├──► graph.py ──► ranked list ──► human ──► feedback_agent.py
festival calendar ► context_agent ┘    (LangGraph)                             (adjusts params)
                                  │                                                   │
                  forecast.py ────┘                                                   │
                  (Prophet, backtest)                                                 ▼
                                                                            store.py (disk)
                                                                                      │
chat question ──► chatbot.py ──────────────────────────────────────────────────────────┘
                  (same facts as the ranker)
```

The orchestrator is a LangGraph state machine:

```
START → gather_signals → reasoning → human_approval (interrupt) → END
```

- `gather_signals` assesses every SKU, nets off stock already in transit, drops
  healthy ones, caps to the eight most severe, and attaches context. No model
  involvement.
- `reasoning` is the single model call: rank, assign urgency, write a rationale.
- `human_approval` is a real `interrupt()`. Nothing is ordered without a decision.

### Why the dashboard bypasses the graph's resume

The graph's interrupt is a single batch-level pause: one `interrupt()` covers the
whole ranked list. It is not shaped for independent per-item clicks, and a
LangGraph interrupt cannot be resumed twice. The dashboard therefore reads the
recommendations off that one interrupt and applies each approve/reject directly
through the Feedback Agent, which is where the per-item logic lives anyway.

Resume itself works — `Command(resume=...)` returns the decision into state and
`test_graph.py` exercises it. It is the dashboard that declines to use it, for the
reason above. There used to be an `apply_decision` node after `human_approval`
that returned `{}` and did nothing; it has been removed, along with an unused
`Command` import that made the file look more resumable than it was. If you later
want the graph to own the whole cycle, the honest shape is one interrupt per item,
not a node bolted onto this one.

---

## 3. The dataset

Synthetic, one store, 25 SKUs, 365 days (2025-09-01 to 2026-08-31), fixed seed.
Regenerate with `python3 data/generate_dataset.py` from the project root — same
seed means byte-identical output.

### Why synthetic

Two reasons, and both are worth being upfront about. There is no public dataset
of a single Indian micro-retailer at daily SKU granularity with festival context.
And validation needs a known answer: because the patterns are deliberately
planted, it is possible to check whether the system found the signal that is
actually there rather than one it invented.

The cost is that effect sizes do not transfer to a real shop. The backtest
demonstrates the pipeline works and that context helps on patterns that genuinely
occur in retail. It does not predict what would happen in Meera's actual store.

### Planted patterns

| Pattern | SKUs | Why it is there |
|---|---|---|
| Steady fast movers | Rice, Atta, Sugar, Salt, Oil | The easy case; anything should handle these |
| Growth trend | Biscuits (+25%/yr), Namkeen (+20%) | A fixed threshold set on stale averages under-orders as demand grows |
| Festival-driven | Diyas, Sweets, Gift Hamper | Multi-festival items; category-level lift is correct for these |
| Festival-exclusive | Rakhi Set, Holi Colors Pack | Sell for exactly one festival. Near-zero the rest of the year. This was aspirational until the generator was fixed — see below |
| Monsoon seasonality | Raincoat, Umbrella | Seasonal ramp with no festival behind it, landing inside the backtest window |
| Declining dead stock | Steel Tiffin, Plastic Storage | The overstock side of the problem |
| Intermittent demand | Mobile Cover, USB Cable, Earphones | Mostly-zero series that break both a naive mean and a naive median |

Three supplier profiles, deliberately distinct: reliable and fast (4±0.8 days),
reliably slow (11±1.5), and high variance (6±3.5). Variance matters as much as
the mean — the `z · σ · √L` term is what covers it.

### Censored demand

`daily_sales.csv` carries two demand columns:

- `units_sold` — what was actually sold, **capped by stock on hand**
- `true_demand_uncensored` — what would have sold with unlimited stock

This distinction is essential and easy to miss. Once an item hits zero stock,
observed sales flatten to zero regardless of real demand. Two consequences run
through the whole system:

1. `demand_rate()` **excludes stockout days** from its trailing window. A day
   where you sold nothing because you had nothing is not evidence of low demand.
2. The backtest scores against `true_demand_uncensored`, because each policy
   maintains its own stock trajectory and would have censored differently.
   Replaying `units_sold` would bake one policy's failures into every policy.

`true_demand_uncensored` is an evaluation oracle only. **It must never be a model
input.** A real deployment does not have it.

### The `exclusive` flag

`festival_item_overrides.csv` has an `exclusive` column. An exclusive SKU responds
only to festivals it has an override for, never to a category match.

This exists because `Raksha Bandhan` declares `affected_categories = "Festive,FMCG"`,
and `Festive` covers diyas, sweets, hampers, rakhis and Holi colours — items driven
by entirely different events. Without the flag, a Holi colours pack inherited a
Raksha Bandhan lift and the system recommended stocking it for a festival nobody
buys it for.

Sweets, diyas and hampers are deliberately **not** exclusive: they genuinely move
at more than one festival, and that lift is correct.

**This flag has to be honoured in two places, and for a long time it was honoured
in only one.** `context_agent._relevant_festivals()` implemented it correctly.
`generate_dataset.festival_uplift()` did not — it fell straight through to the
category match. The producer and the consumer therefore disagreed about what
`exclusive` meant, and the producer won, because it wrote the data.

What that actually put in `daily_sales.csv`, before the fix:

| SKU | Diwali | Christmas | Sankranti | Holi | Raksha Bandhan |
|---|---|---|---|---|---|
| Holi Colors Pack | 311 | 150 | 129 | **103** | 229 |

The SKU whose entire purpose was to sell for one festival sold *less* during its
own festival than during three others. After the fix it reads 97 + 107 units
across the Holi window and ≤4 in every other month, which is the pattern this
table always claimed.

The cost of the mismatch was not cosmetic. 229 units of Rakhi-week demand for Holi
Colors sat inside the holdout window, and the context agent — behaving exactly as
designed — refused to stock for it. The backtest scored the correct behaviour as
that SKU's worst result across all policies. A bug in the fixture made the fix look
like a regression.

If you add SKUs, ask whether each is single-festival, and change exclusivity
semantics in **both** files or neither. Category alone is too coarse to carry the
distinction.

---

## 4. Algorithms and why each was chosen

### Demand rate — two estimators, chosen automatically

`demand_rate()` switches on sparsity, at `intermittent_threshold = 0.5`.

**Dense series → trailing 5% winsorized mean + MAD.** This was a plain trailing
median until an audit against `true_demand_uncensored` showed it biased the whole
system low. The reasoning behind the median was sound as far as it went — retail
sales have outlier days (a bulk buyer, a data-entry error) that a mean chases —
but it answered the wrong question. The reorder point needs *expected demand
summed over the lead time*, and `E[Σ over L days] = L · E[daily]`. Means add that
way; medians do not. On right-skewed count data the median sits below the mean, so
`mu · L` came out low for essentially every SKU.

Measured against the true trailing mean before the change: median ratio 0.87,
with nine of twenty-five SKUs low by more than 20% (Shampoo 0.63, Toothpaste 0.74,
Detergent 0.77). Four estimators were compared:

| Estimator | Median ratio to truth | SKUs >20% low |
|---|---|---|
| Trailing median + MAD (old) | 0.87 | 9/25 |
| Mean of stockout-excluded days | 0.93 | 8/25 |
| Mean of all days, no exclusion | 0.82 | 12/25 |
| Poisson MLE with right-censoring | 0.94 | 6/25 |

Almost the entire gain is the median→mean swap. Winsorizing at 5% keeps the
outlier resistance the median was originally chosen for, at a cost of about one
point (0.98 vs 0.99 for a raw mean) — a single absurd value is clipped to the 95th
percentile instead of moving the estimate. `sigma` stays on MAD, scaled by 1.4826
to be comparable to a standard deviation: it is a spread estimate, where
robustness costs nothing and no additivity argument pulls toward the moment
estimator. Its deviations are taken about the median, not the new `mu`, which is
the MAD's own definition and avoids inflating sigma on skewed series.

The censored MLE is a real further improvement and is the obvious follow-up; see
section 9.

**Sparse series (≥50% zero days) → Croston's method with the Syntetos-Boylan
correction.** This is not a refinement, it fixes a real failure. A median of a
mostly-zero series reads exactly `0` regardless of the true rate, because zeros
are the majority. The accessories were reported as having no demand at all, so
the system recommended ordering nothing for them.

Croston separates the series into *how big a sale is when one happens* and *how
many periods pass between sales*, smooths each independently, and divides. SBA
applies `(1 - α/2)` because Croston is known to be biased upward.

Both paths return `(mu, sigma)`, so callers never need to know which ran.

### Recent trend — a deliberately cheap proxy

`recent_trend_ratio()` compares a 14-day rate against a 180-day baseline, capped
at 20×. It exists to catch a seasonal ramp — the monsoon on raincoats — that a
single trailing window cannot see and that the festival calendar does not know
about, because no festival is involved.

It reuses `demand_rate()` for both windows, so it inherits stockout exclusion and
the estimator switch. No separate estimator to validate.

The cap is deliberate: a finite "20× normal" is as informative to a human as
"infinite" and avoids an unbounded float propagating into places that expect an
ordinary number.

**Known blind spot:** an item already at zero stock shows suppressed recent sales,
so the ratio understates a ramp precisely when the item is in the worst shape.
Fixing this properly needs either a look-back to before the stockout began, or the
Prophet forecast wired into the live loop. Accepted for now.

### Reorder point

```
ROP  = µd · L + z · σd · √L
Qty  = ROP − (on hand + on order)
```

`z = 1.65` (≈95% service level), per-item adjustable and clamped to `[1.00, 2.50]`.

Note that when demand is perfectly steady, `σ = 0` and `z` changes nothing. This
surprises people, and it is correct: with no variability there is nothing for a
safety buffer to cover. It also means supplier *lead-time variance* matters as
much as the average — the high-variance supplier in the dataset exists to make
that visible.

### Risk classification

- `on_hand ≤ ROP` → **stockout risk** (boundary inclusive)
- `days_of_cover ≥ 45` → **overstock**
- otherwise → **healthy**

Healthy items never reach the model. It never has to decide what counts as
at-risk, only how to prioritise what already does.

### Forecasting

Per-SKU Prophet, `seasonality_mode="multiplicative"` because a festival lift is a
percentage of baseline, not a fixed number of units. Festivals enter as a
`holidays` frame with a ramp window (demand climbs for days before an event, not
on the day). Promotions enter as a binary regressor.

**The constraint worth knowing:** a Prophet regressor must be known for the dates
you are forecasting, not just those you trained on. A festival calendar and a
promotion plan are known weeks ahead. Weather is not, beyond a few days — that is
the concrete reason weather was left out, not a hand-wave about scope.

---

## 5. The reasoning layer

The model gets already-computed facts and does three narrow jobs: assign urgency,
write a rationale citing the specific driver, and (in chat) classify question vs
command. Four guards sit around its output, each added after watching a specific
failure:

| Guard | What it prevents |
|---|---|
| Quantities merged from computed facts by item id | A generated number ever being the number acted on |
| Unknown item ids dropped | A hallucinated SKU entering the list |
| Rationale grounding check | A rationale naming a festival that is not this item's own, claiming a trend for an item with no trend signal, or making any supplier claim at all |
| Coverage backfill | Items silently vanishing when the model omits them from its reply |

Each recommendation carries a `source` field — `model`, `fallback` (model
unavailable, ranked by rule), or `backfill` (model omitted it) — so the interface
can label rule-based entries honestly rather than presenting them as reasoning.

### The top-eight cap

`gather_signals` caps to the eight most severe items. Two independent reasons:
eighteen alerts is a list nobody reads, and a longer batch takes measurably longer
and gives the model more room to omit something. The full flagged count travels in
state, so nothing pretends fewer items needed attention than actually did.

### The chat surface

Which item a message refers to is resolved **deterministically**, by scoring
identifying words and dropping size suffixes and packaging nouns. Reject reasons
are read from the seller's own words, never taken from the model — a small model
populates optional enum fields on nearly every call, and a fabricated reason would
land in the audit log as the seller's own words and move a real parameter.

Chat facts are rendered as sentences rather than `key=value` pairs. Terse form is
fine for ranking, where the model sorts known fields. It is not fine for answering
a person, where the model has to know that "flagged" implies "reorder this" before
it can say anything useful.

Commands are **detected, never executed**. `respond()` returns a structured intent
shaped like `submit_feedback()`'s arguments; acting on it is the caller's choice.

---

## 6. Feedback and learning

Four reason codes, three of which move something:

| Reason | Effect |
|---|---|
| `qty_too_high` | Lowers per-item `z` by 0.15 |
| `qty_too_low` | Raises per-item `z` by 0.15 |
| `supplier_unreliable` | Adds 1 day to that item's lead-time buffer |
| `not_needed_now` | Logged, **adjusts nothing** |

`not_needed_now` is deliberately inert. It reflects the seller's circumstances —
cash flow, timing — not a miscalculation. Letting it move a threshold would drift
the system on a signal that says nothing about whether the maths was right.

### Guardrails

- **Hysteresis:** the same reason must appear in this item's last
  `CONSISTENCY_THRESHOLD = 3` **rejections**, with no other reason among them. A
  different rejection reason in between resets the count. Approvals are filtered
  out before the scan and therefore do **not** reset it — three `qty_too_high`
  rejections spread across weeks of otherwise-approved recommendations still trip
  the threshold. That is intended (an approval agrees with a *different*
  recommendation and says nothing about whether this complaint was resolved) but
  it is not three-in-a-row on the calendar, and earlier wording implied it was.
- **Clamping:** `z ∈ [1.00, 2.50]`, lead-time buffer `∈ [0, 10]` days.
- **Known coupling:** `z` feeds both the order quantity and the reorder point, and
  `classify_risk()` compares on-hand against the reorder point. Lowering `z` on
  `qty_too_high` therefore also makes the item less likely to be *flagged*, not
  just cheaper to restock. Defensible — a seller who thinks the quantity is too
  high usually thinks the trigger is too eager — but a real second-order effect,
  recorded here so it is discovered on purpose rather than by surprise.
- **Audit:** every decision is logged whether or not it changed anything, with the
  triggering row carrying the old and new values.

One rejection is ambiguous — quantity wrong, no cash this week, supplier closed.
Moving a threshold on a single signal drifts the system in a direction nobody chose.

### Persistence

`store.py` is the only module that writes to disk. `feedback_agent.py` stays pure
so its hysteresis and clamping can be tested without a filesystem.

Two subtleties that will bite if you rewrite this:

- Timestamps must be parsed back to datetimes on load. The streak logic sorts on
  that column; ISO strings sort correctly *by luck* and would break silently if the
  format ever changed.
- `astype(object)` before restoring `None` on the reason column. An all-NaN column
  reloads as `float64`, which coerces an assigned `None` straight back to `NaN`.

---

## 7. Validation

### Method

A replay backtest over a 45-day holdout (2026-07-18 to 2026-08-31 — chosen because
the monsoon ramp lands inside it). Four policies see the same true demand and the
same supplier lead-time distributions, each maintaining its own stock trajectory.
The forecaster trains only on data before the window opens, so nothing leaks.

| Policy | What it is |
|---|---|
| `seller` | The shop owner's own gut-feel rule, run closed-loop. **The headline baseline.** |
| `actual_replay` | Open-loop replay of the recorded purchase orders. Reference only. |
| `plain_rop` | Textbook reorder point off a trailing rate. No calendar, no forecast. |
| `context_aware` | Safety stock sized off the Prophet forecast plus the calendar uplift. |

**Why there are two baselines.** `actual_replay` used to be the only one, and it
was not a fair comparator. It fires fixed order dates derived from a stock
trajectory that no longer applies once the simulator draws its own lead times, so
an open-loop schedule was competing against two closed-loop policies and losing
partly for that reason. `make_seller_heuristic_policy()` runs the same constants
the generator used (`rop = rate × 5`, `qty = rate × 14`) but reacts to the stock
level it is actually handed. Beating a properly-run version of the seller's own
rule is the only version of the claim worth making. The replay is kept and
reported so the difference between the two stays visible.

**Fill rate is reported two ways, and the weighted one is the headline.**
`weighted_fill_rate` is units served ÷ units demanded across the store.
`sku_mean_fill_rate` is the unweighted average across SKUs, which gives an 8-unit
gift hamper the same say as 486 units of sugar and flatters any policy that wins
on the long tail. Both are printed side by side so the gap cannot hide.

Lead-time draws are seeded per item and policy with a stable checksum, so the
figures reproduce exactly across runs and machines. Reproducible is not the same as
low-variance, so the sensitivity was measured rather than asserted: across ten
alternative seed sets the plain reorder point moved by at most 24 stockout days and
0.022 fill rate — roughly an order of magnitude below the gaps below.

### Results

| Policy | Stockout days | Weighted fill | SKU-mean fill | Capital tied up |
|---|---|---|---|---|
| Seller's own rule, closed-loop | **150** | **0.828** | 0.806 | ₹63,672 |
| Recorded orders, replayed | 178 | 0.807 | 0.774 | ₹55,037 |
| Textbook reorder point | 302 | 0.643 | 0.671 | **₹20,235** |
| SellerSense | 171 | 0.802 | 0.805 | ₹30,837 |

**SellerSense does not beat the seller on availability.** It reaches 97% of the
seller's fill rate on 48% of the working capital. Against the textbook reorder
point it is a straight win — 43% fewer stockout days and 16 more points of fill
rate — for about 1.5× the capital.

State the shape of that plainly rather than defending it after the question comes.
A shop owner's binding constraint is usually cash, not shelf space. The seller's
rule buys its availability by holding ₹63,672 of stock; SellerSense gets within
2.6 points of it on ₹30,837. Freeing ₹33,000 of working capital is the product,
and it is a trade the owner currently has no way to evaluate.

Clearest single case: Raincoat through the monsoon. The textbook reorder point
never caught up with the ramp inside the window and finished at a **0.000** fill
rate, against **0.295** with the seasonal signal and 0.216 for the seller.

### What changed, and why the old numbers were wrong

An earlier version of this table read:

| Policy | Stockout days | Fill rate | Capital |
|---|---|---|---|
| What the seller actually did | 220 | 0.668 | ₹52,010 |
| Textbook reorder point | 338 | 0.567 | ₹18,640 |
| SellerSense | 188 | 0.753 | ₹30,121 |

and claimed a win on all three measures at once. That claim was an artefact of
three separate problems, all now fixed:

1. **The baseline was open-loop** (above). A static schedule cannot recover from a
   stockout; the seller's actual rule can, and does.
2. **Fill rate was averaged across SKUs, not weighted by demand.** On the old
   numbers the weighted figures were 0.750 for the seller against 0.771 for
   SellerSense — a 2.1-point gap being reported as 8.5. Nearly every large win sat
   on items with under 40 units of demand, while SellerSense was *behind* the
   seller on Rice, Atta, Tea, Sugar and Namkeen, which are most of the store's
   volume.
3. **The demand estimator was biased low** (section 4), which depressed every
   policy that computed its own reorder point and depressed them unevenly.

The Umbrella showcase is also gone. It used to read 0.000 for the textbook policy
against 0.446 for SellerSense; with an unbiased demand estimate the textbook policy
now catches the monsoon ramp and posts 0.658 against SellerSense's 0.468 — it wins
there. Raincoat replaces it as the flagship case. Losing a demo example to a
correctness fix is the correct trade, and the fact that the old example depended on
a bug is itself the argument for auditing your own evidence.

### What this does and does not prove

It proves the pipeline works end to end, that the comparison now includes a real
baseline rather than a strawman, and that context-awareness buys a large capital
saving at a small availability cost on patterns that genuinely occur in retail.

It does not prove an effect size for a real shop. One year of data cannot
demonstrate anything about an annual event that occurs once in it. And the festival
uplift multipliers the context-aware policy reads are the *same numbers the
generator used to synthesise demand* — that portion of the margin is oracle access
to the data-generating process, not transferable evidence. The monsoon result is
clean by contrast: `MONSOON_MULTIPLIER` lives only in the generator and appears in
no CSV, so Prophet genuinely learns it from history. Lead with Raincoat.

### Test suite

139 tests, `pytest` from the project root.

| File | Tests | Covers |
|---|---|---|
| `test_engine.py` | 32 | Estimators, estimator bias, ROP, risk, trend proxy, in-transit netting, parameter overrides |
| `test_graph.py` | 24 | Signal gathering, cap, merge safety, all three grounding rules, interrupt/resume |
| `test_chatbot.py` | 21 | Item resolution, intent, grounding, reason extraction |
| `test_feedback_agent.py` | 13 | Hysteresis, clamping, reason mapping |
| `test_forecast.py` | 12 | Policy logic, closed-loop baseline, simulator bookkeeping |
| `test_context_agent.py` | 11 | Festival windows, overrides, exclusivity |
| `test_llm_provider.py` | 10 | Provider selection and error messages |
| `test_store.py` | 9 | Round-trips, and a streak surviving a restart |
| `test_llm_cache.py` | 7 | Record and replay |

A caveat this suite earned the hard way: it was fully green while the dataset
contradicted its own specification (section 3). Every test asserted against
fixtures the tests themselves constructed, so nothing compared the *generator's*
output to the *agent's* assumptions. The estimator bias survived for the same
reason — `test_demand_rate_still_uses_median_for_mostly_nonzero_series` asserted
`mu == 10.0` and passed, because it encoded the behaviour rather than the
requirement. Tests that assert what the code does will never tell you the code is
wrong.

---

## 8. Bugs found, and what they taught

Kept because each represents a class of failure that will recur, and because every
one surfaced from running the real system rather than from a unit test with clean
fixtures.

**A type that passed every value check.** Croston returned `numpy.float64` instead
of a plain float — mixed arithmetic with array elements silently promotes. Every
comparison passed. It only surfaced when the orchestrator serialised its state and
the encoder rejected the type. *The regression test asserts the type, not the
value; a value-only assertion would never have caught it.*

**Cross-item fabrication.** Given eighteen items in one prompt, the model pasted
"Raksha Bandhan" into rationales for raincoats and umbrellas — pattern-matching a
phrase from neighbouring items. Nothing errored. *Found by reading output line by
line. Any rationale naming a festival that is not the item's own is now replaced.*

**Silent omission.** The model returned sixteen of eighteen items. No error, no
warning — two simply were not there, and a seller would never have been told those
items were at risk. *Found by comparing counts between runs. The response is now
diffed against the input set.*

**A non-reproducible backtest.** Lead-time draws were seeded with Python's
`hash()`, which is randomised per process, so the headline numbers changed on every
run. For a project where the backtest is the evidence, that is a credibility
problem. *Switched to a stable checksum.*

**Blaming the model for a matching bug.** Chat failed on "why is toothpaste
recommended". The cause was not the model: item matching required the full name
including the size suffix, so "toothpaste" never matched "Toothpaste 150g". And
separately, the model was handed raw key-value pairs with no explanation of what
"flagged" meant. *Both fixes moved work out of the model.*

**A fabricated reason in the audit log.** The intent classifier populated
`reject_reason` on essentially every call, including plain questions. Inert at the
time, but a reason the seller never gave would have been recorded as their own
words and moved a parameter. *Reasons are now read only from the message.*

**Category granularity.** Holi colours were recommended for Raksha Bandhan because
both sit under `Festive`. Not a model failure — the data model was too coarse.
*Fixed with the `exclusive` flag — in the context agent. See the next entry.*

**A fix applied to one side of an interface.** The `exclusive` fix above landed in
`context_agent.py` and was written up as done. It never landed in
`generate_dataset.py`, whose `festival_uplift()` had no exclusivity check at all.
For months the producer and the consumer of that column disagreed about what it
meant, and the producer won, because it wrote the CSV. The dataset contained 229
units of Rakhi-week demand for Holi Colors Pack — inside the holdout window — that
the agent was deliberately built to ignore, so the backtest scored the correct
behaviour as that SKU's worst result. *The lesson is not "test more." It is that a
fix to a shared contract has a producer side and a consumer side, and closing the
ticket after one of them is how a codebase ends up disagreeing with its own
documentation. Both call sites now carry a comment pointing at the other.*

**An estimator that was right about robustness and wrong about the question.**
`demand_rate()` used a trailing median, justified — correctly — by outlier
resistance. But the reorder point needs expected demand summed over the lead time,
and medians do not add. Measured against `true_demand_uncensored`, `mu` came in at
a 0.87 median ratio with nine of twenty-five SKUs more than 20% low, so every
reorder point in the system was low. *Nothing errored, no test failed, and the
existing test asserted `mu == 10.0` — it encoded the behaviour, so it defended the
bug. Found only by scoring the estimator against the ground-truth column the
dataset already carried. If you have an oracle, use it on your components, not
just on your headline.*

**A parameter that existed, was tested, and was never wired up.**
`order_quantity()` took `on_order`, `test_engine.py` asserted it worked, and
nothing in the live path ever passed it — only the two backtest policies guarded
against double-ordering, via their own separate `if pending` check. On the
dashboard's default business date, 15 of 18 flagged items already had stock in
transit and 12 of them should have been asking for nothing. The product was
telling the seller to double-order on 83% of its recommendations. *A tested
function is not a wired function. Grep for callers, not for tests.*

**Evidence that flattered itself twice over.** The headline backtest compared an
open-loop replay of recorded purchase orders against two closed-loop policies, and
scored fill rate as an unweighted mean across SKUs. Both choices happened to point
the same way. Fixing the baseline and weighting by demand turned "beats the seller
on all three measures at once" into "reaches 97% of the seller's fill rate on 48%
of the capital" — a narrower claim, and the true one. *Neither was a lie; both were
defaults nobody re-examined once the number came out favourable. That is the
direction bias runs in, and it is why the sensitivity of a headline should be
measured rather than assumed.*

**A date picker that killed the app.** Any date outside the data range raised
straight out of `assess_item`. *Clamped to the dataset.*

**Navigation that lost its place.** `st.tabs` cannot hold selection across a rerun,
so approving a recommendation — the core interaction — threw the user back to the
first tab. *Replaced with state-bound navigation.*

The pattern worth carrying forward: **most of these were silent.** Only two raised
an exception. Check for absence and for quiet wrongness, not just for errors.

The later entries sharpen that into something more uncomfortable. The last five
were all silent *and* green — the test suite passed throughout, and in two cases
the tests actively protected the bug by asserting current behaviour. Three of them
were only findable by checking the system against something outside itself: the
generator against the agent, the estimator against the uncensored oracle, the
headline against a fairer baseline. A codebase can be internally consistent, fully
tested, and confidently documented while being wrong about the world. The check
that catches that is never another unit test.

---

## 9. Known limitations

**The cold-start case.** Raksha Bandhan falls entirely inside the holdout window
and occurs once in a year of data. No amount of fitting can learn its effect. For
the SKU that sells almost exclusively during it, no policy beats any other. This
is structural, not a bug: an annual event with no prior occurrence is not
recoverable from a single year. It needs more history or a category-level prior.

**Trend proxy blind during a stockout.** Described in section 4.

**Demand estimation still drops censored days rather than modelling them.**
`demand_rate()` excludes stockout days because those observations are capped by
stock, which is correct reasoning — but deletion is a partial mitigation, not a
fix. Stockouts correlate with demand spikes, so removing them also removes the
high days. This is the classic lost-sales selection bias, and it leaves `mu`
biased low on exactly the SKUs that stock out most: Umbrella and Raincoat come in
at 0.71 and 0.81 of the truth even after the winsorized-mean change. The correct
treatment keeps those days as right-censored observations (`demand ≥ on_hand_start`)
and fits a censored likelihood. A Poisson censored MLE was measured at a 0.94
median ratio against 0.93 for the current estimator, with six SKUs more than 20%
low instead of eight. It is the clearest remaining accuracy win and needs
`on_hand_start` threaded into the estimator plus a one-dimensional optimiser.

**The festival uplift multipliers are the generator's own parameters.**
`generate_dataset.festival_uplift()` and `context_agent.get_context()` read the
same numbers out of the same two CSVs, and the context-aware policy sizes orders
off `baseline_mu × uplift_multiplier`. Calling that "domain knowledge rather than
learned" is true of the *mechanism* and misleading about the *evidence*: in this
dataset it is oracle access to the data-generating process, so the festival
portion of the backtest margin does not transfer to a real shop. The monsoon
result does — `MONSOON_MULTIPLIER` exists only in the generator and appears in no
CSV, so Prophet learns it from history like any real forecaster would. A cleaner
version of this experiment would perturb the calendar multipliers away from the
generator's values and re-measure.

**The `actual_replay` policy discards recorded lead times.** `purchase_orders.csv`
carries `lead_time_actual_days`, but `simulate_policy()` redraws lead times from
the supplier distribution for every policy including the replay. Either use the
recorded values for that one policy or drop the column, which currently looks
load-bearing and is not.

**Latency.** Eight items takes roughly 78 seconds on a local 4B model. The demo
defaults to cached responses for this reason. A hosted API is substantially faster
and is the deployment path regardless.

**Scope.** One store, synthetic data, no live marketplace integration, no WhatsApp
delivery.

---

## 10. Extending it

Roughly in order of value per unit of effort.

**Real marketplace exports.** A column normaliser for Amazon/Flipkart CSVs. Mostly
mapping work, and it converts "synthetic data" into "your data" — the single
biggest credibility gain available.

**Persist across sellers.** Everything is currently single-tenant. `store.py` is
the only writer, so this is a keying change, not an architectural one.

**WhatsApp delivery.** The channel this segment actually uses. The blocker is
business verification and template approval, not code. Keep it behind an interface;
the Twilio sandbox works for development.

**More history.** Most remaining forecasting weakness is a data problem. Two or
three years would let annual events be learned rather than asserted.

**Wire the forecast into the live loop.** `forecast.py` is currently used only by
the backtest. The reasoning layer sees the cheap trend proxy instead. Connecting it
would fix the stockout blind spot, at a latency cost worth measuring first.

**Not worth doing:** a larger model. Every accuracy problem in section 8 was solved
by moving work *out* of the model, not by adding parameters.

---

## 11. File reference

### Source

| File | Responsibility |
|---|---|
| `engine.py` | Demand estimation, ROP, safety stock, risk, trend proxy. Pure functions, no I/O |
| `context_agent.py` | Festival windows, overrides, exclusivity, promo regressors. Pure |
| `forecast.py` | Prophet fitting, the four policies, the simulator |
| `graph.py` | LangGraph orchestrator, prompt construction, output guards |
| `feedback_agent.py` | Reason→parameter mapping, hysteresis, clamping. Pure |
| `chatbot.py` | Item resolution, intent, grounded answers |
| `store.py` | Disk persistence. The only module that writes |
| `llm_provider.py` | Provider selection across Groq, Gemini, OpenAI, Ollama |
| `llm_cache.py` | Record/replay of model responses |
| `dashboard.py` | Streamlit interface |
| `run_*.py` | Standalone checks for each layer |

### Data

| File | Contents |
|---|---|
| `daily_sales.csv` | Per SKU per day: sold, stockout flag, on hand, uncensored demand |
| `items.csv` | SKU, category, supplier, cost, price |
| `suppliers.csv` | Mean and standard deviation of lead time |
| `festival_calendar.csv` | Date, type, affected categories, uplift, ramp days |
| `festival_item_overrides.csv` | Per-item uplift and the `exclusive` flag |
| `promotions.csv` | Seller-run promotion windows |
| `purchase_orders.csv` | What the naive policy actually ordered — the backtest baseline |
| `backtest_results.csv` | Per item and policy results |
| `feedback_log.csv` | Runtime; regenerated. Gitignored |
| `seller_parameters.csv` | Runtime; regenerated. Gitignored |
| `demo_llm_cache.json` | Recorded responses for the demo script |

### Conventions worth preserving

- Core modules are pure: dataframes and dicts in, values out, no file I/O. Driver
  scripts and the dashboard do the reading and writing.
- Provider packages import lazily and are optional; a missing one gives the exact
  `pip install` line rather than breaking import for everyone.
- The cache is keyed by exact prompt, not by model. **Re-record after changing a
  prompt, the dataset, or the provider** — otherwise you are replaying one model's
  answers while claiming to run another.
