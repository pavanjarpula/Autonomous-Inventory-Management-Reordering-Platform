# Audit and fixes

An external review of this project found two correctness bugs, three
methodological problems in the evidence, and a set of places where the
documentation was more confident than the code justified. All are fixed. This
file is the summary; the reasoning lives in `PROJECT_REPORT.md` sections 3, 4, 7,
8 and 9.

## The headline result changed

**Before** — an open-loop replay baseline, fill rate averaged across SKUs:

| Policy | Stockout days | Fill rate | Capital |
|---|---|---|---|
| What the seller actually did | 220 | 0.668 | ₹52,010 |
| Textbook reorder point | 338 | 0.567 | ₹18,640 |
| SellerSense | 188 | 0.753 | ₹30,121 |

*"Beats the seller on all three measures at once."*

**After** — closed-loop baseline, fill rate weighted by demand:

| Policy | Stockout days | Weighted fill | Capital |
|---|---|---|---|
| Seller's own rule, closed-loop | **150** | **0.828** | ₹63,672 |
| Recorded orders, replayed | 178 | 0.807 | ₹55,037 |
| Textbook reorder point | 302 | 0.643 | **₹20,235** |
| SellerSense | 171 | 0.802 | ₹30,837 |

*SellerSense reaches 97% of the seller's fill rate on 48% of the working capital,
and strictly beats a textbook reorder point on availability.*

The old claim was an artefact, not a lie. Two defaults nobody re-examined —
an open-loop baseline and an unweighted average — happened to point the same way.
The new claim is narrower and true, and it is arguably the better pitch: a shop
owner's binding constraint is cash, and this frees ₹33,000 of it.

**The Umbrella showcase is gone.** With an unbiased demand estimate the textbook
policy now catches the monsoon ramp and beats SellerSense there (0.658 vs 0.468).
Raincoat replaces it: 0.000 for the textbook policy, 0.295 for SellerSense, 0.216
for the seller.

## Correctness bugs

**1. `exclusive` was honoured in the agent but not the generator.**
`context_agent._relevant_festivals()` implemented the flag;
`generate_dataset.festival_uplift()` never did. The dataset had Holi Colors Pack
selling at peak rate during five festivals, including 229 units during Raksha
Bandhan inside the holdout window — so the backtest scored the agent's *correct*
refusal to stock it as that SKU's worst result. Fixed in the generator; dataset
regenerated. Holi Colors now sells 97+107 units across the Holi window and ≤4 in
every other month.

**2. In-transit stock was ignored in the live path.** `order_quantity()` took
`on_order` and was tested, but nothing outside the backtest ever passed it. On the
dashboard's default date, 15 of 18 flagged items already had stock in transit and
12 should have been asking for nothing — the product was recommending a duplicate
order on 83% of its recommendations. `on_order_qty()` and `days_to_next_arrival()`
added; threaded through `assess_item`, the graph, the dashboard and all four
`run_*` scripts. Flagged items with stock coming now show a zero quantity and an
arrival date rather than being suppressed.

## Method and estimation

**3. Demand rate was biased low.** A trailing median was replaced with a 5%
winsorized mean: the reorder point needs demand summed over the lead time, and
medians do not add. Measured against `true_demand_uncensored`, the median ratio
moves 0.87 → 0.93 and the count of SKUs more than 20% low moves 9/25 → 8/25.
Winsorizing preserves the outlier resistance the median was chosen for.

**4. Fill rate is now reported weighted by demand,** with the unweighted SKU mean
shown beside it, plus a per-SKU table sorted by volume so a long-tail-driven
headline cannot hide.

**5. The baseline is now closed-loop.** `make_seller_heuristic_policy()` runs the
seller's own rule reactively. The open-loop replay is kept and reported as
`actual_replay` for reference.

## Honesty of the write-up

**6.** Removed the dead `apply_decision` node and unused `Command` import.
Corrected the claim that the graph "is not resumed" — resume works and is tested;
the *dashboard* declines to use it, for a stated reason.

**7.** The grounding check now covers three claim classes (borrowed festival,
invented trend, any supplier claim) instead of one. The README no longer says
rationales are "validated against the underlying facts", which was broader than
what the code did.

**8.** Corrected: the hysteresis streak counts rejections only, so approvals do
not break it. Documented the `z` coupling (lowering it also reduces how often an
item is flagged). Test count 123 → 139. `requirements.txt` pinned with upper
bounds. `assess_all` cached. New limitations recorded: censored-demand deletion,
and the fact that the festival multipliers are the generator's own parameters.

## Before you demo

`data/demo_llm_cache.json` is **stale** — the dataset changed and the prompts
gained an in-transit line, so every entry misses. Run
`python3 src/record_demo_cache.py` against a live provider first.

## Verification

- 139 tests pass (was 125; three that encoded the old median were rewritten with
  hand-verified values, eleven added for the new behaviour).
- `run_backtest.py` reproduces byte-identical across runs.
- Seed sensitivity measured: across ten alternative lead-time seed sets the
  textbook policy moves at most 24 stockout days and 0.022 fill rate — about an
  order of magnitude below the reported gaps.
