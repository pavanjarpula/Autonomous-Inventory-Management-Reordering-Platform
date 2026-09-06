"""
SellerSense POC dataset generator — one store, 25 SKUs, 365 daily days.

Purpose-built so each output file maps directly onto one agent:
  items.csv, suppliers.csv        -> Reasoning/Orchestrator agent (cost, price, lead time)
  daily_sales.csv                 -> Consumption Agent (units_sold time series, ds/y-ready)
  festival_calendar.csv,
  festival_item_overrides.csv,
  promotions.csv                  -> Context Agent (Prophet holidays df + regressor columns)
  purchase_orders.csv             -> the naive "seller's actual behaviour" baseline policy,
                                      ready to drop into the Phase-0 backtest as-is
  feedback_log.csv                -> empty stub for the Feedback Agent to write into

Demand model per item: base rate x day-of-week x trend x festival uplift x promo uplift,
sampled through Poisson noise (Bernoulli-gated Poisson for the intermittent accessories).
Festival dates for 2025-26 are approximate (lunar-calendar festivals shift year to year) —
good enough for a POC, verify against a panchang before using outside a demo.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

START = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2026-08-31")
DATES = pd.date_range(START, END, freq="D")
N_DAYS = len(DATES)

# ---------------------------------------------------------------- suppliers
suppliers = pd.DataFrame([
    dict(supplier_id="S1", supplier_name="Shree Grocers Wholesale",   mean_lead_time_days=4,  lead_time_std_days=0.8),
    dict(supplier_id="S2", supplier_name="Metro FMCG Distributors",   mean_lead_time_days=6,  lead_time_std_days=1.2),
    dict(supplier_id="S3", supplier_name="TechBazaar Accessories",    mean_lead_time_days=8,  lead_time_std_days=2.0),
    dict(supplier_id="S4", supplier_name="Utsav Festive Supplies",    mean_lead_time_days=6,  lead_time_std_days=3.5),  # high-variance, unreliable
    dict(supplier_id="S5", supplier_name="Bharat Home Essentials",    mean_lead_time_days=11, lead_time_std_days=1.5),  # predictably slow
])

# ---------------------------------------------------------------- items
# base_rate = mean units/day before any multiplier. trend = total fractional change
# over the full year (0.25 = +25%, -0.4 = -40%). intermittent items use p/jump instead.
items = pd.DataFrame([
    dict(item_id="I001", item_name="Rice 5kg",             category="Staples",   supplier_id="S1", unit_cost_inr=220, unit_price_inr=260, base_rate=8.0,  trend=0.05),
    dict(item_id="I002", item_name="Wheat Atta 5kg",       category="Staples",   supplier_id="S1", unit_cost_inr=190, unit_price_inr=225, base_rate=7.0,  trend=0.05),
    dict(item_id="I003", item_name="Toor Dal 1kg",         category="Staples",   supplier_id="S1", unit_cost_inr=130, unit_price_inr=155, base_rate=6.0,  trend=0.0),
    dict(item_id="I004", item_name="Sugar 1kg",            category="Staples",   supplier_id="S1", unit_cost_inr=42,  unit_price_inr=50,  base_rate=9.0,  trend=0.0),
    dict(item_id="I005", item_name="Cooking Oil 1L",       category="Staples",   supplier_id="S2", unit_cost_inr=130, unit_price_inr=155, base_rate=7.0,  trend=0.05),
    dict(item_id="I006", item_name="Salt 1kg",             category="Staples",   supplier_id="S1", unit_cost_inr=18,  unit_price_inr=22,  base_rate=10.0, trend=0.0),
    dict(item_id="I007", item_name="Tea 250g",             category="Staples",   supplier_id="S2", unit_cost_inr=90,  unit_price_inr=110, base_rate=5.0,  trend=0.0),
    dict(item_id="I008", item_name="Biscuits Pack",        category="FMCG",      supplier_id="S3", unit_cost_inr=25,  unit_price_inr=30,  base_rate=12.0, trend=0.25),
    dict(item_id="I009", item_name="Namkeen 200g",         category="FMCG",      supplier_id="S3", unit_cost_inr=35,  unit_price_inr=45,  base_rate=8.0,  trend=0.20),
    dict(item_id="I010", item_name="Detergent 1kg",        category="FMCG",      supplier_id="S2", unit_cost_inr=95,  unit_price_inr=115, base_rate=4.0,  trend=0.0),
    dict(item_id="I011", item_name="Shampoo 200ml",        category="FMCG",      supplier_id="S2", unit_cost_inr=110, unit_price_inr=140, base_rate=3.0,  trend=0.0),
    dict(item_id="I012", item_name="Soap Pack of 4",       category="FMCG",      supplier_id="S2", unit_cost_inr=70,  unit_price_inr=90,  base_rate=5.0,  trend=0.0),
    dict(item_id="I013", item_name="Toothpaste 150g",      category="FMCG",      supplier_id="S3", unit_cost_inr=60,  unit_price_inr=75,  base_rate=3.5,  trend=0.0),
    dict(item_id="I014", item_name="Diyas & Candles Set",  category="Festive",   supplier_id="S4", unit_cost_inr=80,  unit_price_inr=120, base_rate=0.3,  trend=0.0),
    dict(item_id="I015", item_name="Sweets Gift Box",      category="Festive",   supplier_id="S4", unit_cost_inr=250, unit_price_inr=350, base_rate=0.2,  trend=0.0),
    dict(item_id="I016", item_name="Rakhi Set",             category="Festive",   supplier_id="S4", unit_cost_inr=60,  unit_price_inr=100, base_rate=0.05, trend=0.0),
    dict(item_id="I017", item_name="Holi Colors Pack",     category="Festive",   supplier_id="S4", unit_cost_inr=40,  unit_price_inr=65,  base_rate=0.05, trend=0.0),
    dict(item_id="I018", item_name="Gift Hamper Small",    category="Festive",   supplier_id="S4", unit_cost_inr=300, unit_price_inr=420, base_rate=0.15, trend=0.0),
    dict(item_id="I019", item_name="Steel Tiffin Box",     category="Durables",  supplier_id="S5", unit_cost_inr=180, unit_price_inr=250, base_rate=0.8,  trend=-0.45),
    dict(item_id="I020", item_name="Umbrella",              category="Durables",  supplier_id="S5", unit_cost_inr=150, unit_price_inr=220, base_rate=0.9,  trend=-0.10),
    dict(item_id="I021", item_name="Raincoat",               category="Durables",  supplier_id="S5", unit_cost_inr=280, unit_price_inr=380, base_rate=0.7,  trend=-0.10),
    dict(item_id="I022", item_name="Plastic Storage Box",  category="Durables",  supplier_id="S5", unit_cost_inr=120, unit_price_inr=170, base_rate=0.5,  trend=-0.45),
    dict(item_id="I023", item_name="Mobile Cover",          category="Accessories", supplier_id="S3", unit_cost_inr=70,  unit_price_inr=150, base_rate=None, trend=0.0),
    dict(item_id="I024", item_name="USB Cable",              category="Accessories", supplier_id="S3", unit_cost_inr=50,  unit_price_inr=99,  base_rate=None, trend=0.0),
    dict(item_id="I025", item_name="Earphones",              category="Accessories", supplier_id="S3", unit_cost_inr=180, unit_price_inr=349, base_rate=None, trend=0.0),
])

# intermittent-demand parameters (Accessories only): probability of a sale that day,
# and mean units when a sale happens
INTERMITTENT = {
    "I023": dict(p=0.35, jump_mean=1.8),
    "I024": dict(p=0.30, jump_mean=1.5),
    "I025": dict(p=0.22, jump_mean=1.3),
}

# monsoon seasonality (Jun-Sep): raincoats and umbrellas spike ahead of/through the rains.
# Lands mostly inside the backtest holdout window (last ~45 days of the dataset).
MONSOON_MULTIPLIER = {"I020": 3.0, "I021": 4.5}
MONSOON_MONTHS = {6, 7, 8, 9}

# festival-exclusive SKUs: near-zero baseline, a direct peak rate during their one window
# (rather than a multiplier on an already-tiny base, which under Poisson noise reads as flat)
FESTIVAL_PEAK_RATE = {"I016": 20.0, "I017": 22.0}  # Rakhi Set, Holi Colors Pack

# ---------------------------------------------------------------- festival calendar
# uplift_multiplier applies to any item whose category is in affected_categories;
# item-specific overrides (below) take precedence for the two festival-exclusive SKUs.
festival_calendar = pd.DataFrame([
    dict(date="2025-10-05", festival_name="Great Indian Festival Sale", event_type="Platform-Sale",
         affected_categories="Staples,FMCG,Durables", uplift_multiplier=1.5, ramp_days_before=3),
    dict(date="2025-10-20", festival_name="Diwali", event_type="Religious",
         affected_categories="Festive,Staples,FMCG", uplift_multiplier=1.6, ramp_days_before=10),
    dict(date="2025-12-25", festival_name="Christmas & New Year", event_type="National",
         affected_categories="Festive,FMCG", uplift_multiplier=1.3, ramp_days_before=5),
    dict(date="2026-01-14", festival_name="Makar Sankranti", event_type="Religious",
         affected_categories="Festive,Staples", uplift_multiplier=1.4, ramp_days_before=3),
    dict(date="2026-03-04", festival_name="Holi", event_type="Religious",
         affected_categories="Festive,FMCG", uplift_multiplier=1.5, ramp_days_before=7),
    dict(date="2026-08-28", festival_name="Raksha Bandhan", event_type="Religious",
         affected_categories="Festive,FMCG", uplift_multiplier=1.4, ramp_days_before=7),
])
festival_calendar["date"] = pd.to_datetime(festival_calendar["date"])

# `exclusive` marks a SKU that sells for exactly one festival and no other. Rakhi
# sets and Holi colours are bought for their own festival or not at all, so they
# must not inherit the category-level lift of an unrelated one -- "Festive" covers
# items driven by completely different events. Sweets, diyas and hampers are not
# exclusive: they genuinely move at more than one festival.
festival_item_overrides = pd.DataFrame([
    dict(festival_name="Holi", item_id="I017", uplift_multiplier=15.0, exclusive=True),   # Holi Colors Pack
    dict(festival_name="Raksha Bandhan", item_id="I016", uplift_multiplier=14.0, exclusive=True),  # Rakhi Set
    dict(festival_name="Diwali", item_id="I014", uplift_multiplier=6.0, exclusive=False),  # Diyas & Candles
    dict(festival_name="Diwali", item_id="I015", uplift_multiplier=5.0, exclusive=False),  # Sweets Gift Box
    dict(festival_name="Diwali", item_id="I018", uplift_multiplier=4.5, exclusive=False),  # Gift Hamper
])

# ---------------------------------------------------------------- promotions (seller-run, independent of festivals)
promo_rows = []
for item_id in items["item_id"]:
    n_promos = rng.integers(2, 5)
    for _ in range(n_promos):
        start = START + pd.Timedelta(days=int(rng.integers(0, N_DAYS - 7)))
        length = int(rng.integers(3, 6))
        uplift = float(rng.uniform(1.3, 1.6))
        for d in pd.date_range(start, periods=length):
            if d <= END:
                promo_rows.append(dict(date=d, item_id=item_id, promo_active=1, promo_uplift=round(uplift, 2)))
promotions = pd.DataFrame(promo_rows)

# ---------------------------------------------------------------- demand simulation
def festival_uplift(item_id, category, date):
    """
    Mirrors context_agent._relevant_festivals(). A SKU marked `exclusive` responds
    ONLY to festivals it has an override row for, never to a category match.

    This check used to be missing here while being present in the context agent,
    so the two disagreed about what `exclusive` meant: the generator gave Holi
    Colors Pack a full Raksha Bandhan peak (both sit under "Festive"), while the
    agent -- correctly -- refused to stock it for a festival nobody buys it for.
    The dataset therefore contained 229 units of demand the system was designed
    to ignore, and the backtest scored the correct behaviour as a failure.
    If you change exclusivity semantics, change it in BOTH places.
    """
    own = festival_item_overrides[festival_item_overrides.item_id == item_id]
    is_exclusive = bool(own["exclusive"].any()) if not own.empty else False
    own_festivals = set(own["festival_name"])

    mult = 1.0
    for f in festival_calendar.itertuples():
        if is_exclusive and f.festival_name not in own_festivals:
            continue
        window_start = f.date - pd.Timedelta(days=int(f.ramp_days_before))
        window_end = f.date + pd.Timedelta(days=1)
        if window_start <= date <= window_end:
            override = festival_item_overrides[
                (festival_item_overrides.festival_name == f.festival_name)
                & (festival_item_overrides.item_id == item_id)
            ]
            if not override.empty:
                mult = max(mult, float(override.uplift_multiplier.iloc[0]))
            elif category in f.affected_categories.split(","):
                mult = max(mult, float(f.uplift_multiplier))
    return mult

promo_lookup = promotions.set_index(["item_id", "date"])["promo_uplift"].to_dict() if not promotions.empty else {}
promo_flag_lookup = set(zip(promotions["item_id"], promotions["date"])) if not promotions.empty else set()

dow_multiplier = {0: 1.0, 1: 0.95, 2: 0.95, 3: 1.0, 4: 1.15, 5: 1.25, 6: 1.15}  # Mon..Sun

sales_rows = []
for item in items.itertuples():
    trend_per_day = item.trend / N_DAYS if item.trend else 0.0
    for i, date in enumerate(DATES):
        f_mult = festival_uplift(item.item_id, item.category, date)
        p_mult = promo_lookup.get((item.item_id, date), 1.0)
        dow_mult = dow_multiplier[date.dayofweek]
        season_mult = MONSOON_MULTIPLIER.get(item.item_id, 1.0) if date.month in MONSOON_MONTHS else 1.0
        trend_mult = 1.0 + trend_per_day * i

        if item.item_id in INTERMITTENT:
            params = INTERMITTENT[item.item_id]
            p = min(0.95, params["p"] * f_mult * p_mult * dow_mult)
            demand = int(rng.poisson(params["jump_mean"])) if rng.random() < p else 0
        elif item.item_id in FESTIVAL_PEAK_RATE and f_mult > 1.0:
            lam = FESTIVAL_PEAK_RATE[item.item_id] * dow_mult
            demand = int(rng.poisson(lam))
        else:
            lam = max(0.0, item.base_rate * dow_mult * trend_mult * f_mult * p_mult * season_mult)
            demand = int(rng.poisson(lam))

        sales_rows.append(dict(date=date, item_id=item.item_id, demand=demand))

demand_df = pd.DataFrame(sales_rows)

# ---------------------------------------------------------------- inventory + naive baseline policy
# "the seller's actual behaviour": reorder reactively off a rough gut-feel threshold,
# fixed order quantity, no anticipation of festivals -> this becomes the backtest baseline.
po_rows = []
sales_out_rows = []
po_counter = 0

for item in items.itertuples():
    supplier = suppliers.set_index("supplier_id").loc[item.supplier_id]
    avg_rate = (
        INTERMITTENT[item.item_id]["p"] * INTERMITTENT[item.item_id]["jump_mean"]
        if pd.isna(item.base_rate) else item.base_rate
    )

    on_hand = max(3, int(np.ceil(avg_rate * 12)))
    reorder_point = max(1, int(np.ceil(avg_rate * 5)))
    order_qty = max(3, int(np.ceil(avg_rate * 14)))
    pending = []  # list of (arrival_date, qty)

    item_demand = demand_df[demand_df.item_id == item.item_id].set_index("date")["demand"]

    for date in DATES:
        arrived_qty = sum(q for arr, q in pending if arr == date)
        if arrived_qty:
            on_hand += arrived_qty
            pending = [(arr, q) for arr, q in pending if arr != date]

        demand_today = int(item_demand.loc[date])
        units_sold = min(demand_today, on_hand)
        stockout_flag = int(demand_today > on_hand)
        on_hand_start = on_hand
        on_hand -= units_sold

        has_pending = len(pending) > 0
        if on_hand <= reorder_point and not has_pending:
            lead_time = max(1, int(round(rng.normal(supplier.mean_lead_time_days, supplier.lead_time_std_days))))
            arrival = date + pd.Timedelta(days=lead_time)
            pending.append((arrival, order_qty))
            po_counter += 1
            po_rows.append(dict(
                po_id=f"PO{po_counter:05d}", item_id=item.item_id, supplier_id=item.supplier_id,
                date_ordered=date, date_expected=arrival, qty_ordered=order_qty,
                lead_time_actual_days=lead_time,
            ))

        sales_out_rows.append(dict(
            date=date, item_id=item.item_id, units_sold=units_sold, stockout_flag=stockout_flag,
            promo_active=int((item.item_id, date) in promo_flag_lookup),
            on_hand_start=on_hand_start, on_hand_end=on_hand,
            # ground truth for backtesting only -- the *uncensored* demand that generated
            # units_sold above (units_sold = min(true_demand_uncensored, on_hand_start)).
            # A real system never has this column; it exists here purely so the backtest can
            # score different simulated policies against the same demand fairly, instead
            # of replaying units_sold, which is itself an artifact of this one naive policy's
            # own stock levels. Never feed this into a forecaster as a training feature.
            true_demand_uncensored=demand_today,
        ))

daily_sales = pd.DataFrame(sales_out_rows)
purchase_orders = pd.DataFrame(po_rows)

# ---------------------------------------------------------------- feedback log stub (Feedback Agent writes here)
feedback_log = pd.DataFrame(columns=[
    "feedback_id", "recommendation_id", "item_id", "seller_decision",
    "reason_code", "timestamp", "parameter_adjusted", "old_value", "new_value",
])

# ---------------------------------------------------------------- write out
OUT = "data"
suppliers.to_csv(f"{OUT}/suppliers.csv", index=False)
items.drop(columns=["base_rate", "trend"]).to_csv(f"{OUT}/items.csv", index=False)
festival_calendar.to_csv(f"{OUT}/festival_calendar.csv", index=False)
festival_item_overrides.to_csv(f"{OUT}/festival_item_overrides.csv", index=False)
promotions.drop(columns=["promo_active"]).to_csv(f"{OUT}/promotions.csv", index=False) if not promotions.empty else None
daily_sales.to_csv(f"{OUT}/daily_sales.csv", index=False)
purchase_orders.to_csv(f"{OUT}/purchase_orders.csv", index=False)
feedback_log.to_csv(f"{OUT}/feedback_log.csv", index=False)

print(f"rows: daily_sales={len(daily_sales):,}  purchase_orders={len(purchase_orders):,}  promotions={len(promotions):,}")
print(f"date range: {DATES.min().date()} to {DATES.max().date()}  ({N_DAYS} days)")
print(f"overall stockout rate: {daily_sales.stockout_flag.mean():.1%}")
print(daily_sales.groupby("item_id").stockout_flag.mean().sort_values(ascending=False).head(6).round(3))
