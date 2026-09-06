"""
LangChain @tool-decorated functions wrapping the deterministic engine.

These expose the same logic as engine.py / context_agent.py / forecast.py
to an LLM's tool-calling interface, enabling agentic workflows where the
model can decide what information to fetch.

Each tool is a thin wrapper: it validates inputs, calls the real function,
and returns a JSON-serializable dict. No LLM calls happen inside tools.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(func):
        func.tool_name = func.__name__
        return func

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_agent import get_context
from engine import assess_item, days_to_next_arrival, on_order_qty, recent_trend_ratio
from forecast import fit_item_forecast, precompute_forecast_table
from logger import get_logger

logger = get_logger(__name__, extra_data={"module": "tools"})


@tool
def assess_inventory_item(
    item_id: str,
    as_of_date: str,
    z: float = 1.65,
    lead_time_buffer_days: float = 0.0,
) -> dict:
    """Assess a single inventory item's current risk, stock level, and suggested order.

    Args:
        item_id: The item to assess (e.g., "I001").
        as_of_date: Date to assess as of (YYYY-MM-DD).
        z: Safety stock z-score (default 1.65 for 95% service level).
        lead_time_buffer_days: Extra buffer added to supplier lead time.

    Returns:
        dict with keys: item_id, item_name, category, risk, on_hand,
        reorder_point, suggested_order_qty, days_of_cover, trend_ratio.
    """
    import streamlit as st

    data = _load_data()
    as_of = pd.Timestamp(as_of_date)
    params_by_item = _load_params()

    params = params_by_item.get(item_id, {})
    return assess_item(
        item_id, as_of, data["sales"], data["items"], data["suppliers"],
        z=params.get("z", z),
        lead_time_buffer_days=params.get("lead_time_buffer_days", lead_time_buffer_days),
        on_order=on_order_qty(item_id, as_of, data.get("purchase_orders")),
        arriving_in_days=days_to_next_arrival(item_id, as_of, data.get("purchase_orders")),
    )


@tool
def get_item_context(
    item_id: str,
    as_of_date: str,
) -> dict:
    """Get contextual signals for an item: festivals, promotions, seasonality.

    Args:
        item_id: The item to get context for.
        as_of_date: Date to check context as of (YYYY-MM-DD).

    Returns:
        dict with keys: is_festival_window, active_festival, uplift_multiplier,
        next_festival_name, days_to_next_festival, has_promo, promo_uplift.
    """
    data = _load_data()
    as_of = pd.Timestamp(as_of_date)
    return get_context(
        item_id, as_of, data["items"], data["festival_calendar"],
        data["festival_overrides"], data["promotions"],
    )


@tool
def get_on_order_status(
    item_id: str,
    as_of_date: str,
) -> dict:
    """Check if an item has stock already on order and when it will arrive.

    Args:
        item_id: The item to check.
        as_of_date: Date to check as of (YYYY-MM-DD).

    Returns:
        dict with keys: item_id, on_order_qty, days_to_next_arrival.
    """
    data = _load_data()
    as_of = pd.Timestamp(as_of_date)
    po = data.get("purchase_orders")
    return {
        "item_id": item_id,
        "on_order_qty": on_order_qty(item_id, as_of, po),
        "days_to_next_arrival": days_to_next_arrival(item_id, as_of, po),
    }


@tool
def get_demand_trend(
    item_id: str,
    recent_days: int = 14,
    baseline_days: int = 60,
) -> dict:
    """Get the recent demand trend for an item compared to its baseline.

    Args:
        item_id: The item to analyze.
        recent_days: Number of recent days to compare (default 14).
        baseline_days: Number of baseline days to compare against (default 60).

    Returns:
        dict with keys: item_id, trend_ratio, interpretation.
        trend_ratio > 1.5 means demand is surging; < 0.5 means declining.
    """
    data = _load_data()
    sales = data["sales"]
    item_sales = sales[sales["item_id"] == item_id].sort_values("date")

    ratio = recent_trend_ratio(item_sales, recent_window=recent_days, baseline_window=baseline_days)

    if ratio is None:
        interpretation = "insufficient data"
    elif ratio >= 1.5:
        interpretation = f"demand surging at {ratio}x baseline"
    elif ratio <= 0.5:
        interpretation = f"demand declining to {ratio}x baseline"
    else:
        interpretation = f"demand stable at {ratio}x baseline"

    return {
        "item_id": item_id,
        "trend_ratio": ratio,
        "interpretation": interpretation,
    }


@tool
def batch_assess_items(
    as_of_date: str,
    risk_filter: str = "all",
    limit: int = 10,
) -> list[dict]:
    """Assess multiple inventory items and return their risk status.

    Args:
        as_of_date: Date to assess as of (YYYY-MM-DD).
        risk_filter: Filter by risk type: "all", "stockout_risk", "overstock", "healthy".
        limit: Maximum number of items to return (default 10).

    Returns:
        List of dicts with item assessment data.
    """
    data = _load_data()
    as_of = pd.Timestamp(as_of_date)
    params_by_item = _load_params()

    results = []
    for item_id in data["items"]["item_id"]:
        params = params_by_item.get(item_id, {})
        assessment = assess_item(
            item_id, as_of, data["sales"], data["items"], data["suppliers"],
            z=params.get("z", 1.65),
            lead_time_buffer_days=params.get("lead_time_buffer_days", 0.0),
            on_order=on_order_qty(item_id, as_of, data.get("purchase_orders")),
            arriving_in_days=days_to_next_arrival(item_id, as_of, data.get("purchase_orders")),
        )
        if risk_filter != "all" and assessment["risk"] != risk_filter:
            continue
        results.append(assessment)

    results.sort(key=lambda x: x["days_of_cover"] if x["days_of_cover"] is not None else float("inf"))
    return results[:limit]


@tool
def get_forecast_summary(
    item_id: str,
    horizon_days: int = 14,
) -> dict:
    """Get a Prophet demand forecast summary for an item.

    Args:
        item_id: The item to forecast.
        horizon_days: Number of days to forecast ahead (default 14).

    Returns:
        dict with keys: item_id, forecast_mean_daily, forecast_total,
        trend, seasonality.
    """
    data = _load_data()
    try:
        model = fit_item_forecast(
            item_id, data["sales"], data["festival_calendar"],
            data["promotions"], train_end_date=data["sales"]["date"].max(),
        )
        last_date = data["sales"]["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days)
        forecast = precompute_forecast_table(model, item_id, future_dates, data["promotions"])

        return {
            "item_id": item_id,
            "forecast_mean_daily": round(forecast.mean(), 2),
            "forecast_total": round(forecast.sum(), 1),
            "horizon_days": horizon_days,
        }
    except Exception as e:
        return {
            "item_id": item_id,
            "error": str(e),
            "forecast_mean_daily": None,
            "forecast_total": None,
        }


# ---- internal helpers ----

_data_cache: dict | None = None


def _load_data() -> dict:
    """Load data files once, cache for the session."""
    global _data_cache
    if _data_cache is not None:
        return _data_cache

    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    _data_cache = {
        "sales": pd.read_csv(data_dir / "daily_sales.csv", parse_dates=["date"]),
        "items": pd.read_csv(data_dir / "items.csv"),
        "suppliers": pd.read_csv(data_dir / "suppliers.csv"),
        "festival_calendar": pd.read_csv(data_dir / "festival_calendar.csv", parse_dates=["date"]),
        "festival_overrides": pd.read_csv(data_dir / "festival_item_overrides.csv"),
        "promotions": pd.read_csv(data_dir / "promotions.csv", parse_dates=["date"]),
        "purchase_orders": pd.read_csv(
            data_dir / "purchase_orders.csv",
            parse_dates=["date_ordered", "date_expected"],
        ),
    }
    return _data_cache


def _load_params() -> dict:
    """Load seller parameters from CSV."""
    from pathlib import Path
    from store import load_parameters
    root = Path(__file__).resolve().parents[1]
    params_path = root / "data" / "seller_parameters.csv"
    if params_path.exists():
        return load_parameters(str(params_path))
    return {}


# Export all tools as a list for easy registration
ALL_TOOLS = [
    assess_inventory_item,
    get_item_context,
    get_on_order_status,
    get_demand_trend,
    batch_assess_items,
    get_forecast_summary,
]
