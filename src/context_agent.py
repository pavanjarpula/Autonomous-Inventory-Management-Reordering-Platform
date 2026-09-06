"""
Context layer. Turns the festival calendar, item-specific overrides, and
promotions into lookups the Reasoning Agent and the forecaster
both consume. Same discipline as engine.py: pure functions, dataframes in, values
out, no file I/O, no LLM.
"""

import pandas as pd


def _relevant_festivals(item_id: str, category: str, festival_calendar: pd.DataFrame,
                         festival_overrides: pd.DataFrame) -> pd.DataFrame:
    """Festivals that touch this item: either its category is listed, or it has its
    own override entry.

    A SKU marked `exclusive` is the exception: it is relevant ONLY to the festivals
    it has overrides for, never to a category match. Without this, a Holi colours
    pack inherits the category-level lift of Raksha Bandhan purely because both sit
    under "Festive", and the system recommends stocking it for a festival nobody
    buys it for. A category is too coarse to carry that distinction on its own."""
    own = festival_overrides[festival_overrides.item_id == item_id]
    override_names = set(own["festival_name"])

    if not own.empty and "exclusive" in own.columns and own["exclusive"].any():
        return festival_calendar[festival_calendar["festival_name"].isin(override_names)]

    category_match = festival_calendar["affected_categories"].str.split(",").apply(lambda cats: category in cats)
    return festival_calendar[festival_calendar["festival_name"].isin(override_names) | category_match]


def _resolved_uplift(item_id: str, festival_name: str, category_uplift: float,
                      festival_overrides: pd.DataFrame) -> float:
    """Item-specific override wins over the category default — Holi Colors Pack gets
    its own 15x, not Festive's generic 1.5x."""
    override = festival_overrides[
        (festival_overrides.festival_name == festival_name) & (festival_overrides.item_id == item_id)
    ]
    return float(override.uplift_multiplier.iloc[0]) if not override.empty else float(category_uplift)


def get_context(
    item_id: str,
    date: pd.Timestamp,
    items: pd.DataFrame,
    festival_calendar: pd.DataFrame,
    festival_overrides: pd.DataFrame,
    promotions: pd.DataFrame,
) -> dict:
    category = items.set_index("item_id").loc[item_id, "category"]
    relevant = _relevant_festivals(item_id, category, festival_calendar, festival_overrides)

    is_festival_window = False
    active_festival = None
    uplift_multiplier = 1.0
    for f in relevant.itertuples():
        window_start = f.date - pd.Timedelta(days=int(f.ramp_days_before))
        window_end = f.date + pd.Timedelta(days=1)
        if window_start <= date <= window_end:
            resolved = _resolved_uplift(item_id, f.festival_name, f.uplift_multiplier, festival_overrides)
            if resolved > uplift_multiplier:
                is_festival_window = True
                active_festival = f.festival_name
                uplift_multiplier = resolved

    upcoming = relevant[relevant["date"] >= date].sort_values("date")
    days_to_next_festival = int((upcoming.iloc[0].date - date).days) if not upcoming.empty else None
    next_festival_name = str(upcoming.iloc[0].festival_name) if not upcoming.empty else None

    promo_row = promotions[(promotions.item_id == item_id) & (promotions.date == date)]
    promo_active = not promo_row.empty
    promo_uplift = float(promo_row.promo_uplift.iloc[0]) if promo_active else 1.0

    return dict(
        item_id=item_id,
        date=date,
        is_festival_window=is_festival_window,
        active_festival=active_festival,
        uplift_multiplier=uplift_multiplier,
        days_to_next_festival=days_to_next_festival,
        next_festival_name=next_festival_name,
        promo_active=promo_active,
        promo_uplift=promo_uplift,
    )


def to_prophet_holidays(festival_calendar: pd.DataFrame) -> pd.DataFrame:
    """Reshape the festival calendar into Prophet's holidays= dataframe format."""
    df = festival_calendar.rename(columns={"festival_name": "holiday", "date": "ds"})[
        ["holiday", "ds", "ramp_days_before"]
    ].copy()
    df["lower_window"] = -df["ramp_days_before"]
    df["upper_window"] = 1
    return df[["holiday", "ds", "lower_window", "upper_window"]]


def promo_flag_series(item_id: str, dates: pd.DatetimeIndex, promotions: pd.DataFrame) -> pd.Series:
    """Binary promo_active regressor over `dates` — ready for Prophet's add_regressor()."""
    active_dates = set(promotions.loc[promotions.item_id == item_id, "date"])
    return pd.Series([1 if d in active_dates else 0 for d in dates], index=dates, name="promo_active")
