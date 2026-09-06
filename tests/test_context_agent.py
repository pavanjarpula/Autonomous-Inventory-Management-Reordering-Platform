import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_agent import get_context, promo_flag_series, to_prophet_holidays


@pytest.fixture
def items():
    return pd.DataFrame([
        dict(item_id="FESTIVE_EXCLUSIVE", item_name="Holi Colors", category="Festive"),
        dict(item_id="FESTIVE_GENERIC", item_name="Sweets Box", category="Festive"),
        dict(item_id="STAPLE", item_name="Rice", category="Staples"),
    ])


@pytest.fixture
def festival_calendar():
    return pd.DataFrame([
        dict(date=pd.Timestamp("2026-03-04"), festival_name="Holi",
             affected_categories="Festive,FMCG", uplift_multiplier=1.5, ramp_days_before=7),
    ])


@pytest.fixture
def festival_overrides():
    return pd.DataFrame([
        dict(festival_name="Holi", item_id="FESTIVE_EXCLUSIVE", uplift_multiplier=15.0),
    ])


@pytest.fixture
def promotions():
    return pd.DataFrame([
        dict(date=pd.Timestamp("2026-01-10"), item_id="STAPLE", promo_uplift=1.4),
    ])


def test_an_exclusive_item_ignores_an_unrelated_festival_in_its_own_category():
    """A Holi colours pack and a rakhi set both sit under "Festive", but neither
    sells for the other's festival. Without the exclusive flag the category-level
    lift swept each into the other's window."""
    items = pd.DataFrame([
        dict(item_id="HOLI_ONLY", item_name="Holi Colors", category="Festive"),
        dict(item_id="SWEETS", item_name="Sweets Box", category="Festive"),
    ])
    calendar = pd.DataFrame([
        dict(date=pd.Timestamp("2026-03-04"), festival_name="Holi",
             affected_categories="Festive", uplift_multiplier=1.5, ramp_days_before=7),
        dict(date=pd.Timestamp("2026-08-28"), festival_name="Raksha Bandhan",
             affected_categories="Festive", uplift_multiplier=1.4, ramp_days_before=7),
    ])
    overrides = pd.DataFrame([
        dict(festival_name="Holi", item_id="HOLI_ONLY", uplift_multiplier=15.0, exclusive=True),
    ])
    promotions = pd.DataFrame(columns=["date", "item_id", "promo_uplift"])
    during_rakhi = pd.Timestamp("2026-08-26")

    exclusive = get_context("HOLI_ONLY", during_rakhi, items, calendar, overrides, promotions)
    assert exclusive["is_festival_window"] is False
    assert exclusive["uplift_multiplier"] == 1.0

    # a non-exclusive festive item still picks up the category lift, correctly
    shared = get_context("SWEETS", during_rakhi, items, calendar, overrides, promotions)
    assert shared["is_festival_window"] is True
    assert shared["active_festival"] == "Raksha Bandhan"


def test_an_exclusive_item_still_fires_for_its_own_festival():
    items = pd.DataFrame([dict(item_id="HOLI_ONLY", item_name="Holi Colors", category="Festive")])
    calendar = pd.DataFrame([
        dict(date=pd.Timestamp("2026-03-04"), festival_name="Holi",
             affected_categories="Festive", uplift_multiplier=1.5, ramp_days_before=7),
    ])
    overrides = pd.DataFrame([
        dict(festival_name="Holi", item_id="HOLI_ONLY", uplift_multiplier=15.0, exclusive=True),
    ])
    promotions = pd.DataFrame(columns=["date", "item_id", "promo_uplift"])

    ctx = get_context("HOLI_ONLY", pd.Timestamp("2026-03-04"), items, calendar, overrides, promotions)
    assert ctx["is_festival_window"] is True
    assert ctx["uplift_multiplier"] == 15.0


def test_item_override_wins_over_category_default(items, festival_calendar, festival_overrides, promotions):
    ctx = get_context("FESTIVE_EXCLUSIVE", pd.Timestamp("2026-03-04"), items, festival_calendar, festival_overrides, promotions)
    assert ctx["is_festival_window"] is True
    assert ctx["uplift_multiplier"] == 15.0


def test_category_default_applies_without_override(items, festival_calendar, festival_overrides, promotions):
    ctx = get_context("FESTIVE_GENERIC", pd.Timestamp("2026-03-04"), items, festival_calendar, festival_overrides, promotions)
    assert ctx["is_festival_window"] is True
    assert ctx["uplift_multiplier"] == 1.5


def test_irrelevant_category_is_unaffected(items, festival_calendar, festival_overrides, promotions):
    ctx = get_context("STAPLE", pd.Timestamp("2026-03-04"), items, festival_calendar, festival_overrides, promotions)
    assert ctx["is_festival_window"] is False
    assert ctx["uplift_multiplier"] == 1.0


def test_window_boundary_day_before_ramp_starts_is_not_active(items, festival_calendar, festival_overrides, promotions):
    # ramp_days_before=7, festival on 2026-03-04 -> window starts 2026-02-25
    ctx = get_context("FESTIVE_EXCLUSIVE", pd.Timestamp("2026-02-24"), items, festival_calendar, festival_overrides, promotions)
    assert ctx["is_festival_window"] is False
    assert ctx["days_to_next_festival"] == 8


def test_window_boundary_first_ramp_day_is_active(items, festival_calendar, festival_overrides, promotions):
    ctx = get_context("FESTIVE_EXCLUSIVE", pd.Timestamp("2026-02-25"), items, festival_calendar, festival_overrides, promotions)
    assert ctx["is_festival_window"] is True
    assert ctx["active_festival"] == "Holi"


def test_no_relevant_festival_left_returns_none(items, festival_calendar, festival_overrides, promotions):
    ctx = get_context("STAPLE", pd.Timestamp("2026-06-01"), items, festival_calendar, festival_overrides, promotions)
    assert ctx["days_to_next_festival"] is None
    assert ctx["next_festival_name"] is None


def test_promo_active_only_on_its_own_date(items, festival_calendar, festival_overrides, promotions):
    on_day = get_context("STAPLE", pd.Timestamp("2026-01-10"), items, festival_calendar, festival_overrides, promotions)
    off_day = get_context("STAPLE", pd.Timestamp("2026-01-11"), items, festival_calendar, festival_overrides, promotions)
    assert on_day["promo_active"] is True
    assert on_day["promo_uplift"] == 1.4
    assert off_day["promo_active"] is False
    assert off_day["promo_uplift"] == 1.0


def test_to_prophet_holidays_shape(festival_calendar):
    holidays = to_prophet_holidays(festival_calendar)
    assert list(holidays.columns) == ["holiday", "ds", "lower_window", "upper_window"]
    assert holidays.iloc[0]["lower_window"] == -7
    assert holidays.iloc[0]["upper_window"] == 1


def test_promo_flag_series_marks_only_active_dates(promotions):
    dates = pd.date_range("2026-01-08", "2026-01-12")
    flags = promo_flag_series("STAPLE", dates, promotions)
    assert flags.loc[pd.Timestamp("2026-01-10")] == 1
    assert flags.loc[pd.Timestamp("2026-01-09")] == 0
    assert flags.sum() == 1
