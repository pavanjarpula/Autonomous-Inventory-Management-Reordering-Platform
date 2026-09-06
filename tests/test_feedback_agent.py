import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feedback_agent import (
    CONSISTENCY_THRESHOLD,
    Z_BOUNDS,
    apply_feedback,
    default_parameters,
    record_feedback,
    submit_feedback,
)


@pytest.fixture
def empty_log():
    return pd.DataFrame(columns=[
        "feedback_id", "recommendation_id", "item_id", "seller_decision",
        "reason_code", "timestamp", "parameter_adjusted", "old_value", "new_value",
    ])


def _ts(i):
    return pd.Timestamp("2026-08-01") + pd.Timedelta(days=i)


# ---------------------------------------------------------------- record_feedback

def test_record_feedback_always_appends_regardless_of_reason(empty_log):
    log = record_feedback(empty_log, "R1", "I001", "approve", None, _ts(0))
    assert len(log) == 1
    assert log.iloc[0]["seller_decision"] == "approve"


def test_record_feedback_rejects_an_unknown_reason_code(empty_log):
    with pytest.raises(ValueError):
        record_feedback(empty_log, "R1", "I001", "reject", "not_a_real_reason", _ts(0))


# ---------------------------------------------------------------- hysteresis

def test_single_rejection_never_moves_anything(empty_log):
    log = record_feedback(empty_log, "R1", "I001", "reject", "qty_too_high", _ts(0))
    params, audit = apply_feedback(log, default_parameters("I001"), "I001", "qty_too_high")
    assert audit is None
    assert params == default_parameters("I001")


def test_two_rejections_still_do_not_move_it(empty_log):
    log = empty_log
    for i in range(2):
        log = record_feedback(log, f"R{i}", "I001", "reject", "qty_too_high", _ts(i))
    params, audit = apply_feedback(log, default_parameters("I001"), "I001", "qty_too_high")
    assert audit is None


def test_third_consistent_rejection_moves_the_parameter(empty_log):
    log = empty_log
    params = default_parameters("I001")
    for i in range(CONSISTENCY_THRESHOLD):
        log = record_feedback(log, f"R{i}", "I001", "reject", "qty_too_high", _ts(i))
    params, audit = apply_feedback(log, params, "I001", "qty_too_high")
    assert audit is not None
    assert audit["parameter_adjusted"] == "z"
    assert params["z"] < default_parameters("I001")["z"]


def test_a_different_reason_in_between_resets_the_streak(empty_log):
    log = empty_log
    log = record_feedback(log, "R1", "I001", "reject", "qty_too_high", _ts(0))
    log = record_feedback(log, "R2", "I001", "reject", "supplier_unreliable", _ts(1))  # breaks the streak
    log = record_feedback(log, "R3", "I001", "reject", "qty_too_high", _ts(2))
    params, audit = apply_feedback(log, default_parameters("I001"), "I001", "qty_too_high")
    assert audit is None  # only 1 consistent qty_too_high since the interruption, not 3


def test_streaks_are_tracked_independently_per_item(empty_log):
    log = empty_log
    for i in range(CONSISTENCY_THRESHOLD):
        log = record_feedback(log, f"R{i}", "I001", "reject", "qty_too_high", _ts(i))
    # I002 has never been rejected -- must not inherit I001's streak
    params, audit = apply_feedback(log, default_parameters("I002"), "I002", "qty_too_high")
    assert audit is None


# ---------------------------------------------------------------- reason -> parameter mapping

def test_qty_too_low_raises_z(empty_log):
    log = empty_log
    for i in range(CONSISTENCY_THRESHOLD):
        log = record_feedback(log, f"R{i}", "I001", "reject", "qty_too_low", _ts(i))
    params, audit = apply_feedback(log, default_parameters("I001"), "I001", "qty_too_low")
    assert params["z"] > default_parameters("I001")["z"]


def test_supplier_unreliable_raises_the_lead_time_buffer_not_z(empty_log):
    log = empty_log
    for i in range(CONSISTENCY_THRESHOLD):
        log = record_feedback(log, f"R{i}", "I001", "reject", "supplier_unreliable", _ts(i))
    params, audit = apply_feedback(log, default_parameters("I001"), "I001", "supplier_unreliable")
    assert audit["parameter_adjusted"] == "lead_time_buffer_days"
    assert params["lead_time_buffer_days"] > 0
    assert params["z"] == default_parameters("I001")["z"]  # z is untouched by this reason


def test_not_needed_now_never_adjusts_anything_no_matter_how_often(empty_log):
    log = empty_log
    for i in range(10):  # far past the consistency threshold
        log = record_feedback(log, f"R{i}", "I001", "reject", "not_needed_now", _ts(i))
    params, audit = apply_feedback(log, default_parameters("I001"), "I001", "not_needed_now")
    assert audit is None
    assert params == default_parameters("I001")


# ---------------------------------------------------------------- clamping

def test_z_never_moves_past_its_lower_bound():
    params = dict(item_id="I001", z=Z_BOUNDS[0], lead_time_buffer_days=0.0)
    log = pd.DataFrame([
        dict(feedback_id=f"FB{i}", recommendation_id=f"R{i}", item_id="I001", seller_decision="reject",
             reason_code="qty_too_high", timestamp=_ts(i), parameter_adjusted=None, old_value=None, new_value=None)
        for i in range(CONSISTENCY_THRESHOLD)
    ])
    updated, audit = apply_feedback(log, params, "I001", "qty_too_high")
    assert audit is None  # already at the floor -- nothing to move, so no audit event either
    assert updated["z"] == Z_BOUNDS[0]


# ---------------------------------------------------------------- submit_feedback (the full loop)

def test_submit_feedback_end_to_end_three_rejections_then_it_moves(empty_log):
    log, params_by_item = empty_log, {}
    for i in range(CONSISTENCY_THRESHOLD):
        log, params_by_item, audit = submit_feedback(
            log, params_by_item, f"R{i}", "I001", "reject", "qty_too_high", _ts(i)
        )
    assert audit is not None
    assert params_by_item["I001"]["z"] < Z_BOUNDS[1]
    # the audit trail on the log itself reflects the change, on the row that triggered it
    assert log.iloc[-1]["parameter_adjusted"] == "z"
    assert log.iloc[-1]["old_value"] is not None


def test_submit_feedback_approve_never_touches_parameters_or_requires_a_reason(empty_log):
    log, params_by_item, audit = submit_feedback(
        empty_log, {}, "R1", "I001", "approve", None, _ts(0)
    )
    assert audit is None
    assert log.iloc[0]["seller_decision"] == "approve"
