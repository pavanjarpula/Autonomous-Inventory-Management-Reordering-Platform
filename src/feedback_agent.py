"""
The Feedback Agent. Closes the loop from a seller's approve/reject
decision back into the per-item parameters engine.py's assess_item() actually
uses (z, lead_time_buffer_days) -- no LLM anywhere in this file, same discipline
as every earlier phase.

Two guardrails, both deliberate:
  - hysteresis: a single rejection never moves anything. The SAME reason has to
    appear in the last CONSISTENCY_THRESHOLD REJECTIONS for this item, with no
    other reason among them, before a parameter actually adjusts -- a seller
    flip-flopping between complaints never accidentally accumulates toward a
    threshold for any one of them.

    Read that precisely: the streak is over rejections only. Approvals in between
    do NOT reset it, so three "qty_too_high" rejections spread across weeks of
    otherwise-approved recommendations still trip the threshold. That is the
    intended behaviour -- an approval is agreement with a DIFFERENT
    recommendation and says nothing about whether this complaint was resolved --
    but it is not the same thing as three-in-a-row on the calendar, and earlier
    wording that said "unbroken run" was misleading about it.
  - not every rejection reason adjusts a parameter at all. "qty_too_high" and
    "supplier_unreliable" indicate the MATH was wrong, so they earn a parameter
    nudge once consistent. Note the coupling this creates: z feeds BOTH the
    order quantity and the reorder point, and classify_risk() compares on_hand
    against the reorder point. So lowering z on "qty_too_high" also makes the
    item less likely to be FLAGGED at all, not just cheaper to restock. That is
    defensible -- a seller who thinks the quantity is too high usually thinks the
    trigger is too eager as well -- but it is a real second-order effect and
    should be visible rather than discovered. "not_needed_now" is about the seller's own
    circumstances (cash flow, unrelated timing), not a miscalibration -- it's
    logged for visibility but never adjusts anything, so one seller's cash-flow
    week doesn't quietly warp the reorder math for every future cycle.

Pure functions throughout, same as engine.py and context_agent.py: dataframes
and dicts in, updated dataframes and dicts out, no file I/O in this module --
a driver script owns loading/saving feedback_log.csv and the parameter store.
"""

import pandas as pd

REASON_CODES = ("qty_too_high", "qty_too_low", "supplier_unreliable", "not_needed_now")

Z_DEFAULT = 1.65
Z_BOUNDS = (1.00, 2.50)
Z_STEP = 0.15

LEAD_TIME_BUFFER_BOUNDS = (0.0, 10.0)
LEAD_TIME_BUFFER_STEP = 1.0

CONSISTENCY_THRESHOLD = 3


def default_parameters(item_id: str) -> dict:
    return dict(item_id=item_id, z=Z_DEFAULT, lead_time_buffer_days=0.0)


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def record_feedback(
    feedback_log: pd.DataFrame,
    recommendation_id: str,
    item_id: str,
    decision: str,
    reason_code: str | None,
    timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Appends one row -- always logs, independent of whether it ends up moving
    a parameter. The audit trail is complete even for rejections that don't
    (yet, or ever) change anything."""
    if reason_code is not None and reason_code not in REASON_CODES:
        raise ValueError(f"unknown reason_code: {reason_code!r} -- must be one of {REASON_CODES}")

    new_row = dict(
        feedback_id=f"FB{len(feedback_log) + 1:05d}",
        recommendation_id=recommendation_id, item_id=item_id,
        seller_decision=decision, reason_code=reason_code, timestamp=timestamp,
        parameter_adjusted=None, old_value=None, new_value=None,
    )
    if feedback_log.empty:
        # concat onto a genuinely empty frame is a pandas dtype-inference
        # deprecation as of 2.x -- the log starts empty every time a seller's
        # very first feedback event is recorded, so this isn't a rare path
        return pd.DataFrame([new_row])
    return pd.concat([feedback_log, pd.DataFrame([new_row])], ignore_index=True)


def _consistent_streak(feedback_log: pd.DataFrame, item_id: str, reason_code: str) -> int:
    """Length of the unbroken run of this exact reason at the END of this item's
    REJECTION history -- a different rejection reason anywhere in between resets
    the count back to zero for everything before it. Approvals are filtered out
    before the scan and therefore do not break a run; see the module docstring."""
    rejections = feedback_log[
        (feedback_log.item_id == item_id) & (feedback_log.seller_decision == "reject")
    ].sort_values("timestamp")
    streak = 0
    for reason in rejections["reason_code"].iloc[::-1]:
        if reason != reason_code:
            break
        streak += 1
    return streak


def apply_feedback(
    feedback_log: pd.DataFrame,
    parameters: dict,
    item_id: str,
    reason_code: str,
) -> tuple[dict, dict | None]:
    """
    Returns (parameters, audit) -- parameters unchanged and audit=None unless the
    streak has just reached the threshold. Expects feedback_log to already
    include the event being evaluated (call record_feedback() first) so the Nth
    identical rejection in a row is the one that actually moves something.
    """
    if reason_code == "not_needed_now":
        return parameters, None  # by design: circumstantial, never adjusts anything

    if _consistent_streak(feedback_log, item_id, reason_code) < CONSISTENCY_THRESHOLD:
        return parameters, None

    updated = dict(parameters)
    if reason_code == "qty_too_high":
        old = parameters["z"]
        updated["z"] = _clamp(old - Z_STEP, Z_BOUNDS)
        param_name, new = "z", updated["z"]
    elif reason_code == "qty_too_low":
        old = parameters["z"]
        updated["z"] = _clamp(old + Z_STEP, Z_BOUNDS)
        param_name, new = "z", updated["z"]
    elif reason_code == "supplier_unreliable":
        old = parameters["lead_time_buffer_days"]
        updated["lead_time_buffer_days"] = _clamp(old + LEAD_TIME_BUFFER_STEP, LEAD_TIME_BUFFER_BOUNDS)
        param_name, new = "lead_time_buffer_days", updated["lead_time_buffer_days"]
    else:
        raise ValueError(f"unknown reason_code: {reason_code!r}")

    if new == old:
        return parameters, None  # already sitting at the clamp boundary -- nothing actually moved

    return updated, dict(parameter_adjusted=param_name, old_value=old, new_value=new)


def submit_feedback(
    feedback_log: pd.DataFrame,
    parameters_by_item: dict,
    recommendation_id: str,
    item_id: str,
    decision: str,
    reason_code: str | None,
    timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, dict, dict | None]:
    """The convenience entry point: log the decision, then adjust a parameter if
    (and only if) the reason has now repeated consistently enough. Returns the
    updated feedback log, the updated per-item parameter store, and an audit
    record (or None if nothing moved)."""
    feedback_log = record_feedback(feedback_log, recommendation_id, item_id, decision, reason_code, timestamp)
    parameters_by_item = dict(parameters_by_item)
    params = parameters_by_item.get(item_id, default_parameters(item_id))

    if decision != "reject" or reason_code is None:
        return feedback_log, parameters_by_item, None

    updated_params, audit = apply_feedback(feedback_log, params, item_id, reason_code)
    parameters_by_item[item_id] = updated_params

    if audit:
        last = feedback_log.index[-1]
        feedback_log.loc[last, "parameter_adjusted"] = audit["parameter_adjusted"]
        feedback_log.loc[last, "old_value"] = audit["old_value"]
        feedback_log.loc[last, "new_value"] = audit["new_value"]

    return feedback_log, parameters_by_item, audit
