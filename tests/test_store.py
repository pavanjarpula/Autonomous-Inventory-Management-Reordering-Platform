import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feedback_agent import CONSISTENCY_THRESHOLD, default_parameters, submit_feedback
from store import (
    empty_feedback_log,
    load_feedback_log,
    load_parameters,
    save_feedback_log,
    save_parameters,
)


def _ts(i):
    return pd.Timestamp("2026-08-01") + pd.Timedelta(days=i)


# ---------------------------------------------------------------- feedback log

def test_load_feedback_log_returns_empty_when_file_is_missing(tmp_path):
    log = load_feedback_log(tmp_path / "nope.csv")
    assert log.empty
    assert list(log.columns) == list(empty_feedback_log().columns)


def test_load_feedback_log_handles_a_headers_only_file(tmp_path):
    path = tmp_path / "feedback.csv"
    save_feedback_log(empty_feedback_log(), path)
    assert load_feedback_log(path).empty


def test_feedback_log_survives_a_save_load_round_trip(tmp_path):
    path = tmp_path / "feedback.csv"
    log, params = empty_feedback_log(), {}
    log, params, _ = submit_feedback(log, params, "R1", "I001", "reject", "qty_too_high", _ts(0))
    save_feedback_log(log, path)

    reloaded = load_feedback_log(path)
    assert len(reloaded) == 1
    assert reloaded.iloc[0]["item_id"] == "I001"
    assert reloaded.iloc[0]["reason_code"] == "qty_too_high"


def test_reloaded_timestamps_come_back_as_datetimes_not_strings(tmp_path):
    # the consistency streak sorts on this column; strings would sort by luck
    path = tmp_path / "feedback.csv"
    log, params = empty_feedback_log(), {}
    log, params, _ = submit_feedback(log, params, "R1", "I001", "reject", "qty_too_high", _ts(0))
    save_feedback_log(log, path)
    assert pd.api.types.is_datetime64_any_dtype(load_feedback_log(path)["timestamp"])


def test_an_approve_rows_missing_reason_reloads_as_none_not_nan(tmp_path):
    path = tmp_path / "feedback.csv"
    log, params = empty_feedback_log(), {}
    log, params, _ = submit_feedback(log, params, "R1", "I001", "approve", None, _ts(0))
    save_feedback_log(log, path)
    assert load_feedback_log(path).iloc[0]["reason_code"] is None


# ---------------------------------------------------------------- parameters

def test_load_parameters_returns_empty_dict_when_file_is_missing(tmp_path):
    assert load_parameters(tmp_path / "nope.csv") == {}


def test_parameters_survive_a_save_load_round_trip(tmp_path):
    path = tmp_path / "params.csv"
    params = {"I001": dict(item_id="I001", z=1.35, lead_time_buffer_days=2.0)}
    save_parameters(params, path)

    reloaded = load_parameters(path)
    assert reloaded["I001"]["z"] == 1.35
    assert reloaded["I001"]["lead_time_buffer_days"] == 2.0


def test_saving_no_parameters_writes_a_readable_empty_file(tmp_path):
    path = tmp_path / "params.csv"
    save_parameters({}, path)
    assert load_parameters(path) == {}


# ---------------------------------------------------------------- the point of all this

def test_a_rejection_streak_survives_a_restart(tmp_path):
    """The whole reason persistence matters: two rejections, a restart, then a
    third must still cross the threshold. Before this, a reload reset the streak
    to zero and the parameter never moved."""
    log_path, params_path = tmp_path / "feedback.csv", tmp_path / "params.csv"

    log, params = empty_feedback_log(), {}
    for i in range(CONSISTENCY_THRESHOLD - 1):
        log, params, audit = submit_feedback(log, params, f"R{i}", "I001", "reject", "qty_too_high", _ts(i))
        assert audit is None
    save_feedback_log(log, log_path)
    save_parameters(params, params_path)

    # --- restart: nothing in memory, everything from disk ---
    log = load_feedback_log(log_path)
    params = load_parameters(params_path)

    log, params, audit = submit_feedback(
        log, params, "R-final", "I001", "reject", "qty_too_high", _ts(CONSISTENCY_THRESHOLD)
    )
    assert audit is not None, "the streak was lost across the restart"
    assert audit["parameter_adjusted"] == "z"
    assert params["I001"]["z"] < default_parameters("I001")["z"]
