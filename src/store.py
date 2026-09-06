"""
Disk persistence for the two pieces of state that have to survive a restart:
the feedback log, and the per-item parameters that feedback adjusts. Everything
else the system shows is recomputed from the source data on demand.

Kept out of feedback_agent.py deliberately -- that module stays pure
(dataframes and dicts in, updated ones out) so its hysteresis and clamping can
be tested without touching a filesystem. This is the only module that writes.
"""

from pathlib import Path

import pandas as pd

FEEDBACK_COLUMNS = [
    "feedback_id", "recommendation_id", "item_id", "seller_decision",
    "reason_code", "timestamp", "parameter_adjusted", "old_value", "new_value",
]
PARAMETER_COLUMNS = ["item_id", "z", "lead_time_buffer_days"]


def empty_feedback_log() -> pd.DataFrame:
    return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def load_feedback_log(path: Path) -> pd.DataFrame:
    """Missing file, or a headers-only file, both mean 'no history yet'."""
    path = Path(path)
    if not path.exists():
        return empty_feedback_log()

    df = pd.read_csv(path)
    if df.empty:
        return empty_feedback_log()

    # timestamps drive the consistency-streak ordering, so they have to come
    # back as datetimes, not strings that merely happen to sort correctly
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # a CSV round-trip turns the "no reason given" of an approve row into NaN;
    # the streak logic compares against None, so restore that shape. The
    # astype(object) is load-bearing: an all-NaN column comes back as float64,
    # which silently coerces an assigned None straight back to NaN.
    df["reason_code"] = df["reason_code"].astype(object).where(df["reason_code"].notna(), None)
    return df[FEEDBACK_COLUMNS]


def save_feedback_log(feedback_log: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feedback_log.to_csv(path, index=False)


def load_parameters(path: Path) -> dict:
    """Returns item_id -> parameter dict. An item with no saved row simply
    isn't here; callers fall back to feedback_agent.default_parameters()."""
    path = Path(path)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if df.empty:
        return {}

    return {
        row["item_id"]: dict(item_id=row["item_id"], z=float(row["z"]),
                             lead_time_buffer_days=float(row["lead_time_buffer_days"]))
        for _, row in df.iterrows()
    }


def save_parameters(params_by_item: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(params_by_item.values())
    df = pd.DataFrame(rows, columns=PARAMETER_COLUMNS) if rows else pd.DataFrame(columns=PARAMETER_COLUMNS)
    df.to_csv(path, index=False)
