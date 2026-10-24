"""Structural timetable opportunities for the separately frozen V4 experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .frequency_v1 import _airing_weights, _integer_values
from .prepare import conflict_indices, seconds

DAY = 86400
TRAIN_START = seconds("2020-03-09")
SCHEDULE_COLUMNS = ["air_id", "channel_id", "tv_show_id", "start", "end"]


@dataclass(frozen=True)
class ExposureState:
    candidate_ids: np.ndarray
    e_long_raw: np.ndarray
    e_last28_raw: np.ndarray
    f_raw: np.ndarray
    e_long_slots: np.ndarray
    e_last28_slots: np.ndarray
    f_slots: np.ndarray
    denominator_weight: np.ndarray
    probability_airings: pd.DataFrame
    diagnostics: dict[str, Any]


def validate_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Retain only structural fields; require exact, precleaned airing rows."""
    missing = set(SCHEDULE_COLUMNS) - set(schedule.columns)
    if missing:
        raise ValueError(f"Missing schedule columns: {sorted(missing)}")
    data = pd.DataFrame({name: _integer_values(schedule[name], f"schedule.{name}") for name in SCHEDULE_COLUMNS})
    if data.air_id.duplicated().any():
        raise ValueError("Schedule air_id must be unique")
    if (data.tv_show_id <= 0).any() or (data.channel_id <= 0).any() or (data.end <= data.start).any():
        raise ValueError("Schedule must have positive labels/channels and positive full durations")
    if data.duplicated(["channel_id", "start", "end", "tv_show_id"]).any():
        raise ValueError("Exact schedule duplicates must be removed before feature building")
    return data


def read_future_schedule(path: Path) -> pd.DataFrame:
    """Read no content metadata; apply the original labelled/positive rule."""
    data = pd.read_csv(path, usecols=["channel_id", "tv_show_id", "start_time", "duration"],
                       dtype=str, keep_default_na=False)
    duration = data.duration.astype(np.int64)
    data = data.loc[(data.tv_show_id != "0") & (duration > 0)].copy()
    start = pd.to_datetime(data.start_time, format="%d.%m.%Y %H:%M:%S").astype("int64") // 1_000_000_000
    return pd.DataFrame({"channel_id": data.channel_id.astype(np.int64),
                         "tv_show_id": data.tv_show_id.astype(np.int64),
                         "start": start, "end": start + data.duration.astype(np.int64)}).reset_index(drop=True)


def prepare_window_schedule(
    training: pd.DataFrame, train_start: int, target_end: int,
    future: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preserve old air IDs and apply SAME exact-duplicate/conflict policies.

    The original training parquet is already cleaned. For final, combine it
    with original31-42 structural rows before duplicate/conflict exclusion.
    Candidate availability is a separate immutable input and is not rebuilt.
    """
    data = validate_schedule(training)
    old_ids = set(map(int, data.air_id))
    future_rows = 0
    if future is not None:
        future = future[["channel_id", "tv_show_id", "start", "end"]].copy()
        for name in future.columns:
            future[name] = _integer_values(future[name], f"future.{name}")
        if (future.tv_show_id <= 0).any() or (future.channel_id <= 0).any() or (future.end <= future.start).any():
            raise ValueError("Future schedule must already exclude zero labels and nonpositive durations")
        next_id = int(data.air_id.max()) + 1 if len(data) else 0
        future["air_id"] = np.arange(next_id, next_id + len(future), dtype=np.int64)
        future_rows = len(future)
        data = pd.concat([data, future[SCHEDULE_COLUMNS]], ignore_index=True)
    before_duplicates = len(data)
    data = data.drop_duplicates(["channel_id", "start", "end", "tv_show_id"])
    duplicates = before_duplicates - len(data)
    data = data.loc[(data.start < target_end) & (data.end > train_start)].reset_index(drop=True)
    conflicts = conflict_indices(data) if future is not None else set()
    retained = data.drop(index=list(conflicts)).reset_index(drop=True)
    retained = validate_schedule(retained)
    return retained, {
        "training_source_rows": len(training), "future_labelled_positive_rows": future_rows,
        "exact_channel_item_time_duplicates_removed": duplicates,
        "conflicting_labelled_rows_excluded": len(conflicts), "retained_rows": len(retained),
        "retained_old_air_ids": int(retained.air_id.isin(old_ids).sum()),
        "old_air_ids_preserved": True, "train_start": int(train_start), "target_end": int(target_end),
        "schedule_rows_overlap_scope": True, "boundary_full_end_preserved": True,
        "content_metadata_used": False, "candidate_pool_changed": False,
    }


def eligible_airings(schedule: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    """Opportunity to satisfy >=80% of the original full airing duration."""
    overlap = np.maximum(0, np.minimum(schedule.end.to_numpy(), right) - np.maximum(schedule.start.to_numpy(), left))
    duration = schedule.end.to_numpy() - schedule.start.to_numpy()
    return schedule.loc[5 * overlap >= 4 * duration]


def build_exposure_state(
    schedule: pd.DataFrame, candidates: list[int], cutoff: int, targetend: int,
    train_start: int = TRAIN_START,
) -> ExposureState:
    """Cache raw-channel and unique(item,start,end) opportunity counts.

    Full end is retained for boundary duration tests. A row starting at or after
    targetend is rejected, so no later-period timetable aggregate can enter.
    """
    for value in (train_start, cutoff, targetend):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise TypeError("Window boundaries must be exact integer seconds")
    if not train_start < cutoff < targetend:
        raise ValueError("Require train_start < cutoff < targetend")
    data = validate_schedule(schedule)
    if ((data.start >= targetend) | (data.end <= train_start)).any():
        raise ValueError("Schedule contains rows outside the permitted train/target overlap scope")
    candidate_ids = np.sort(_integer_values(candidates, "candidates"))
    if len(candidate_ids) < 5 or len(np.unique(candidate_ids)) != len(candidate_ids) or (candidate_ids <= 0).any():
        raise ValueError("Require at least five unique positive candidates")
    data = data.loc[data.tv_show_id.isin(candidate_ids)].copy()
    windows = [(train_start, cutoff), (max(train_start, cutoff - 28 * DAY), cutoff), (cutoff, targetend)]
    raw, slots = [], []
    for left, right in windows:
        eligible = eligible_airings(data, left, right)
        raw_counts = eligible.groupby("tv_show_id").size().to_dict()
        slot_counts = eligible.drop_duplicates(["tv_show_id", "start", "end"]).groupby("tv_show_id").size().to_dict()
        raw.append(np.array([raw_counts.get(int(item), 0) for item in candidate_ids], dtype=np.float64))
        slots.append(np.array([slot_counts.get(int(item), 0) for item in candidate_ids], dtype=np.float64))
    complete = eligible_airings(data.loc[data.end <= cutoff], train_start, cutoff).copy()
    complete["weight"] = _airing_weights(complete.end.to_numpy(), int(cutoff), 28.)
    denominator_counts = complete.groupby("tv_show_id").weight.sum().to_dict()
    denominator = np.array([denominator_counts.get(int(item), 0.) for item in candidate_ids], dtype=np.float64)
    arrays = [candidate_ids, *raw, *slots, denominator]
    for array in arrays:
        if not np.isfinite(array).all() or (array < 0).any():
            raise ArithmeticError("Invalid exposure cache")
        array.setflags(write=False)
    diagnostics = {
        "train_start": int(train_start), "cutoff": int(cutoff), "target_end": int(targetend),
        "train_days": (cutoff - train_start) / DAY, "target_days": (targetend - cutoff) / DAY,
        "schedule_rows_within_scope": len(schedule), "candidate_schedule_rows": len(data),
        "eligible_training_raw_airings": int(raw[0].sum()), "eligible_target_raw_airings": int(raw[2].sum()),
        "eligible_training_slot_airings": int(slots[0].sum()), "eligible_target_slot_airings": int(slots[2].sum()),
        "complete_probability_training_airings": len(complete),
        "eligibility": "overlap(window)>=.8*(original end-original start)",
        "raw": "distinct channel,start,end,item", "slots": "distinct item,start,end across channels",
        "probability_weight": "2**(-(cutoff-airing_end)/(28*86400)); complete eligible training airings only",
        "future_viewing_used": False, "future_content_metadata_used": False,
    }
    return ExposureState(candidate_ids, *raw, *slots, denominator,
                         complete[["air_id", "tv_show_id", "end", "weight"]].reset_index(drop=True), diagnostics)


def exposure_ratio(state: ExposureState, mode: str) -> np.ndarray:
    if mode == "raw":
        long, last28, future = state.e_long_raw, state.e_last28_raw, state.f_raw
    elif mode == "slots":
        long, last28, future = state.e_long_slots, state.e_last28_slots, state.f_slots
    else:
        raise ValueError("exposure_mode must be 'raw' or 'slots'")
    d = .5 * (last28 + long * 28. / state.diagnostics["train_days"])
    return (future * 28. / state.diagnostics["target_days"] + 1.) / (d + 1.)
