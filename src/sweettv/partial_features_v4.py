"""Train-only partial coverage features, kept separate from qualified history.

No model or target labels are accepted. Repeated/nested source fragments are
explicitly unioned within user-airing before clipping to an observation window.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from .prepare import merge_intervals


FRAGMENT_COLUMNS = ["uid", "air_id", "tv_show_id", "start", "end", "duration"]
AIRING_COLUMNS = ["user_id", "uid", "air_id", "tv_show_id", "watch_seconds", "duration", "last_time", "coverage"]
STRONG_EVENT_COLUMNS = ["user_id", "uid", "air_id", "tv_show_id", "last_time"]


def _require_integers(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame or not is_integer_dtype(frame[column].dtype) or frame[column].isna().any():
            raise TypeError(f"{column} must contain exact nonmissing integers")


def mapped_user_names(user_map: pd.DataFrame) -> np.ndarray:
    _require_integers(user_map, ["uid"])
    if "user_id" not in user_map or not user_map.user_id.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise TypeError("User IDs must be exact nonempty strings, never inferred floats")
    ordered = user_map.sort_values("uid").reset_index(drop=True)
    if ordered.empty or ordered.user_id.duplicated().any() or not np.array_equal(ordered.uid.to_numpy(), np.arange(len(ordered))):
        raise ValueError("User map must have unique string IDs and dense zero-based uid")
    return ordered.user_id.to_numpy(dtype=object)


def union_partial_fragments(fragments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return explicit unions; source rows/seconds are never counted twice."""
    started = time.perf_counter()
    _require_integers(fragments, FRAGMENT_COLUMNS)
    if not (fragments.uid.ge(0) & fragments.tv_show_id.gt(0) & fragments.end.gt(fragments.start) & fragments.duration.gt(0)).all():
        raise ValueError("Only positive labelled source fragments are supported")
    definitions = fragments[["air_id", "tv_show_id", "duration"]].drop_duplicates()
    if definitions.air_id.duplicated().any():
        raise ValueError("One air_id must have one programme and one full duration")
    # Reuse the already audited interval union with air_id as the group key;
    # no assumptions about fragment order, duplicates or nesting are needed.
    temporary = fragments[["uid", "air_id", "start", "end"]].rename(columns={"air_id": "channel_id"})
    merged = merge_intervals(temporary).rename(columns={"channel_id": "air_id"})
    merged = merged.merge(definitions, on="air_id", how="left", validate="many_to_one")[FRAGMENT_COLUMNS]
    raw_seconds = int((fragments.end - fragments.start).sum())
    union_seconds = int((merged.end - merged.start).sum())
    if union_seconds > raw_seconds:
        raise AssertionError("Union introduced seconds absent from the source")
    return merged, {"source_rows": len(fragments), "union_rows": len(merged),
                    "source_fragment_seconds": raw_seconds, "union_fragment_seconds": union_seconds,
                    "duplicate_seconds_removed": raw_seconds - union_seconds,
                    "explicit_union_performed": True, "elapsed_seconds": time.perf_counter() - started}


def aggregate_unioned_airings(unioned: pd.DataFrame, user_map: pd.DataFrame,
                              start: int, cutoff: int) -> pd.DataFrame:
    """Clip an explicit union and retain every positive user-airing coverage.

    Caller must pass output of union_partial_fragments. No qualification filter
    is applied here. Full airing duration is unchanged at both boundaries.
    """
    names = mapped_user_names(user_map)
    if not isinstance(start, (int, np.integer)) or not isinstance(cutoff, (int, np.integer)) or cutoff <= start:
        raise ValueError("Window requires exact integer start < cutoff")
    _require_integers(unioned, FRAGMENT_COLUMNS)
    if len(unioned) and (unioned.uid.min() < 0 or unioned.uid.max() >= len(names)):
        raise ValueError("Fragment uid missing from the exact user map")
    clipped = unioned.loc[unioned.start.lt(cutoff) & unioned.end.gt(start)].copy()
    clipped["start"] = np.maximum(clipped.start, start)
    clipped["end"] = np.minimum(clipped.end, cutoff)
    clipped["watch_seconds"] = clipped.end - clipped.start
    airings = clipped.groupby(["uid", "air_id", "tv_show_id"], as_index=False).agg(
        watch_seconds=("watch_seconds", "sum"), duration=("duration", "first"), last_time=("end", "max"))
    if not (airings.watch_seconds.gt(0) & airings.watch_seconds.le(airings.duration) &
            airings.last_time.gt(start) & airings.last_time.le(cutoff)).all():
        raise AssertionError("Partial airing coverage escaped source duration or temporal bounds")
    airings["user_id"] = names[airings.uid.to_numpy(dtype=np.int64)]
    airings["coverage"] = airings.watch_seconds.astype(np.float64) / airings.duration
    return airings[AIRING_COLUMNS].sort_values(["uid", "air_id"]).reset_index(drop=True)


def aggregate_partial_airings(fragments: pd.DataFrame, user_map: pd.DataFrame,
                              start: int, cutoff: int) -> tuple[pd.DataFrame, dict]:
    """Safe public API: explicit union then fresh clipping at the given cutoff."""
    unioned, audit = union_partial_fragments(fragments)
    return aggregate_unioned_airings(unioned, user_map, start, cutoff), audit


def qualified_event_rows(airings: pd.DataFrame) -> pd.DataFrame:
    """Exactly the old >=0.8 qualified history; never part of weak profiles."""
    _require_integers(airings, ["uid", "air_id", "tv_show_id", "watch_seconds", "duration", "last_time"])
    mask = 5 * airings.watch_seconds >= 4 * airings.duration
    return airings.loc[mask, STRONG_EVENT_COLUMNS].sort_values(["uid", "air_id"]).reset_index(drop=True)


def eligible_weak_airings(airings: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Weak only: 60+ seconds, 5%+ coverage, strictly below 80%."""
    _require_integers(airings, ["uid", "air_id", "tv_show_id", "watch_seconds", "duration", "last_time"])
    if not airings.user_id.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise TypeError("User IDs must remain exact strings")
    if not (airings.watch_seconds.gt(0) & airings.watch_seconds.le(airings.duration) & airings.duration.gt(0) & airings.last_time.le(cutoff)).all():
        raise ValueError("Invalid or future user-airing supplied to weak feature builder")
    if airings.duplicated(["user_id", "air_id"]).any():
        raise ValueError("Weak feature input must contain one row per user-airing")
    mask = (airings.watch_seconds.ge(60) & (20 * airings.watch_seconds >= airings.duration) &
            (5 * airings.watch_seconds < 4 * airings.duration))
    return airings.loc[mask, AIRING_COLUMNS].copy()


def build_weak_profile(airings: pd.DataFrame, cutoff: int, power: int = 2,
                       half_life_days: float = 28) -> pd.DataFrame:
    """Separate user-item sum of coverage**p times HL28 event recency.

    This creates features only; it produces neither a model score nor a ranked
    list. Qualified airings cannot enter this profile. p=2/4 and HL=28 are fixed.
    """
    if power not in (2, 4) or isinstance(power, bool) or half_life_days != 28:
        raise ValueError("Frozen weak feature design requires power 2 or 4 and half-life 28 days")
    weak = eligible_weak_airings(airings, cutoff)
    age = cutoff - weak.last_time.to_numpy(dtype=np.int64)
    coverage = weak.watch_seconds.to_numpy(dtype=np.float64) / weak.duration.to_numpy(dtype=np.float64)
    weak["weak_weight"] = np.power(coverage, power) * np.exp2(-age / (28 * 86400))
    result = weak.groupby(["user_id", "tv_show_id"], as_index=False).agg(
        weak_weight=("weak_weight", "sum"), weak_airings=("air_id", "size"), last_time=("last_time", "max"))
    return result.sort_values(["user_id", "tv_show_id"]).reset_index(drop=True)
