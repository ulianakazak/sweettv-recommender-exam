"""Prepare a separate, fixed rolling validation design from immutable caches.

Every fold recomputes qualified airings from clipped matched fragments through
the original aggregate_window. Full-period positives/counts are never filtered
to obtain earlier training histories. Future schedule columns are read only as
item availability; content metadata is not loaded by this module.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from .prepare import aggregate_window, seconds


FOLDS = (
    ("fold_a", "2020-03-09", "2020-05-04", "2020-06-01"),
    ("fold_b", "2020-03-09", "2020-06-01", "2020-06-29"),
    ("fold_c", "2020-03-09", "2020-06-29", "2020-07-27"),
)
WATCH_FRACTION_MIN = 0.8
FRAGMENT_COLUMNS = ["uid", "air_id", "tv_show_id", "start", "end", "duration"]
SCHEDULE_COLUMNS = ["air_id", "start", "end", "tv_show_id"]
EVENT_COLUMNS = ["user_id", "uid", "air_id", "tv_show_id", "last_time"]


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _file_info(path: Path, rows: int) -> dict[str, Any]:
    return {"path": str(path.resolve()), "rows": int(rows),
            "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _check_integer_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        if not is_integer_dtype(frame[column].dtype) or frame[column].isna().any():
            raise TypeError(f"{label}.{column} must contain exact, nonmissing integers")


def _check_period(counts: pd.DataFrame, positive: pd.DataFrame,
                  lower: int, upper: int) -> dict[str, Any]:
    if positive.duplicated(["uid", "air_id"]).any():
        raise AssertionError("A qualified user-airing must occur exactly once")
    if not positive.empty:
        if not (positive.last_time.gt(lower) & positive.last_time.le(upper)).all():
            raise AssertionError("Observed positive-airing time escaped its temporal window")
        if not (positive.watch_seconds.le(positive.duration) &
                positive.watch_seconds.ge(WATCH_FRACTION_MIN * positive.duration)).all():
            raise AssertionError("A positive airing violates the full-duration threshold")
    if counts.duplicated(["user_id", "tv_show_id"]).any():
        raise AssertionError("Aggregated user-programme pairs are not unique")
    if not counts.empty:
        if not counts.user_id.map(lambda value: isinstance(value, str)).all():
            raise TypeError("Large user IDs must remain exact strings")
        if not (counts["count"].gt(0) & counts.watch_seconds.gt(0) &
                counts.tv_show_id.gt(0) & counts.last_time.gt(lower) &
                counts.last_time.le(upper)).all():
            raise AssertionError("Counts violate positivity or temporal bounds")
    if int(counts["count"].sum()) != len(positive):
        raise AssertionError("Counts must count qualified airings, not fragments")
    return {"user_item_pairs": len(counts), "qualified_user_airings": len(positive),
            "users_with_positive_history": int(counts.user_id.nunique()),
            "programmes_with_positive_history": int(counts.tv_show_id.nunique()),
            "min_observed_end": int(positive.last_time.min()) if len(positive) else None,
            "max_observed_end": int(positive.last_time.max()) if len(positive) else None,
            "unique_user_airings": True, "times_within_clipped_window": True,
            "full_duration_threshold_checked": True}


def prepare_folds(prepared_dir: Path, target_dir: Path) -> dict[str, Any]:
    """Write three new rolling folds and return their sizes, bounds and hashes.

    Existing fold directories are rejected rather than overwritten. The three
    source caches are hashed before/after processing. user_map.csv is read with
    an explicit string user_id dtype and checked for a dense integer uid map.
    Returned train_events has one row per positive user-airing and can be passed
    directly to frequency_v1; last_time is the observed clipped endpoint.
    """
    started = time.perf_counter()
    prepared_dir, target_dir = Path(prepared_dir).resolve(), Path(target_dir).resolve()
    if target_dir == prepared_dir or target_dir.is_relative_to(prepared_dir):
        raise ValueError("Target must be separate from the immutable prepared directory")
    for name, *_ in FOLDS:
        if (target_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing fold: {target_dir / name}")
    sources = {name: prepared_dir / name for name in (
        "matched_fragments.parquet", "user_map.csv", "train_schedule.parquet")}
    hashes_before = {name: _sha256(path) for name, path in sources.items()}
    user_map = pd.read_csv(sources["user_map.csv"], dtype={"uid": "int64", "user_id": str},
                           keep_default_na=False).sort_values("uid").reset_index(drop=True)
    if user_map.empty or user_map.user_id.eq("").any() or user_map.user_id.duplicated().any():
        raise ValueError("User map must contain unique nonempty string identifiers")
    if not np.array_equal(user_map.uid.to_numpy(), np.arange(len(user_map))):
        raise ValueError("uid map must be unique and dense from zero")
    names = user_map.user_id.to_numpy(dtype=object)
    fragments = pd.read_parquet(sources["matched_fragments.parquet"], columns=FRAGMENT_COLUMNS)
    schedule = pd.read_parquet(sources["train_schedule.parquet"], columns=SCHEDULE_COLUMNS)
    _check_integer_columns(fragments, FRAGMENT_COLUMNS, "fragments")
    _check_integer_columns(schedule, SCHEDULE_COLUMNS, "schedule")
    if len(fragments) and not ((fragments.uid >= 0) & (fragments.uid < len(names)) &
                               fragments.tv_show_id.gt(0) & fragments.end.gt(fragments.start) &
                               fragments.duration.gt(0)).all():
        raise ValueError("Fragments contain invalid user mappings, items or intervals")
    if schedule.air_id.duplicated().any() or not schedule.end.gt(schedule.start).all():
        raise ValueError("Schedule requires unique air_id and positive intervals")
    # Cross-check that the denominator really is the original full airing
    # duration. Do not silently trust an already clipped fragment length.
    definitions = fragments[["air_id", "tv_show_id", "duration"]].drop_duplicates()
    original = schedule.assign(full_duration=schedule.end - schedule.start)
    definitions = definitions.merge(original[["air_id", "tv_show_id", "full_duration"]],
                                    on=["air_id", "tv_show_id"], how="left", validate="many_to_one")
    if definitions.full_duration.isna().any() or not definitions.duration.eq(definitions.full_duration).all():
        raise ValueError("Fragment denominator does not match the original airing duration")
    target_dir.mkdir(parents=True, exist_ok=True)
    folds: dict[str, Any] = {}
    for label, train_start, cutoff, valid_end in FOLDS:
        fold_started = time.perf_counter()
        lower, middle, upper = map(seconds, [train_start, cutoff, valid_end])
        train_counts, _, train_positive = aggregate_window(
            fragments, lower, middle, names, WATCH_FRACTION_MIN)
        valid_counts, _, valid_positive = aggregate_window(
            fragments, middle, upper, names, WATCH_FRACTION_MIN)
        train_checks = _check_period(train_counts, train_positive, lower, middle)
        valid_checks = _check_period(valid_counts, valid_positive, middle, upper)
        train_events = train_positive[["uid", "air_id", "tv_show_id", "last_time"]].copy()
        train_events["user_id"] = names[train_events.uid.to_numpy(dtype=np.int64)]
        train_events = train_events[EVENT_COLUMNS].sort_values(
            ["uid", "last_time", "air_id"]).reset_index(drop=True)
        available = schedule.loc[(schedule.start < upper) & (schedule.end > middle) &
                                 schedule.tv_show_id.ne(0), "tv_show_id"]
        candidates = pd.DataFrame({"tv_show_id": np.sort(available.unique())})
        if not set(valid_counts.tv_show_id).issubset(set(candidates.tv_show_id)):
            raise AssertionError("Validation truth contains programmes unavailable in the target window")
        directory = target_dir / label
        directory.mkdir(exist_ok=False)
        train_counts.to_csv(directory / "train_counts.csv", index=False)
        valid_counts.to_csv(directory / "valid_counts.csv", index=False)
        train_events.to_parquet(directory / "train_events.parquet", index=False)
        candidates.to_csv(directory / "candidates.csv", index=False)
        folds[label] = {
            "directory": str(directory),
            "train_start_inclusive": train_start, "train_end_exclusive": cutoff,
            "valid_start_inclusive": cutoff, "valid_end_exclusive": valid_end,
            "encoded_bounds": {"train_start": lower, "cutoff": middle, "valid_end": upper},
            "train": train_checks, "valid": valid_checks,
            "candidate_count": len(candidates),
            "validation_truth_ids_available": True,
            "files": {
                name: _file_info(directory / name, len(frame))
                for name, frame in [("train_counts.csv", train_counts), ("valid_counts.csv", valid_counts),
                                    ("train_events.parquet", train_events), ("candidates.csv", candidates)]},
            "elapsed_seconds": time.perf_counter() - fold_started,
        }
    hashes_after = {name: _sha256(path) for name, path in sources.items()}
    if hashes_after != hashes_before:
        raise AssertionError("Immutable prepared inputs changed during fold preparation")
    return {"target_dir": str(target_dir), "watch_fraction_min": WATCH_FRACTION_MIN,
            "time_basis": "timezone-naive wallclock; half-open observation windows",
            "denominator": "original full airing duration, checked against train_schedule",
            "frequency_unit": "one qualified user-airing after clipping/aggregation of fragments",
            "future_schedule_use": "availability only; no content columns loaded",
            "train_recomputed_from_clipped_fragments": True,
            "source_files": {name: {"path": str(path), "bytes": path.stat().st_size,
                                     "sha256": hashes_before[name]} for name, path in sources.items()},
            "source_hashes_unchanged": True, "folds": folds,
            "elapsed_seconds": time.perf_counter() - started}
