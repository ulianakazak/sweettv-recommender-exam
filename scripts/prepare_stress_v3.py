"""Prepare one isolated early stress fold; no models or quality scores.

The label construction is unchanged: reuse original matched fragments, clip
observed seconds separately to train/target windows, aggregate with the existing
aggregate_window at 0.8 of the full original airing duration. Future schedule
columns are used only for availability.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from sweettv.prepare import aggregate_window, seconds


ROOT = Path(__file__).resolve().parents[1]
TRAIN_START, CUTOFF, VALID_END = "2020-03-09", "2020-04-06", "2020-05-04"
FRAGMENT_COLUMNS = ["uid", "air_id", "tv_show_id", "start", "end", "duration"]
EVENT_COLUMNS = ["user_id", "uid", "air_id", "tv_show_id", "last_time"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _count_dict(frame):
    return {(row.user_id, int(row.tv_show_id)): (int(row.count), int(row.watch_seconds), int(row.last_time))
            for row in frame.itertuples(index=False)}


def _independent_window(rows, start, end, names):
    """Independent integer clipping/union/Counter, without aggregate_window."""
    intervals, durations = defaultdict(list), {}
    for uid, air_id, item, lo, hi, duration in rows:
        clipped_lo, clipped_hi = max(int(lo), start), min(int(hi), end)
        if clipped_hi <= clipped_lo:
            continue
        key = int(uid), int(air_id), int(item)
        if key in durations and durations[key] != int(duration):
            raise AssertionError("Inconsistent full duration for one user-airing")
        durations[key] = int(duration)
        intervals[key].append((clipped_lo, clipped_hi))
    count, watched, last, events = Counter(), Counter(), {}, set()
    for (uid, air_id, item), pieces in intervals.items():
        union = []
        for lo, hi in sorted(pieces):
            if union and lo <= union[-1][1]:
                union[-1] = union[-1][0], max(union[-1][1], hi)
            else:
                union.append((lo, hi))
        coverage = sum(hi - lo for lo, hi in union)
        assert coverage == sum(hi - lo for lo, hi in pieces), "Duplicate source fragment seconds"
        duration = durations[uid, air_id, item]
        assert 0 < coverage <= duration
        if 5 * coverage < 4 * duration:
            continue
        user, endpoint = names[uid], max(hi for lo, hi in union)
        pair = user, item
        count[pair] += 1
        watched[pair] += coverage
        last[pair] = max(last.get(pair, start), endpoint)
        events.add((user, uid, air_id, item, endpoint))
    return {pair: (number, watched[pair], last[pair]) for pair, number in count.items()}, events


def prepare_stress(prepared_dir: Path, target_dir: Path) -> dict:
    """Create new caches and return a report suitable for freezing pre-score."""
    started = time.perf_counter()
    prepared_dir, target_dir = Path(prepared_dir).resolve(), Path(target_dir).resolve()
    if target_dir == prepared_dir or target_dir.is_relative_to(prepared_dir):
        raise ValueError("Stress outputs must be separate from immutable inputs")
    if target_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing stress directory: {target_dir}")
    source_paths = {name: prepared_dir / name for name in ["matched_fragments.parquet", "user_map.csv", "train_schedule.parquet"]}
    before = {name: sha256(path) for name, path in source_paths.items()}
    user_map = pd.read_csv(source_paths["user_map.csv"], dtype={"uid": "int64", "user_id": str}, keep_default_na=False).sort_values("uid").reset_index(drop=True)
    if user_map.empty or user_map.user_id.eq("").any() or user_map.user_id.duplicated().any():
        raise ValueError("User IDs must be unique nonempty exact strings")
    if not np.array_equal(user_map.uid.to_numpy(), np.arange(len(user_map))):
        raise ValueError("uid must be a unique dense zero-based mapping")
    names = user_map.user_id.to_numpy(dtype=object)
    fragments = pd.read_parquet(source_paths["matched_fragments.parquet"], columns=FRAGMENT_COLUMNS)
    schedule = pd.read_parquet(source_paths["train_schedule.parquet"], columns=["air_id", "start", "end", "tv_show_id"])
    for label, frame in [("fragments", fragments), ("schedule", schedule)]:
        if any(not is_integer_dtype(frame[column].dtype) or frame[column].isna().any() for column in frame.columns):
            raise TypeError(f"{label} must contain exact nonmissing integer columns")
    assert ((fragments.uid >= 0) & (fragments.uid < len(names)) & fragments.tv_show_id.gt(0) & fragments.end.gt(fragments.start) & fragments.duration.gt(0)).all()
    assert schedule.air_id.is_unique and schedule.end.gt(schedule.start).all()
    durations = fragments[["air_id", "tv_show_id", "duration"]].drop_duplicates().merge(
        schedule.assign(full_duration=schedule.end - schedule.start)[["air_id", "tv_show_id", "full_duration"]],
        on=["air_id", "tv_show_id"], how="left", validate="many_to_one")
    assert durations.full_duration.notna().all() and durations.duration.eq(durations.full_duration).all(), "Full airing denominator mismatch"
    lower, middle, upper = map(seconds, [TRAIN_START, CUTOFF, VALID_END])
    train, _, train_positive = aggregate_window(fragments, lower, middle, names, threshold=0.8)
    valid, _, valid_positive = aggregate_window(fragments, middle, upper, names, threshold=0.8)
    events = train_positive[["uid", "air_id", "tv_show_id", "last_time"]].copy()
    events["user_id"] = names[events.uid.to_numpy(dtype=np.int64)]
    events = events[EVENT_COLUMNS].sort_values(["uid", "last_time", "air_id"]).reset_index(drop=True)
    candidates = pd.DataFrame({"tv_show_id": np.sort(schedule.loc[schedule.start.lt(upper) & schedule.end.gt(middle) & schedule.tv_show_id.ne(0), "tv_show_id"].unique())})
    assert set(valid.tv_show_id).issubset(set(candidates.tv_show_id))
    assert not events.duplicated(["user_id", "air_id"]).any()
    assert events.user_id.map(lambda value: isinstance(value, str)).all()
    assert events.last_time.gt(lower).all() and events.last_time.le(middle).all()
    assert valid.last_time.gt(middle).all() and valid.last_time.le(upper).all()
    assert int(train["count"].sum()) == len(events)
    event_counts = events.groupby(["user_id", "tv_show_id"], as_index=False).agg(count=("air_id", "size"), last_time=("last_time", "max"))
    merged = train.merge(event_counts, on=["user_id", "tv_show_id"], how="outer", suffixes=("_saved", "_event"), validate="one_to_one")
    assert merged.count_saved.eq(merged.count_event).all() and merged.last_time_saved.eq(merged.last_time_event).all()
    # Prefer actual airing crossings at each boundary, then deterministic uid
    # extrema/fill. This sample is selected without recommendation scores.
    sample_uids, crossing_details = set(), {}
    for date, boundary in [(CUTOFF, middle), (VALID_END, upper)]:
        crossing_ids = schedule.loc[schedule.start.lt(boundary) & schedule.end.gt(boundary), "air_id"]
        possible = sorted(set(map(int, fragments.loc[fragments.air_id.isin(crossing_ids), "uid"])))
        selected = possible[:4]
        sample_uids.update(selected)
        crossing_details[date] = {"crossing_airings": len(crossing_ids), "selected_uids": selected}
    sample_uids.update([0, len(names) // 2, len(names) - 1])
    for uid in range(len(names)):
        if len(sample_uids) >= min(12, len(names)):
            break
        sample_uids.add(uid)
    sample_uids = sorted(sample_uids)
    sample_names = set(names[sample_uids])
    sample = fragments.loc[fragments.uid.isin(sample_uids)]
    rows = list(sample.itertuples(index=False, name=None))
    expected_train, expected_events = _independent_window(rows, lower, middle, names)
    expected_valid, _ = _independent_window(rows, middle, upper, names)
    assert _count_dict(train.loc[train.user_id.isin(sample_names)]) == expected_train
    assert _count_dict(valid.loc[valid.user_id.isin(sample_names)]) == expected_valid
    assert set(events.loc[events.user_id.isin(sample_names)].itertuples(index=False, name=None)) == expected_events
    # All saved train endpoints also agree with the clipped fragment source.
    observed = fragments.loc[fragments.start.lt(middle) & fragments.end.gt(lower), ["uid", "air_id", "end"]].copy()
    observed["observed_end"] = np.minimum(observed.end, middle)
    max_end = observed.groupby(["uid", "air_id"], as_index=False).observed_end.max()
    endpoints = events.merge(max_end, on=["uid", "air_id"], how="left", validate="one_to_one")
    assert endpoints.observed_end.notna().all() and endpoints.last_time.eq(endpoints.observed_end).all()
    target_dir.mkdir(parents=True, exist_ok=False)
    train.to_csv(target_dir / "train_counts.csv", index=False)
    valid.to_csv(target_dir / "valid_counts.csv", index=False)
    events.to_parquet(target_dir / "train_events.parquet", index=False)
    candidates.to_csv(target_dir / "candidates.csv", index=False)
    reopened = pd.read_parquet(target_dir / "train_events.parquet")
    pd.testing.assert_frame_equal(events, reopened)
    after = {name: sha256(path) for name, path in source_paths.items()}
    assert before == after, "Immutable prepared inputs changed"
    files = {name: {"path": str(target_dir / name), "rows": len(frame), "bytes": (target_dir / name).stat().st_size,
                    "sha256": sha256(target_dir / name), "columns": frame.columns.tolist()}
             for name, frame in [("train_counts.csv", train), ("valid_counts.csv", valid), ("train_events.parquet", events), ("candidates.csv", candidates)]}
    return {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "quality_scores_computed": False,
            "model_scoring_performed": False, "target_rule_changed": False,
            "window": {"train_start_inclusive": TRAIN_START, "train_end_exclusive": CUTOFF,
                       "valid_start_inclusive": CUTOFF, "valid_end_exclusive": VALID_END,
                       "encoded_bounds": {"start": lower, "cutoff": middle, "valid_end": upper}},
            "label_policy": "existing aggregate_window, threshold 0.8 of full original airing duration; separately clip observed fragment seconds in each half-open window; distinct qualified airings; no filtering of full-period positives",
            "candidate_policy": "nonzero programme IDs of any train_schedule airing overlapping target; availability only; no future content columns loaded",
            "inherited_cache_policy": "same immutable matched_fragments, including prior exclusion of 117 duration-ambiguous raw rows; no raw rejoin or new timezone assumption",
            "source_files": {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": before[name]} for name, path in source_paths.items()},
            "source_hashes_unchanged": True, "files": files,
            "train": {"user_item_pairs": len(train), "qualified_user_airings": len(events), "users_with_positive": int(train.user_id.nunique()), "positive_items": int(train.tv_show_id.nunique()), "max_last_time": int(events.last_time.max()) if len(events) else None},
            "valid": {"user_item_pairs": len(valid), "qualified_user_airings": len(valid_positive), "users_with_positive": int(valid.user_id.nunique()), "positive_items": int(valid.tv_show_id.nunique()), "max_last_time": int(valid.last_time.max()) if len(valid) else None},
            "candidate_count": len(candidates),
            "checks": {"independent_python_union_counter_sample_matches": True, "sample_uids": sample_uids,
                       "sample_user_ids": [str(names[uid]) for uid in sample_uids], "sample_source_fragments": len(sample),
                       "sample_train_positive_airings": len(expected_events), "sample_train_pairs": len(expected_train),
                       "sample_valid_pairs": len(expected_valid), "cutoff_sample_selection": crossing_details,
                       "all_train_endpoints_match_clipped_source": True, "all_train_event_counts_reaggregate_exactly": True,
                       "all_user_ids_exact_strings": True, "all_valid_items_available": True,
                       "all_observation_times_within_window": True, "full_duration_denominators_checked": True,
                       "parquet_readback_equal": True},
            "elapsed_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "work/prepared")
    parser.add_argument("--target-dir", type=Path, default=ROOT / "work/improvements_v3/stress")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/improvements_v3/stress_preparation_checks.json")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Refusing to overwrite the already prepared stress report")
    result = prepare_stress(args.prepared_dir, args.target_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "quality_scores_computed": False, "train": result["train"],
                      "valid": result["valid"], "candidate_count": result["candidate_count"],
                      "source_hashes_unchanged": True, "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
