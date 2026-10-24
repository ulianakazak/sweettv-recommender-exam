"""Prepare audited, temporally bounded interactions without future content features."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

META = ["tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
        "tv_show_genre_3", "year_of_production", "director", "actors"]


def seconds(value: str) -> int:
    """Integer encoding of naive wallclock, not a claim of UTC timezone."""
    return int(pd.Timestamp(value).value // 1_000_000_000)


def merge_intervals(views: pd.DataFrame) -> pd.DataFrame:
    """Union positive intervals within (uid, channel_id), including nested ones.

    Input/output: uid, channel_id, start, end (all exact integers).
    Adjacent intervals can be merged because their union contains no gap.
    """
    if views.empty:
        return views[["uid", "channel_id", "start", "end"]].copy()
    data = views.loc[views.end > views.start, ["uid", "channel_id", "start", "end"]]
    if data.empty:
        return data.copy()
    order = np.lexsort((data.end.to_numpy(), data.start.to_numpy(),
                        data.channel_id.to_numpy(), data.uid.to_numpy()))
    uid, channel, start, end = (data[c].to_numpy()[order] for c in data.columns)
    new_group = np.r_[True, (uid[1:] != uid[:-1]) | (channel[1:] != channel[:-1])]
    group = np.cumsum(new_group, dtype=np.int64) - 1
    # Encode groups into disjoint ranges so an accumulate resets at each group.
    minimum, span = int(end.min()), int(end.max() - end.min()) + 1
    if int(group[-1]) * span + span >= np.iinfo(np.int64).max:
        raise OverflowError("Interval group encoding would overflow int64")
    running_end = np.maximum.accumulate(end - minimum + group * span) - group * span + minimum
    previous_end = np.r_[end[0], running_end[:-1]]
    begin = np.flatnonzero(new_group | (start > previous_end))
    result = pd.DataFrame({"uid": uid[begin], "channel_id": channel[begin],
                           "start": start[begin], "end": np.maximum.reduceat(end, begin)})
    return result


def conflict_indices(airings: pd.DataFrame) -> set[int]:
    conflicts: set[int] = set()
    for _, part in airings.groupby("channel_id", sort=False):
        active: list[tuple[int, int]] = []
        for idx, start, end in part.sort_values(["start", "end", "tv_show_id"])[["start", "end"]].itertuples(name=None):
            active = [(j, e) for j, e in active if e > start]
            if active:
                conflicts.add(int(idx))
                conflicts.update(int(j) for j, _ in active)
            active.append((int(idx), int(end)))
    return conflicts


def match_intervals(views: pd.DataFrame, airings: pd.DataFrame, include_view_index: bool = False) -> pd.DataFrame:
    """Robust interval join. No assumption that programme endings are sorted.

    Views must already be unioned within uid/channel. Airings have air_id,
    channel_id, start, end, tv_show_id. Return disjoint matched fragments.
    """
    pieces = []
    for channel, group in views.groupby("channel_id", sort=True):
        schedule = airings.loc[airings.channel_id == channel].sort_values(["start", "end", "air_id"])
        if schedule.empty:
            continue
        astart, aend = schedule.start.to_numpy(), schedule.end.to_numpy()
        vstart, vend = group.start.to_numpy(), group.end.to_numpy()
        # prefix_max_end is monotone even with nested/overlapping airings.
        lower = np.searchsorted(np.maximum.accumulate(aend), vstart, side="right")
        upper = np.searchsorted(astart, vend, side="left")
        lengths = np.maximum(0, upper - lower)
        total = int(lengths.sum())
        if not total:
            continue
        vi = np.repeat(np.arange(len(group)), lengths)
        offsets = np.cumsum(lengths) - lengths
        ai = np.repeat(lower, lengths) + np.arange(total) - np.repeat(offsets, lengths)
        start = np.maximum(vstart[vi], astart[ai])
        end = np.minimum(vend[vi], aend[ai])
        valid = end > start
        values = {
            "uid": group.uid.to_numpy()[vi[valid]],
            "air_id": schedule.air_id.to_numpy()[ai[valid]],
            "tv_show_id": schedule.tv_show_id.to_numpy()[ai[valid]],
            "start": start[valid], "end": end[valid],
            "duration": aend[ai[valid]] - astart[ai[valid]],
        }
        if include_view_index:
            values["view_index"] = group.index.to_numpy()[vi[valid]]
        pieces.append(pd.DataFrame(values))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["uid", "air_id", "tv_show_id", "start", "end", "duration"])


def aggregate_window(fragments: pd.DataFrame, start: int, end: int,
                     user_names: np.ndarray, threshold: float = 0.8):
    """Clip observation to window; full airing duration is always denominator."""
    f = fragments.loc[(fragments.start < end) & (fragments.end > start)].copy()
    f["start"] = np.maximum(f.start, start)
    f["end"] = np.minimum(f.end, end)
    f["watch_seconds"] = f.end - f.start
    per_air = f.groupby(["uid", "air_id", "tv_show_id"], as_index=False).agg(
        watch_seconds=("watch_seconds", "sum"), duration=("duration", "first"),
        last_time=("end", "max"))
    if len(per_air) and not (per_air.watch_seconds <= per_air.duration).all():
        raise AssertionError("Viewing union failed: coverage exceeds airing duration")
    positive = per_air.loc[per_air.watch_seconds >= threshold * per_air.duration].copy()
    counts = positive.groupby(["uid", "tv_show_id"], as_index=False).agg(
        count=("air_id", "size"), watch_seconds=("watch_seconds", "sum"), last_time=("last_time", "max"))
    counts["user_id"] = user_names[counts.uid.to_numpy(dtype=int)]
    counts = counts[["user_id", "tv_show_id", "count", "watch_seconds", "last_time"]]
    counts = counts.sort_values(["user_id", "tv_show_id"]).reset_index(drop=True)
    return counts, per_air, positive


def catalogue(airings: pd.DataFrame, before: int, after: int | None = None) -> pd.DataFrame:
    mask = airings.start < before
    if after is not None:
        mask &= airings.end > after
    data = airings.loc[mask].sort_values(["start", "channel_id", "tv_show_id"])
    # Last nonempty value of each field, using train schedule only.
    result = data[["tv_show_id"] + META].replace("", pd.NA).groupby("tv_show_id", sort=True).last()
    return result.fillna("").reset_index()


def brute_force_check(views: pd.DataFrame, airings: pd.DataFrame, matched: pd.DataFrame):
    """Independent nested loops on a deterministic real sample, not same algorithm."""
    sampled = []
    for _, part in views.groupby("channel_id", sort=True):
        sampled.append(part.iloc[np.linspace(0, len(part) - 1, min(3, len(part)), dtype=int)])
    sample = pd.concat(sampled, ignore_index=True)
    expected = []
    for row in sample.itertuples(index=False):
        relevant = airings.loc[airings.channel_id == row.channel_id]
        for a in relevant.itertuples(index=False):
            lo, hi = max(row.start, a.start), min(row.end, a.end)
            if hi > lo:
                expected.append((row.uid, a.air_id, a.tv_show_id, lo, hi, a.end - a.start))
    actual = match_intervals(sample, airings)
    expected = sorted(expected)
    actual_list = sorted(actual.itertuples(index=False, name=None))
    if actual_list != expected:
        raise AssertionError("Real sample interval join differs from brute force")
    return {"real_sample_intervals": len(sample), "matched_fragments": len(expected),
            "independent_brute_force_match": True}


def run(config_path: Path) -> None:
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    raw = Path(config["data_dir"])
    prepared, output = Path(config["prepared_dir"]), Path(config["output_dir"])
    prepared.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    boundaries = {key: seconds(config[key]) for key in ["start", "tune_cutoff", "validation_cutoff", "test_cutoff", "test_end"]}
    periods = {
        "tune_train": (boundaries["start"], boundaries["tune_cutoff"]),
        "tune_valid": (boundaries["tune_cutoff"], boundaries["validation_cutoff"]),
        "train": (boundaries["start"], boundaries["validation_cutoff"]),
        "valid": (boundaries["validation_cutoff"], boundaries["test_cutoff"]),
        "full": (boundaries["start"], boundaries["test_cutoff"]),
    }
    print("Loading original viewing log", flush=True)
    log = pd.read_csv(raw / "dataset11-30.csv", dtype={"user_id": str, "vsetv_id": np.int32,
                      "duraton": np.int64}, keep_default_na=False)
    start = pd.to_datetime(log.start_time, format="%Y-%m-%d %H:%M:%S").astype("int64").to_numpy() // 1_000_000_000
    end = pd.to_datetime(log.stop_time, format="%Y-%m-%d %H:%M:%S").astype("int64").to_numpy() // 1_000_000_000
    duration_ok = (end - start == log.duraton.to_numpy()) & (end > start)
    names = np.array(sorted(log.user_id.unique()), dtype=object)
    mapping = {name: i for i, name in enumerate(names)}
    all_views = pd.DataFrame({"uid": log.user_id.map(mapping).to_numpy(dtype=np.int32),
                              "channel_id": log.vsetv_id.to_numpy(), "start": start, "end": end})
    raw_rows, raw_seconds = len(log), int((end - start).sum())
    excluded = int((~duration_ok).sum())
    raw_examples = log.loc[~duration_ok].head(3).to_dict("records")
    retained = all_views.loc[duration_ok].copy()
    del log, start, end
    gc.collect()
    print("Unioning same-user same-channel intervals", flush=True)
    views = merge_intervals(retained)
    union_seconds = int((views.end - views.start).sum())
    retained_seconds = int((retained.end - retained.start).sum())
    print(f"Retained rows={len(retained)}; union segments={len(views)}", flush=True)
    parts = []
    for filename in ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv"]:
        data = pd.read_csv(raw / filename, dtype=str, keep_default_na=False)
        data = data.loc[(data.tv_show_id != "0") & (data.duration.astype("int64") > 0)].copy()
        data["channel_id"] = data.channel_id.astype(np.int32)
        data["tv_show_id"] = data.tv_show_id.astype(np.int64)
        data["start"] = pd.to_datetime(data.start_time, format="%d.%m.%Y %H:%M:%S").astype("int64") // 1_000_000_000
        data["end"] = data.start + data.duration.astype(np.int64)
        parts.append(data[["channel_id", "tv_show_id", "start", "end"] + META])
    airings = pd.concat(parts, ignore_index=True).drop_duplicates(["channel_id", "start", "end", "tv_show_id"])
    airings = airings.loc[(airings.start < boundaries["test_cutoff"]) & (airings.end > boundaries["start"])].copy()
    conflicts = conflict_indices(airings)
    schedule_conflict_rows = len(conflicts)
    clean = airings.drop(index=list(conflicts)).reset_index(drop=True)
    clean["air_id"] = np.arange(len(clean), dtype=np.int64)
    # Ordering these columns is the public join interface.
    join_airings = clean[["air_id", "channel_id", "start", "end", "tv_show_id"]]
    print(f"Joining {len(views)} union intervals to {len(clean)} labelled airings", flush=True)
    fragments = match_intervals(views, join_airings)
    print(f"Matched fragments={len(fragments)}", flush=True)
    check = brute_force_check(views, join_airings, fragments)
    raw_hits = match_intervals(retained, join_airings, include_view_index=True)
    fragments.to_parquet(prepared / "matched_fragments.parquet", index=False)
    views.to_parquet(prepared / "view_unions.parquet", index=False)
    clean.to_parquet(prepared / "train_schedule.parquet", index=False)
    pd.DataFrame({"uid": np.arange(len(names)), "user_id": names}).to_csv(prepared / "user_map.csv", index=False)
    stats = {}
    for name, (lo, hi) in periods.items():
        counts, per_air, positive = aggregate_window(fragments, lo, hi, names, config["watch_fraction_min"])
        counts.to_csv(prepared / f"{name}_counts.csv", index=False)
        clipped_view = views.loc[(views.start < hi) & (views.end > lo)]
        view_seconds = int((np.minimum(clipped_view.end, hi) - np.maximum(clipped_view.start, lo)).sum())
        raw_window = retained.loc[(retained.start < hi) & (retained.end > lo)]
        # Count original eligible records with positive overlap INSIDE this window.
        window_raw_hits = raw_hits.loc[(raw_hits.start < hi) & (raw_hits.end > lo)]
        raw_matched_records = int(window_raw_hits.view_index.nunique())
        matched_seconds = int(per_air.watch_seconds.sum())
        stats[name] = {"window_start": config[{"tune_train": "start", "tune_valid": "tune_cutoff", "train": "start", "valid": "validation_cutoff", "full": "start"}[name]],
                       "window_end_exclusive": pd.to_datetime(hi, unit="s").isoformat(),
                       "user_item_pairs": len(counts), "positive_airings_user_pairs": len(positive),
                       "users_with_positive": int(counts.user_id.nunique()),
                       "positive_programmes": int(counts.tv_show_id.nunique()),
                       "all_union_view_seconds": view_seconds, "labelled_matched_union_seconds": matched_seconds,
                       "labelled_second_fraction": matched_seconds / view_seconds if view_seconds else 0,
                       "positive_union_seconds": int(positive.watch_seconds.sum()),
                       "positive_second_fraction": int(positive.watch_seconds.sum()) / view_seconds if view_seconds else 0,
                       "matched_airings_user_pairs": len(per_air),
                       "positive_airing_pair_fraction": len(positive) / len(per_air) if len(per_air) else 0,
                       "raw_eligible_records_in_window": len(raw_window),
                       "raw_records_with_labelled_overlap": raw_matched_records,
                       "raw_labelled_record_fraction": raw_matched_records / len(raw_window) if len(raw_window) else 0,
                       "raw_users_in_window": int(raw_window.uid.nunique()),
                       "positive_user_fraction": int(counts.user_id.nunique()) / raw_window.uid.nunique() if len(raw_window) else 0,
                       "max_positive_observed_time": int(positive.last_time.max()) if len(positive) else None}
        assert not len(counts) or counts.last_time.max() <= hi
        print(f"{name}: {stats[name]}", flush=True)
    for label, hi in [("tune", boundaries["tune_cutoff"]), ("train", boundaries["validation_cutoff"]), ("full", boundaries["test_cutoff"])]:
        catalogue(clean, hi).to_csv(prepared / f"catalog_{label}.csv", index=False)
    catalogue(clean, boundaries["test_cutoff"], boundaries["validation_cutoff"]).to_csv(prepared / "catalog_display_valid.csv", index=False)
    for label, lo, hi in [("tune", boundaries["tune_cutoff"], boundaries["validation_cutoff"]),
                          ("valid", boundaries["validation_cutoff"], boundaries["test_cutoff"])]:
        ids = np.sort(airings.loc[(airings.start < hi) & (airings.end > lo), "tv_show_id"].unique())
        pd.DataFrame({"tv_show_id": ids}).to_csv(prepared / f"candidates_{label}.csv", index=False)
    future = pd.read_csv(raw / "export_arh_31-42-final.csv", usecols=["channel_id", "start_time", "duration", "tv_show_id"], dtype=str, keep_default_na=False)
    future = future.loc[(future.tv_show_id != "0") & (future.duration.astype(np.int64) > 0)].copy()
    future["start"] = pd.to_datetime(future.start_time, format="%d.%m.%Y %H:%M:%S").astype("int64") // 1_000_000_000
    future["end"] = future.start + future.duration.astype(np.int64)
    future_ids = np.union1d(
        future.loc[(future.start < boundaries["test_end"]) & (future.end > boundaries["test_cutoff"]), "tv_show_id"].astype(np.int64).unique(),
        airings.loc[(airings.start < boundaries["test_end"]) & (airings.end > boundaries["test_cutoff"]), "tv_show_id"].unique())
    pd.DataFrame({"tv_show_id": np.sort(future_ids)}).to_csv(prepared / "candidates_test.csv", index=False)
    template = pd.read_csv(raw / "submission.csv", dtype=str, keep_default_na=False)
    template.to_csv(prepared / "template.csv", index=False)
    # Real numerical example: choose fragmented positive airing if present.
    full_counts, per_air, positive = aggregate_window(fragments, boundaries["start"], boundaries["test_cutoff"], names, config["watch_fraction_min"])
    multiplicity = fragments.groupby(["uid", "air_id"]).size()
    fragmented_keys = set(multiplicity.loc[multiplicity > 1].index.tolist())
    selected = next((r for r in positive.itertuples(index=False) if (r.uid, r.air_id) in fragmented_keys), next(positive.itertuples(index=False)))
    sf = fragments.loc[(fragments.uid == selected.uid) & (fragments.air_id == selected.air_id)]
    sa = clean.loc[clean.air_id == selected.air_id].iloc[0]
    overlapping_raw = all_views.loc[(all_views.uid == selected.uid) & (all_views.channel_id == sa.channel_id) &
                                    (all_views.start < sa.end) & (all_views.end > sa.start)]
    example = {"user_id": str(names[selected.uid]), "channel_id": int(sa.channel_id),
               "tv_show_id": int(sa.tv_show_id), "title": sa.tv_show_title,
               "air_start": pd.to_datetime(sa.start, unit="s").isoformat(),
               "air_end": pd.to_datetime(sa.end, unit="s").isoformat(),
               "duration_seconds": int(selected.duration), "watch_seconds_after_union": int(selected.watch_seconds),
               "watch_fraction": float(selected.watch_seconds / selected.duration),
               "qualifies_at_0_8": True,
               "raw_overlapping_view_intervals_including_any_excluded": overlapping_raw[["start", "end"]].to_dict("records"),
               "merged_fragments": sf[["start", "end"]].to_dict("records"),
               "verification": "AI computed union/overlap; not student personal verification"}
    (output / "join_example.json").write_text(json.dumps(example, ensure_ascii=False, indent=2) + "\n")
    prepared_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(prepared.glob("*.csv"))}
    result = {"config": config, "boundaries_encoding": boundaries,
              "time_encoding": "naive wallclock encoded as integer seconds, no timezone conversion",
              "raw_log_rows": raw_rows, "excluded_duration_ambiguity_rows": excluded,
              "excluded_fraction_rows": excluded / raw_rows, "duration_ambiguity_examples": raw_examples,
              "raw_seconds_before_exclusion": raw_seconds,
              "eligible_raw_seconds": retained_seconds, "union_seconds": union_seconds,
              "same_channel_double_seconds_removed": retained_seconds - union_seconds,
              "union_interval_count": len(views), "matched_fragments": len(fragments),
              "labelled_conflicting_training_airings_excluded": schedule_conflict_rows,
              "periods": stats, "join_checks": check,
              "prepared_csv_sha256": prepared_hashes,
              "test_candidates": len(future_ids), "template_users": len(template),
              "elapsed_seconds": time.perf_counter() - started,
              "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "versions": {"numpy": np.__version__, "pandas": pd.__version__},
              "human_verification": "not_confirmed"}
    (output / "preparation_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Preparation complete, seconds={result['elapsed_seconds']:.2f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/run.json"))
    run(parser.parse_args().config)
