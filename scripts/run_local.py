"""Portable local commands. No network, authentication or submission API."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def path(value):
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def digest(filename):
    result = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def dump(filename, value):
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def users(prepared):
    import pandas as pd
    result = pd.read_csv(prepared / "template.csv", dtype=str, keep_default_na=False).user_id.tolist()
    if len(result) != len(set(result)) or not all(isinstance(user, str) for user in result):
        raise ValueError("Template user IDs must be exact unique strings")
    return result


def pool(filename):
    import pandas as pd
    return sorted(map(int, pd.read_csv(filename).tv_show_id))


def prepare(args):
    from sweettv.prepare import run
    prepared, audit = path(args.prepared_dir), path(args.output_dir)
    if prepared.exists() or audit.exists():
        raise FileExistsError("Use fresh preparation/output directories")
    conf = json.loads((ROOT / "configs/run.json").read_text())
    conf.update(data_dir=str(path(args.raw_dir).resolve()), prepared_dir=str(prepared), output_dir=str(audit))
    generated = audit.parent / (audit.name + "_run_config.json")
    dump(generated, conf)
    run(generated)


def baselines(args):
    output = path(args.output_dir)
    if output.exists():
        raise FileExistsError("Use a fresh baseline output directory")
    command = [sys.executable, "-m", "sweettv.evaluate", "--prepared-dir", str(path(args.prepared_dir)),
               "--output-dir", str(output), "--predictions-dir", str(output / "per_user")]
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def prepare_v4(args):
    """Rebuild caches from external-data preparation, without model fitting."""
    import numpy as np
    import pandas as pd
    from sweettv.prepare import aggregate_window, seconds
    from sweettv.partial_features_v4 import union_partial_fragments, aggregate_unioned_airings, qualified_event_rows
    from sweettv.exposure_v4 import prepare_window_schedule, read_future_schedule

    prepared, destination = path(args.prepared_dir), path(args.cache_dir)
    if destination.exists():
        raise FileExistsError("Use a fresh V4 cache directory")
    design = json.loads((ROOT / "configs/v4.json").read_text())
    fragments = pd.read_parquet(prepared / "matched_fragments.parquet")
    mapping = pd.read_csv(prepared / "user_map.csv", dtype={"uid": "int64", "user_id": str}, keep_default_na=False)
    mapping = mapping.sort_values("uid").reset_index(drop=True)
    if not np.array_equal(mapping.uid.to_numpy(), np.arange(len(mapping))):
        raise ValueError("User-map uid must be dense and zero-based")
    names = mapping.user_id.to_numpy(dtype=object)
    timetable = pd.read_parquet(prepared / "train_schedule.parquet")
    future = read_future_schedule(path(args.raw_dir) / "export_arh_31-42-final.csv")
    unioned, union_audit = union_partial_fragments(fragments)
    windows = design["folds"] + design["diagnostics"] + [dict(id="final", **design["final"])]
    records = []
    for window in windows:
        phase = window["id"]
        start, cutoff, end = map(seconds, [design["train_start"], window["cutoff"], window["target_end"]])
        folder = destination / phase
        folder.mkdir(parents=True)
        partial = aggregate_unioned_airings(unioned, mapping, start, cutoff)
        strong = qualified_event_rows(partial)
        # Match original input order: rolling/stress sorted by observed endpoint;
        # original long/full histories came from uid/air_id groupby order.
        ordering = ["uid", "last_time", "air_id"] if phase in {"fold_a", "fold_b", "fold_c", "stress"} else ["uid", "air_id"]
        strong = strong.sort_values(ordering).reset_index(drop=True)
        train, _, original_positive = aggregate_window(fragments, start, cutoff, names, .8)
        assert int(train["count"].sum()) == len(strong) == len(original_positive)
        schedule, schedule_audit = prepare_window_schedule(timetable, start, end, future if phase == "final" else None)
        if phase == "final":
            candidates = pool(prepared / "candidates_test.csv")
        elif phase == "old_window":
            candidates = pool(prepared / "candidates_valid.csv")
        else:
            candidates = sorted(map(int, timetable.loc[(timetable.start < end) & (timetable.end > cutoff), "tv_show_id"].unique()))
        strong.to_parquet(folder / "train_events.parquet", index=False)
        partial.to_parquet(folder / "partial_airings.parquet", index=False)
        schedule.to_parquet(folder / "schedule.parquet", index=False)
        train.to_csv(folder / "train_counts.csv", index=False)
        pd.DataFrame({"tv_show_id": candidates}).to_csv(folder / "candidates.csv", index=False)
        if phase != "final":
            valid, _, _ = aggregate_window(fragments, cutoff, end, names, .8)
            assert set(valid.tv_show_id) <= set(candidates)
            valid.to_csv(folder / "valid_counts.csv", index=False)
        records.append({"phase": phase, "qualified_events": len(strong), "partial_rows": len(partial),
                        "candidates": len(candidates), "schedule": schedule_audit})
    dump(destination / "preparation.json", {"windows": records, "union": union_audit,
        "model_fitting_or_quality_scoring": False, "source_prepared": str(prepared), "seed": 42})


def make_state(args, spec):
    import pandas as pd
    from sweettv.prepare import seconds
    from sweettv.signals_v4 import build_features
    folder = path(args.cache_dir) / spec["id"]
    cohort = users(path(args.prepared_dir))
    candidates = pool(folder / "candidates.csv")
    state = build_features(pd.read_parquet(folder / "train_events.parquet"),
                           pd.read_parquet(folder / "partial_airings.parquet"),
                           pd.read_parquet(folder / "schedule.parquet"), cohort, candidates,
                           seconds(spec["cutoff"]), seconds(spec["target_end"]))
    return state, cohort, candidates, folder


def evaluate_v4(args):
    import numpy as np
    import pandas as pd
    from sweettv.evaluate import build_truth, score_predictions
    from sweettv.signals_v4 import predict_v4
    design = json.loads((ROOT / "configs/v4.json").read_text())
    destination = path(args.output_dir)
    if destination.exists():
        raise FileExistsError("Use a fresh evaluation output directory")
    destination.mkdir(parents=True)
    records = []
    for spec in design["folds"]:
        state, cohort, candidates, folder = make_state(args, spec)
        train = pd.read_csv(folder / "train_counts.csv", dtype={"user_id": str}, keep_default_na=False)
        valid = pd.read_csv(folder / "valid_counts.csv", dtype={"user_id": str}, keep_default_na=False)
        truth = build_truth(valid, cohort)
        for conf in design["configs"]:
            predictions, fallback, _ = predict_v4(state, conf)
            metrics, detail = score_predictions(truth, predictions, train, fallback, len(candidates))
            detail.to_csv(destination / f"{spec['id']}__{conf['id']}.csv", index=False)
            records.append({"phase": spec["id"], "config_id": conf["id"], "params": conf, "metrics": metrics})
    lookup = {(r["phase"], r["config_id"]): r["metrics"]["map_at_5"] for r in records}
    ranked = []
    for conf in design["configs"]:
        values = [lookup[f["id"], conf["id"]] for f in design["folds"]]
        d2 = [value - lookup[f["id"], "v2_control"] for value, f in zip(values, design["folds"])]
        d3 = [value - lookup[f["id"], "v3_control"] for value, f in zip(values, design["folds"])]
        eligible = conf["family"] not in {"v2", "v3"} and np.mean(d2) > 0 and np.mean(d3) > 0 and sum(x > 0 for x in d2) >= 2 and min(d2) >= -.002
        ranked.append({"config_id": conf["id"], "fold_maps": values, "mean_map": float(np.mean(values)), "eligible": bool(eligible)})
    ranked.sort(key=lambda row: (-row["mean_map"], row["config_id"]))
    dump(destination / "results.json", {"records": records, "ranked_configs": ranked,
        "same_predeclared_design_replay": True, "new_independent_test_claimed": False})


def fit_final(args):
    import pandas as pd
    from sweettv.evaluate import validate_submission
    from sweettv.signals_v4 import predict_v4
    design = json.loads((ROOT / "configs/v4.json").read_text())
    state, cohort, candidates, _ = make_state(args, dict(id="final", **design["final"]))
    predictions, _, diagnostics = predict_v4(state, design["selected_config"])
    frame = pd.DataFrame({"user_id": cohort, "tv_show_id": [" ".join(map(str, predictions[user])) for user in cohort]})
    validation = validate_submission(frame, cohort, candidates)
    filename = path(args.output)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("x") as stream:
        frame.to_csv(stream, index=False)
    dump(filename.with_suffix(".validation.json"), {**validation, "sha256": digest(filename),
         "expected_original_sha256": design["expected_submission_sha256"],
         "matches_original_bytes": digest(filename) == design["expected_submission_sha256"], "diagnostics": diagnostics})


def verify(args):
    import csv
    expected = json.loads((ROOT / "configs/v4.json").read_text())["expected_submission_sha256"]
    filename = path(args.csv)
    with filename.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["user_id", "tv_show_id"]:
            raise ValueError("Invalid submission columns")
        rows = list(reader)
    if len(rows) != 1260 or len({row["user_id"] for row in rows}) != 1260:
        raise ValueError("Expected1260 unique exact-string users")
    for row in rows:
        values = row["tv_show_id"].split(" ")
        if len(values) != len(set(values)) or len(values) != 5 or any(not token.isdecimal() or int(token) <= 0 for token in values):
            raise ValueError("Expected five unique positive programme IDs")
    actual = digest(filename)
    if actual != expected:
        raise AssertionError("CSV hash differs from the confirmed original")
    print(json.dumps({"valid": True, "rows": len(rows), "sha256": actual,
                      "candidate_pool_membership": "original independent check, not repeated without external data"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare", help="External raw CSV -> audited baseline caches")
    a.add_argument("--raw-dir", required=True); a.add_argument("--prepared-dir", default="work/prepared")
    a.add_argument("--output-dir", default="work/preparation_audit"); a.set_defaults(action=prepare)
    a = sub.add_parser("baselines", help="Original unchanged popularity/CBF/implicit ALS experiment")
    a.add_argument("--prepared-dir", default="work/prepared"); a.add_argument("--output-dir", default="outputs/baselines")
    a.set_defaults(action=baselines)
    a = sub.add_parser("prepare-v4", help="Prepare six windows without fitting/scoring")
    a.add_argument("--raw-dir", required=True); a.add_argument("--prepared-dir", default="work/prepared")
    a.add_argument("--cache-dir", default="work/v4"); a.set_defaults(action=prepare_v4)
    a = sub.add_parser("evaluate-v4", help="Replay all26 fixed configs in three development folds")
    a.add_argument("--prepared-dir", default="work/prepared"); a.add_argument("--cache-dir", default="work/v4")
    a.add_argument("--output-dir", default="outputs/v4_replay"); a.set_defaults(action=evaluate_v4)
    a = sub.add_parser("fit-final", help="Fit frozen selected config; never submits")
    a.add_argument("--prepared-dir", default="work/prepared"); a.add_argument("--cache-dir", default="work/v4")
    a.add_argument("--output", default="outputs/best_local_v4_submission.csv"); a.set_defaults(action=fit_final)
    a = sub.add_parser("verify", help="Check required CSV bytes/format with standard library only")
    a.add_argument("--csv", default="artifacts/best_local_v4_submission.csv"); a.set_defaults(action=verify)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
