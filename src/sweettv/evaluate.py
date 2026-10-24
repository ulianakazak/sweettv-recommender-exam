"""Tune on an earlier window, compare on one later hold-out, then refit.

Run with: PYTHONPATH=src .venv/bin/python -m sweettv.evaluate
No network, Kaggle submission or repository publication is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

# Set limits before numpy/scipy/implicit are imported. Explicit CLI settings are
# recorded below and threadpool_limits is additionally used during fitting.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from .models import fit_model, predict

METHODS = ("popularity", "cbf", "implicit_mf")
SEED = 42
K = 5
# A deliberately small grid selected before evaluating the main hold-out.
GRID = {
    "popularity": [{}],
    "cbf": [{"profile_weighting": "binary"}, {"profile_weighting": "log_count"}],
    "implicit_mf": [
        {"factors": 32, "regularization": 0.1, "alpha": 10.0, "iterations": 20},
        {"factors": 64, "regularization": 0.1, "alpha": 10.0, "iterations": 20},
        {"factors": 32, "regularization": 1.0, "alpha": 40.0, "iterations": 20},
    ],
}


def read_counts(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"user_id": str, "tv_show_id": "int64", "count": "int64"}, keep_default_na=False)
    if data[["user_id", "tv_show_id"]].duplicated().any():
        raise ValueError(f"Duplicate user-item pairs in {path}")
    return data


def read_catalog(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"tv_show_id": "int64"}, keep_default_na=False)


def candidates_from(path: Path) -> list[int]:
    result = sorted(pd.read_csv(path, dtype={"tv_show_id": "int64"})["tv_show_id"].tolist())
    if len(result) < K or len(result) != len(set(result)) or 0 in result:
        raise ValueError(f"Invalid candidate pool {path}")
    return result


def build_truth(counts: pd.DataFrame, users: list[str]) -> dict[str, list[int]]:
    """Local target: five most frequent qualified distinct airings; ID tie break."""
    order = counts.sort_values(["user_id", "count", "tv_show_id"], ascending=[True, False, True])
    grouped = order.groupby("user_id", sort=False)["tv_show_id"].apply(lambda x: [int(v) for v in x.head(K)]).to_dict()
    return {str(user): grouped.get(str(user), []) for user in users}


def ap_at_k(actual: list[int], predicted: list[int], k: int = K) -> float:
    truth = set(map(int, actual))
    if not truth:
        return 0.0
    hits, total, seen = 0, 0.0, set()
    for rank, item in enumerate(predicted[:k], 1):
        item = int(item)
        if item in truth and item not in seen:
            hits += 1
            total += hits / rank
        seen.add(item)
    return total / min(k, len(truth))


def score_predictions(truth: dict[str, list[int]], predictions: dict[str, list[int]],
                      train: pd.DataFrame, fallback_slots: dict[str, int], candidate_count: int) -> tuple[dict[str, Any], pd.DataFrame]:
    train_sizes = train.groupby("user_id")["count"].sum().to_dict()
    positive_sizes = [train_sizes.get(user, 0) for user in truth if train_sizes.get(user, 0) > 0]
    median = float(np.median(positive_sizes)) if positive_sizes else 0.0
    rows = []
    for user, actual in truth.items():
        history_size = int(train_sizes.get(user, 0))
        group = "cold" if history_size == 0 else "light" if history_size <= median else "heavy"
        rows.append({"user_id": user, "ap_at_5": ap_at_k(actual, predictions[user]),
                     "truth_size": len(actual), "train_airings": history_size, "history_group": group,
                     "fallback_slots": fallback_slots[user],
                     "predicted_ids": " ".join(map(str, predictions[user])),
                     "truth_ids": " ".join(map(str, actual))})
    frame = pd.DataFrame(rows)
    nonempty = frame[frame["truth_size"] > 0]
    unique_recommended = {item for recs in predictions.values() for item in recs}
    summary = {"map_at_5": float(frame["ap_at_5"].mean()),
               "map_at_5_nonempty_truth": float(nonempty["ap_at_5"].mean()) if len(nonempty) else None,
               "users": len(frame), "users_nonempty_truth": len(nonempty),
               "users_empty_truth": int((frame["truth_size"] == 0).sum()),
               "unique_recommended_items": len(unique_recommended),
               "catalogue_coverage_at_5": len(unique_recommended) / candidate_count,
               "fallback_slot_fraction": float(frame["fallback_slots"].sum() / (len(frame) * K)),
               "positive_history_median_airings": median,
               "subgroups": {}}
    for group in ("cold", "light", "heavy"):
        part = frame[frame["history_group"] == group]
        summary["subgroups"][group] = {"users": len(part), "map_at_5": float(part["ap_at_5"].mean()) if len(part) else None,
                                      "nonempty_truth_users": int((part["truth_size"] > 0).sum())}
    return summary, frame


def run_one(method: str, params: dict, train: pd.DataFrame, catalog: pd.DataFrame,
            users: list[str], candidates: list[int], truth: dict[str, list[int]] | None,
            phase: str) -> tuple[dict, dict[str, list[int]], dict[str, int], pd.DataFrame | None]:
    started = time.perf_counter()
    model = fit_model(method, train, catalog, params, SEED)
    trained = time.perf_counter()
    predictions, fallback = predict(model, train, users, candidates)
    finished = time.perf_counter()
    record = {"phase": phase, "method": method, "params": params, "seed": SEED,
              "fit_seconds": trained - started, "predict_seconds": finished - trained,
              "rss_bytes_after_prediction": psutil.Process().memory_info().rss,
              "candidates": len(candidates), "prediction_users": len(users),
              "diagnostics": model.diagnostics}
    details = None
    if truth is not None:
        metrics, details = score_predictions(truth, predictions, train, fallback, len(candidates))
        record["metrics"] = metrics
    return record, predictions, fallback, details


def validate_submission(frame: pd.DataFrame, users: list[str], candidates: list[int]) -> dict:
    if list(frame.columns) != ["user_id", "tv_show_id"]:
        raise ValueError("Wrong submission columns/order")
    if frame["user_id"].tolist() != users or frame["user_id"].duplicated().any():
        raise ValueError("Submission user order/uniqueness differs from template")
    pool = set(candidates)
    for user, value in frame.itertuples(index=False, name=None):
        text = str(value)
        tokens = text.split(" ")
        if len(tokens) != K or any(not token.isdecimal() for token in tokens):
            raise ValueError(f"Wrong item field format for user {user}")
        ids = list(map(int, tokens))
        if len(set(ids)) != K or any(item == 0 or item not in pool for item in ids):
            raise ValueError(f"Invalid ranked item set for user {user}")
    return {"valid": True, "users": len(users), "columns": frame.columns.tolist(),
            "template_order_preserved": True, "exact_string_user_ids": True,
            "unique_nonzero_5_ids_per_user": True, "all_ids_in_target_candidates": True,
            "rank_order_preserved_from_model": True}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def example_users(train: pd.DataFrame, users: list[str]) -> list[tuple[str, str]]:
    """Fix selection using training activity, before looking at ranked output."""
    activity = train.groupby("user_id")["count"].sum().to_dict()
    positive = [(str(user), int(activity[user])) for user in users if activity.get(user, 0) > 0]
    if len(positive) < 2:
        raise ValueError("Need two positive-history template users for examples")
    heavy = sorted(positive, key=lambda pair: (-pair[1], pair[0]))[0][0]
    light = sorted((pair for pair in positive if pair[0] != heavy), key=lambda pair: (pair[1], pair[0]))[0][0]
    return [(heavy, "highest_positive_train_airing_count"), (light, "lowest_positive_train_airing_count")]


def make_examples(output_dir: Path, selected_users: list[tuple[str, str]], train: pd.DataFrame,
                  truth: dict[str, list[int]], predictions: dict[str, dict[str, list[int]]],
                  display_catalog: pd.DataFrame, chosen: str) -> None:
    catalogue = display_catalog.drop_duplicates("tv_show_id").set_index("tv_show_id").to_dict("index")

    def metadata(item: int) -> dict:
        row = catalogue.get(int(item), {})
        return {"tv_show_id": int(item), "title": str(row.get("tv_show_title", "[not in training catalogue]")),
                "category": str(row.get("tv_show_category", "[not in training catalogue]"))}

    all_rows, json_users = [], []
    for user, rule in selected_users:
        history = train[train["user_id"] == user].sort_values(["count", "tv_show_id"], ascending=[False, True])
        watched = set(history["tv_show_id"].astype(int))
        details = {"user_id": user, "selection_rule": rule,
                   "train_airings": int(history["count"].sum()), "train_unique_programmes": len(history),
                   "train_history": [{**metadata(int(row.tv_show_id)), "qualified_airing_count": int(row.count)}
                                     for row in history.itertuples(index=False)],
                   "validation_truth_top5": [metadata(item) for item in truth[user]], "methods": {}}
        for method in METHODS:
            recs = predictions[method][user]
            details["methods"][method] = {"ap_at_5": ap_at_k(truth[user], recs), "top5": []}
            for rank, item in enumerate(recs, 1):
                record = {"user_id": user, "selection_rule": rule, "method": method, "rank": rank,
                          **metadata(item), "seen_in_train": item in watched, "in_validation_truth": item in truth[user]}
                all_rows.append(record)
                details["methods"][method]["top5"].append(record)
        json_users.append(details)
    pd.DataFrame(all_rows).to_csv(output_dir / "recommendation_examples.csv", index=False)
    write_json(output_dir / "recommendation_examples.json", {"scope": "main temporal hold-out", "users_selected_before_scoring": True,
               "metadata_is_for_display_only": True, "users": json_users})

    # Supply a concrete worked example for AI QA and a separate blank human task.
    # Prefer one of the two activity-selected examples with nonempty truth;
    # never choose this verification case for an attractive metric value.
    nonempty_users = [user for user, actual in truth.items() if actual]
    selected_nonempty = [entry["user_id"] for entry in json_users if truth[entry["user_id"]]]
    if selected_nonempty:
        user = selected_nonempty[0]
    elif nonempty_users:
        user = sorted(nonempty_users)[0]
    else:
        user = json_users[0]["user_id"]
    recs = predictions[chosen][user]
    actual = set(truth[user])
    steps, hits = [], 0
    for rank, item in enumerate(recs, 1):
        relevant = item in actual
        hits += int(relevant)
        steps.append({"rank": rank, "tv_show_id": item, "relevant": relevant,
                      "cumulative_hits": hits, "precision_at_rank": hits / rank,
                      "ap_numerator_contribution": hits / rank if relevant else 0.0})
    write_json(output_dir / "manual_ap5_example.json", {
        "user_id": user, "method": chosen, "truth_ids": sorted(actual), "predicted_ids": recs,
        "steps": steps, "denominator": min(K, len(actual)), "ap_at_5_ai_computed": ap_at_k(list(actual), recs),
        "ai_check_only": True, "student_personal_verification": "not_confirmed",
        "example_selection": "first nonempty truth among the two train-activity-selected users; otherwise first nonempty truth user by ID; no selection by AP",
        "human_task": "Independently check relevance at each rank, sum precision at relevant ranks, divide by min(5, truth size) when truth is nonempty; empty truth is defined as AP=0. Confirm or correct this value.",
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("work/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("work/predictions"))
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    p = args.prepared_dir
    template = pd.read_csv(p / "template.csv", dtype=str, keep_default_na=False)
    if list(template.columns) != ["user_id", "tv_show_id"] or template["user_id"].duplicated().any():
        raise ValueError("Invalid template")
    users = template["user_id"].tolist()
    train = read_counts(p / "train_counts.csv")
    valid = read_counts(p / "valid_counts.csv")
    tune_train = read_counts(p / "tune_train_counts.csv")
    tune_valid = read_counts(p / "tune_valid_counts.csv")
    catalog_train = read_catalog(p / "catalog_train.csv")
    catalog_tune = read_catalog(p / "catalog_tune.csv")
    candidates_tune = candidates_from(p / "candidates_tune.csv")
    candidates_valid = candidates_from(p / "candidates_valid.csv")
    tune_truth = build_truth(tune_valid, users)
    valid_truth = build_truth(valid, users)
    for label, truth, candidates in (("tune", tune_truth, candidates_tune), ("main", valid_truth, candidates_valid)):
        inaccessible = {item for actual in truth.values() for item in actual} - set(candidates)
        if inaccessible:
            raise ValueError(f"{label} truth includes {len(inaccessible)} IDs outside the common candidate pool")
    selected_users = example_users(train, users)
    records, selected_params, main_predictions, main_records = [], {}, {}, []
    sources = {str(path): file_sha256(path) for path in sorted(p.glob("*.csv"))}
    immutable_plan = {
        "seed": SEED, "k": K, "threads": 1, "grid": GRID,
        "tuning_window": "2020-03-09 <= t < 2020-04-27 -> 2020-04-27 <= t < 2020-05-18",
        "main_window": "2020-03-09 <= t < 2020-05-18 -> 2020-05-18 <= t < 2020-07-27",
        "main_method_selection": "highest main hold-out MAP@5, stable method order for ties",
        "hyperparameter_selection": "earlier tuning MAP@5 only, stable grid order for ties",
        "local_truth": "top 5 by qualified airing count DESC, ID ASC ties; empty truth AP=0",
        "same_example_users_chosen_before_scoring": selected_users,
        "input_sha256": sources,
    }
    plan_path = args.output_dir / "evaluation_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != json.loads(json.dumps(immutable_plan)):
            raise ValueError("Existing evaluation plan differs; use a new output directory for an explicitly revised run")
    else:
        write_json(plan_path, immutable_plan)
    print(f"template_users={len(users)} main_train_pairs={len(train)} main_candidates={len(candidates_valid)}", flush=True)

    with threadpool_limits(limits=1):
        for method in METHODS:
            trials = []
            for trial_id, params in enumerate(GRID[method]):
                record, _, _, details = run_one(method, params, tune_train, catalog_tune, users, candidates_tune, tune_truth, "early_tuning")
                record["trial_id"] = trial_id
                records.append(record)
                trials.append(record)
                details.to_csv(args.predictions_dir / f"tune_{method}_{trial_id}.csv", index=False)
                print(f"tune {method} {params}: MAP@5={record['metrics']['map_at_5']:.8f}", flush=True)
            # Stable grid order resolves exact ties without consulting main truth.
            best = max(trials, key=lambda record: record["metrics"]["map_at_5"])
            selected_params[method] = best["params"]

        for method in METHODS:
            record, predictions, _, details = run_one(method, selected_params[method], train, catalog_train, users,
                                                     candidates_valid, valid_truth, "main_holdout")
            records.append(record)
            main_records.append(record)
            main_predictions[method] = predictions
            details.to_csv(args.predictions_dir / f"holdout_{method}.csv", index=False)
            print(f"main {method}: MAP@5={record['metrics']['map_at_5']:.8f} fit={record['fit_seconds']:.3f}s predict={record['predict_seconds']:.3f}s", flush=True)

        # The main hold-out selects a final strategy; this selected score is not
        # an unbiased estimate of a strategy selected independently of it.
        chosen_record = max(main_records, key=lambda record: record["metrics"]["map_at_5"])
        chosen = chosen_record["method"]
        full = read_counts(p / "full_counts.csv")
        catalog_full = read_catalog(p / "catalog_full.csv")
        candidates_test = candidates_from(p / "candidates_test.csv")
        final_record, final_predictions, final_fallback, _ = run_one(chosen, selected_params[chosen], full, catalog_full,
                                                                   users, candidates_test, None, "final_refit")
        final_record["fallback_slot_fraction"] = sum(final_fallback.values()) / (len(users) * K)
        records.append(final_record)
        final = pd.DataFrame({"user_id": users, "tv_show_id": [" ".join(map(str, final_predictions[user])) for user in users]})
        validation = validate_submission(final, users, candidates_test)
        final_path = args.output_dir / "final_submission.csv"
        final.to_csv(final_path, index=False)
        # Reopen the actual saved bytes with exact string IDs.
        validation.update(validate_submission(pd.read_csv(final_path, dtype=str, keep_default_na=False), users, candidates_test))
        validation["sha256"] = file_sha256(final_path)
        validation["selected_method"] = chosen
        write_json(args.output_dir / "submission_validation.json", validation)

    comparison = []
    for record in main_records:
        metrics = record["metrics"]
        comparison.append({"method": record["method"], "map_at_5": metrics["map_at_5"],
                           "map_at_5_nonempty_truth": metrics["map_at_5_nonempty_truth"],
                           "fit_seconds": record["fit_seconds"], "predict_seconds": record["predict_seconds"],
                           "users": metrics["users"], "users_nonempty_truth": metrics["users_nonempty_truth"],
                           "catalogue_coverage_at_5": metrics["catalogue_coverage_at_5"],
                           "fallback_slot_fraction": metrics["fallback_slot_fraction"],
                           "selected_for_final": record["method"] == chosen,
                           "params_json": json.dumps(record["params"], sort_keys=True)})
    pd.DataFrame(comparison).to_csv(args.output_dir / "model_comparison.csv", index=False)
    display_catalog = catalog_train
    # Optional later-period metadata is accepted exclusively for annotation,
    # after all ranks and the selected final strategy have already been fixed.
    if (p / "catalog_display_valid.csv").is_file():
        display_catalog = pd.concat([catalog_train, read_catalog(p / "catalog_display_valid.csv")], ignore_index=True)
    make_examples(args.output_dir, selected_users, train, valid_truth, main_predictions, display_catalog, chosen)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl", "psutil")}
    write_json(args.output_dir / "experiments.json", {
        "run_id": started_at.replace(":", "").replace("+", "_"), "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.perf_counter() - started,
        "seed": SEED, "threads": 1, "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
        "prepared_input_sha256": sources, "selected_hyperparameters": selected_params,
        "protocol": {"tuning": "11-17 -> 18-20", "main_holdout": "11-20 -> 21-30", "final": "11-30 -> 31-42",
                     "truth": "Top5 programmes by count of distinct airings with unioned coverage >=0.8; count DESC, programme ID ASC ties",
                     "metric": "AP@5=sum(precision at each unique relevant rank)/min(5, unique truth size); empty truth=0; MAP averages all template users",
                     "user_id_type": "str", "history_repeats_allowed": True,
                     "candidate_pool": "unique labelled nonzero IDs airing in target window", "future_content_features": False,
                     "fallback": "train distinct-user popularity then ascending remaining candidate ID",
                     "selection_caveat": "Hyperparameters use earlier tuning only; the final method is selected on main hold-out, so its winning score has selection optimism and is not an independent final test.",
                     "target_reproduction_caveat": "Kaggle description does not fully specify fragments/frequency/ties/empty truth; local truth is an explicit approximation, not verified Kaggle labels."},
        "records": records, "selected_final_method": chosen,
        "student_personal_verification": "not_confirmed", "kaggle_submission_performed": False,
        "github_publication_performed": False, "kaggle_private_score": None,
        "failure_cases": [{
            "phase": "dependency_setup", "reported_by": "root agent during this work session",
            "failure": "implicit 0.7.2 source build failed on arm64 at clang++ linking: symbol(s) not found for architecture arm64",
            "root_cause": "The compiled extension could not link against the bundled CPython environment; no algorithmic score had been computed.",
            "fix": "Replaced the optional compiled package with portable NumPy/SciPy Hu-Koren-Volinsky implicit ALS normal equations.",
            "verification": "Independent dense-objective unit test and nonincreasing objective checks on every real ALS iteration; actual tests/logs must accompany the claim.",
        }],
    })
    print(f"selected_final_method={chosen} submission={final_path} users={len(users)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
