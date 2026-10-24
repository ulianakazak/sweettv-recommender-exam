"""A separate frequency/recency iteration using historical qualified airings.

Each input row represents exactly one qualified user-airing interaction. This
module accepts no target behaviour, labels, content attributes or evaluation
scores. It never changes its input DataFrame or any baseline implementation.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400
K = 5


def _airing_weights(last_time: np.ndarray, cutoff: int, half_life_days: float | None) -> np.ndarray:
    """One unit per airing, optionally halved after each half-life interval."""
    if half_life_days is None:
        return np.ones(len(last_time), dtype=np.float64)
    age_seconds = cutoff - last_time.astype(np.int64)
    return np.exp2(-age_seconds.astype(np.float64) / (half_life_days * SECONDS_PER_DAY))


def _shrink_scores(personal_weights: np.ndarray, prior: np.ndarray, prior_strength: float) -> np.ndarray:
    """Normalised mixture; only used for users with positive available history.

    With n=sum(r), the equivalent mixture is n/(n+tau) * (r/n) +
    tau/(n+tau) * g. Keeping the denominator makes the score interpretation
    explicit even though it is constant when ranking items for one user.
    """
    denominator = float(personal_weights.sum()) + prior_strength
    if denominator <= 0:
        return np.zeros_like(personal_weights, dtype=np.float64)
    return (personal_weights + prior_strength * prior) / denominator


def _integer_values(values: list | pd.Series, name: str) -> np.ndarray:
    """Do not silently accept float-valued identifiers or timestamps."""
    series = pd.Series(values, copy=False)
    if series.isna().any():
        raise ValueError(f"{name} contains missing values")
    if not series.map(lambda value: isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))).all():
        raise TypeError(f"{name} must contain exact integers")
    return series.to_numpy(dtype=np.int64)


def _rank_and_fill(candidate_ids: np.ndarray, scores: np.ndarray | None, fallback: list[int]) -> tuple[list[int], int]:
    ranked: list[int] = []
    if scores is not None:
        supported = np.flatnonzero(np.isfinite(scores) & (scores > 0))
        order = np.lexsort((candidate_ids[supported], -scores[supported]))
        ranked = candidate_ids[supported[order[:K]]].tolist()
    seen = set(ranked)
    fallback_slots = 0
    for item in fallback:
        if len(ranked) == K:
            break
        if item not in seen:
            ranked.append(item)
            seen.add(item)
            fallback_slots += 1
    if len(ranked) != K:
        raise ValueError("Fewer than five candidates after ranking and fallback")
    return ranked, fallback_slots


def predict_frequency(
    events: pd.DataFrame,
    users: list[str],
    candidates: list[int],
    cutoff: int,
    half_life_days: float | None = None,
    prior_strength: float = 0,
    mode: str = "personal",
) -> tuple[dict[str, list[int]], dict[str, int], dict[str, Any]]:
    """Return ranked lists, fallback counts and auditable aggregate diagnostics.

    Required events columns: user_id (str), tv_show_id (exact integer), last_time
    (integer seconds). A row is one qualified airing. If count is provided every
    value must be one; pre-aggregated user-programme counts are not accepted.
    Optional air_id is checked for duplicate (user_id, air_id) pairs. Without
    air_id, the caller must have validated unique qualified airings upstream.

    All events must satisfy last_time <= cutoff. Items unavailable in candidates
    contribute neither personal history mass n_u nor normalised prior mass g_i.
    The prior counts distinct historical users, including users outside the
    requested prediction cohort. User-history repeats are allowed.

    mode='personal': r_ui is the sum of per-airing weights; rank by
    (r_ui + tau*g_i)/(n_u + tau). mode='popularity' ranks only the distinct-user
    prior and requires half_life_days=None and prior_strength=0. A user without
    available positive history receives the same train-popularity/ID fallback
    even when tau>0. Positive scores tie by ascending integer programme ID.
    """
    started = time.perf_counter()
    required = {"user_id", "tv_show_id", "last_time"}
    if not required.issubset(events.columns):
        raise ValueError(f"Missing event columns: {sorted(required - set(events.columns))}")
    if mode not in {"personal", "popularity"}:
        raise ValueError("mode must be 'personal' or 'popularity'")
    if not isinstance(cutoff, (int, np.integer)) or isinstance(cutoff, (bool, np.bool_)):
        raise TypeError("cutoff must be integer seconds")
    cutoff = int(cutoff)
    if not all(isinstance(user, str) and user for user in users) or len(users) != len(set(users)):
        raise ValueError("Requested user IDs must be unique nonempty strings")
    if events["user_id"].isna().any() or not events["user_id"].map(lambda user: isinstance(user, str) and bool(user)).all():
        raise TypeError("Event user IDs must be nonempty strings, never float")
    if half_life_days is not None:
        half_life_days = float(half_life_days)
        if not math.isfinite(half_life_days) or half_life_days <= 0:
            raise ValueError("half_life_days must be finite and positive")
    prior_strength = float(prior_strength)
    if not math.isfinite(prior_strength) or prior_strength < 0:
        raise ValueError("prior_strength must be finite and nonnegative")
    if mode == "popularity" and (half_life_days is not None or prior_strength != 0):
        raise ValueError("Popularity uses no personal half-life or shrinkage strength")
    candidate_ids = _integer_values(candidates, "candidates")
    if len(candidate_ids) < K or len(np.unique(candidate_ids)) != len(candidate_ids) or np.any(candidate_ids <= 0):
        raise ValueError("Candidates must contain at least five unique positive programme IDs")
    candidate_ids = np.sort(candidate_ids)
    item_ids = _integer_values(events["tv_show_id"], "events.tv_show_id")
    last_times = _integer_values(events["last_time"], "events.last_time")
    if np.any(item_ids <= 0):
        raise ValueError("Events must contain qualified, labelled nonzero programme IDs")
    if np.any(last_times > cutoff):
        raise ValueError("Future events rejected: every last_time must be <= cutoff")
    if "count" in events and not events["count"].eq(1).all():
        raise ValueError("Each event must be one qualified airing; aggregated count rows are not accepted")
    if "air_id" in events:
        if events["air_id"].isna().any():
            raise ValueError("air_id contains missing values")
        if events.duplicated(["user_id", "air_id"]).any():
            raise ValueError("Duplicate user-airing events rejected")

    available_mask = np.isin(item_ids, candidate_ids)
    available = events.loc[available_mask, ["user_id", "tv_show_id"]].copy()
    available["weight"] = _airing_weights(last_times[available_mask], cutoff, half_life_days)
    distinct_users = available.groupby("tv_show_id", sort=False)["user_id"].nunique().to_dict()
    prior_counts = np.array([distinct_users.get(int(item), 0) for item in candidate_ids], dtype=np.float64)
    total_prior_count = float(prior_counts.sum())
    prior = prior_counts / total_prior_count if total_prior_count else np.zeros(len(candidate_ids), dtype=np.float64)
    fallback_order = np.lexsort((candidate_ids, -prior_counts))
    fallback = candidate_ids[fallback_order].tolist()

    aggregated = available.groupby(["user_id", "tv_show_id"], sort=False)["weight"].sum()
    requested_users = set(users)
    personal: dict[str, dict[int, float]] = {}
    for (user, item), weight in aggregated.items():
        if user in requested_users and weight > 0:
            personal.setdefault(user, {})[int(item)] = float(weight)
    candidate_index = {int(item): index for index, item in enumerate(candidate_ids)}
    predictions: dict[str, list[int]] = {}
    fallback_slots: dict[str, int] = {}
    available_mass: list[float] = []
    for user in users:
        weighted_history = personal.get(user, {})
        mass = float(sum(weighted_history.values()))
        available_mass.append(mass)
        if mode == "popularity":
            predictions[user], fallback_slots[user] = _rank_and_fill(candidate_ids, prior, fallback)
        elif mass <= 0:
            predictions[user], fallback_slots[user] = _rank_and_fill(candidate_ids, None, fallback)
        else:
            weights = np.zeros(len(candidate_ids), dtype=np.float64)
            for item, weight in weighted_history.items():
                weights[candidate_index[item]] = weight
            scores = _shrink_scores(weights, prior, prior_strength)
            predictions[user], fallback_slots[user] = _rank_and_fill(candidate_ids, scores, fallback)

    diagnostics = {
        "implementation": "frequency_v1",
        "mode": mode,
        "cutoff": cutoff,
        "half_life_days": half_life_days,
        "prior_strength": prior_strength,
        "train_event_rows": len(events),
        "train_users": int(events["user_id"].nunique()),
        "train_programmes": int(events["tv_show_id"].nunique()),
        "available_event_rows": int(available_mask.sum()),
        "available_programmes_with_train_events": len(distinct_users),
        "candidate_count": len(candidate_ids),
        "requested_users": len(users),
        "users_with_available_positive_history": sum(mass > 0 for mass in available_mass),
        "users_without_available_positive_history": sum(mass <= 0 for mass in available_mass),
        "prior_distinct_user_count_sum": int(total_prior_count),
        "normalised_prior_sum": float(prior.sum()),
        "available_personal_weight_sum": float(sum(available_mass)),
        "max_train_last_time": int(last_times.max()) if len(last_times) else None,
        "future_events_used": False,
        "duplicate_airings_check": "user_id_air_id_checked" if "air_id" in events else "upstream_unique_qualified_airings_required",
        "per_airing_weight": "1" if half_life_days is None else "2**(-(cutoff-last_time)/(half_life_days*86400))",
        "personal_score": "(r_ui + prior_strength*g_i)/(sum_available_r_ui + prior_strength)",
        "prior": "distinct train users per candidate item / sum over all candidate items",
        "fallback": "train distinct-user popularity DESC, programme ID ASC; no-history users always use fallback",
        "tie_break": "score DESC, programme ID ASC",
        "history_repeats_allowed": True,
        "fallback_slots_total": int(sum(fallback_slots.values())),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return predictions, fallback_slots, diagnostics
