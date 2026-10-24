"""Reusable historical state for the isolated mass-preserving power iteration.

No labels, validation scores or future programme metadata enter this API.
Power changes the shape of a user's available history while retaining its total
mass, so the prior strength keeps the same pseudo-airing interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from .frequency_v1 import _airing_weights, _integer_values, _rank_and_fill


@dataclass(frozen=True)
class PowerFrequencyState:
    """Train-only sufficient statistics; numeric arrays are read-only."""

    users: tuple[str, ...]
    candidate_ids: np.ndarray
    personal_matrix: np.ndarray
    personal_mass: np.ndarray
    prior: np.ndarray
    prior_counts: np.ndarray
    fallback: tuple[int, ...]
    diagnostics: dict[str, Any]


def _mass_preserving_power(personal: np.ndarray, mass: np.ndarray, power: float) -> np.ndarray:
    """q_ui=n_u*r_ui**power/sum_j r_uj**power, with an exact p=1 branch."""
    if power == 1.0:
        return personal
    shaped = np.power(personal, power)
    shaped_mass = shaped.sum(axis=1)
    scale = np.divide(mass, shaped_mass, out=np.zeros_like(mass), where=shaped_mass > 0)
    shaped *= scale[:, None]
    return shaped


def build_state(
    events: pd.DataFrame,
    users: list[str],
    candidates: list[int],
    cutoff: int,
    half_life_days: float | None = None,
) -> PowerFrequencyState:
    """Build one reusable 2-D state for one fold and one decay half-life.

    One events row must be one qualified user-airing interaction. Required
    columns are user_id:str, tv_show_id:int and last_time:int seconds. Optional
    air_id is checked for duplicate (user_id, air_id); optional count must be 1.
    Every last_time is checked against cutoff before availability filtering.
    The global prior uses all historical users, including users outside `users`.
    Only available candidate items enter personal history mass or prior mass.
    """
    started = time.perf_counter()
    required = {"user_id", "tv_show_id", "last_time"}
    if not required.issubset(events.columns):
        raise ValueError(f"Missing event columns: {sorted(required - set(events.columns))}")
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
    candidate_ids = _integer_values(candidates, "candidates")
    if len(candidate_ids) < 5 or len(np.unique(candidate_ids)) != len(candidate_ids) or np.any(candidate_ids <= 0):
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
    prior_total = float(prior_counts.sum())
    prior = prior_counts / prior_total if prior_total else np.zeros(len(candidate_ids), dtype=np.float64)
    order = np.lexsort((candidate_ids, -prior_counts))
    fallback = tuple(map(int, candidate_ids[order]))

    user_index = {user: index for index, user in enumerate(users)}
    candidate_index = {int(item): index for index, item in enumerate(candidate_ids)}
    personal = np.zeros((len(users), len(candidate_ids)), dtype=np.float64)
    mass = np.zeros(len(users), dtype=np.float64)
    aggregated = available.groupby(["user_id", "tv_show_id"], sort=False)["weight"].sum()
    for (user, item), weight in aggregated.items():
        if user in user_index and weight > 0:
            row, column = user_index[user], candidate_index[int(item)]
            personal[row, column] = float(weight)
            # This follows the same first-appearance aggregation order as V1,
            # including for noninteger decay weights, instead of re-summing a
            # candidate-sorted matrix row to obtain a slightly different n_u.
            mass[row] += float(weight)
    if not np.isfinite(personal).all() or not np.isfinite(mass).all() or not np.isfinite(prior).all():
        raise ArithmeticError("Nonfinite historical state")
    diagnostics = {
        "implementation": "power_frequency_v2",
        "cutoff": cutoff,
        "half_life_days": half_life_days,
        "train_event_rows": len(events),
        "train_users": int(events["user_id"].nunique()),
        "train_programmes": int(events["tv_show_id"].nunique()),
        "available_event_rows": int(available_mask.sum()),
        "available_programmes_with_train_events": len(distinct_users),
        "requested_users": len(users),
        "candidate_count": len(candidate_ids),
        "personal_matrix_shape": list(personal.shape),
        "personal_matrix_dtype": "float64",
        "personal_matrix_bytes": int(personal.nbytes),
        "users_without_available_positive_history": int((mass <= 0).sum()),
        "users_with_available_positive_history": int((mass > 0).sum()),
        "available_personal_weight_sum": float(mass.sum()),
        "prior_distinct_user_count_sum": int(prior_total),
        "normalised_prior_sum": float(prior.sum()),
        "max_train_last_time": int(last_times.max()) if len(last_times) else None,
        "future_events_used": False,
        "duplicate_airings_check": "user_id_air_id_checked" if "air_id" in events else "upstream_unique_qualified_airings_required",
        "per_airing_weight": "1" if half_life_days is None else "2**(-(cutoff-last_time)/(half_life_days*86400))",
        "prior": "distinct train users per available candidate / sum of those distinct-user counts",
        "build_state_seconds": time.perf_counter() - started,
    }
    for array in (candidate_ids, personal, mass, prior, prior_counts):
        array.setflags(write=False)
    return PowerFrequencyState(tuple(users), candidate_ids, personal, mass, prior, prior_counts, fallback, diagnostics)


def predict_power(
    state: PowerFrequencyState,
    prior_strength: float,
    power: float,
) -> tuple[dict[str, list[int]], dict[str, int], dict[str, Any]]:
    """Score a fixed state; no state arrays or previous results are modified."""
    started = time.perf_counter()
    prior_strength, power = float(prior_strength), float(power)
    if not math.isfinite(prior_strength) or prior_strength < 0:
        raise ValueError("prior_strength must be finite and nonnegative")
    if not math.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    shaped = _mass_preserving_power(state.personal_matrix, state.personal_mass, power)
    if not np.isfinite(shaped).all():
        raise ArithmeticError("Nonfinite power-transformed history")
    positive = state.personal_mass > 0
    denominator = state.personal_mass + prior_strength
    scores = np.divide(shaped + prior_strength * state.prior[None, :], denominator[:, None],
                       out=np.zeros_like(shaped), where=denominator[:, None] > 0)
    if not np.isfinite(scores).all():
        raise ArithmeticError("Nonfinite ranking scores")
    predictions, fallback_slots = {}, {}
    fallback = list(state.fallback)
    for row, user in enumerate(state.users):
        predictions[user], fallback_slots[user] = _rank_and_fill(
            state.candidate_ids, scores[row] if positive[row] else None, fallback,
        )
    # Float addition/normalisation can incur roundoff; disclose the actual error
    # rather than asserting bit-exact mass sums for the non-identity powers.
    mass_error = np.abs(shaped.sum(axis=1) - state.personal_mass)
    max_absolute_error = float(mass_error.max()) if len(mass_error) else 0.0
    if len(mass_error) and np.any(mass_error > 1e-10 * np.maximum(1.0, state.personal_mass)):
        raise ArithmeticError("Power transformation did not preserve personal history mass")
    diagnostics = {
        **state.diagnostics,
        "prior_strength": prior_strength,
        "power": power,
        "mass_transform": "q_ui=n_u*r_ui**power/sum_available_j(r_uj**power); exact q=r branch for power=1",
        "score": "(q_ui + prior_strength*g_i)/(n_u + prior_strength)",
        "power_one_exact_identity_branch": power == 1.0,
        "maximum_mass_absolute_roundoff": max_absolute_error,
        "scores_finite": True,
        "history_repeats_allowed": True,
        "fallback": "train distinct-user popularity DESC then candidate ID ASC; every zero-history user uses this exact fallback",
        "tie_break": "score DESC, numeric programme ID ASC",
        "fallback_slots_total": int(sum(fallback_slots.values())),
        "predict_seconds": time.perf_counter() - started,
    }
    return predictions, fallback_slots, diagnostics
