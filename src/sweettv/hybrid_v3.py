"""Two bounded, train-only V3 mechanisms on the unchanged V2 control.

Daily caps change either profile shape alone or shape and confidence mass.
User neighbours use binary all-history cosine and candidate-only preference
profiles. This module accepts no truth, future behaviour or evaluation scores.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .frequency_v1 import _airing_weights, _rank_and_fill
from .power_frequency_v2 import (
    PowerFrequencyState, _mass_preserving_power, build_state, predict_power,
)

HALF_LIFE_DAYS = 28.0
PRIOR_STRENGTH = 30.0
POWER = .5
NEIGHBOUR_COUNT = 50
OVERLAP_SHRINKAGE = 10.0


@dataclass(frozen=True)
class HybridFeatures:
    """Reusable sufficient statistics; arrays are read-only, IDs stay strings."""

    base: PowerFrequencyState
    daily_matrix: np.ndarray
    cf_raw: np.ndarray
    cf_idf: np.ndarray
    cf_raw_supported: np.ndarray
    cf_idf_supported: np.ndarray
    train_users: tuple[str, ...]
    raw_neighbor_indices: np.ndarray
    idf_neighbor_indices: np.ndarray
    raw_neighbor_weights: np.ndarray
    idf_neighbor_weights: np.ndarray
    diagnostics: dict[str, Any]


def _numeric_user_key(user: str) -> tuple[int, str]:
    """Parse with Python integers, never float or a fixed-width integer type."""
    try:
        value = int(user)
    except (ValueError, TypeError) as exc:
        raise ValueError("V3 user IDs must be exact integer strings") from exc
    if not isinstance(user, str) or user.strip() != user:
        raise ValueError("V3 user IDs must be exact integer strings")
    # The string is a deterministic secondary key if distinct strings encode
    # the same integer; identity itself is never normalised or merged.
    return value, user


def _scores(profile: np.ndarray, mass: np.ndarray, prior: np.ndarray) -> np.ndarray:
    shaped = _mass_preserving_power(profile, mass, POWER)
    result = (shaped + PRIOR_STRENGTH * prior[None, :]) / (mass[:, None] + PRIOR_STRENGTH)
    if not np.isfinite(result).all():
        raise ArithmeticError("Nonfinite V3 scores")
    return result


def _neighbours(
    binary: sparse.csr_matrix,
    query_selector: sparse.csr_matrix,
    profiles: sparse.csr_matrix,
    profile_supported: np.ndarray,
    own_indices: np.ndarray,
    use_idf: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cached h, support, top-50 indices and shrunk cosine weights.

    Training rows are already in numeric user-ID order. Consequently row index
    is the exact numeric-ID tie key, including for IDs above 2**64.
    """
    train_count, item_count = binary.shape
    query_count = query_selector.shape[0]
    query_binary = (query_selector @ binary).tocsr()
    overlap = (query_binary @ binary.T).toarray()
    if use_idf:
        df = np.asarray(binary.sum(axis=0)).ravel()
        idf = 1.0 + np.log((train_count + 1.0) / (df + 1.0))
        # sqrt(IDF) on columns makes the dot product IDF-weighted, not IDF².
        weighted = binary.multiply(np.sqrt(idf)[None, :]).tocsr()
        query_weighted = query_selector @ weighted
        dot = (query_weighted @ weighted.T).toarray()
        norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
        query_norms = np.sqrt(np.asarray(query_weighted.multiply(query_weighted).sum(axis=1)).ravel())
    else:
        dot = overlap.copy()
        norms = np.sqrt(np.asarray(binary.sum(axis=1)).ravel())
        query_norms = np.sqrt(np.asarray(query_binary.sum(axis=1)).ravel())
    denominator = query_norms[:, None] * norms[None, :]
    similarity = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
    similarity *= overlap / (overlap + OVERLAP_SHRINKAGE)
    # Unsupported candidate profiles must not consume any of the 50 slots.
    similarity[:, ~profile_supported] = 0.0
    known = np.flatnonzero(own_indices >= 0)
    similarity[known, own_indices[known]] = 0.0
    if not np.isfinite(similarity).all():
        raise ArithmeticError("Nonfinite neighbour similarity")

    distribution = np.zeros((query_count, profiles.shape[1]), dtype=np.float64)
    indices = np.full((query_count, NEIGHBOUR_COUNT), -1, dtype=np.int64)
    weights = np.zeros((query_count, NEIGHBOUR_COUNT), dtype=np.float64)
    supported = np.zeros(query_count, dtype=bool)
    for row in range(query_count):
        positive = np.flatnonzero(similarity[row] > 0)
        order = np.lexsort((positive, -similarity[row, positive]))
        chosen = positive[order[:NEIGHBOUR_COUNT]]
        if not len(chosen):
            continue
        selected_weights = similarity[row, chosen]
        distribution[row] = np.asarray(profiles[chosen].T @ selected_weights).ravel() / selected_weights.sum()
        supported[row] = True
        indices[row, :len(chosen)] = chosen
        weights[row, :len(chosen)] = selected_weights
    if not np.isfinite(distribution).all():
        raise ArithmeticError("Nonfinite neighbour distribution")
    return distribution, supported, indices, weights


def build_features(
    events: pd.DataFrame,
    users: list[str],
    candidates: list[int],
    cutoff: int,
) -> HybridFeatures:
    """Validate airing rows and cache daily profiles and both CF distributions.

    Existing V2 validation rejects future rows, floats in IDs/times, duplicate
    user-airings and aggregated counts. Similarity uses *all* TRAIN titles and
    users; candidate availability gates personal mass and neighbour profiles.
    """
    started = time.perf_counter()
    base = build_state(events, users, candidates, cutoff, HALF_LIFE_DAYS)
    for user in users:
        _numeric_user_key(user)
    train_users = tuple(sorted(events["user_id"].unique(), key=_numeric_user_key))
    user_index = {user: index for index, user in enumerate(train_users)}
    requested_index = {user: index for index, user in enumerate(users)}
    candidate_index = {int(item): index for index, item in enumerate(base.candidate_ids)}
    all_items = sorted(map(int, events["tv_show_id"].unique()))
    item_index = {item: index for index, item in enumerate(all_items)}
    pairs = events[["user_id", "tv_show_id"]].drop_duplicates()
    pair_rows = np.array([user_index[user] for user in pairs.user_id], dtype=np.int64)
    pair_columns = np.array([item_index[int(item)] for item in pairs.tv_show_id], dtype=np.int64)
    binary = sparse.csr_matrix((np.ones(len(pairs)), (pair_rows, pair_columns)),
                               shape=(len(train_users), len(all_items)))
    own_indices = np.array([user_index.get(user, -1) for user in users], dtype=np.int64)
    known = np.flatnonzero(own_indices >= 0)
    selector = sparse.csr_matrix((np.ones(len(known)), (known, own_indices[known])),
                                shape=(len(users), len(train_users)))

    available = events.loc[events.tv_show_id.isin(candidate_index), ["user_id", "tv_show_id", "last_time"]].copy()
    last_times = available.last_time.to_numpy(dtype=np.int64)
    available["weight"] = _airing_weights(last_times, int(cutoff), HALF_LIFE_DAYS)
    # Equivalent to (last_time-1)//86400 without int64 underflow at its minimum.
    available["day"] = last_times // 86400 - (last_times % 86400 == 0)
    daily = available.groupby(["user_id", "tv_show_id", "day"], sort=False).weight.max()
    daily = daily.groupby(level=[0, 1], sort=False).sum()
    daily_matrix = np.zeros_like(base.personal_matrix)
    for (user, item), weight in daily.items():
        if user in requested_index:
            daily_matrix[requested_index[user], candidate_index[int(item)]] = float(weight)

    counts = available.groupby(["user_id", "tv_show_id"], sort=False).weight.sum()
    profile_rows, profile_columns, profile_values = [], [], []
    for (user, item), weight in counts.items():
        if weight > 0:
            profile_rows.append(user_index[user])
            profile_columns.append(candidate_index[int(item)])
            profile_values.append(math.sqrt(float(weight)))
    profiles = sparse.csr_matrix((profile_values, (profile_rows, profile_columns)),
                                 shape=(len(train_users), len(base.candidate_ids)), dtype=np.float64)
    profile_mass = np.asarray(profiles.sum(axis=1)).ravel()
    profile_supported = profile_mass > 0
    inverse_mass = np.divide(1.0, profile_mass, out=np.zeros_like(profile_mass), where=profile_supported)
    profiles = profiles.multiply(inverse_mass[:, None]).tocsr()
    raw, raw_supported, raw_indices, raw_weights = _neighbours(
        binary, selector, profiles, profile_supported, own_indices, False)
    idf, idf_supported, idf_indices, idf_weights = _neighbours(
        binary, selector, profiles, profile_supported, own_indices, True)
    arrays = (daily_matrix, raw, idf, raw_supported, idf_supported,
              raw_indices, idf_indices, raw_weights, idf_weights)
    for array in arrays:
        if not np.isfinite(array).all():
            raise ArithmeticError("Nonfinite cached V3 features")
        array.setflags(write=False)
    diagnostics = {
        "implementation": "hybrid_v3", "feature_build_seconds": time.perf_counter() - started,
        "cutoff": int(cutoff), "half_life_days": HALF_LIFE_DAYS,
        "train_users": len(train_users), "similarity_train_items": len(all_items),
        "similarity_binary_user_item_pairs": len(pairs),
        "candidate_count": len(base.candidate_ids), "requested_users": len(users),
        "train_neighbours_with_available_profile": int(profile_supported.sum()),
        "requested_users_with_raw_cf_support": int(raw_supported.sum()),
        "requested_users_with_idf_cf_support": int(idf_supported.sum()),
        "neighbor_count_limit": NEIGHBOUR_COUNT, "overlap_shrinkage": OVERLAP_SHRINKAGE,
        "day_bucket": "floor((last_time-1)/86400); naive timestamp scale",
        "idf": "1+ln((N+1)/(df+1)); binary columns multiplied by sqrt(idf)",
        "neighbor_profile": "L1-normalised sqrt of available HL28 per-airing weighted counts; no IDF",
        "zero_candidate_profile_excluded_before_top_k": True,
        "self_excluded": True, "neighbor_tie_break": "numeric user ID ASC, parsed with Python int",
        "future_events_used": False, "evaluation_labels_used": False,
    }
    return HybridFeatures(base, daily_matrix, raw, idf, raw_supported, idf_supported,
                          train_users, raw_indices, idf_indices, raw_weights, idf_weights, diagnostics)


def _unit_interval(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")
    return value


def predict_v3(
    state: HybridFeatures,
    conf: dict[str, Any],
) -> tuple[dict[str, list[int]], dict[str, int], dict[str, Any]]:
    """Score only the supplied mechanism, with fixed HL28, tau30 and power .5.

    family v2 delegates directly to the previous implementation. For neighbours,
    unsupported rows use V2 scores in standalone and mixture configurations.
    Every configuration uses the exact common fallback for zero own available
    history, even when a neighbour has relevant candidate history.
    """
    started = time.perf_counter()
    family = conf.get("family")
    if family == "v2":
        return predict_power(state.base, PRIOR_STRENGTH, POWER)
    base = state.base
    extra: dict[str, Any]
    if family == "daycap":
        fraction = _unit_interval(conf["fraction"], "fraction")
        mass_mode = conf["mass_mode"]
        if mass_mode not in {"original", "capped"}:
            raise ValueError("mass_mode must be 'original' or 'capped'")
        profile = (1.0 - fraction) * base.personal_matrix + fraction * state.daily_matrix
        mass = base.personal_mass if mass_mode == "original" else profile.sum(axis=1)
        scores = _scores(profile, mass, base.prior)
        extra = {"fraction": fraction, "mass_mode": mass_mode,
                 "personal_mass_sum_used": float(mass.sum()),
                 "day_profile": "rprime=(1-fraction)*r+fraction*sum_day(max event decay28)",
                 "score": "(m*sqrt(rprime)/sum(sqrt(rprime))+30*g)/(m+30)"}
    elif family == "neighbors":
        use_idf = conf["idf"]
        if not isinstance(use_idf, (bool, np.bool_)):
            raise TypeError("idf must be boolean")
        mix = _unit_interval(conf["mix"], "mix")
        supported = state.cf_idf_supported if use_idf else state.cf_raw_supported
        distribution = state.cf_idf if use_idf else state.cf_raw
        scores = _scores(base.personal_matrix, base.personal_mass, base.prior)
        # Preserve V2 scores exactly for unsupported rows, including at mix=1.
        scores[supported] = (1.0 - mix) * scores[supported] + mix * distribution[supported]
        extra = {"idf": bool(use_idf), "mix": mix,
                 "users_with_cf_support": int(supported.sum()),
                 "users_without_cf_support": int((~supported).sum()),
                 "unsupported_cf": "unchanged V2 scores for standalone and mixtures",
                 "score": "(1-mix)*V2_score+mix*similarity_weighted_neighbour_profile"}
    else:
        raise ValueError("family must be 'v2', 'daycap' or 'neighbors'")
    if not np.isfinite(scores).all():
        raise ArithmeticError("Nonfinite V3 ranking scores")
    predictions, fallback_slots = {}, {}
    fallback = list(base.fallback)
    for row, user in enumerate(base.users):
        predictions[user], fallback_slots[user] = _rank_and_fill(
            base.candidate_ids, scores[row] if base.personal_mass[row] > 0 else None, fallback)
    diagnostics = {
        **state.diagnostics, **extra, "config_id": conf.get("id"), "family": family,
        "prior_strength": PRIOR_STRENGTH, "power": POWER, "scores_finite": True,
        "fallback": "unchanged V2 train distinct-user popularity DESC then item ID ASC",
        "zero_own_available_history": "exact V2 fallback regardless of CF support",
        "fallback_slots_total": int(sum(fallback_slots.values())),
        "tie_break": "score DESC, numeric item ID ASC", "history_repeats_allowed": True,
        "predict_seconds": time.perf_counter() - started,
    }
    return predictions, fallback_slots, diagnostics
