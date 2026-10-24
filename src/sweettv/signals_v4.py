"""Frozen Phase A signals: partial coverage, opportunities and airing rates.

Only qualified/partial TRAIN observations and structural known timetables enter.
The previous V2 and V3 controls delegate directly to their frozen code.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from .exposure_v4 import ExposureState, build_exposure_state, exposure_ratio
from .frequency_v1 import _integer_values, _rank_and_fill
from .hybrid_v3 import HybridFeatures, _scores, build_features as build_v3_features, predict_v3
from .partial_features_v4 import build_weak_profile, eligible_weak_airings
from .power_frequency_v2 import PowerFrequencyState, predict_power


@dataclass(frozen=True)
class SignalFeatures:
    base: PowerFrequencyState
    v3: HybridFeatures
    exposure: ExposureState
    weak_profiles: dict[int, np.ndarray]
    weak_supported: dict[int, np.ndarray]
    qualified_y: np.ndarray
    population_mu: np.ndarray
    diagnostics: dict[str, Any]


def _normalise_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mass = matrix.sum(axis=1)
    result = np.divide(matrix, mass[:, None], out=np.zeros_like(matrix), where=mass[:, None] > 0)
    return result, mass > 0


def _probability_history(
    events: pd.DataFrame, schedule: pd.DataFrame, exposure: ExposureState, users: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if "air_id" not in events:
        raise ValueError("Probability history requires exact air_id on qualified events")
    air_ids = _integer_values(events.air_id, "events.air_id")
    candidate_index = {int(item): i for i, item in enumerate(exposure.candidate_ids)}
    user_index = {user: i for i, user in enumerate(users)}
    available = events.loc[events.tv_show_id.isin(candidate_index), ["user_id", "tv_show_id"]].copy()
    available["air_id"] = air_ids[events.tv_show_id.isin(candidate_index).to_numpy()]
    structural = schedule[["air_id", "tv_show_id", "end"]]
    joined = available.merge(structural, on="air_id", how="left", suffixes=("", "_schedule"), validate="many_to_one")
    if joined.end.isna().any():
        raise ValueError("Qualified available air_id missing from structural schedule")
    if not joined.tv_show_id.eq(joined.tv_show_id_schedule).all():
        raise ValueError("Qualified air_id programme does not match structural schedule")
    boundary_rows = int((joined.end > exposure.diagnostics["cutoff"]).sum())
    # Joining the precomputed eligibility set drops crossing-cutoff airings and
    # any full-duration airing that cannot qualify inside the TRAIN window.
    matched = available.merge(exposure.probability_airings[["air_id", "tv_show_id", "weight"]],
                              on=["air_id", "tv_show_id"], how="inner", validate="many_to_one")
    grouped = matched.groupby(["user_id", "tv_show_id"], sort=False).weight.sum()
    y = np.zeros((len(users), len(exposure.candidate_ids)), dtype=np.float64)
    global_y = np.zeros(len(exposure.candidate_ids), dtype=np.float64)
    for (user, item), weight in grouped.items():
        column = candidate_index[int(item)]
        if float(weight) > exposure.denominator_weight[column] + 1e-10 * max(1., exposure.denominator_weight[column]):
            raise AssertionError("Qualified numerator Y exceeds complete-airing opportunity D")
        global_y[column] += float(weight)
        if user in user_index:
            y[user_index[user], column] = float(weight)
    train_users = int(events.user_id.nunique())
    total_opportunity = train_users * exposure.denominator_weight
    if (global_y > total_opportunity + 1e-10 * np.maximum(1., total_opportunity)).any():
        raise AssertionError("Population qualified numerator exceeds population opportunities")
    mu = np.divide(global_y, total_opportunity, out=np.zeros_like(global_y), where=total_opportunity > 0)
    if not np.isfinite(y).all() or not np.isfinite(mu).all() or (mu < 0).any() or (mu > 1 + 1e-12).any():
        raise AssertionError("Invalid probability sufficient statistics")
    return y, mu, {
        "probability_training_users_before_candidate_gating": train_users,
        "probability_qualified_available_events": len(available),
        "probability_matched_complete_eligible_events": len(matched),
        "probability_crossing_cutoff_qualified_events_dropped": boundary_rows,
        "probability_other_ineligible_events_dropped": len(available) - len(matched) - boundary_rows,
        "probability_numerator_and_denominator_same_airing_end_weights": True,
        "maximum_population_mu": float(mu.max()) if len(mu) else 0.,
    }


def build_features(
    events: pd.DataFrame,
    partial_airings: pd.DataFrame,
    schedule_state: pd.DataFrame,
    users: list[str],
    candidates: list[int],
    cutoff: int,
    targetend: int,
) -> SignalFeatures:
    """Cache train-only features; `schedule_state` is structural schedule.parquet.

    Partial airings may contain every positive coverage row from the audited
    cache. Only 60+s, coverage>=.05 and coverage<.8 rows enter weak profiles.
    Full qualified history, its mass and global prior are never redefined.
    """
    started = time.perf_counter()
    v3 = build_v3_features(events, users, candidates, cutoff)
    base = v3.base
    exposure = build_exposure_state(schedule_state, candidates, cutoff, targetend)
    np.testing.assert_array_equal(base.candidate_ids, exposure.candidate_ids)
    weak_rows = eligible_weak_airings(partial_airings, int(cutoff))
    if len(partial_airings):
        observed_coverage = partial_airings.watch_seconds.to_numpy(dtype=np.float64) / partial_airings.duration.to_numpy(dtype=np.float64)
        if not np.allclose(partial_airings.coverage.to_numpy(dtype=np.float64), observed_coverage, rtol=0, atol=1e-14):
            raise ValueError("Partial coverage differs from watched/full original duration")
    if len(weak_rows):
        weak_keys = pd.MultiIndex.from_frame(weak_rows[["user_id", "air_id"]])
        if "air_id" not in events:
            raise ValueError("Qualified events require air_id")
        strong_keys = pd.MultiIndex.from_frame(events[["user_id", "air_id"]])
        if weak_keys.isin(strong_keys).any():
            raise ValueError("One user-airing cannot be both weak and qualified")
    user_index = {user: row for row, user in enumerate(users)}
    candidate_index = {int(item): column for column, item in enumerate(base.candidate_ids)}
    weak_profiles, weak_supported = {}, {}
    for power in (2, 4):
        aggregated = build_weak_profile(partial_airings, int(cutoff), power=power)
        matrix = np.zeros_like(base.personal_matrix)
        for user, item, weight in aggregated[["user_id", "tv_show_id", "weak_weight"]].itertuples(index=False, name=None):
            if user in user_index and int(item) in candidate_index:
                matrix[user_index[user], candidate_index[int(item)]] = float(weight)
        profile, supported = _normalise_rows(matrix)
        profile.setflags(write=False); supported.setflags(write=False)
        weak_profiles[power], weak_supported[power] = profile, supported
    y, mu, probability_diag = _probability_history(events, schedule_state, exposure, users)
    y.setflags(write=False); mu.setflags(write=False)
    diagnostics = {
        "implementation": "signals_v4_phase_a", "cutoff": int(cutoff), "target_end": int(targetend),
        "half_life_days": 28., "power": .5, "prior_strength": 30.,
        "qualified_event_rows": len(events), "partial_source_airing_rows": len(partial_airings),
        "eligible_weak_airing_rows": len(weak_rows),
        "users_with_available_weak_p2": int(weak_supported[2].sum()),
        "users_with_available_weak_p4": int(weak_supported[4].sum()),
        "users_without_available_qualified_history": int((base.personal_mass <= 0).sum()),
        "weak_and_qualified_airings_disjoint": True,
        "weak_profile": "sum coverage**p*decay28(last_time), L1 normalised directly; no sqrt",
        "exposure": exposure.diagnostics, **probability_diag,
        "feature_build_seconds": time.perf_counter() - started,
        "future_viewing_used": False, "target_truth_used": False, "future_content_metadata_used": False,
    }
    return SignalFeatures(base, v3, exposure, weak_profiles, weak_supported, y, mu, diagnostics)


def _mix(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("mix must be finite and between zero and one")
    return value


def _partial_scores(state: SignalFeatures, power: int, mix: float) -> np.ndarray:
    if power not in (2, 4) or isinstance(power, bool):
        raise ValueError("Partial p must be 2 or 4")
    base = state.base
    scores = _scores(base.personal_matrix, base.personal_mass, base.prior)
    supported = state.weak_supported[power] & (base.personal_mass > 0)
    own_shape, _ = _normalise_rows(np.sqrt(base.personal_matrix))
    shape = (1. - mix) * own_shape[supported] + mix * state.weak_profiles[power][supported]
    mass = base.personal_mass[supported, None]
    scores[supported] = (mass * shape + 30. * base.prior[None, :]) / (mass + 30.)
    return scores


def _exposure_correct(scores: np.ndarray, state: SignalFeatures, mode: str, beta: float) -> np.ndarray:
    beta = float(beta)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    ratio = exposure_ratio(state.exposure, mode)
    return scores * np.clip(ratio, .25, 4.)[None, :] ** beta


def _probability_scores(state: SignalFeatures, kappa: float, mix: float) -> tuple[np.ndarray, dict[str, Any]]:
    kappa = float(kappa)
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    d = state.exposure.denominator_weight
    probability = (state.qualified_y + kappa * state.population_mu[None, :]) / (d[None, :] + kappa)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1 + 1e-12).any():
        raise AssertionError("Per-airing probability must be finite and between zero and one")
    # Only roundoff beyond one is clipped after the strict tolerance check.
    probability = np.minimum(probability, 1.)
    expected = probability * state.exposure.f_raw[None, :]
    distribution, supported = _normalise_rows(expected)
    base = state.base
    scores = _scores(base.personal_matrix, base.personal_mass, base.prior)
    scores[supported] = (1. - mix) * scores[supported] + mix * distribution[supported]
    return scores, {"kappa": kappa, "mix": mix, "maximum_per_airing_probability": float(probability.max()),
                    "probability_users_with_supported_expected_profile": int(supported.sum()),
                    "unsupported_probability_profile": "unchanged V2 scores",
                    "probability_score": "L1(p_ui*Fraw_i), mixed with V2 scores"}


def predict_v4(
    state: SignalFeatures, conf: dict[str, Any],
) -> tuple[dict[str, list[int]], dict[str, int], dict[str, Any]]:
    """Score one frozen Phase A config; zero own qualified mass always falls back."""
    started = time.perf_counter()
    family = conf.get("family")
    if family == "v2":
        return predict_power(state.base, 30., .5)
    if family == "v3":
        return predict_v3(state.v3, {"id": "cf_idf_l015", "family": "neighbors", "idf": True, "mix": .15})
    base = state.base
    extra: dict[str, Any] = {}
    if family in {"partial", "combo"}:
        power, mix = conf["p"], _mix(conf["mix"])
        scores = _partial_scores(state, power, mix)
        extra.update({"partial_power": power, "mix": mix,
                      "partial_mass": "unchanged original V2 n; prior30 unchanged",
                      "unsupported_weak": "unchanged V2 personal score before any exposure correction"})
    elif family == "exposure":
        scores = _scores(base.personal_matrix, base.personal_mass, base.prior)
    elif family == "probability":
        scores, extra = _probability_scores(state, conf["kappa"], _mix(conf["mix"]))
    else:
        raise ValueError("Unknown Phase A family")
    if family in {"exposure", "combo"}:
        scores = _exposure_correct(scores, state, conf["exposure_mode"], conf["beta"])
        ratio = exposure_ratio(state.exposure, conf["exposure_mode"])
        extra.update({"exposure_mode": conf["exposure_mode"], "beta": float(conf["beta"]),
                      "exposure_ratio_minimum": float(ratio.min()), "exposure_ratio_maximum": float(ratio.max()),
                      "exposure_score": "personal_score*clip((F*28/target_days+1)/(D+1),.25,4)**beta",
                      "exposure_denominator": ".5*(E_last28+E_long*28/train_days)"})
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ArithmeticError("Phase A scores must be finite and nonnegative")
    predictions, fallback_slots = {}, {}
    fallback = list(base.fallback)
    for row, user in enumerate(base.users):
        predictions[user], fallback_slots[user] = _rank_and_fill(
            base.candidate_ids, scores[row] if base.personal_mass[row] > 0 else None, fallback)
    diagnostics = {**state.diagnostics, **extra, "family": family, "config_id": conf.get("id"),
                   "scores_finite": True, "fallback_slots_total": int(sum(fallback_slots.values())),
                   "zero_own_available_qualified_history": "exact original V2 popularity fallback",
                   "tie_break": "score DESC numeric item ID ASC; five unique nonzero available items",
                   "history_repeats_allowed": True, "predict_seconds": time.perf_counter() - started}
    return predictions, fallback_slots, diagnostics
