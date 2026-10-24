"""Train-only baselines, content profiles and implicit ALS with common ranking.

User identifiers are always strings. All content dictionaries and IDF statistics
are fitted on the supplied training catalogue; target metadata is never accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

K = 5


def implicit_objective(matrix: sparse.csr_matrix, users: np.ndarray, items: np.ndarray, regularization: float) -> float:
    """Exact HKV objective without allocating a dense user-item matrix.

    matrix stores confidence c_ui for observed positives; unobserved p=0,c=1.
    trace(X'X Y'Y) accounts for all unobserved squared predictions, then observed
    entries replace their squared prediction by c_ui*(1-prediction)^2.
    """
    observed = matrix.tocoo()
    dots = np.einsum("ij,ij->i", users[observed.row], items[observed.col])
    all_squared_scores = np.sum((users.T @ users) * (items.T @ items))
    correction = np.sum(observed.data * np.square(1.0 - dots) - np.square(dots))
    penalty = regularization * (np.sum(np.square(users)) + np.sum(np.square(items)))
    return float(all_squared_scores + correction + penalty)


def solve_implicit_rows(matrix: sparse.csr_matrix, fixed: np.ndarray, regularization: float) -> np.ndarray:
    """Minimise one side of the weighted least-squares objective exactly."""
    factors = fixed.shape[1]
    base = fixed.T @ fixed + regularization * np.eye(factors, dtype=np.float64)
    solved = np.zeros((matrix.shape[0], factors), dtype=np.float64)
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row:row+2]
        indices = matrix.indices[start:stop]
        confidences = matrix.data[start:stop]
        if len(indices) == 0:
            continue
        selected = fixed[indices]
        lhs = base + (selected.T * (confidences - 1.0)) @ selected
        rhs = selected.T @ confidences
        solved[row] = np.linalg.solve(lhs, rhs)
    return solved


def fit_implicit_als(matrix: sparse.csr_matrix, factors: int, regularization: float,
                     iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deterministic NumPy/SciPy implementation of Hu-Koren-Volinsky ALS.

    This portable direct solver replaces the unavailable compiled implicit
    package. The model and objective are unchanged; we use float64 normal
    equations and one BLAS thread to make the small-scale exam run reproducible.
    """
    if regularization <= 0 or np.any(matrix.data < 1):
        raise ValueError("Regularization must be positive and confidence >=1")
    rng = np.random.default_rng(seed)
    users = rng.normal(0.0, 0.01, size=(matrix.shape[0], factors))
    items = rng.normal(0.0, 0.01, size=(matrix.shape[1], factors))
    transposed = matrix.T.tocsr()
    objectives = [implicit_objective(matrix, users, items, regularization)]
    iteration_seconds = []
    for _ in range(iterations):
        started = time.perf_counter()
        users = solve_implicit_rows(matrix, items, regularization)
        items = solve_implicit_rows(transposed, users, regularization)
        objective = implicit_objective(matrix, users, items, regularization)
        if objective > objectives[-1] + max(1e-6, abs(objectives[-1]) * 1e-8):
            raise ArithmeticError("HKV ALS objective increased after a complete iteration")
        objectives.append(objective)
        iteration_seconds.append(time.perf_counter() - started)
    return users, items, {"implementation": "portable_numpy_scipy_direct_hkv_als",
                           "factor_dtype": "float64", "objective_by_iteration": objectives,
                           "iteration_seconds": iteration_seconds, "objective_monotone_nonincreasing": True}


def user_item_matrix(counts: pd.DataFrame, users: list[str], items: list[int], weighting: str) -> sparse.csr_matrix:
    user_index = {str(user): i for i, user in enumerate(users)}
    item_index = {int(item): i for i, item in enumerate(items)}
    selected = counts[counts["tv_show_id"].isin(item_index)].copy()
    if selected.empty:
        return sparse.csr_matrix((len(users), len(items)), dtype=np.float32)
    rows = selected["user_id"].map(user_index).to_numpy(dtype=np.int64)
    cols = selected["tv_show_id"].map(item_index).to_numpy(dtype=np.int64)
    values = selected["count"].to_numpy(dtype=np.float32)
    if weighting == "binary":
        values = np.ones_like(values)
    elif weighting == "log_count":
        values = np.log1p(values)
    elif weighting != "count":
        raise ValueError(f"Unknown weighting {weighting}")
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(len(users), len(items)), dtype=np.float32).tocsr()
    matrix.sum_duplicates()
    return matrix


def popularity_order(counts: pd.DataFrame, candidates: list[int]) -> list[int]:
    """Distinct positive training users; never target frequency or duration."""
    popularity = counts.groupby("tv_show_id")["user_id"].nunique().to_dict()
    return sorted(set(map(int, candidates)), key=lambda item: (-popularity.get(item, 0), item))


def rank_with_fallback(candidates: list[int], scores: np.ndarray | None, fallback: list[int], k: int = K,
                       positive_only: bool = False) -> tuple[list[int], int]:
    """Rank supported candidates, then append one shared deterministic fallback.

    ALS signed scores are meaningful relative scores and may be negative.
    Content zero scores mean no content match and are delegated to fallback.
    """
    candidate_array = np.asarray(candidates, dtype=np.int64)
    if len(candidate_array) != len(np.unique(candidate_array)) or np.any(candidate_array == 0):
        raise ValueError("Candidates must be unique, nonzero programme IDs")
    if len(candidate_array) < k:
        raise ValueError(f"Only {len(candidate_array)} candidates, need {k}")
    ranked: list[int] = []
    if scores is not None:
        score_array = np.asarray(scores, dtype=np.float64)
        if len(score_array) != len(candidate_array):
            raise ValueError("Candidate and score lengths differ")
        valid = np.isfinite(score_array)
        if positive_only:
            valid &= score_array > 0
        selected = np.flatnonzero(valid)
        order = np.lexsort((candidate_array[selected], -score_array[selected]))
        ranked = candidate_array[selected[order[:k]]].tolist()
    added = 0
    candidate_set = set(candidate_array.tolist())
    seen = set(ranked)
    for item in fallback:
        item = int(item)
        if len(ranked) == k:
            break
        if item in candidate_set and item not in seen:
            ranked.append(item)
            seen.add(item)
            added += 1
    if len(ranked) != k:
        raise ValueError("Shared fallback failed to fill five candidates")
    return ranked, added


def content_attributes(record: dict[str, Any]) -> dict[str, float]:
    """A small interpretable vocabulary: no titles, item IDs or future fields."""
    result: dict[str, float] = {}

    def clean(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        return re.sub(r"\s+", " ", str(value)).strip(" \t\n'\"").casefold()

    category = clean(record.get("tv_show_category", ""))
    if category:
        result[f"category={category}"] = 1.0
    for column in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
        value = clean(record.get(column, ""))
        if value:
            result[f"genre={value}"] = 1.0
    year = re.search(r"(?:18|19|20)\d{2}", clean(record.get("year_of_production", "")))
    if year:
        result[f"decade={int(year.group()) // 10 * 10}"] = 0.25
    for column, label in (("director", "director"), ("actors", "actor")):
        value = clean(record.get(column, ""))
        for token in re.split(r"[,;|]", value):
            token = token.strip()
            if token:
                result[f"{label}={token}"] = 1.0
    return result


@dataclass
class FittedModel:
    name: str
    params: dict[str, Any]
    diagnostics: dict[str, Any]
    users: list[str]
    items: list[int]
    user_vectors: Any = None
    item_vectors: Any = None
    popularity: dict[int, int] | None = None

    def score(self, users: list[str], candidates: list[int]) -> np.ndarray:
        scores = np.full((len(users), len(candidates)), np.nan, dtype=np.float32)
        if self.name == "popularity":
            values = np.array([self.popularity.get(int(item), 0) for item in candidates], dtype=np.float32)
            values[values == 0] = np.nan
            scores[:] = values
            return scores
        u_index = {str(user): i for i, user in enumerate(self.users)}
        i_index = {int(item): i for i, item in enumerate(self.items)}
        known_users = [(out, u_index[str(user)]) for out, user in enumerate(users) if str(user) in u_index]
        known_items = [(out, i_index[int(item)]) for out, item in enumerate(candidates) if int(item) in i_index]
        if not known_users or not known_items:
            return scores
        out_rows, rows = zip(*known_users)
        out_cols, cols = zip(*known_items)
        product = self.user_vectors[list(rows)] @ self.item_vectors[list(cols)].T
        if sparse.issparse(product):
            product = product.toarray()
        scores[np.ix_(out_rows, out_cols)] = product
        return scores


def fit_model(name: str, counts: pd.DataFrame, catalog: pd.DataFrame, params: dict[str, Any], seed: int = 42) -> FittedModel:
    if (counts["count"] <= 0).any() or (counts["tv_show_id"] == 0).any():
        raise ValueError("Only positive nonzero programme interactions may be fitted")
    if not counts["user_id"].map(lambda x: isinstance(x, str)).all():
        raise TypeError("user_id must be strings, never inferred float64")
    users = sorted(counts["user_id"].unique().tolist())
    observed_items = sorted(map(int, counts["tv_show_id"].unique()))
    diag = {"train_users": len(users), "train_observed_items": len(observed_items),
            "train_user_item_pairs": len(counts), "positive_airing_count": int(counts["count"].sum())}
    if name == "popularity":
        values = counts.groupby("tv_show_id")["user_id"].nunique().to_dict()
        return FittedModel(name, params, diag, users, observed_items, popularity=values)
    if name == "cbf":
        item_catalog = catalog[catalog["tv_show_id"] != 0].drop_duplicates("tv_show_id").set_index("tv_show_id")
        items = sorted(set(map(int, item_catalog.index)) | set(observed_items))
        features = [content_attributes(item_catalog.loc[item].to_dict()) if item in item_catalog.index else {} for item in items]
        vectorizer = DictVectorizer(sparse=True, dtype=np.float32, sort=True)
        raw = vectorizer.fit_transform(features)
        if raw.shape[1] == 0:
            raise ValueError("No usable content metadata in the training catalogue")
        item_vectors = TfidfTransformer(norm="l2", smooth_idf=True).fit_transform(raw).astype(np.float32)
        histories = user_item_matrix(counts, users, items, params["profile_weighting"])
        profiles = normalize(histories @ item_vectors, norm="l2", axis=1).tocsr()
        diag.update({"catalogue_items": len(items), "content_features": raw.shape[1],
                     "items_without_features": int(np.sum(np.asarray(raw.getnnz(axis=1)) == 0)),
                     "content_fit_scope": "training_catalogue_only", "future_metadata_used": False})
        return FittedModel(name, params, diag, users, items, profiles, item_vectors)
    if name == "implicit_mf":
        matrix = user_item_matrix(counts, users, observed_items, "log_count")
        matrix = matrix.astype(np.float64)
        matrix.data = 1.0 + float(params["alpha"]) * matrix.data
        user_factors, item_factors, als_diag = fit_implicit_als(
            matrix, int(params["factors"]), float(params["regularization"]), int(params["iterations"]), seed,
        )
        diag.update({"confidence": "1 + alpha * log1p(positive_airing_count)",
                     "matrix_shape": list(matrix.shape), "matrix_nnz": int(matrix.nnz),
                     "threads": 1, "random_seed": seed, **als_diag})
        return FittedModel(name, params, diag, users, observed_items, user_factors, item_factors)
    raise ValueError(f"Unknown method {name}")


def predict(model: FittedModel, counts: pd.DataFrame, users: list[str], candidates: list[int]) -> tuple[dict[str, list[int]], dict[str, int]]:
    fallback = popularity_order(counts, candidates)
    scores = model.score(users, candidates)
    predictions, fallback_slots = {}, {}
    for row, user in enumerate(users):
        predictions[str(user)], fallback_slots[str(user)] = rank_with_fallback(
            candidates, scores[row], fallback, positive_only=model.name in ("cbf", "popularity"),
        )
    return predictions, fallback_slots
