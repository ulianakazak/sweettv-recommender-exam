"""Synthetic Phase A formulas, boundaries and frozen-control equivalence."""
import unittest

import numpy as np
import pandas as pd

from sweettv.exposure_v4 import DAY, TRAIN_START, exposure_ratio
from sweettv.hybrid_v3 import _scores, build_features as build_v3, predict_v3
from sweettv.power_frequency_v2 import build_state, predict_power
from sweettv.signals_v4 import (_partial_scores, _probability_scores, build_features, predict_v4)


CUTOFF = TRAIN_START + 56 * DAY
TARGET_END = CUTOFF + 28 * DAY
POOL = [1, 2, 3, 4, 5, 6]


def events(rows):
    return pd.DataFrame(rows, columns=["user_id", "tv_show_id", "last_time", "air_id"]).astype(
        {"user_id": str, "tv_show_id": "int64", "last_time": "int64", "air_id": "int64"})


def partial(rows):
    frame = pd.DataFrame(rows, columns=["user_id", "air_id", "tv_show_id", "watch_seconds", "duration", "last_time"])
    frame["uid"] = [int(user) % 100 for user in frame.user_id]
    frame = frame.astype({"user_id": str, "uid": "int64", "air_id": "int64", "tv_show_id": "int64",
                          "watch_seconds": "int64", "duration": "int64", "last_time": "int64"})
    frame["coverage"] = frame.watch_seconds.astype(float) / frame.duration
    return frame


def schedule(rows):
    return pd.DataFrame(rows, columns=["air_id", "channel_id", "tv_show_id", "start", "end"]).astype("int64")


def configs():
    result = [{"family": "v2"}, {"family": "v3"}]
    result += [{"family": "partial", "p": p, "mix": mix} for p in (2, 4) for mix in (.1, .25, .5)]
    result += [{"family": "exposure", "exposure_mode": mode, "beta": beta}
               for mode in ("raw", "slots") for beta in (.25, .5, 1.)]
    result += [{"family": "combo", "p": p, "mix": .25, "exposure_mode": mode, "beta": beta}
               for p in (2, 4) for mode in ("raw", "slots") for beta in (.25, .5)]
    result += [{"family": "probability", "kappa": kappa, "mix": mix}
               for kappa in (10, 50) for mix in (.15, .3)]
    return result


def fixture():
    strong = events([("10", 1, CUTOFF, air) for air in range(1, 5)]
                    + [("10", 2, CUTOFF, 5), ("20", 2, CUTOFF, 6)])
    weak = partial([("10", 101, 3, 500, 1000, CUTOFF), ("10", 102, 4, 250, 1000, CUTOFF),
                    ("30", 103, 3, 500, 1000, CUTOFF)])
    rows = [(row.air_id, row.air_id, row.tv_show_id, CUTOFF - 100, CUTOFF) for row in strong.itertuples()]
    rows += [(row.air_id, row.air_id, row.tv_show_id, CUTOFF - row.duration, CUTOFF) for row in weak.itertuples()]
    rows += [(201, 201, 1, CUTOFF + DAY, CUTOFF + DAY + 100),
             (202, 202, 3, CUTOFF + DAY, CUTOFF + DAY + 100),
             (203, 203, 3, CUTOFF + DAY, CUTOFF + DAY + 100)]
    return strong, weak, schedule(rows)


class SignalsV4Tests(unittest.TestCase):
    def test_partial_direct_l1_no_sqrt_and_original_mass_prior(self):
        strong, weak, timetable = fixture()
        state = build_features(strong, weak, timetable, ["10", "20", "30", "90"], POOL, CUTOFF, TARGET_END)
        np.testing.assert_allclose(state.weak_profiles[2][0], [0, 0, .8, .2, 0, 0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(state.weak_profiles[4][0], [0, 0, 16 / 17, 1 / 17, 0, 0], rtol=0, atol=1e-15)
        self.assertEqual(state.base.personal_mass[0], 5.)
        own = np.array([2 / 3, 1 / 3, 0, 0, 0, 0])
        shape = .75 * own + .25 * np.array([0, 0, .8, .2, 0, 0])
        expected = (5 * shape + 30 * np.array([1 / 3, 2 / 3, 0, 0, 0, 0])) / 35
        np.testing.assert_allclose(_partial_scores(state, 2, .25)[0], expected, rtol=0, atol=1e-15)
        self.assertAlmostEqual(expected.sum(), 1.)
        original = _scores(state.base.personal_matrix, state.base.personal_mass, state.base.prior)
        np.testing.assert_array_equal(_partial_scores(state, 2, .25)[1], original[1])

    def test_weak_decay_per_airing_before_direct_l1_and_strong_excluded(self):
        strong = events([("10", 1, CUTOFF, 1)])
        weak = partial([("10", 101, 2, 500, 1000, CUTOFF - 28 * DAY),
                        ("10", 102, 3, 500, 1000, CUTOFF),
                        ("10", 103, 4, 800, 1000, CUTOFF),
                        ("10", 104, 5, 59, 1000, CUTOFF)])
        timetable = schedule([(1, 1, 1, CUTOFF - 100, CUTOFF)])
        state = build_features(strong, weak, timetable, ["10"], POOL, CUTOFF, TARGET_END)
        np.testing.assert_allclose(state.weak_profiles[2][0], [0, 1 / 3, 2 / 3, 0, 0, 0], rtol=0, atol=1e-15)
        self.assertEqual(state.diagnostics["eligible_weak_airing_rows"], 2)

    def test_exposure_and_combo_are_exact_multiplicative_corrections(self):
        strong, weak, timetable = fixture()
        state = build_features(strong, weak, timetable, ["10", "20", "30"], POOL, CUTOFF, TARGET_END)
        for mode in ("raw", "slots"):
            ratio = np.clip(exposure_ratio(state.exposure, mode), .25, 4.)
            for family in ("exposure", "combo"):
                conf = {"family": family, "exposure_mode": mode, "beta": .5, "p": 2, "mix": .25}
                scores = _partial_scores(state, 2, .25) if family == "combo" else _scores(
                    state.base.personal_matrix, state.base.personal_mass, state.base.prior)
                expected = scores * np.sqrt(ratio)[None, :]
                pred, _, _ = predict_v4(state, conf)
                for row, user in enumerate(["10", "20"]):
                    ranked = sorted([item for item in POOL if expected[row, item - 1] > 0],
                                    key=lambda item: (-expected[row, item - 1], item))
                    ranked += [item for item in state.base.fallback if item not in ranked]
                    self.assertEqual(pred[user], ranked[:5])
        # User20 has no weak profile; combo still applies the exposure term.
        self.assertFalse(state.weak_supported[2][1])

    def test_probability_exact_airing_end_weights_boundary_drop_and_mu(self):
        old_end = CUTOFF - 28 * DAY
        timetable = schedule([(1, 1, 1, old_end - 100, old_end),
                              (2, 2, 1, CUTOFF - 100, CUTOFF),
                              (3, 3, 2, CUTOFF - 100, CUTOFF),
                              (4, 4, 2, CUTOFF - 80, CUTOFF + 20),
                              (5, 5, 1, CUTOFF + DAY, CUTOFF + DAY + 100),
                              (6, 6, 1, CUTOFF + 2 * DAY, CUTOFF + 2 * DAY + 100),
                              (7, 7, 2, CUTOFF + DAY, CUTOFF + DAY + 100)])
        strong = events([("10", 1, old_end - 20, 1), ("20", 1, CUTOFF, 2),
                         ("20", 2, CUTOFF, 3), ("10", 2, CUTOFF, 4)])
        state = build_features(strong, partial([]), timetable, ["10", "20", "90"], POOL, CUTOFF, TARGET_END)
        np.testing.assert_allclose(state.exposure.denominator_weight[:2], [1.5, 1], rtol=0, atol=1e-15)
        np.testing.assert_allclose(state.qualified_y[:2, :2], [[.5, 0], [1, 1]], rtol=0, atol=1e-15)
        np.testing.assert_allclose(state.population_mu[:2], [.5, .5], rtol=0, atol=1e-15)
        self.assertEqual(state.diagnostics["probability_crossing_cutoff_qualified_events_dropped"], 1)
        probability = np.array([11 / 23, 5 / 11])
        expected_counts = probability * np.array([2, 1])
        expected_distribution = expected_counts / expected_counts.sum()
        scores, diag = _probability_scores(state, 10, .3)
        base = _scores(state.base.personal_matrix, state.base.personal_mass, state.base.prior)
        expected = .7 * base[0]
        expected[:2] += .3 * expected_distribution
        np.testing.assert_allclose(scores[0], expected, rtol=0, atol=1e-15)
        self.assertLessEqual(diag["maximum_per_airing_probability"], 1.)
        self.assertGreater(abs(state.base.personal_matrix[0, 0] - .5), 1e-8)

    def test_probability_population_includes_unavailable_item_users(self):
        strong = events([("10", 1, CUTOFF, 1), ("20", 99, CUTOFF, 2)])
        timetable = schedule([(1, 1, 1, CUTOFF - 100, CUTOFF), (2, 2, 99, CUTOFF - 100, CUTOFF),
                              (3, 3, 1, CUTOFF + DAY, CUTOFF + DAY + 100)])
        state = build_features(strong, partial([]), timetable, ["10"], POOL, CUTOFF, TARGET_END)
        self.assertEqual(state.diagnostics["probability_training_users_before_candidate_gating"], 2)
        self.assertEqual(state.population_mu[0], .5)

    def test_controls_direct_and_zero_qualified_history_fallback_all26(self):
        strong, weak, timetable = fixture()
        big1, big2 = "1000000000000000000003000", "1000000000000000000004000"
        renames = {"10": big1, "20": big2}
        strong.user_id = strong.user_id.replace(renames)
        weak.user_id = weak.user_id.replace(renames)
        users = [big1, big2, "30", "90"]
        state = build_features(strong, weak, timetable, users, POOL, CUTOFF, TARGET_END)
        direct_v2 = predict_power(build_state(strong, users, POOL, CUTOFF, 28), 30, .5)
        direct_v3 = predict_v3(build_v3(strong, users, POOL, CUTOFF),
                               {"id": "cf_idf_l015", "family": "neighbors", "idf": True, "mix": .15})
        self.assertEqual(predict_v4(state, {"family": "v2"})[:2], direct_v2[:2])
        self.assertEqual(predict_v4(state, {"family": "v3"})[:2], direct_v3[:2])
        self.assertEqual(len(configs()), 26)
        for conf in configs():
            pred, fallback, diag = predict_v4(state, conf)
            for user in ("30", "90"):
                self.assertEqual(pred[user], direct_v2[0][user])
                self.assertEqual(fallback[user], 5)
            for values in pred.values():
                self.assertEqual(len(values), len(set(values)))
                self.assertEqual(len(values), 5)
                self.assertLessEqual(set(values), set(POOL))
            self.assertTrue(diag["scores_finite"])

    def test_probability_unsupported_exact_v2_and_empty_state(self):
        strong = events([("10", 1, CUTOFF, 1)])
        timetable = schedule([(1, 1, 1, CUTOFF - 100, CUTOFF)])
        state = build_features(strong, partial([]), timetable, ["10", "90"], POOL, CUTOFF, TARGET_END)
        expected = predict_power(state.base, 30, .5)
        self.assertEqual(predict_v4(state, {"family": "probability", "kappa": 10, "mix": .3})[:2], expected[:2])
        empty = build_features(events([]), partial([]), schedule([]), ["90"], POOL, CUTOFF, TARGET_END)
        for conf in configs():
            pred, fallback, _ = predict_v4(empty, conf)
            self.assertEqual(pred["90"], [1, 2, 3, 4, 5])
            self.assertEqual(fallback["90"], 5)

    def test_permutation_finite_readonly_and_inputs_unchanged(self):
        strong, weak, timetable = fixture()
        originals = [frame.copy(deep=True) for frame in (strong, weak, timetable)]
        users = ["10", "20", "30", "90"]
        first = build_features(strong, weak, timetable, users, POOL, CUTOFF, TARGET_END)
        second = build_features(strong.sample(frac=1, random_state=42), weak.sample(frac=1, random_state=42),
                                timetable.sample(frac=1, random_state=42), users[::-1], POOL[::-1], CUTOFF, TARGET_END)
        for conf in configs():
            self.assertEqual(predict_v4(first, conf)[:2], predict_v4(second, conf)[:2])
        for array in [first.qualified_y, first.population_mu, *first.weak_profiles.values(), *first.weak_supported.values()]:
            self.assertFalse(array.flags.writeable)
            self.assertTrue(np.isfinite(array).all())
        for frame, original in zip((strong, weak, timetable), originals):
            pd.testing.assert_frame_equal(frame, original)

    def test_future_duplicate_mismatch_and_invalid_config_rejected(self):
        strong, weak, timetable = fixture()
        with self.assertRaisesRegex(ValueError, "future"):
            build_features(strong, weak.assign(last_time=CUTOFF + 1), timetable, ["10"], POOL, CUTOFF, TARGET_END)
        with self.assertRaisesRegex(ValueError, "Future"):
            build_features(strong.assign(last_time=CUTOFF + 1), weak, timetable, ["10"], POOL, CUTOFF, TARGET_END)
        with self.assertRaisesRegex(ValueError, "coverage"):
            build_features(strong, weak.assign(coverage=.1), timetable, ["10"], POOL, CUTOFF, TARGET_END)
        with self.assertRaisesRegex(ValueError, "match"):
            build_features(strong.assign(tv_show_id=6), weak, timetable, ["10"], POOL, CUTOFF, TARGET_END)
        state = build_features(strong, weak, timetable, ["10"], POOL, CUTOFF, TARGET_END)
        for conf in ({"family": "partial", "p": 3, "mix": .25},
                     {"family": "partial", "p": 2, "mix": 2},
                     {"family": "exposure", "exposure_mode": "raw", "beta": float("nan")},
                     {"family": "probability", "kappa": 0, "mix": .3}, {"family": "wrong"}):
            with self.assertRaises(ValueError):
                predict_v4(state, conf)


if __name__ == "__main__":
    unittest.main()
