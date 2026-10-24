"""Synthetic, hand-worked V3 tests; no real data or evaluation windows."""
import math
import unittest

import numpy as np
import pandas as pd

from sweettv.hybrid_v3 import _scores, build_features, predict_v3
from sweettv.power_frequency_v2 import build_state, predict_power


def events(rows):
    frame = pd.DataFrame(rows, columns=["user_id", "tv_show_id", "last_time"])
    frame["air_id"] = range(len(frame))
    return frame.astype({"user_id": str, "tv_show_id": "int64", "last_time": "int64"})


def configurations():
    result = [{"id": "v2_control", "family": "v2"}]
    result += [{"family": "daycap", "fraction": fraction, "mass_mode": mass_mode}
               for fraction in (.5, 1.) for mass_mode in ("original", "capped")]
    result += [{"family": "neighbors", "idf": idf, "mix": mix}
               for idf in (False, True) for mix in (.15, .3, 1.)]
    return result


class HybridV3Tests(unittest.TestCase):
    def test_daily_midnight_boundary_uses_max_weight_per_day(self):
        cutoff = 2 * 86400
        # Endpoint 86400 belongs to day 0; 86401 and 172800 to day 1.
        frame = events([("10", 1, 86400), ("10", 1, 86401),
                        ("10", 1, cutoff), ("10", 2, cutoff)])
        state = build_features(frame, ["10"], [1, 2, 3, 4, 5], cutoff)
        old_weight = 2 ** (-1 / 28)
        next_weight = 2 ** (-(86400 - 1) / (28 * 86400))
        np.testing.assert_allclose(state.base.personal_matrix[0],
                                   [old_weight + next_weight + 1, 1, 0, 0, 0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(state.daily_matrix[0], [old_weight + 1, 1, 0, 0, 0],
                                   rtol=0, atol=1e-15)
        # Plain last_time//86400 would put 86400 and 86401 together, retaining
        # next_weight+1 across two days instead of the required old_weight+1.
        self.assertGreater(state.daily_matrix[0, 0], 1.9)

    def test_daily_shape_and_mass_ablation_match_hand_formula(self):
        frame = events([("10", 1, 0)] * 9 + [("10", 2, 0), ("20", 2, 0), ("30", 2, 0)])
        state = build_features(frame, ["10"], [1, 2, 3, 4, 5], 0)
        np.testing.assert_array_equal(state.daily_matrix[0], [1, 1, 0, 0, 0])
        np.testing.assert_allclose(state.base.prior, [.25, .75, 0, 0, 0])
        for fraction in (.5, 1.):
            profile = (1 - fraction) * np.array([9., 1, 0, 0, 0]) + fraction * np.array([1., 1, 0, 0, 0])
            for mode in ("original", "capped"):
                mass = 10. if mode == "original" else float(profile.sum())
                q = mass * np.sqrt(profile) / np.sqrt(profile).sum()
                self.assertAlmostEqual(q.sum(), mass)
                expected = (q + 30 * np.array([.25, .75, 0, 0, 0])) / (mass + 30)
                actual = _scores(profile[None, :], np.array([mass]), state.base.prior)
                np.testing.assert_allclose(actual[0], expected, rtol=0, atol=1e-15)
                pred, _, diag = predict_v3(state, {"family": "daycap", "fraction": fraction, "mass_mode": mode})
                self.assertEqual(pred["10"], sorted(range(1, 6), key=lambda item: (-expected[item - 1], item)))
                self.assertEqual(diag["personal_mass_sum_used"], mass)
        # At fraction=1 the same shape has mass10 versus mass2, changing the
        # influence of the fixed 30 pseudo-airing popularity prior.
        self.assertNotEqual(_scores(state.daily_matrix, np.array([10.]), state.base.prior)[0, 0],
                            _scores(state.daily_matrix, np.array([2.]), state.base.prior)[0, 0])

    def test_raw_and_idf_cosine_shrink_self_exclusion_and_profiles(self):
        frame = events([("10", 1, 0), ("10", 99, 0), ("20", 99, 0)]
                       + [("20", 2, 0)] * 4
                       + [("30", 1, 0), ("30", 3, 0), ("40", 99, 0), ("50", 4, 0)])
        state = build_features(frame, ["10"], [1, 2, 3, 4, 5], 0)
        raw_ids = [state.train_users[index] for index in state.raw_neighbor_indices[0] if index >= 0]
        self.assertEqual(raw_ids, ["20", "30"])
        np.testing.assert_allclose(state.raw_neighbor_weights[0, :2], [1 / 22, 1 / 22], rtol=0, atol=1e-15)
        # Neighbour20's sqrt(4) profile normalises to [0,1,0]; neighbour30's
        # profile is [.5,0,.5]. Repeated airings never change binary cosine.
        np.testing.assert_allclose(state.cf_raw[0], [.25, .5, .25, 0, 0], rtol=0, atol=1e-15)
        a, b, c = 1 + math.log(6 / 3), 1 + math.log(6 / 4), 1 + math.log(6 / 2)
        s20 = b / math.sqrt((a + b) * (b + c)) / 11
        s30 = a / math.sqrt((a + b) * (a + c)) / 11
        idf_ids = [state.train_users[index] for index in state.idf_neighbor_indices[0] if index >= 0]
        self.assertEqual(idf_ids, ["30", "20"])
        np.testing.assert_allclose(state.idf_neighbor_weights[0, :2], [s30, s20], rtol=0, atol=1e-15)
        expected = np.array([.5 * s30, s20, .5 * s30, 0, 0]) / (s20 + s30)
        np.testing.assert_allclose(state.cf_idf[0], expected, rtol=0, atol=1e-15)
        incorrect_squared = b ** 2 / math.sqrt((a ** 2 + b ** 2) * (b ** 2 + c ** 2)) / 11
        self.assertGreater(abs(s20 - incorrect_squared), 1e-4)
        self.assertNotIn("10", raw_ids)
        self.assertNotIn("40", raw_ids)  # Has similarity but no available profile.

    def test_top50_filters_empty_profiles_first_and_uses_exact_numeric_ids(self):
        own = "1000000000000000000008000"
        supported_ids = ["2", "10"] + [str(1000000000000000000007000 + i) for i in range(49)]
        rows = [(own, 1, 0), (own, 99, 0)]
        # Sixty users with larger cosine but no available candidate profile.
        rows += [(str(100 + i), 99, 0) for i in range(60)]
        for user in supported_ids:
            rows += [(user, 99, 0), (user, 2, 0)]
        state = build_features(events(rows), [own], [1, 2, 3, 4, 5], 0)
        chosen = [state.train_users[index] for index in state.raw_neighbor_indices[0] if index >= 0]
        self.assertEqual(chosen, sorted(supported_ids, key=int)[:50])
        self.assertEqual(chosen[:2], ["2", "10"])
        self.assertEqual(len(chosen), 50)
        self.assertNotIn(supported_ids[-1], chosen)
        np.testing.assert_allclose(state.cf_raw[0], [0, 1, 0, 0, 0], rtol=0, atol=1e-15)

    def test_unsupported_cf_uses_v2_and_zero_own_history_always_falls_back(self):
        frame = events([("10", 1, 0), ("10", 99, 0), ("20", 99, 0), ("20", 2, 0),
                        ("60", 98, 0), ("70", 98, 0), ("70", 4, 0), ("80", 5, 0)])
        users = ["10", "60", "80", "90"]
        state = build_features(frame, users, [1, 2, 3, 4, 5, 6], 0)
        control, control_fallback, _ = predict_power(state.base, 30, .5)
        self.assertTrue(state.cf_raw_supported[1])  # User60 has neighbours, but n=0.
        self.assertFalse(state.cf_raw_supported[2])  # User80 has own history only.
        for conf in configurations():
            pred, fallback, diag = predict_v3(state, conf)
            for user in ("60", "90"):
                self.assertEqual(pred[user], control[user])
                self.assertEqual(fallback[user], control_fallback[user])
                self.assertEqual(fallback[user], 5)
            if conf["family"] == "neighbors":
                self.assertEqual(pred["80"], control["80"])
                self.assertEqual(fallback["80"], control_fallback["80"])
            self.assertTrue(diag["scores_finite"])

    def test_neighbor_mixture_ranking_matches_manual_distribution(self):
        frame = events([("10", 1, 0), ("10", 99, 0), ("20", 99, 0), ("20", 2, 0)])
        state = build_features(frame, ["10"], [1, 2, 3, 4, 5, 6], 0)
        # g=[.5,.5], n=1, q=[1,0] so V2=[16/31,15/31].
        for mix in (.15, .3, 1.):
            expected = (1 - mix) * np.array([16 / 31, 15 / 31, 0, 0, 0, 0]) + mix * np.array([0., 1, 0, 0, 0, 0])
            pred, _, _ = predict_v3(state, {"family": "neighbors", "idf": False, "mix": mix})
            self.assertEqual(pred["10"], sorted(range(1, 7), key=lambda item: (-expected[item - 1], item))[:5])

    def test_exact_control_in_three_scenarios_and_read_only_reuse(self):
        cutoff = 60 * 86400
        large1, large2 = "1000000000000000000003000", "1000000000000000000004000"
        cases = [
            (events([("10", 1, 0), ("10", 1, 0), ("20", 2, 0)]), ["10", "20", "90"], 0),
            (events([(large1, 1, cutoff - 28 * 86400), (large1, 2, cutoff), (large2, 2, cutoff)]),
             [large1, large2, "90"], cutoff),
            (events([("10", 99, 0), ("20", 98, 0)]), ["10", "20", "90"], 0),
        ]
        for frame, users, end in cases:
            with self.subTest(users=users):
                before = frame.copy(deep=True)
                state = build_features(frame, users, [6, 5, 4, 3, 2, 1], end)
                expected, expected_fallback, _ = predict_power(build_state(frame, users, [6, 5, 4, 3, 2, 1], end, 28), 30, .5)
                actual, actual_fallback, _ = predict_v3(state, {"family": "v2"})
                self.assertEqual(actual, expected)
                self.assertEqual(actual_fallback, expected_fallback)
                saved = state.daily_matrix.copy()
                for conf in configurations():
                    pred, _, diag = predict_v3(state, conf)
                    self.assertTrue(diag["scores_finite"])
                    for values in pred.values():
                        self.assertEqual(len(values), 5)
                        self.assertEqual(len(set(values)), 5)
                        self.assertLessEqual(set(values), set(range(1, 7)))
                np.testing.assert_array_equal(state.daily_matrix, saved)
                for array in (state.daily_matrix, state.cf_raw, state.cf_idf, state.cf_raw_supported,
                              state.raw_neighbor_indices, state.raw_neighbor_weights):
                    self.assertFalse(array.flags.writeable)
                    self.assertTrue(np.isfinite(array).all())
                pd.testing.assert_frame_equal(frame, before)

    def test_event_user_and_candidate_permutation_invariance(self):
        frame = events([("10", 1, 86400), ("10", 1, 86401), ("10", 99, 0),
                        ("20", 99, 0), ("20", 2, 86400), ("30", 1, 86401), ("30", 3, 86401)])
        users = ["30", "10", "20", "90"]
        first = build_features(frame, users, [1, 2, 3, 4, 5, 6], 2 * 86400)
        second = build_features(frame.sample(frac=1, random_state=42), users[::-1], [6, 4, 2, 1, 5, 3], 2 * 86400)
        for conf in configurations():
            pred1, fall1, _ = predict_v3(first, conf)
            pred2, fall2, _ = predict_v3(second, conf)
            self.assertEqual(pred1, pred2)
            self.assertEqual(fall1, fall2)

    def test_empty_state_and_input_validation(self):
        state = build_features(events([]), ["90"], [8, 2, 1, 6, 5, 3], 0)
        for conf in configurations():
            pred, fallback, _ = predict_v3(state, conf)
            self.assertEqual(pred["90"], [1, 2, 3, 5, 6])
            self.assertEqual(fallback["90"], 5)
        with self.assertRaisesRegex(ValueError, "Future"):
            build_features(events([("10", 99, 1)]), ["10"], [1, 2, 3, 4, 5], 0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_features(events([("10", 1, 0), ("10", 1, 0)]).assign(air_id=0), ["10"], [1, 2, 3, 4, 5], 0)
        with self.assertRaises(TypeError):
            build_features(events([("10", 1, 0)]).assign(user_id=10.), ["10"], [1, 2, 3, 4, 5], 0)
        with self.assertRaisesRegex(ValueError, "integer strings"):
            build_features(events([("abc", 1, 0)]), ["10"], [1, 2, 3, 4, 5], 0)
        with self.assertRaisesRegex(ValueError, "one qualified"):
            build_features(events([("10", 1, 0)]).assign(count=2), ["10"], [1, 2, 3, 4, 5], 0)
        for conf in ({"family": "daycap", "fraction": 2, "mass_mode": "original"},
                     {"family": "daycap", "fraction": .5, "mass_mode": "wrong"},
                     {"family": "neighbors", "idf": False, "mix": float("nan")},
                     {"family": "wrong"}):
            with self.assertRaises(ValueError):
                predict_v3(state, conf)
        with self.assertRaises(TypeError):
            predict_v3(state, {"family": "neighbors", "idf": 1, "mix": .3})


if __name__ == "__main__":
    unittest.main()
