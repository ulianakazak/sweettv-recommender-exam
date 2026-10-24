"""Hand-worked tests for mass-preserving history power and V1 equivalence."""
import unittest

import numpy as np
import pandas as pd

from sweettv.frequency_v1 import predict_frequency
from sweettv.power_frequency_v2 import _mass_preserving_power, build_state, predict_power


def events(rows):
    result = pd.DataFrame(rows, columns=["user_id", "tv_show_id", "last_time"])
    result["air_id"] = range(len(result))
    return result.astype({"user_id": str, "tv_show_id": "int64", "last_time": "int64"})


class PowerFrequencyV2Tests(unittest.TestCase):
    def test_power_shape_preserves_mass_and_matches_hand_values(self):
        history = np.array([[1.,3.], [1.,4.], [0.,0.]])
        mass = np.array([4.,5.,0.])
        squared = _mass_preserving_power(history, mass, 2.)
        np.testing.assert_allclose(squared[0], [.4,3.6], rtol=0, atol=1e-15)
        rooted = _mass_preserving_power(history, mass, .5)
        np.testing.assert_allclose(rooted[1], [5/3,10/3], rtol=0, atol=1e-15)
        np.testing.assert_allclose(squared.sum(axis=1), mass, rtol=0, atol=1e-15)
        np.testing.assert_allclose(rooted.sum(axis=1), mass, rtol=0, atol=1e-15)
        # A power above one makes the larger component more concentrated, while
        # a power below one makes it less concentrated, at the same total mass.
        self.assertGreater(squared[0,1]/4, history[0,1]/4)
        self.assertLess(rooted[0,1]/4, history[0,1]/4)
        self.assertIs(_mass_preserving_power(history, mass, 1.), history)

    def test_shrinkage_uses_unchanged_pseudo_airing_mass(self):
        frame = events([("u",1,0), ("u",1,0), ("u",1,0), ("u",2,0),
                        ("v",2,0), ("w",2,0), ("x",2,0)])
        state = build_state(frame, ["u"], [1,2,3,4,5], 0)
        np.testing.assert_allclose(state.prior, [.2,.8,0,0,0])
        np.testing.assert_allclose(state.personal_matrix[0], [3,1,0,0,0])
        self.assertEqual(state.personal_mass[0], 4)
        # q=[3.6,.4], n=4, tau=10, g=[.2,.8] gives scores [.4,.6].
        shaped = _mass_preserving_power(state.personal_matrix, state.personal_mass, 2.)
        scores = (shaped+10*state.prior)/(state.personal_mass[:,None]+10)
        np.testing.assert_allclose(scores[0,:2], [.4,.6], rtol=0, atol=1e-15)
        pure, _, _ = predict_power(state, 0, 2)
        shrink, _, _ = predict_power(state, 10, 2)
        self.assertEqual(pure["u"][:2], [1,2])
        self.assertEqual(shrink["u"][:2], [2,1])

    def test_decay_is_summed_per_airing_before_power(self):
        cutoff = 40*86400
        frame = events([("u",1,cutoff-28*86400), ("u",1,cutoff-28*86400), ("u",2,cutoff)])
        state = build_state(frame, ["u"], [1,2,3,4,5], cutoff, half_life_days=14)
        np.testing.assert_allclose(state.personal_matrix[0], [.5,1,0,0,0], rtol=0, atol=1e-15)
        self.assertEqual(state.personal_mass[0], 1.5)
        predictions, _, _ = predict_power(state, 0, 1.5)
        self.assertEqual(predictions["u"][:2], [2,1])

    def test_power_one_exact_v1_rank_and_fallback_in_three_scenarios(self):
        large1, large2 = "1000000000000000000003000", "1000000000000000000004000"
        cutoff = 60*86400
        cases = [
            (events([("u",1,0), ("u",1,0), ("u",2,0), ("v",3,0), ("v",2,0)]),
             ["u","v","cold"], [6,5,4,3,2,1], 0, None, 100),
            (events([(large1,1,cutoff-14*86400), (large1,2,cutoff), (large2,2,cutoff-28*86400), (large2,99,cutoff)]),
             [large1,large2,"cold"], [1,2,3,4,5,6], cutoff, 14, 30),
            (events([("u",99,0), ("v",98,0)]), ["u","v","cold"], [1,2,3,4,5], 0, None, 100),
        ]
        for frame, users, candidates, end, half_life, tau in cases:
            with self.subTest(half_life=half_life, tau=tau, users=users):
                expected, expected_fallback, _ = predict_frequency(frame, users, candidates, end,
                                                                  half_life_days=half_life, prior_strength=tau)
                state = build_state(frame, users, candidates, end, half_life)
                actual, actual_fallback, diagnostics = predict_power(state, tau, 1.)
                self.assertEqual(actual, expected)
                self.assertEqual(actual_fallback, expected_fallback)
                self.assertTrue(diagnostics["power_one_exact_identity_branch"])

    def test_prior_is_distinct_users_and_unavailable_history_is_excluded(self):
        frame = events([("u",1,0)]*10 + [("a",2,0), ("b",2,0), ("c",2,0), ("u",99,0)])
        state = build_state(frame, ["u","cold"], [1,2,3,4,5], 0)
        np.testing.assert_array_equal(state.prior_counts, [1,3,0,0,0])
        self.assertEqual(state.personal_mass[0], 10.)
        self.assertEqual(state.diagnostics["available_event_rows"], 13)
        predicted, fallback, _ = predict_power(state, 100, 1.)
        self.assertEqual(predicted["u"][:2], [2,1])
        self.assertEqual(predicted["cold"], [2,1,3,4,5])
        self.assertEqual(fallback["cold"], 5)

    def test_reusable_state_is_finite_read_only_and_not_mutated(self):
        frame = events([("u",2,0), ("u",10,0), ("v",3,0)])
        before = frame.copy(deep=True)
        state = build_state(frame, ["u","cold"], [10,3,2,4,1,5], 0)
        saved = state.personal_matrix.copy()
        for power in (.5,.75,1,1.25,1.5):
            predicted, _, diagnostics = predict_power(state, 100, power)
            self.assertTrue(diagnostics["scores_finite"])
            self.assertEqual(predicted["cold"], [2,3,10,1,4])
            for values in predicted.values():
                self.assertEqual(len(values), 5)
                self.assertEqual(len(set(values)), 5)
            np.testing.assert_array_equal(state.personal_matrix, saved)
        for array in (state.personal_matrix, state.personal_mass, state.prior, state.prior_counts, state.candidate_ids):
            self.assertTrue(np.isfinite(array).all())
            self.assertFalse(array.flags.writeable)
        pd.testing.assert_frame_equal(frame, before)

    def test_empty_state_cold_fallback(self):
        state = build_state(events([]), ["cold"], [8,2,1,6,5,3], 0)
        predicted, fallback, diagnostics = predict_power(state, 1000, .5)
        self.assertEqual(predicted["cold"], [1,2,3,5,6])
        self.assertEqual(fallback["cold"], 5)
        self.assertEqual(diagnostics["maximum_mass_absolute_roundoff"], 0.)

    def test_future_duplicate_float_ids_and_invalid_parameters_rejected(self):
        with self.assertRaisesRegex(ValueError, "Future"):
            build_state(events([("u",99,101)]), ["u"], [1,2,3,4,5], 100)
        duplicate = events([("u",1,0), ("u",1,0)]).assign(air_id=0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_state(duplicate, ["u"], [1,2,3,4,5], 0)
        with self.assertRaises(TypeError):
            build_state(events([("u",1,0)]).assign(user_id=1.0), ["u"], [1,2,3,4,5], 0)
        with self.assertRaises(TypeError):
            build_state(events([("u",1,0)]), ["u"], [1.,2.,3.,4.,5.], 0)
        state = build_state(events([("u",1,0)]), ["u"], [1,2,3,4,5], 0)
        for tau, power in [(-1,1),(10,0),(10,float('nan')),(10,float('inf'))]:
            with self.subTest(tau=tau,power=power), self.assertRaises(ValueError):
                predict_power(state,tau,power)


if __name__ == "__main__":
    unittest.main()
