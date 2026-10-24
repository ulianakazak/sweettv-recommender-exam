"""Hand-worked verification of the separate frequency/recency API."""
import unittest

import numpy as np
import pandas as pd

from sweettv.frequency_v1 import _airing_weights, _shrink_scores, predict_frequency


def events(rows):
    """Rows are user, programme, timestamp; each is a different qualified airing."""
    frame = pd.DataFrame(rows, columns=["user_id", "tv_show_id", "last_time"])
    frame["air_id"] = range(len(frame))
    return frame


class FrequencyV1Tests(unittest.TestCase):
    def test_half_life_hand_computation_and_ranking(self):
        cutoff = 30 * 86400
        weights = _airing_weights(np.array([cutoff, cutoff-14*86400, cutoff-28*86400]), cutoff, 14)
        np.testing.assert_allclose(weights, [1, .5, .25], rtol=0, atol=1e-15)
        frame = events([("u", 1, cutoff-28*86400), ("u", 1, cutoff-28*86400), ("u", 2, cutoff)])
        plain, _, _ = predict_frequency(frame, ["u"], [1,2,3,4,5], cutoff)
        decay, _, diag = predict_frequency(frame, ["u"], [1,2,3,4,5], cutoff, half_life_days=14)
        self.assertEqual(plain["u"][:2], [1,2])
        self.assertEqual(decay["u"][:2], [2,1])
        self.assertAlmostEqual(diag["available_personal_weight_sum"], 1.5)

    def test_shrink_mixture_matches_hand_math_and_changes_ranking(self):
        r = np.array([3., 1.])
        prior = np.array([.1, .9])
        scores = _shrink_scores(r, prior, 6.)
        np.testing.assert_allclose(scores, [.36, .64], rtol=0, atol=1e-15)
        np.testing.assert_allclose(scores, .4*(r/4)+.6*prior, rtol=0, atol=1e-15)
        self.assertAlmostEqual(float(scores.sum()), 1.)
        frame = events([("u",1,0), ("u",1,0), ("u",2,0), ("a",2,0), ("b",2,0), ("c",2,0)])
        personal, _, _ = predict_frequency(frame, ["u"], [1,2,3,4,5], 0)
        shrunk, _, _ = predict_frequency(frame, ["u"], [1,2,3,4,5], 0, prior_strength=10)
        self.assertEqual(personal["u"][:2], [1,2])
        self.assertEqual(shrunk["u"][:2], [2,1])

    def test_cold_users_get_exact_same_popularity_fallback(self):
        frame = events([("a",4,0), ("b",4,0), ("a",2,0), ("c",1,0)])
        popularity, _, _ = predict_frequency(frame, ["cold"], [1,2,3,4,5,6], 0, mode="popularity")
        personal, slots, _ = predict_frequency(frame, ["cold"], [1,2,3,4,5,6], 0)
        shrink, shrink_slots, _ = predict_frequency(frame, ["cold"], [1,2,3,4,5,6], 0, half_life_days=28, prior_strength=100)
        self.assertEqual(popularity["cold"], [4,1,2,3,5])
        self.assertEqual(personal, popularity)
        self.assertEqual(shrink, popularity)
        self.assertEqual(slots["cold"], 5)
        self.assertEqual(shrink_slots["cold"], 5)

    def test_precise_big_string_user_ids_and_float_rejection(self):
        first = "1000000000000000000003000"
        second = "1000000000000000000004000"
        frame = events([(first,2,0), (second,1,0)])
        predicted, _, _ = predict_frequency(frame, [first,second], [1,2,3,4,5], 0)
        self.assertEqual(predicted[first][0], 2)
        self.assertEqual(predicted[second][0], 1)
        with self.assertRaises(TypeError):
            predict_frequency(frame.assign(user_id=frame.user_id.astype(float)), [first], [1,2,3,4,5], 0)

    def test_future_events_rejected_even_when_item_is_unavailable(self):
        frame = events([("u",999,101)])
        with self.assertRaisesRegex(ValueError, "Future events"):
            predict_frequency(frame, ["u"], [1,2,3,4,5], 100)

    def test_availability_gates_personal_mass_and_prior(self):
        frame = events([("u",99,0), ("u",99,0), ("a",2,0), ("b",2,0), ("c",1,0)])
        predicted, slots, diag = predict_frequency(frame, ["u"], [1,2,3,4,5], 0, prior_strength=5)
        self.assertEqual(predicted["u"], [2,1,3,4,5])
        self.assertEqual(slots["u"], 5)
        self.assertEqual(diag["available_personal_weight_sum"], 0.)
        self.assertEqual(diag["prior_distinct_user_count_sum"], 3)
        self.assertEqual(diag["available_event_rows"], 3)

    def test_tied_frequency_and_popularity_use_numeric_id_order(self):
        frame = events([("u",10,0), ("u",2,0), ("v",3,0), ("v",3,0)])
        personal, _, _ = predict_frequency(frame, ["u"], [20,10,4,3,2,1], 0)
        population, _, _ = predict_frequency(frame, ["u"], [20,10,4,3,2,1], 0, mode="popularity")
        self.assertEqual(personal["u"][:2], [2,10])
        self.assertEqual(population["u"], [2,3,10,1,4])

    def test_duplicate_airing_and_aggregated_count_rejected(self):
        frame = events([("u",1,0), ("u",1,0)])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            predict_frequency(frame.assign(air_id=0), ["u"], [1,2,3,4,5], 0)
        with self.assertRaisesRegex(ValueError, "aggregated"):
            predict_frequency(frame.assign(count=2), ["u"], [1,2,3,4,5], 0)

    def test_empty_history_can_still_fill_candidates_and_does_not_mutate_input(self):
        frame = events([]).astype({"user_id":str,"tv_show_id":"int64","last_time":"int64"})
        before = frame.copy(deep=True)
        predictions, slots, diag = predict_frequency(frame, ["u"], [10,2,8,1,3,9], 0)
        self.assertEqual(predictions["u"], [1,2,3,8,9])
        self.assertEqual(slots["u"], 5)
        self.assertEqual(diag["normalised_prior_sum"], 0.)
        pd.testing.assert_frame_equal(frame,before)


if __name__ == "__main__":
    unittest.main()
