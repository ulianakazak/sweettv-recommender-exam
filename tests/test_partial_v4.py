"""Hand-computed coverage, boundary and weak/qualified separation checks."""
import unittest

import numpy as np
import pandas as pd

from sweettv.partial_features_v4 import (
    aggregate_partial_airings, build_weak_profile, eligible_weak_airings,
    qualified_event_rows,
)


FIRST, SECOND = "1000000000000000000006000", "1000000000000000000005000"


class PartialV4Tests(unittest.TestCase):
    def mapping(self):
        return pd.DataFrame({"uid": [1, 0], "user_id": [SECOND, FIRST]})

    def test_duplicate_nested_union_clipping_and_full_denominator(self):
        fragments = pd.DataFrame([
            [0, 10, 91, -40, 30, 100], [0, 10, 91, -40, 30, 100],
            [0, 10, 91, 0, 50, 100], [0, 10, 91, 50, 60, 100],
            [0, 11, 92, 0, 30, 100], [0, 11, 92, 40, 90, 100],
            [1, 12, 93, 10, 110, 100],
        ], columns=["uid", "air_id", "tv_show_id", "start", "end", "duration"])
        before = fragments.copy(deep=True)
        early, audit = aggregate_partial_airings(fragments, self.mapping(), 0, 50)
        pd.testing.assert_frame_equal(fragments, before)
        row = early.loc[early.air_id.eq(10)].iloc[0]
        self.assertEqual((row.user_id, row.watch_seconds, row.duration, row.last_time, row.coverage), (FIRST, 50, 100, 50, .5))
        self.assertGreater(audit["duplicate_seconds_removed"], 0)
        self.assertTrue(qualified_event_rows(early).empty)
        complete, _ = aggregate_partial_airings(fragments, self.mapping(), 0, 110)
        self.assertEqual(set(qualified_event_rows(complete).air_id), {11, 12})
        self.assertEqual(set(complete.user_id), {FIRST, SECOND})

    def test_future_fragments_cannot_make_earlier_partial_or_qualified_history(self):
        base = pd.DataFrame([[0, 1, 91, 0, 50, 100]], columns=["uid", "air_id", "tv_show_id", "start", "end", "duration"])
        future = pd.DataFrame([[0, 1, 91, 50, 100, 100], [1, 2, 92, 100, 200, 100]], columns=base.columns)
        a, _ = aggregate_partial_airings(base, self.mapping(), 0, 50)
        b, _ = aggregate_partial_airings(pd.concat([base, future], ignore_index=True), self.mapping(), 0, 50)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(qualified_event_rows(b).empty)

    def test_weak_thresholds_no_qualified_leak_and_hl28_hand_math(self):
        cutoff = 56 * 86400
        rows = [
            [FIRST, 0, 0, 91, 100, 200, cutoff, .5],
            [FIRST, 0, 1, 91, 100, 400, cutoff-28*86400, .25],
            [FIRST, 0, 2, 92, 80, 100, cutoff, .8],
            [FIRST, 0, 3, 93, 59, 1000, cutoff, .059],
            [SECOND, 1, 4, 94, 60, 1201, cutoff, 60/1201],
            [SECOND, 1, 5, 95, 60, 1200, cutoff, .05],
        ]
        airings = pd.DataFrame(rows, columns=["user_id", "uid", "air_id", "tv_show_id", "watch_seconds", "duration", "last_time", "coverage"])
        selected = eligible_weak_airings(airings, cutoff)
        self.assertEqual(set(selected.air_id), {0, 1, 5})
        self.assertEqual(set(qualified_event_rows(airings).air_id), {2})
        p2 = build_weak_profile(airings, cutoff, power=2).set_index(["user_id", "tv_show_id"])
        p4 = build_weak_profile(airings, cutoff, power=4).set_index(["user_id", "tv_show_id"])
        self.assertAlmostEqual(p2.loc[(FIRST, 91), "weak_weight"], .25 + .0625*.5)
        self.assertAlmostEqual(p4.loc[(FIRST, 91), "weak_weight"], .0625 + .00390625*.5)
        self.assertAlmostEqual(p2.loc[(SECOND, 95), "weak_weight"], .0025)
        self.assertEqual(int(p2.loc[(FIRST, 91), "weak_airings"]), 2)
        self.assertNotIn(92, p2.index.get_level_values("tv_show_id"))

    def test_reject_future_rows_float_ids_and_unfrozen_parameters(self):
        airings = pd.DataFrame([[FIRST, 0, 1, 91, 60, 100, 101, .6]], columns=["user_id", "uid", "air_id", "tv_show_id", "watch_seconds", "duration", "last_time", "coverage"])
        with self.assertRaises(ValueError):
            build_weak_profile(airings, 100)
        with self.assertRaises(TypeError):
            build_weak_profile(airings.assign(user_id=1.844383217889249e19), 101)
        with self.assertRaises(ValueError):
            build_weak_profile(airings, 101, power=3)
        with self.assertRaises(ValueError):
            build_weak_profile(airings, 101, half_life_days=14)


if __name__ == "__main__":
    unittest.main()
