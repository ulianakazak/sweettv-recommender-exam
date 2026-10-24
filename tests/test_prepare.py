import unittest

import numpy as np
import pandas as pd

from sweettv.prepare import aggregate_window, conflict_indices, match_intervals, merge_intervals


class IntervalTests(unittest.TestCase):
    def test_union_nested_duplicates_and_groups(self):
        v = pd.DataFrame([[0, 1, 0, 10], [0, 1, 2, 3], [0, 1, 9, 12],
                          [0, 1, 0, 10], [1, 1, 2, 3], [0, 2, 3, 4]],
                         columns=["uid", "channel_id", "start", "end"])
        result = merge_intervals(v)
        self.assertEqual(sorted(result.itertuples(index=False, name=None)),
                         [(0, 1, 0, 12), (0, 2, 3, 4), (1, 1, 2, 3)])

    def test_interval_join_nested_endpoints_and_halfopen(self):
        # Ends 20,5,10 are not monotone; simple searchsorted(end) is invalid.
        a = pd.DataFrame([[0, 1, 0, 20, 10], [1, 1, 2, 5, 11], [2, 1, 6, 10, 12]],
                         columns=["air_id", "channel_id", "start", "end", "tv_show_id"])
        v = pd.DataFrame([[0, 1, 5, 6], [1, 1, 9, 11], [2, 2, 0, 20]],
                         columns=["uid", "channel_id", "start", "end"])
        rows = sorted(match_intervals(v, a).itertuples(index=False, name=None))
        self.assertEqual(rows, [(0, 0, 10, 5, 6, 20), (1, 0, 10, 9, 11, 20), (1, 2, 12, 9, 10, 4)])
        self.assertEqual(conflict_indices(a), {0, 1, 2})

    def test_fragment_union_before_threshold(self):
        a = pd.DataFrame([[0, 1, 0, 100, 9]], columns=["air_id", "channel_id", "start", "end", "tv_show_id"])
        v = pd.DataFrame([[0, 1, 0, 45], [0, 1, 40, 60], [0, 1, 70, 100]],
                         columns=["uid", "channel_id", "start", "end"])
        f = match_intervals(merge_intervals(v), a)
        counts, _, positive = aggregate_window(f, 0, 100, np.array(["1000000000000000000006000"], object))
        self.assertEqual(int(positive.watch_seconds.iloc[0]), 90)
        self.assertEqual(int(counts['count'].iloc[0]), 1)
        self.assertEqual(counts.user_id.iloc[0], "1000000000000000000006000")

    def test_boundary_full_denominator_and_future_perturbation(self):
        a = pd.DataFrame([[0, 1, 0, 100, 9]], columns=["air_id", "channel_id", "start", "end", "tv_show_id"])
        base = pd.DataFrame([[0, 1, 0, 50]], columns=["uid", "channel_id", "start", "end"])
        with_future = pd.concat([base, pd.DataFrame([[0, 1, 50, 100]], columns=base.columns)], ignore_index=True)
        before = aggregate_window(match_intervals(merge_intervals(base), a), 0, 50, np.array(["u"]))
        after = aggregate_window(match_intervals(merge_intervals(with_future), a), 0, 50, np.array(["u"]))
        self.assertTrue(before[0].empty)
        self.assertTrue(after[0].empty)
        self.assertEqual(int(after[1].watch_seconds.iloc[0]), 50)
        self.assertEqual(int(after[1].duration.iloc[0]), 100)
        pd.testing.assert_frame_equal(before[1], after[1])

    def test_random_join_against_brute_force(self):
        rng = np.random.default_rng(42)
        vstart = rng.integers(0, 100, 50)
        astart = rng.integers(0, 100, 40)
        v = pd.DataFrame({"uid": np.arange(50), "channel_id": rng.integers(1, 4, 50),
                          "start": vstart, "end": vstart + rng.integers(1, 30, 50)})
        a = pd.DataFrame({"air_id": np.arange(40), "channel_id": rng.integers(1, 4, 40),
                          "start": astart, "end": astart + rng.integers(1, 40, 40), "tv_show_id": np.arange(1, 41)})
        expected = []
        for row in v.itertuples(index=False):
            for air in a.itertuples(index=False):
                lo, hi = max(row.start, air.start), min(row.end, air.end)
                if row.channel_id == air.channel_id and hi > lo:
                    expected.append((row.uid, air.air_id, air.tv_show_id, lo, hi, air.end - air.start))
        self.assertEqual(sorted(expected), sorted(match_intervals(v, a).itertuples(index=False, name=None)))


if __name__ == "__main__":
    unittest.main()
