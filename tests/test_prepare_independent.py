"""Independent invariants for preprocessing; no access to original CSV files."""
import unittest

import numpy as np
import pandas as pd

from sweettv.prepare import (
    META,
    aggregate_window,
    catalogue,
    match_intervals,
    merge_intervals,
)


VIEW_COLUMNS = ["uid", "channel_id", "start", "end"]
AIR_COLUMNS = ["air_id", "channel_id", "start", "end", "tv_show_id"]


class IndependentPreparationInvariants(unittest.TestCase):
    def test_union_is_idempotent_and_permutation_invariant(self):
        """Rereading duplicate records and input order cannot change history."""
        data = pd.DataFrame(
            [[0, 2, 20, 30], [0, 2, 0, 5], [0, 2, 4, 20],
             [0, 2, 3, 4], [0, 2, 40, 40], [0, 2, 50, 45],
             [0, 3, 10, 15], [1, 2, 10, 15]], columns=VIEW_COLUMNS)
        base = merge_intervals(data)
        expanded = pd.concat([data, data, data], ignore_index=True).sample(
            frac=1, random_state=913).reset_index(drop=True)
        pd.testing.assert_frame_equal(base, merge_intervals(expanded))
        pd.testing.assert_frame_equal(base, merge_intervals(base))
        # Same person on a different channel and a different person remain separate.
        self.assertEqual(list(base.itertuples(index=False, name=None)),
                         [(0, 2, 0, 30), (0, 3, 10, 15), (1, 2, 10, 15)])

    def test_union_conserves_observed_coverage_by_user_and_channel(self):
        """Small integer-time occupancy is an independent reference for union."""
        rng = np.random.default_rng(913)
        starts = rng.integers(-30, 30, 90)
        data = pd.DataFrame({"uid": rng.integers(0, 3, 90),
                             "channel_id": rng.integers(10, 13, 90),
                             "start": starts,
                             "end": starts + rng.integers(1, 20, 90)})
        union = merge_intervals(data)
        for key, original in data.groupby(["uid", "channel_id"]):
            expected = set()
            for row in original.itertuples(index=False):
                expected.update(range(row.start, row.end))
            selected = union.loc[(union.uid == key[0]) & (union.channel_id == key[1])]
            self.assertEqual(int((selected.end - selected.start).sum()), len(expected))
            if len(selected) > 1:
                self.assertTrue(np.all(selected.start.to_numpy()[1:] >
                                       selected.end.to_numpy()[:-1]))

    def test_frequency_counts_distinct_airings_and_not_fragments(self):
        airings = pd.DataFrame([[0, 2, 0, 100, 91], [1, 2, 100, 200, 91],
                                [2, 2, 200, 300, 92]], columns=AIR_COLUMNS)
        views = pd.DataFrame([[0, 2, 0, 30], [0, 2, 40, 100],
                              [0, 2, 110, 150], [0, 2, 150, 190],
                              [0, 2, 200, 240], [0, 2, 250, 289]],
                             columns=VIEW_COLUMNS)
        names = np.array(["1000000000000000000006000", "1000000000000000000001000"], dtype=object)
        counts, per_air, positives = aggregate_window(
            match_intervals(merge_intervals(views), airings), 0, 300, names)
        self.assertEqual(len(per_air), 3)
        self.assertEqual(len(positives), 2)
        self.assertEqual(counts.to_dict("records"), [
            {"user_id": "1000000000000000000006000", "tv_show_id": 91,
             "count": 2, "watch_seconds": 170, "last_time": 190}])

    def test_clipped_coverage_is_additive_across_adjacent_windows(self):
        """Observed seconds conserve across cutoffs; positivity need not do so."""
        airings = pd.DataFrame([[0, 2, 0, 100, 91]], columns=AIR_COLUMNS)
        views = pd.DataFrame([[0, 2, -10, 30], [0, 2, 40, 110]], columns=VIEW_COLUMNS)
        fragments = match_intervals(merge_intervals(views), airings)
        names = np.array(["1000000000000000000006000"], dtype=object)
        full, all_air, _ = aggregate_window(fragments, 0, 100, names)
        left, left_air, _ = aggregate_window(fragments, 0, 50, names)
        right, right_air, _ = aggregate_window(fragments, 50, 100, names)
        self.assertEqual(int(all_air.watch_seconds.sum()),
                         int(left_air.watch_seconds.sum() + right_air.watch_seconds.sum()))
        self.assertEqual(int(full["count"].sum()), 1)
        self.assertTrue(left.empty)
        self.assertTrue(right.empty)
        self.assertEqual(int(left_air.duration.iloc[0]), 100)
        self.assertEqual(int(right_air.duration.iloc[0]), 100)

    def test_catalogue_is_unchanged_by_future_content(self):
        """Train metadata remains identical when target content is perturbed."""
        records = []
        for item, start, title, genre in [(91, 0, "old title", "drama"),
                                           (91, 20, "", "comedy"),
                                           (92, 30, "other", "news")]:
            row = {field: "" for field in META}
            row.update(tv_show_id=item, channel_id=2, start=start, end=start + 10,
                       tv_show_title=title, tv_show_genre_1=genre)
            records.append(row)
        train = pd.DataFrame(records)
        future = train.copy()
        future["start"] = 100
        future["end"] = 110
        future["tv_show_title"] = "future-only title"
        future["tv_show_genre_1"] = "future-only genre"
        actual = catalogue(pd.concat([future, train], ignore_index=True), 100)
        pd.testing.assert_frame_equal(actual, catalogue(train, 100))
        row = actual.loc[actual.tv_show_id == 91].iloc[0]
        self.assertEqual(row.tv_show_title, "old title")
        self.assertEqual(row.tv_show_genre_1, "comedy")

    def test_no_matching_channel_returns_empty_outputs_without_dropping_interface(self):
        views = pd.DataFrame([[0, 99, 0, 100]], columns=VIEW_COLUMNS)
        airings = pd.DataFrame([[0, 2, 0, 100, 91]], columns=AIR_COLUMNS)
        fragments = match_intervals(merge_intervals(views), airings)
        counts, per_air, positives = aggregate_window(
            fragments, 0, 100, np.array(["1000000000000000000006000"], dtype=object))
        self.assertTrue(counts.empty and per_air.empty and positives.empty)
        self.assertEqual(list(counts.columns),
                         ["user_id", "tv_show_id", "count", "watch_seconds", "last_time"])


if __name__ == "__main__":
    unittest.main()
