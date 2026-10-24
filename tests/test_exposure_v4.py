"""Hand-worked timetable opportunity and original-policy tests."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from sweettv.exposure_v4 import (DAY, build_exposure_state, eligible_airings, exposure_ratio,
                                prepare_window_schedule, read_future_schedule)


def schedule(rows):
    return pd.DataFrame(rows, columns=["air_id", "channel_id", "tv_show_id", "start", "end"]).astype("int64")


class ExposureV4Tests(unittest.TestCase):
    def test_full_duration_eighty_percent_and_half_open_boundaries(self):
        data = schedule([(1, 1, 1, -20, 80), (2, 2, 2, -21, 79),
                         (3, 3, 3, 20, 120), (4, 4, 4, 21, 121),
                         (5, 5, 5, 100, 200), (6, 6, 6, -100, 0)])
        self.assertEqual(eligible_airings(data, 0, 100).air_id.tolist(), [1, 3])
        self.assertEqual(eligible_airings(data, 100, 200).air_id.tolist(), [5])

    def test_raw_channels_and_unique_simultaneous_slots_are_separate(self):
        cutoff = 56 * DAY
        data = schedule([(1, 1, 1, 40 * DAY, 40 * DAY + 100),
                         (2, 2, 1, 40 * DAY, 40 * DAY + 100),
                         (3, 3, 1, 10 * DAY, 10 * DAY + 100),
                         (4, 1, 1, 60 * DAY, 60 * DAY + 100),
                         (5, 2, 1, 60 * DAY, 60 * DAY + 100),
                         (6, 3, 2, 70 * DAY, 70 * DAY + 100)])
        state = build_exposure_state(data, [1, 2, 3, 4, 5], cutoff, 84 * DAY, train_start=0)
        self.assertEqual(state.e_long_raw[0], 3)
        self.assertEqual(state.e_last28_raw[0], 2)
        self.assertEqual(state.f_raw[0], 2)
        self.assertEqual(state.e_long_slots[0], 2)
        self.assertEqual(state.e_last28_slots[0], 1)
        self.assertEqual(state.f_slots[0], 1)
        # Raw D=.5*(2+3*28/56)=1.75; target has exactly28 days.
        self.assertAlmostEqual(exposure_ratio(state, "raw")[0], 3 / 2.75)
        self.assertAlmostEqual(exposure_ratio(state, "slots")[0], 2 / 2)
        for values in (state.e_long_raw, state.e_long_slots, state.denominator_weight):
            self.assertFalse(values.flags.writeable)

    def test_long_target_normalisation_and_complete_end_weight(self):
        cutoff = 56 * DAY
        data = schedule([(1, 1, 1, cutoff - 28 * DAY - 100, cutoff - 28 * DAY),
                         (2, 2, 1, cutoff - 100, cutoff),
                         (3, 3, 1, cutoff - 80, cutoff + 20),
                         (4, 4, 1, cutoff + DAY, cutoff + DAY + 100)])
        state = build_exposure_state(data, [1, 2, 3, 4, 5], cutoff, cutoff + 84 * DAY, train_start=0)
        self.assertAlmostEqual(state.denominator_weight[0], 1.5)
        self.assertEqual(state.probability_airings.air_id.tolist(), [1, 2])
        # Crossing row3 is train-eligible (.8), but cannot enter probability D.
        self.assertEqual(state.e_long_raw[0], 3)
        self.assertEqual(state.f_raw[0], 1)
        expected_d = .5 * (2 + 3 * .5)
        self.assertAlmostEqual(exposure_ratio(state, "raw")[0], (1 / 3 + 1) / (expected_d + 1))

    def test_same_duplicate_conflict_policy_and_stable_old_air_ids(self):
        old = schedule([(7, 1, 1, 0, 100), (8, 2, 2, 0, 100)])
        future = pd.DataFrame([(1, 1, 0, 100), (2, 3, 50, 150), (3, 4, 150, 250)],
                              columns=["channel_id", "tv_show_id", "start", "end"])
        combined, diag = prepare_window_schedule(old, 0, 200, future)
        self.assertEqual(combined.tv_show_id.tolist(), [1, 4])
        self.assertEqual(combined.loc[combined.tv_show_id == 1, "air_id"].tolist(), [7])
        self.assertEqual(combined.loc[combined.tv_show_id == 4, "end"].tolist(), [250])
        self.assertEqual(diag["exact_channel_item_time_duplicates_removed"], 1)
        self.assertEqual(diag["conflicting_labelled_rows_excluded"], 2)
        self.assertTrue(diag["old_air_ids_preserved"])

    def test_future_parser_ignores_metadata_zero_and_nonpositive_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "future.csv"
            pd.DataFrame({"channel_id": [1, 2, 3], "tv_show_id": [10, 0, 20],
                          "start_time": ["27.07.2020 00:00:00"] * 3, "duration": [100, 100, 0],
                          "actors": ["unused"] * 3}).to_csv(path, index=False)
            result = read_future_schedule(path)
        self.assertEqual(result.columns.tolist(), ["channel_id", "tv_show_id", "start", "end"])
        self.assertEqual(result.tv_show_id.tolist(), [10])
        self.assertEqual(int(result.end.iloc[0] - result.start.iloc[0]), 100)

    def test_future_scope_duplicate_ids_and_invalid_types_rejected(self):
        data = schedule([(1, 1, 1, 0, 100)])
        with self.assertRaisesRegex(ValueError, "outside"):
            build_exposure_state(schedule([(1, 1, 1, 200, 300)]), [1, 2, 3, 4, 5], 50, 100, train_start=0)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_exposure_state(schedule([(1, 1, 1, 0, 100), (1, 2, 2, 0, 100)]), [1, 2, 3, 4, 5], 50, 200, train_start=0)
        with self.assertRaises(TypeError):
            build_exposure_state(data.assign(start=0.), [1, 2, 3, 4, 5], 50, 200, train_start=0)
        with self.assertRaises(ValueError):
            exposure_ratio(build_exposure_state(data, [1, 2, 3, 4, 5], 50, 200, train_start=0), "wrong")


if __name__ == "__main__":
    unittest.main()
