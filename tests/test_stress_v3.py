"""Synthetic early-stress boundaries; original data and labels are not opened."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.prepare_stress_v3 import prepare_stress
from sweettv.prepare import seconds


FIRST = "1000000000000000000006000"
SECOND = "1000000000000000000005000"


class StressV3Tests(unittest.TestCase):
    def fixtures(self, target, add_future=False):
        target.mkdir()
        cutoff, end = seconds("2020-04-06"), seconds("2020-05-04")
        airings = [(0, cutoff-20, cutoff+80, 11), (1, cutoff-300, cutoff-200, 10),
                   (2, cutoff-500, cutoff-400, 10), (3, end-80, end+20, 12),
                   (4, cutoff, cutoff+100, 13), (5, cutoff-100, cutoff, 14),
                   (6, end, end+100, 15), (7, cutoff, cutoff+100, 0),
                   (8, cutoff-700, cutoff-600, 16)]
        fragments = [(0, 0, 11, cutoff-20, cutoff+80, 100),
                     (0, 1, 10, cutoff-300, cutoff-270, 100),
                     (0, 1, 10, cutoff-250, cutoff-200, 100),
                     (0, 2, 10, cutoff-500, cutoff-420, 100),
                     (0, 3, 12, end-80, end+20, 100),
                     (1, 8, 16, cutoff-700, cutoff-600, 100)]
        if add_future:
            airings.append((9, cutoff+1000, cutoff+1100, 99))
            fragments.append((0, 9, 99, cutoff+1000, cutoff+1100, 100))
        pd.DataFrame(airings, columns=["air_id", "start", "end", "tv_show_id"]).to_parquet(target / "train_schedule.parquet", index=False)
        pd.DataFrame(fragments, columns=["uid", "air_id", "tv_show_id", "start", "end", "duration"]).to_parquet(target / "matched_fragments.parquet", index=False)
        pd.DataFrame({"uid": [1, 0], "user_id": [SECOND, FIRST]}).to_csv(target / "user_map.csv", index=False)

    def test_separate_clipping_exact_threshold_full_duration_and_big_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixtures(root / "source")
            result = prepare_stress(root / "source", root / "stress")
            train = pd.read_csv(root / "stress/train_counts.csv", dtype={"user_id": str})
            valid = pd.read_csv(root / "stress/valid_counts.csv", dtype={"user_id": str})
            events = pd.read_parquet(root / "stress/train_events.parquet")
            self.assertEqual(train.loc[train.user_id.eq(FIRST), ["tv_show_id", "count"]].to_dict("records"), [{"tv_show_id": 10, "count": 2}])
            self.assertNotIn(11, train.tv_show_id.tolist())  # only 20/100 before cutoff
            self.assertEqual(set(valid.tv_show_id), {11, 12})  # exactly 80/100 each
            self.assertEqual(valid.loc[valid.tv_show_id.eq(12), "last_time"].iloc[0], seconds("2020-05-04"))
            self.assertEqual(set(events.user_id), {FIRST, SECOND})
            self.assertEqual(events.columns.tolist(), ["user_id", "uid", "air_id", "tv_show_id", "last_time"])
            self.assertFalse(events.duplicated(["user_id", "air_id"]).any())
            self.assertEqual(set(pd.read_csv(root / "stress/candidates.csv").tv_show_id), {11, 12, 13})
            self.assertTrue(result["checks"]["independent_python_union_counter_sample_matches"])
            self.assertFalse(result["quality_scores_computed"])
            self.assertTrue(result["source_hashes_unchanged"])

    def test_future_views_do_not_change_training_and_outputs_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixtures(root / "base")
            self.fixtures(root / "changed", add_future=True)
            prepare_stress(root / "base", root / "base_stress")
            prepare_stress(root / "changed", root / "changed_stress")
            self.assertEqual((root / "base_stress/train_counts.csv").read_bytes(), (root / "changed_stress/train_counts.csv").read_bytes())
            pd.testing.assert_frame_equal(pd.read_parquet(root / "base_stress/train_events.parquet"), pd.read_parquet(root / "changed_stress/train_events.parquet"))
            with self.assertRaises(FileExistsError):
                prepare_stress(root / "base", root / "base_stress")
            with self.assertRaises(ValueError):
                prepare_stress(root / "base", root / "base/new")


if __name__ == "__main__":
    unittest.main()
