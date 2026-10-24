"""Synthetic boundary checks for new rolling folds; no original CSV access."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from sweettv.prepare import seconds
from sweettv.rolling_v1 import FOLDS, prepare_folds


BIG_A = "1000000000000000000006000"
BIG_B = "1000000000000000000005000"


class RollingV1Tests(unittest.TestCase):
    def fixtures(self, directory: Path, add_future: bool = False):
        directory.mkdir()
        cutoff = seconds("2020-05-04")
        june = seconds("2020-06-01")
        schedule = [
            (0, cutoff-100, cutoff, 10),
            (1, cutoff-300, cutoff-200, 10),
            (2, cutoff-30, cutoff+70, 11),
            (3, cutoff, cutoff+100, 12),
            (4, cutoff-500, cutoff-400, 13),
            (5, june-80, june+20, 14),
            (6, cutoff-600, cutoff-500, 15),
            (7, cutoff-20, cutoff+80, 0),
            (8, june, june+100, 16),
        ]
        fragments = [
            (0, 0, 10, cutoff-80, cutoff, 100),
            (0, 1, 10, cutoff-300, cutoff-270, 100),
            (0, 1, 10, cutoff-250, cutoff-200, 100),
            (0, 2, 11, cutoff-30, cutoff+70, 100),
            (0, 3, 12, cutoff+20, cutoff+100, 100),
            (0, 4, 13, cutoff-500, cutoff-421, 100),
            (1, 5, 14, june-80, june+20, 100),
            (1, 6, 15, cutoff-600, cutoff-500, 100),
        ]
        if add_future:
            schedule.append((9, june+1000, june+1100, 99))
            fragments.append((0, 9, 99, june+1000, june+1100, 100))
        pd.DataFrame(fragments, columns=["uid", "air_id", "tv_show_id", "start", "end", "duration"]).to_parquet(directory / "matched_fragments.parquet", index=False)
        pd.DataFrame(schedule, columns=["air_id", "start", "end", "tv_show_id"]).to_parquet(directory / "train_schedule.parquet", index=False)
        # Deliberately unsorted map and adjacent huge IDs catch float coercion.
        pd.DataFrame({"uid": [1, 0], "user_id": [BIG_B, BIG_A]}).to_csv(directory / "user_map.csv", index=False)

    def test_boundaries_full_denominator_distinct_airings_and_big_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixtures(root / "prepared")
            summary = prepare_folds(root / "prepared", root / "rolling")
            self.assertTrue(summary["source_hashes_unchanged"])
            self.assertEqual(set(summary["folds"]), {x[0] for x in FOLDS})
            train = pd.read_csv(root / "rolling/fold_a/train_counts.csv", dtype={"user_id": str})
            self.assertEqual(train.loc[train.user_id.eq(BIG_A), ["tv_show_id", "count"]].to_dict("records"), [{"tv_show_id": 10, "count": 2}])
            events = pd.read_parquet(root / "rolling/fold_a/train_events.parquet")
            self.assertEqual(set(events.user_id), {BIG_A, BIG_B})
            self.assertEqual(set(events.loc[events.user_id.eq(BIG_A), "air_id"]), {0, 1})
            self.assertFalse(events.duplicated(["user_id", "air_id"]).any())
            valid = pd.read_csv(root / "rolling/fold_a/valid_counts.csv", dtype={"user_id": str})
            # 70/100 from airing 2 does not qualify: denominator is not clipped.
            self.assertNotIn(11, valid.tv_show_id.tolist())
            self.assertEqual(set(valid.tv_show_id), {12, 14})
            self.assertEqual(valid.loc[valid.tv_show_id.eq(14), "last_time"].iloc[0], seconds("2020-06-01"))
            candidates = set(pd.read_csv(root / "rolling/fold_a/candidates.csv").tv_show_id)
            self.assertEqual(candidates, {11, 12, 14})
            # The 100-second airing crossing May 4 is positive in later full
            # training history, but must not become a fold-a train positive.
            later = pd.read_csv(root / "rolling/fold_b/train_counts.csv", dtype={"user_id": str})
            self.assertIn(11, later.tv_show_id.tolist())

    def test_future_behaviour_cannot_change_earlier_training_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixtures(root / "base")
            self.fixtures(root / "changed", add_future=True)
            prepare_folds(root / "base", root / "base_folds")
            prepare_folds(root / "changed", root / "changed_folds")
            for fold in ["fold_a", "fold_b"]:
                pd.testing.assert_frame_equal(
                    pd.read_csv(root / f"base_folds/{fold}/train_counts.csv", dtype={"user_id": str}),
                    pd.read_csv(root / f"changed_folds/{fold}/train_counts.csv", dtype={"user_id": str}))

    def test_rejects_overwrite_and_wrong_full_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixtures(root / "prepared")
            with self.assertRaisesRegex(ValueError, "separate"):
                prepare_folds(root / "prepared", root / "prepared/new")
            prepare_folds(root / "prepared", root / "folds")
            with self.assertRaises(FileExistsError):
                prepare_folds(root / "prepared", root / "folds")
            fragments = pd.read_parquet(root / "prepared/matched_fragments.parquet")
            fragments.loc[0, "duration"] = 80
            fragments.to_parquet(root / "prepared/matched_fragments.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "denominator"):
                prepare_folds(root / "prepared", root / "other_folds")


if __name__ == "__main__":
    unittest.main()
