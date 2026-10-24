"""Small independent checks for ranking, AP, sparse orientation and exact IDs."""
import unittest
import numpy as np
import pandas as pd
from scipy import sparse

from sweettv.evaluate import ap_at_k, build_truth, validate_submission
from sweettv.models import content_attributes, fit_model, predict, rank_with_fallback, implicit_objective, fit_implicit_als


class MetricAndRankingTests(unittest.TestCase):
    def test_hand_metric_and_duplicate(self):
        # Relevant IDs at ranks 1,3,5: numerator=1+2/3+3/5, denom=3.
        self.assertAlmostEqual(ap_at_k([10, 30, 50], [10, 20, 30, 40, 50]), (1+2/3+3/5)/3)
        self.assertAlmostEqual(ap_at_k([10], [10, 10, 20, 30, 40]), 1.0)
        self.assertEqual(ap_at_k([], [10, 20, 30, 40, 50]), 0.0)

    def test_truth_top5_and_count_id_ties(self):
        frame = pd.DataFrame({"user_id": ["u"]*6, "tv_show_id": [6,5,4,3,2,1], "count": [2,2,2,2,2,1]})
        self.assertEqual(build_truth(frame, ["u", "cold"]), {"u": [2,3,4,5,6], "cold": []})

    def test_shared_fallback_ties_and_missing_scores(self):
        ids = [6,1,3,2,4,5]
        ranked, slots = rank_with_fallback(ids, np.array([np.nan, 1, 1, 0, -1, np.nan]), [6,5,4,3,2,1], positive_only=True)
        self.assertEqual(ranked, [1,3,6,5,4])
        self.assertEqual(slots, 3)
        ranked, _ = rank_with_fallback(ids, None, [6,5,4,3,2,1])
        self.assertEqual(ranked, [6,5,4,3,2])

    def test_exact_string_user_and_template_order(self):
        users = ["1000000000000000000003000", "1000000000000000000002000"]
        frame = pd.DataFrame({"user_id": users, "tv_show_id": ["1 2 3 4 5", "5 4 3 2 1"]})
        self.assertTrue(validate_submission(frame, users, [1,2,3,4,5])["valid"])
        with self.assertRaises(ValueError):
            validate_submission(frame.iloc[::-1], users, [1,2,3,4,5])

    def test_future_metadata_not_accepted_by_prediction(self):
        # Two users and six items yield visibly different content profiles.
        counts = pd.DataFrame({"user_id": ["1000000000000000000003000", "b"], "tv_show_id": [1,2], "count": [3,2]})
        catalog = pd.DataFrame({"tv_show_id": [1,2,3,4,5,6], "tv_show_category": ["sport","news","sport","news","film","music"]})
        model = fit_model("cbf", counts, catalog, {"profile_weighting": "binary"})
        first, _ = predict(model, counts, ["1000000000000000000003000", "cold"], [1,2,3,4,5,6,7])
        self.assertEqual(first["1000000000000000000003000"][:2], [1,3])
        self.assertEqual(first["cold"], [1,2,3,4,5])
        # Prediction takes IDs only; changing target behaviour cannot alter it.
        second, _ = predict(model, counts, ["1000000000000000000003000", "cold"], [1,2,3,4,5,6,7])
        self.assertEqual(first, second)
        self.assertFalse(model.diagnostics["future_metadata_used"])

    def test_title_and_programme_id_do_not_become_content_tokens(self):
        attributes = content_attributes({"tv_show_id": 999, "tv_show_title": "future secret", "tv_show_category": "Спорт", "year_of_production": "1996-"})
        self.assertEqual(set(attributes), {"category=спорт", "decade=1990"})

    def test_implicit_objective_matches_dense_and_decreases(self):
        matrix = sparse.csr_matrix(np.array([[5.,0.,3.],[0.,8.,0.]]))
        users = np.array([[.2,.3],[.1,.5]])
        items = np.array([[.4,.2],[.1,.6],[.7,.1]])
        regularization = .1
        confidence = np.where(matrix.toarray() > 0, matrix.toarray(), 1.)
        preference = (matrix.toarray() > 0).astype(float)
        dense = np.sum(confidence*(preference-users@items.T)**2)+regularization*(np.sum(users**2)+np.sum(items**2))
        self.assertAlmostEqual(implicit_objective(matrix, users, items, regularization), dense)
        user_factors, item_factors, diag = fit_implicit_als(matrix, 2, regularization, 3, 42)
        objective = diag["objective_by_iteration"]
        self.assertTrue(all(after <= before+1e-9 for before, after in zip(objective, objective[1:])))
        self.assertEqual(user_factors.shape, (2,2))
        self.assertEqual(item_factors.shape, (3,2))


if __name__ == "__main__":
    unittest.main()
