import types
import unittest

import numpy as np

from scripts.run_specificity_controls import EmpiricalLengthCDF, evaluate_gallery, nonlinear_features


class SpecificityControlsTest(unittest.TestCase):
    def test_length_transform_uses_only_training_reference(self):
        cdf = EmpiricalLengthCDF().fit([1, 3, 7, 15])
        original = cdf.reference_.copy()
        self.assertEqual(cdf.transform([3])[0], .5)
        self.assertEqual(cdf.transform([3, 10**9])[0], .5)
        np.testing.assert_array_equal(original, cdf.reference_)
        np.testing.assert_array_equal(cdf.transform([0, 15, 16]), [0, 1, 1])

    def test_heldout_gallery_removes_training_negative(self):
        # A query's best-scoring negative is a training task; it must not be
        # available in the held-out-only protocol. Positive and other negative stay.
        rows = [types.SimpleNamespace(task=t, label=l, benchmark="b", source=s)
                for t, l, s in (("q", "human", "h"), ("q", "agent", "a"),
                                ("other", "agent", "a"), ("training", "agent", "a"))]
        scores = {(0, 1): {"fixed": .8}, (0, 2): {"fixed": .2}, (0, 3): {"fixed": .9}}
        full = evaluate_gallery(rows, scores, {"q", "other"}, False)
        heldout = evaluate_gallery(rows, scores, {"q", "other"}, True)
        select = lambda rr: next(r for r in rr if r["task"] == "q" and r["direction"] == "human_agent")
        self.assertEqual(select(full)["recall_at_1"], 0)
        self.assertEqual(select(full)["candidate_tasks"], 3)
        self.assertEqual(select(heldout)["recall_at_1"], 1)
        self.assertEqual(select(heldout)["candidate_tasks"], 2)
        self.assertEqual(select(heldout)["random_recall_at_1"], .5)

    def test_no_result_when_gallery_has_no_negative(self):
        rows = [types.SimpleNamespace(task="q", label=l, benchmark="b", source=l)
                for l in ("human", "agent")]
        self.assertEqual(evaluate_gallery(rows, {(0, 1): {"fixed": .8}}, {"q"}, True), [])

    def test_nonlinear_control_has_no_mode_or_length_inputs(self):
        np.testing.assert_allclose(nonlinear_features(np.array([[.2, .5]])),
                                   [[.2, .5, .04, .25, .1, .3]])


if __name__ == "__main__":
    unittest.main()
