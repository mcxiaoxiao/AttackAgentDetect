import unittest

import numpy as np

from attackagent.prototype import ModeAwareAssociationPrototype, TraceRecord


class PrototypeSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = []
        for task in range(5):
            for n in range(2):
                records.append(TraceRecord(
                    trace_id=f"h-{task}-{n}", benchmark="synthetic-test", activity_id=f"task-{task}",
                    mode="human", commands=("ls", "grep", f"cat{task}"),
                    actions=("discover", "search", "read"), source=f"operator-{n}",
                ))
            for n in range(2):
                records.append(TraceRecord(
                    trace_id=f"a-{task}-{n}", benchmark="synthetic-test", activity_id=f"task-{task}",
                    mode="agent", commands=("pwd", "find", f"strings{task}"),
                    actions=("environment", "discover", "read"), source=f"model-{n}",
                ))

        cls.records = records
        cls.model = ModeAwareAssociationPrototype(seed=7).fit(records)

    def test_fit_and_score_with_grouped_trace_records(self):
        model, records = self.model, self.records
        modes = model.predict_mode(records[:2])
        self.assertEqual(len(modes), 2)
        self.assertTrue(all(0 <= row["agent_like_score"] <= 1 for row in modes))
        scores = model.predict_pair_scores(records[:4], [(0, 2), (0, 1)])
        self.assertEqual(set(scores), {"similarity_only", "mode_aware"})
        self.assertTrue(all(len(values) == 2 for values in scores.values()))
        self.assertTrue(all(0 <= value <= 1 for values in scores.values() for value in values))
        self.assertGreater(model.training_pair_counts_["same_activity"], 0)
        self.assertGreater(model.training_pair_counts_["different_activity"], 0)

    def test_script_label_is_not_silently_treated_as_human_or_agent(self):
        record = TraceRecord("x", "b", "t", "script", ("ls",), ("discover",))
        with self.assertRaisesRegex(ValueError, "human or agent"):
            ModeAwareAssociationPrototype().fit([record] * 8)

    def test_empty_and_unseen_commands_abstain_without_a_numeric_score(self):
        records = [
            TraceRecord("empty", "new", "new", "unknown", (), ()),
            TraceRecord("unseen", "new", "new", "unknown", ("unseencommandxyz",), ("read",)),
            TraceRecord("script", "new", "new", "script", ("ls",), ("discover",)),
        ]
        predictions = self.model.predict_mode(iter(records))
        self.assertEqual([r["abstention_reason"] for r in predictions], [
            "empty_commands", "out_of_vocabulary", "unsupported_mode_class",
        ])
        self.assertEqual([r["command_count"] for r in predictions], [0, 1, 1])
        for prediction in predictions:
            self.assertEqual(prediction["supported_prediction"], "unknown")
            self.assertIsNone(prediction["agent_like_score"])
            self.assertIsNone(prediction["confidence"])
        self.assertEqual([r["recognized_feature_count"] for r in predictions], [0, 0, 1])
        self.assertEqual(self.model.predict_mode([]), [])

    def test_evidence_gate_preserves_known_input_scores(self):
        predictions = self.model.predict_mode(self.records)
        raw_scores = self.model.mode_classifier_.predict_proba(
            self.model.mode_vectorizer_.transform([" ".join(r.commands) for r in self.records])
        )[:, 1]
        np.testing.assert_array_equal([r["agent_like_score"] for r in predictions], raw_scores)
        self.assertTrue(all(r["abstention_reason"] is None for r in predictions))

    def test_pair_scoring_rejects_missing_evidence_even_with_supplied_probabilities(self):
        records = self.records[:2] + [TraceRecord("empty", "new", "new", "unknown", (), ())]
        for probabilities in (None, [0.2, 0.3, 0.5]):
            with self.subTest(probabilities=probabilities):
                with self.assertRaisesRegex(ValueError, "insufficient command evidence at record indices 2"):
                    self.model.predict_pair_scores(records, [(0, 2)], agent_probabilities=probabilities)
        # An unused empty record must not disrupt supported pairs.
        with_empty = self.model.predict_pair_scores(records, [(0, 1)])
        reference = self.model.predict_pair_scores(records[:2], [(0, 1)])
        for method in reference:
            np.testing.assert_array_equal(with_empty[method], reference[method])

    def test_probability_vectors_must_match_records_and_be_finite(self):
        invalid = ([0.1], [[0.1], [0.2]], [float("nan"), 0.1], [float("inf"), 0.1],
                   [-0.1, 0.2], [0.2, 1.1], [None, 0.2])
        for argument in ("agent_probabilities", "oracle_probabilities"):
            for values in invalid:
                with self.subTest(argument=argument, values=values):
                    with self.assertRaisesRegex(ValueError, argument):
                        self.model.predict_pair_scores(self.records[:2], [(0, 1)], **{argument: values})

    def test_empty_pair_lists_and_invalid_indices(self):
        result = self.model.predict_pair_scores([], [], oracle_probabilities=[])
        self.assertEqual(set(result), {"similarity_only", "mode_aware", "oracle_mode_upper_bound"})
        self.assertTrue(all(len(values) == 0 for values in result.values()))
        for pair in ((-1, 0), (0, 2), (0,), (0, 1.0), (False, 1)):
            with self.subTest(pair=pair):
                with self.assertRaisesRegex(ValueError, "valid non-negative record indices"):
                    self.model.predict_pair_scores(self.records[:2], [pair])


if __name__ == "__main__":
    unittest.main()
