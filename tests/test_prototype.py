import unittest

from attackagent.prototype import ModeAwareAssociationPrototype, TraceRecord


class PrototypeSmokeTest(unittest.TestCase):
    def test_fit_and_score_with_grouped_trace_records(self):
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

        model = ModeAwareAssociationPrototype(seed=7).fit(records)
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


if __name__ == "__main__":
    unittest.main()
