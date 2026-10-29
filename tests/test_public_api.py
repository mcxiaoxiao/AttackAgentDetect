import unittest

from attackagent import LabeledEvidence, TraceEvidence, fit_detector, list_methods, predict_modes, rank_candidates


def evidence(trace_id, commands, *, source_format="jsonl-command-v1", actions=("inspect", "read")):
    return TraceEvidence(trace_id, source_format, "demo", tuple(commands), tuple(actions))


class PublicApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = []
        for index in range(6):
            rows.append(LabeledEvidence(evidence(f"h{index}", ("ls", "grep", "cat")), "human", f"task{index}"))
            rows.append(LabeledEvidence(evidence(f"a{index}", ("pwd", "find", "python3")), "agent", f"task{index}"))
        cls.detector = fit_detector(rows, seed=3)

    def test_mode_inference_accepts_evidence_without_any_annotation_channel(self):
        query = evidence("q", ("pwd", "find", "python3"))
        result = predict_modes(self.detector, [query])[0]
        self.assertTrue(result.available)
        self.assertFalse(result.calibrated)
        self.assertIsNotNone(result.agent_like_score)

    def test_incompatible_detector_keeps_behavior_ranking_and_gallery_intact(self):
        query = evidence("q", ("curl", "cat"), source_format="linux-auditd-v1")
        candidates = [
            evidence("same-a", ("curl", "cat"), source_format="linux-auditd-v1"),
            evidence("other", ("python3", "find"), source_format="linux-auditd-v1"),
        ]
        baseline = rank_candidates(query, candidates, method="command_cosine")
        adjusted = rank_candidates(query, candidates, method="mode_adjusted", detector=self.detector)
        self.assertEqual([row.trace_id for row in adjusted], [row.trace_id for row in baseline])
        self.assertEqual([row.trace_id for row in adjusted], ["same-a", "other"])
        self.assertTrue(all("fallback" in row.score_semantics for row in adjusted))

    def test_empty_or_oov_candidates_are_not_removed(self):
        query = evidence("q", ("ls", "grep"))
        candidates = [evidence("known", ("ls", "grep")), evidence("oov", ("brandnewtoolxyz",), actions=())]
        ranked = rank_candidates(query, candidates, method="mode_adjusted", detector=self.detector)
        self.assertEqual({row.trace_id for row in ranked}, {"known", "oov"})
        self.assertTrue(all(row.method == "command_cosine" for row in ranked))

    def test_procgrep_adapter_runs_without_treating_ids_as_labels(self):
        query = evidence("q", ("ls", "grep", "cat"), actions=("discover", "search", "read"))
        candidates = [
            evidence("near", ("ls", "grep", "cat"), actions=("discover", "search", "read")),
            evidence("far", ("curl", "python3"), actions=("network", "execute")),
        ]
        ranked = rank_candidates(query, candidates, method="procgrep_jsd")
        self.assertEqual(ranked[0].trace_id, "near")
        self.assertTrue(all(0 <= row.score <= 1 for row in ranked))

    def test_method_inventory_separates_implementations_from_publications(self):
        rows = list_methods()
        kinds = {row["kind"] for row in rows}
        self.assertIn("external_method_adapter", kinds)
        self.assertIn("external_research", kinds)
        self.assertIn("experimental_baseline", kinds)


if __name__ == "__main__":
    unittest.main()
