import json
import math
import unittest
from attackagent import TraceEvidence, ModeScore
from attackagent.profiles import FEATURE_NAMES, extract_behavior_features, build_behavior_profile


class BehaviorProfileTest(unittest.TestCase):
    def test_empty_and_singleton_have_explicit_missing_transition_evidence(self):
        for commands in ((), ("ls",)):
            values = extract_behavior_features(commands, ())
            self.assertEqual(len(values), 24)
            self.assertEqual(set(values), set(FEATURE_NAMES))
            self.assertIsNone(values["adjacent_command_repeat"])
            self.assertIsNone(values["command_half_jsd"])
            self.assertEqual(values["short_trace_flag"], 1)
            self.assertTrue(all(x is None or math.isfinite(x) for x in values.values()))
            json.dumps(values, allow_nan=False)

    def test_repeated_and_diverse_sequences_have_expected_statistics(self):
        repeated = extract_behavior_features(("ls",) * 6, ("discover",) * 6)
        self.assertEqual(repeated["command_entropy_bits"], 0)
        self.assertEqual(repeated["adjacent_command_repeat"], 1)
        self.assertEqual(repeated["command_bigram_repeat"], 4 / 5)
        self.assertEqual(repeated["command_singleton_bigram_rate"], 0)
        self.assertEqual(repeated["max_action_run_share"], 1)
        self.assertEqual(repeated["action_run_cv"], 0)
        self.assertEqual(repeated["command_half_jsd"], 0)
        changed = extract_behavior_features(("ls", "ls", "cat", "cat"), ("discover", "discover", "read", "read"))
        self.assertEqual(changed["command_half_jsd"], 1)
        self.assertEqual(changed["action_half_jsd"], 1)
        self.assertEqual(changed["command_entropy_bits"], 1)

    def test_singletons_count_occurrences_not_only_distinct_support(self):
        # aa occurs twice; ab and ba once each: singleton occurrence fraction = 2/4.
        values = extract_behavior_features(("a", "a", "b", "a", "a"), ())
        self.assertEqual(values["command_singleton_bigram_rate"], .5)

    def test_profile_does_not_learn_identity_or_truth_from_metadata(self):
        first = TraceEvidence("a", "test", "first-task", ("ls", "cat"), ("discover", "read"))
        other = TraceEvidence("b", "other-format", "second-task", first.commands, first.actions)
        a, b = build_behavior_profile(first), build_behavior_profile(other)
        self.assertEqual(a["feature_values"], b["feature_values"])
        self.assertEqual(a["mode_estimate"]["prediction"], "unknown")
        self.assertNotIn("benchmark", a)
        self.assertNotIn("mode", a)
        self.assertNotIn("commands", a)
        self.assertNotIn("actions", a)
        score = ModeScore("b", "agent", .9, False, True, None, 2, "test")
        with self.assertRaises(ValueError):
            build_behavior_profile(first, score)

    def test_missing_action_stream_does_not_create_zero_behavior(self):
        values = extract_behavior_features(("ls", "cat"), ())
        self.assertIsNone(values["action_entropy_bits"])
        unknown = extract_behavior_features(("ls", "cat"), ("custom-action",) * 2)
        self.assertIsNone(unknown["action_coverage"])
        self.assertEqual(unknown["action_top1_share"], 1)


if __name__ == "__main__":
    unittest.main()
