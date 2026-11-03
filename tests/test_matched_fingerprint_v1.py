"""Contracts for the matched association experiment."""
import collections
import importlib.util
import json
import pathlib
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "matched_fingerprint", ROOT / "research/experiments/matched_fingerprint_v1/run.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class MatchedFingerprintProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.tasks, cls.folds = study.build_cth()
        cls.pairs, cls.roles = study.cth_pairs(cls.rows, cls.tasks, cls.folds)

    def test_role_allocations_are_exact_and_shared_by_gallery(self):
        roles = collections.defaultdict(list)
        for r in self.roles:
            roles[r["replicate"], r["agent_slots"]].append(r)
        self.assertEqual(len(roles), 170 * 5)
        for (rep, count), allocation in roles.items():
            self.assertEqual(len(allocation), 32)
            self.assertEqual(sum(r["mode"] == "agent" for r in allocation), count)
            self.assertEqual(len({(r["task"], r["role"]) for r in allocation}), 32)
        self.assertEqual(len(self.pairs), 170 * 5 * 16 * 4)
        for p in self.pairs:
            self.assertEqual(self.folds[p["query_task"]], p["fold"])
            self.assertEqual(self.folds[p["candidate_task"]], p["fold"])
            self.assertEqual(p["same_task"], int(p["query_task"] == p["candidate_task"]))
            self.assertNotEqual(p["query_id"], p["candidate_id"])

    def test_frozen_pair_list_reproduces_original_baselines(self):
        p = ROOT / "research/results/matched_fingerprint_v1/baseline_reproduction.json"
        if not p.exists():
            self.skipTest("run reproduce_baselines.py to produce the reproducibility audit")
        audit = json.loads(p.read_text())
        self.assertLess(audit["max_error"], 1e-12)
        self.assertEqual(len(audit["per_pair_reconstruction"]), 10)

    def test_gallery_has_one_positive_and_three_distinct_negative_tasks(self):
        rows = []
        for t in range(4):
            for j in range(3):
                rows.append({"id": f"{t}-{j}", "task": f"t{t}",
                             "mode": "human" if j == 0 else "agent",
                             "raw_hash": f"{t}-{j}"})
        pairs = study.gallery(rows, {f"t{t}" for t in range(4)}, 0, 2, 19)
        by_query = collections.defaultdict(list)
        for p in pairs:
            by_query[p["replicate"], p["query_id"]].append(p)
        self.assertEqual(len(by_query), 4 * 3 * 2)
        for group in by_query.values():
            self.assertEqual(len(group), 4)
            self.assertEqual(sum(p["same_task"] for p in group), 1)
            self.assertEqual(len({p["candidate_task"] for p in group}), 4)
            self.assertTrue(all(p["query_id"] != p["candidate_id"] for p in group))

    def test_threshold_respects_baseline_fallback(self):
        y = np.array([1, 1] + [0] * 20)
        scores = np.array([.8, .7] + [.95, .92, .85, .7] + [.1] * 16)
        fallback = np.zeros(len(y), dtype=bool)
        fallback[0] = True
        fallback[-1] = True
        base_pred = np.zeros(len(y), dtype=bool)
        base_pred[0] = True
        t = study.threshold(y, scores, fallback, base_pred)
        pred = study.decision(scores, t, fallback, base_pred)
        self.assertTrue(pred[0])
        self.assertFalse(pred[-1])
        self.assertLessEqual(pred[y == 0].mean(), .1)
        self.assertTrue(np.array_equal(pred[fallback], base_pred[fallback]))

    def test_removing_s_keeps_exact_b_columns_and_missing_mask(self):
        rows = [
            {"commands": ["ls", "cat", "ls", "id"], "actions": ["read", "read", "read", "discover"],
             "timestamps": None},
            {"commands": ["id", "ls", "cat", "cat"], "actions": ["discover", "read", "read", "read"],
             "timestamps": None},
            {"commands": ["ls"], "actions": ["read"], "timestamps": None},
        ]
        profiles = study.profile_cache(rows, 2)
        features = study.Features(rows, [0, 1], profiles, 8, "mean_std_trend")
        a, b = np.array([0, 0]), np.array([1, 2])
        evidence = features.evidence(a, b)
        both, missing_both = features.full(evidence, a, b, ("B", "S"), "conditional")
        only_b, missing_b = features.full(evidence, a, b, ("B",), "conditional")
        np.testing.assert_allclose(both[:, :only_b.shape[1]], only_b)
        np.testing.assert_array_equal(missing_both, missing_b)
        self.assertFalse(missing_both[0])
        self.assertTrue(missing_both[1])
        self.assertTrue(all(not p["T_ok"] for p in profiles[8]))


if __name__ == "__main__":
    unittest.main()
