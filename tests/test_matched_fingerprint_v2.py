import importlib.util
import json
import pathlib
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = load_module("research/experiments/matched_fingerprint_v2/run_role_balanced.py", "role_balanced")
ARG = load_module("research/experiments/matched_fingerprint_v2/argument_profile.py", "argument_profile")
PAIR_SUPPORT = load_module("research/experiments/matched_fingerprint_v2/run_pair_support.py",
                           "pair_support")
NYU_RUN = load_module("research/experiments/matched_fingerprint_v2/run_independent_nyu_agent.py",
                       "independent_nyu_run")
NYU_ANALYSIS = load_module(
    "research/experiments/matched_fingerprint_v2/analyze_independent_nyu_agent.py",
    "independent_nyu_analysis")


class MatchedFingerprintV2Tests(unittest.TestCase):
    def test_pair_type_weights_equalize_each_label(self):
        rows = [{"mode": m} for m in ("human", "agent", "human", "agent", "agent")]
        a = [0, 0, 1, 2, 3, 4, 1, 2]
        b = [2, 1, 3, 4, 4, 3, 4, 3]
        y = [1, 1, 0, 0, 0, 0, 1, 1]
        w = RUN.pair_type_weights(rows, a, b, y)
        modes = [RUN.pair_type(rows[x]["mode"], rows[z]["mode"]) for x, z in zip(a, b)]
        for label in (0, 1):
            totals = {}
            for kind in set(modes):
                totals[kind] = sum(weight for weight, mode, cls in zip(w, modes, y)
                                   if mode == kind and cls == label)
            present = [value for value in totals.values() if value > 0]
            self.assertTrue(present)
            self.assertLess(max(present) - min(present), 1e-12)

    def test_pair_type_weights_can_be_disabled(self):
        rows = [{"mode": "human"}, {"mode": "agent"}]
        self.assertIsNone(RUN.pair_type_weights(rows, [0], [1], [1], enabled=False))

    def test_role_balancing_preserves_frozen_cth_pair_manifest(self):
        old = load_module("research/experiments/matched_fingerprint_v1/run.py", "matched_v1")
        old_rows, old_tasks, old_folds = old.build_cth()
        new_rows, new_tasks, new_folds = RUN.build_cth()
        self.assertEqual(old_tasks, new_tasks)
        self.assertEqual(old_folds, new_folds)
        old_pairs, old_roles = old.cth_pairs(old_rows, old_tasks, old_folds)
        new_pairs, new_roles = RUN.cth_pairs(new_rows, new_tasks, new_folds)
        self.assertEqual(old_pairs, new_pairs)
        self.assertEqual(old_roles, new_roles)

    def test_abstract_command_profile_redacts_argument_values(self):
        raw = "grep --output=/home/alice/secrets/foo.txt 10.1.2.3 4141414141414141 && curl -I https://secret.example/path"
        features, count = ARG.profile_for_chunks([raw])
        joined = " ".join(features).lower()
        self.assertEqual(count, len(ARG.command_heads(raw)))
        for value in ("alice", "secrets", "foo.txt", "10.1.2.3", "4141414141414141", "secret.example"):
            self.assertNotIn(value, joined)
        self.assertTrue(any(token.startswith("flag__grep__") for token in features))
        self.assertTrue(any("argtype__grep__option_value_path" == token for token in features))

    def test_pair_support_expansion_is_exhaustive_and_labelled_by_task(self):
        rows = [{"task": task, "mode": mode}
                for task in ("a", "b", "c", "d")
                for mode in ("human", "human", "agent", "agent")]
        pairs = PAIR_SUPPORT.exhaustive_pairs(rows, {"a", "b", "c", "d"})
        self.assertEqual(len(pairs), 120)
        self.assertEqual(sum(label for _, _, label in pairs), 24)
        self.assertEqual(len({(a, b) for a, b, _ in pairs}), len(pairs))
        for a, b, label in pairs:
            self.assertEqual(label, int(rows[a]["task"] == rows[b]["task"]))

    def test_pair_support_weights_match_label_and_role_type_mass(self):
        rows = [{"task": task, "mode": mode}
                for task in ("a", "b", "c", "d")
                for mode in ("human", "human", "agent", "agent")]
        pairs = PAIR_SUPPORT.exhaustive_pairs(rows, {"a", "b", "c", "d"})
        weights = PAIR_SUPPORT.pair_type_weights(rows, pairs, total_mass=120)
        labels = [label for _, _, label in pairs]
        kinds = [PAIR_SUPPORT.M.pair_type(rows[a]["mode"], rows[b]["mode"])
                 for a, b, _ in pairs]
        for label in (0, 1):
            self.assertAlmostEqual(sum(w for w, y in zip(weights, labels) if y == label), 60)
            masses = [sum(w for w, y, kind in zip(weights, labels, kinds)
                          if y == label and kind == role_type)
                      for role_type in ("HH", "HA", "AA")]
            self.assertTrue(all(abs(mass - 20) < 1e-9 for mass in masses))

    def test_nyu_gallery_builder_uses_one_positive_and_three_distinct_negative_tasks(self):
        rows = []
        manifest_tasks = []
        for partition in range(3):
            for task_index in range(4):
                task = f"p{partition}-task{task_index}"
                manifest_tasks.append({"task": task, "candidate_partition": partition})
                for run_index in range(2):
                    rows.append({"id": f"{task}-run{run_index}", "task": task,
                                 "mode": "agent"})
        pairs = NYU_RUN.external_pairs(
            rows, {"eligible_tasks": manifest_tasks, "eligible_trace_count": len(rows)},
            expected_pair_count=9_600)
        self.assertEqual(len(pairs), 9_600)  # 100 repetitions × 24 queries × 4 candidates
        grouped = {}
        for pair in pairs:
            grouped.setdefault((pair["replicate"], pair["query_id"]), []).append(pair)
        self.assertEqual(len(grouped), 2_400)
        for query_pairs in grouped.values():
            self.assertEqual(len(query_pairs), 4)
            positives = [p for p in query_pairs if p["same_task"]]
            negatives = [p for p in query_pairs if not p["same_task"]]
            self.assertEqual(len(positives), 1)
            self.assertEqual(len(negatives), 3)
            self.assertEqual(len({p["candidate_task"] for p in negatives}), 3)
            self.assertTrue(all(p["pair_type"] == "AA" for p in query_pairs))

    def test_nyu_task_macro_error_metrics(self):
        frame = pd.DataFrame([
            {"query_task": "a", "same_task": 1, "predicted_link": 1, "query_id": "a1"},
            {"query_task": "a", "same_task": 1, "predicted_link": 0, "query_id": "a2"},
            {"query_task": "a", "same_task": 0, "predicted_link": 1, "query_id": "a3"},
            {"query_task": "a", "same_task": 0, "predicted_link": 0, "query_id": "a4"},
            {"query_task": "b", "same_task": 1, "predicted_link": 1, "query_id": "b1"},
            {"query_task": "b", "same_task": 0, "predicted_link": 0, "query_id": "b2"},
        ])
        matrix, counts = NYU_ANALYSIS.metric_task_matrix(frame, ["a", "b"])
        self.assertEqual(matrix.shape, (2, 3))
        self.assertAlmostEqual(matrix[:, 0].mean(), 0.25)  # FMR
        self.assertAlmostEqual(matrix[:, 1].mean(), 0.25)  # FSR
        self.assertAlmostEqual(matrix[:, 2].mean(), 0.75)  # BA
        self.assertEqual(counts["a"]["positive_pairs"], 2)

    def test_nyu_rank_metrics_handle_ties_consistently(self):
        frame = pd.DataFrame([
            {"fold": 0, "replicate": 0, "query_id": "q", "query_task": "t",
             "same_task": label, "score": score}
            for label, score in ((1, .5), (0, .7), (0, .5), (0, .2))
        ])
        result = NYU_ANALYSIS.rank_rows(frame).iloc[0]
        self.assertEqual(result.hit1, 0.0)
        self.assertAlmostEqual(result.mrr, (1 / 2 + 1 / 3) / 2)
        self.assertAlmostEqual(result.auc, 0.5)

    def test_cth_five_share_metrics_match_existing_summary_and_pair_type_counts(self):
        results = json.loads((ROOT / "research/results/matched_fingerprint_v2/five_share/"
                             "five_share_analysis.json").read_text())
        legacy = json.loads((ROOT / "research/results/matched_fingerprint_v1/metrics.json")
                            .read_text())
        for learner in ("svm", "tree", "xgb"):
            row = results["metrics_by_share_and_pair_type"][learner]["ALL"]["0.5"]
            old = next(x for x in legacy["cth"]["methods"]
                       if x["learner"] == learner and x["variant"] == "evidence"
                       and x["fraction"] == "0.5" and x["pair_type"] == "ALL")
            for metric in ("ba", "fmr", "fsr"):
                self.assertAlmostEqual(row["evidence"]["metrics"][metric],
                                       old["metrics"][metric]["point"], places=12)
            for share in ("0.0", "0.25", "0.5", "0.75", "1.0"):
                all_counts = results["metrics_by_share_and_pair_type"][learner]["ALL"][share][
                    "counts"]["BS"]
                type_counts = [results["metrics_by_share_and_pair_type"][learner][kind][share]
                               ["counts"]["BS"] for kind in ("HH", "HA", "AA")]
                for label in ("positive_pair_decisions", "negative_pair_decisions"):
                    self.assertEqual(all_counts[label], sum(x[label] for x in type_counts))

    def test_unified_ablation_has_frozen_pair_counts_and_expected_blocks(self):
        result = json.loads((ROOT / "research/results/matched_fingerprint_v2/unified_ablation/"
                             "unified_bst_ablation.json").read_text())
        self.assertTrue(result["pair_lists_identical_within_dataset"])
        self.assertEqual(set(result["results"]["cth"]), {"BS", "S", "B", "evidence"})
        self.assertEqual(set(result["results"]["ait"]),
                         {"BST", "ST", "BT", "BS", "evidence"})
        cth = result["results"]["cth"]
        self.assertAlmostEqual(cth["BS"]["metrics"]["ba"], 0.5368, places=3)
        self.assertAlmostEqual(result["results"]["ait"]["BST"]["metrics"]["ba"],
                               0.5192, places=3)


if __name__ == "__main__":
    unittest.main()
