#!/usr/bin/env python3
"""Check dataset, split, and reported-count invariants without exposing traces."""

import json
import hashlib
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attackagent.traces import load_observations  # noqa: E402
from scripts.run_exploration import SEED, split_tasks  # noqa: E402


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def metrics_from_confusion(matrix):
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    recall_human = tn / (tn + fp) if tn + fp else 0.0
    recall_agent = tp / (tp + fn) if tp + fn else 0.0
    f1_human = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_agent = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return (recall_human + recall_agent) / 2, (f1_human + f1_agent) / 2


def verify_sha256(relative_path, expected):
    path = ROOT / relative_path
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, f"SHA-256 mismatch: {relative_path}"


def main():
    audit = json.loads((ROOT / "results/data-audit.json").read_text(encoding="utf-8"))
    primary = json.loads((ROOT / "results/preliminary.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((ROOT / "results/sensitivity.json").read_text(encoding="utf-8"))
    trace = json.loads((ROOT / "results/trace-baseline.json").read_text(encoding="utf-8"))
    split_diagnostics = json.loads((ROOT / "results/split-diagnostics.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "results/quick-summary.json").read_text(encoding="utf-8"))
    local_models = json.loads((ROOT / "results/local_model_comparison.json").read_text(encoding="utf-8"))
    permutation = json.loads((ROOT / "results/mode_label_permutation.json").read_text(encoding="utf-8"))
    unseen_model = json.loads((ROOT / "results/unseen_agent_model_generalization.json").read_text(encoding="utf-8"))
    specificity = json.loads((ROOT / "results/specificity_controls.json").read_text(encoding="utf-8"))

    assert audit["revision"] == primary["dataset_revision"]
    assert audit["human_files"] == 48 and audit["human_samples"] == 47
    assert audit["agent_samples"] == 194
    assert audit["agent_samples_by_model"] == {
        "claude-opus-4.6": 74,
        "claude-sonnet-4.6": 74,
        "gpt-5.5": 46,
    }
    assert audit["matched_task_count"] == 31
    assert (audit["matched_human_samples"], audit["matched_agent_samples"]) == (45, 93)

    observations = load_observations()
    partitions = split_tasks(observations)
    assert len(partitions["train"]) == 16
    assert len(partitions["validation"]) == 5
    assert len(partitions["test"]) == 6
    assert not (partitions["train"] & partitions["validation"])
    assert not (partitions["train"] & partitions["test"])
    assert not (partitions["validation"] & partitions["test"])
    assert primary["seed"] == SEED
    assert primary["matched_nonempty_tasks"] == 27
    assert primary["python_tool_calls_excluded"] == 291

    test = primary["splits"]["test"]
    assert test["task_count"] == 6
    assert test["human_samples"] == 9 and test["agent_samples"] == 18
    assert sensitivity["test_observation_count"] == 27
    assert sensitivity["test_task_count"] == 6
    assert sensitivity["bootstrap_resamples"] == 5000

    for method, result in primary["association"].items():
        common = result["downstream_test"]["common"]
        assert common["pairs"] == 351 and common["same_task_pairs"] == 48
        bacc, _ = metrics_from_confusion(common["confusion_matrix_different_same"])
        assert close(bacc, common["balanced_accuracy"]), method

    assert trace["test_task_count"] == 6 and trace["test_samples"] == 27
    bacc, macro_f1 = metrics_from_confusion(trace["metrics"]["confusion_matrix_human_agent"])
    assert close(bacc, trace["metrics"]["balanced_accuracy"])
    assert close(macro_f1, trace["metrics"]["macro_f1"])

    assert split_diagnostics["matched_nonempty_task_count"] == 27
    assert split_diagnostics["session_count"] == 121
    assert split_diagnostics["repeats"] == 50
    assert all(row["overlap_task_count"] == 0
               for row in split_diagnostics["task_group_split"]["per_split"])
    assert split_diagnostics["random_session_split"]["mean_train_test_task_overlap"] > 0
    assert summary["revision"] == audit["revision"]
    assert summary["sessions"] == {"human": 47, "model_agent": 194, "script": 0}
    assert summary["test"]["held_out_tasks"] == 6
    assert close(summary["test"]["mode_classifier"]["balanced_accuracy"],
                 round(primary["classification"]["test"]["balanced_accuracy"], 3))
    assert close(summary["test"]["same_task_retrieval"]["auc"],
                 round(primary["association"]["command_cosine"]["test"]["agent-human"]["pair_auc"], 3))

    # Validate the extended task-macro classifier comparison and its provenance.
    assert local_models["data"]["records"] == 177
    assert local_models["data"]["task_groups"] == 40
    assert local_models["data"]["source_label_mismatch_count"] == 0
    assert local_models["protocol"]["uncertainty"].startswith("5000 benchmark-stratified")
    assert len(local_models["folds"]) == 5
    assert all(len(set(fold["train_tasks"]) & set(fold["test_tasks"])) == 0
               for fold in local_models["folds"])
    assert local_models["models"]["command_tfidf_logistic"]["task_count"] == 40
    assert local_models["models"]["command_tfidf_logistic"]["balanced_accuracy"] > max(
        result["balanced_accuracy"] for name, result in local_models["models"].items()
        if name != "command_tfidf_logistic"
    )
    verify_sha256("scripts/compare_trace_mode_models.py", local_models["source_script_sha256"])
    verify_sha256(local_models["task_metrics_csv"]["path"], local_models["task_metrics_csv"]["sha256"])

    # Keep the post-hoc permutation result tied to the fixed-fold reproduction.
    assert permutation["data"]["records"] == 177
    assert permutation["data"]["benchmark_task_groups"] == 40
    assert permutation["protocol"]["permutations"] == 999
    assert permutation["failed_permutations"] == 0
    assert permutation["fixed_evaluation_reproduction"]["status"] == "passed"
    assert permutation["fixed_evaluation_reproduction"]["max_absolute_metric_difference"] == 0.0
    assert permutation["permutation_summary"]["full_gallery"]["permutations_at_or_above_observed"] == 5
    assert permutation["permutation_summary"]["heldout_gallery"]["permutations_at_or_above_observed"] == 323
    for gallery, key in (("full_gallery", "full_gallery"), ("heldout_gallery", "heldout_gallery")):
        permuted = permutation["observed"][gallery]["mode_minus_similarity_delta"]
        paired = specificity["paired_deltas"][f"{key}|human_agent|mode_aware_minus_similarity_only"]["mean_delta"]["recall_at_1"]
        assert close(permuted, paired), f"Permutation observed result differs from specificity controls: {gallery}"
    verify_sha256("scripts/run_mode_label_permutation.py", permutation["source_script_sha256"])
    verify_sha256(permutation["task_metrics_csv"]["path"], permutation["task_metrics_csv"]["sha256"])

    # Confirm that unseen-source stress tests use the same trace manifest and grouped split.
    assert unseen_model["data"]["traces"] == 177
    assert unseen_model["data"]["matched_task_groups"] == 40
    assert unseen_model["data"]["trace_manifest_sha256"] == permutation["data"]["record_manifest_sha256"]
    assert unseen_model["data"]["capture_path_label_audit"]["mismatches"] == 0
    assert set(unseen_model["data"]["agent_models"]) == {"claude-opus-4.6", "claude-sonnet-4.6", "gpt-5.5"}
    verify_sha256("scripts/run_unseen_agent_model.py", unseen_model["source_script_sha256"])
    verify_sha256(unseen_model["outputs"]["task_metrics_csv"], unseen_model["outputs"]["task_metrics_csv_sha256"])

    print(json.dumps({
        "status": "passed",
        "dataset_revision": audit["revision"],
        "human_samples": audit["human_samples"],
        "agent_samples": audit["agent_samples"],
        "model_counts": audit["agent_samples_by_model"],
        "nonempty_matched_tasks": primary["matched_nonempty_tasks"],
        "split_sizes": {key: value["task_count"] for key, value in primary["splits"].items()},
        "test_pairs_per_association_method": 351,
        "bootstrap_resamples": sensitivity["bootstrap_resamples"],
        "trace_test_balanced_accuracy": trace["metrics"]["balanced_accuracy"],
        "split_diagnostic_repeats": split_diagnostics["repeats"],
        "task_group_split_overlap": split_diagnostics["task_group_split"]["mean_train_test_task_overlap"],
        "extended_results": {
            "local_model_comparison": "passed; 4 models, 40 task groups, source hash and CSV hash verified",
            "mode_label_permutation": "passed; 999 permutations and fixed-evaluation reproduction verified",
            "unseen_agent_source": "passed; trace manifest, source hash, and task-metric hash verified",
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
