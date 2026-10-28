#!/usr/bin/env python3
"""Check dataset, split, and reported-count invariants without exposing traces."""

import json
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


def main():
    audit = json.loads((ROOT / "results/data-audit.json").read_text(encoding="utf-8"))
    primary = json.loads((ROOT / "results/preliminary.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((ROOT / "results/sensitivity.json").read_text(encoding="utf-8"))
    trace = json.loads((ROOT / "results/trace-baseline.json").read_text(encoding="utf-8"))
    split_diagnostics = json.loads((ROOT / "results/split-diagnostics.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "results/quick-summary.json").read_text(encoding="utf-8"))

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
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
