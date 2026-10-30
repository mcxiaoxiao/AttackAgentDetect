#!/usr/bin/env python3
"""Unseen-agent-model transfer with benchmark/task-disjoint folds."""
from __future__ import annotations
import collections, csv, datetime as dt, hashlib, json, pathlib, subprocess, sys
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_mixed_gallery as gallery  # noqa: E402
from run_feature_ablation import grouped_folds, make_design, calculate_task_metrics, aggregate_task_metrics, task_bootstrap, BASE_SEED  # noqa: E402

OUT = ROOT / "results" / "agent_model_transfer_v1.json"
TASK_CSV = ROOT / "results" / "agent_model_transfer_tasks_v1.csv"
PRED_CSV = ROOT / "results" / "agent_model_transfer_predictions_v1.csv"
FEATURES = ("F1_command_tfidf", "F2_sequence_structure", "F3_command_plus_structure")


def run(bootstraps=5000):
    rows = gallery.load_rows()
    if len(rows) != 177 or len({(r.benchmark, r.task) for r in rows}) != 40:
        raise SystemExit("Pinned CTH cohort changed; stop and re-audit before model transfer")
    models = sorted({r.source for r in rows if r.label == "agent"})
    task_metrics, predictions, fold_audit = [], [], []
    for model_index, heldout_model in enumerate(models):
        folds = grouped_folds(rows, BASE_SEED + 900 + model_index)
        for fold_index, (train_keys, test_keys) in enumerate(folds, 1):
            agent_test = [r for r in rows if r.label == "agent" and r.source == heldout_model
                          and (r.benchmark, r.task) in test_keys]
            heldout_tasks = {(r.benchmark, r.task) for r in agent_test}
            human_test = [r for r in rows if r.label == "human" and (r.benchmark, r.task) in heldout_tasks]
            paired_tasks = {(r.benchmark, r.task) for r in human_test} & heldout_tasks
            test_rows = [r for r in (*agent_test, *human_test) if (r.benchmark, r.task) in paired_tasks]
            train_rows = [r for r in rows if (r.benchmark, r.task) in train_keys
                          and not (r.label == "agent" and r.source == heldout_model)]
            if {(r.benchmark, r.task) for r in train_rows} & {(r.benchmark, r.task) for r in test_rows}:
                raise AssertionError("Task leakage in unseen-model evaluation")
            if len({r.label for r in train_rows}) < 2 or len({r.label for r in test_rows}) < 2:
                fold_audit.append({"heldout_model": heldout_model, "fold": fold_index,
                                   "status": "not_evaluable_missing_class", "n_train": len(train_rows), "n_test": len(test_rows)})
                continue
            y_train = np.asarray([int(r.label == "agent") for r in train_rows])
            y_test = np.asarray([int(r.label == "agent") for r in test_rows])
            for feature in FEATURES:
                X_train, X_test, names = make_design(train_rows, test_rows, feature, BASE_SEED + fold_index + model_index * 10)
                model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                                           random_state=BASE_SEED + fold_index + model_index * 10)
                model.fit(X_train, y_train)
                probabilities = model.predict_proba(X_test)[:, 1]
                for metric in calculate_task_metrics(test_rows, y_test, probabilities):
                    task_metrics.append({"heldout_model": heldout_model, "fold": fold_index,
                                         "feature": feature, **metric})
                for i, row in enumerate(test_rows):
                    predictions.append({"heldout_model": heldout_model, "fold": fold_index,
                                        "feature": feature, "trace_id": row.trace_id,
                                        "benchmark": row.benchmark, "task": row.task,
                                        "label_for_evaluation_only": row.label,
                                        "probability": float(probabilities[i])})
            fold_audit.append({"heldout_model": heldout_model, "fold": fold_index, "status": "evaluated",
                               "n_train": len(train_rows), "n_test": len(test_rows),
                               "train_task_groups": len({(r.benchmark, r.task) for r in train_rows}),
                               "test_task_groups": len(paired_tasks), "vocabulary_train_only": True,
                               "heldout_model_excluded_from_training": True,
                               "human_test_rows_reused_across_model_panels": True})

    summaries = []
    for heldout_model in models:
        model_index = models.index(heldout_model)
        for feature in FEATURES:
            selected = [r for r in task_metrics if r["heldout_model"] == heldout_model and r["feature"] == feature]
            # Average duplicate task outcomes across folds only if any task is repeated (normally each occurs once).
            by_task = collections.defaultdict(list)
            for row in selected:
                by_task[(row["benchmark"], row["task"])].append(row)
            averaged = [{"benchmark": b, "task": task, **{
                key: float(np.mean([r[key] for r in values]))
                for key in ("balanced_accuracy", "macro_f1", "auroc", "human_recall", "agent_recall")
            }} for (b, task), values in sorted(by_task.items())]
            summaries.append({"heldout_model": heldout_model, "feature": feature,
                              "metrics": aggregate_task_metrics(averaged) if averaged else None,
                              "ci95_task_cluster": {key: task_bootstrap(averaged, key, draws=bootstraps,
                                                                         seed=BASE_SEED + i + model_index * 100)
                                                   for i, key in enumerate(("balanced_accuracy", "macro_f1", "auroc", "human_recall", "agent_recall"))}
                              if averaged else None,
                              "task_groups": len(averaged), "records": sum(r["n"] for r in selected)})

    raw_manifest = ROOT / "data" / "raw" / "cyber-task-horizons" / "download_manifest.json"
    cross_manifest = ROOT / "data" / "manifests" / "crossbenchmark_public_subset.json"
    result = {
        "status": "completed exploratory task-disjoint unseen-model transfer",
        "protocol_version": "agent-model-transfer-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"name": "Cyber Task Horizons public CTF traces", "revision": "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae",
                    "license": "CC BY 4.0", "records": len(rows), "task_groups": 40,
                    "mode_capture_confound": "Human terminal records and Agent tool-call records remain capture-path confounded."},
        "evaluation": {"split": "three benchmark-stratified grouped folds; no benchmark-qualified task appears in both train and test",
                       "test": "held-out Agent model records plus same-task human records only; all other Agent models and training tasks form the training set",
                       "human_row_reuse": "a human record can appear in the test panel for each held-out Agent model; panels are dependent and are not pooled as independent trials",
                       "features": "command TF-IDF, fixed sequence structure, their concatenation; balanced logistic regression",
                       "metrics": "task-macro and equal benchmark weighting; 5,000 task-cluster bootstrap draws within benchmark",
                       "interpretation": "This tests transfer across the three CTH Agent model sources under task separation, but cannot separate model effects from tool-call versus terminal capture."},
        "fold_audit": fold_audit, "summary": summaries,
        "provenance": {"git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "input_manifest_sha256": {"intercode": hashlib.sha256(raw_manifest.read_bytes()).hexdigest(),
                                                  "crossbenchmark": hashlib.sha256(cross_manifest.read_bytes()).hexdigest()},
                       "code_sha256": {"runner": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                                       "features": hashlib.sha256((ROOT / "scripts" / "run_feature_ablation.py").read_bytes()).hexdigest(),
                                       "parser": hashlib.sha256((ROOT / "attackagent" / "traces.py").read_bytes()).hexdigest()},
                       "python": sys.version.split()[0], "numpy": np.__version__},
        "interpretation_guard": "A held-out Agent model score is not an APT or actor-identification result; source-label and capture path are still confounded.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path, records in ((TASK_CSV, task_metrics), (PRED_CSV, predictions)):
        if records:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--bootstraps", type=int, default=5000); args = parser.parse_args()
    print(json.dumps(run(args.bootstraps), ensure_ascii=False, indent=2))
