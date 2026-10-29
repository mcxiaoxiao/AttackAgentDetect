#!/usr/bin/env python3
"""Grouped, CPU-only comparison of local Human/Agent trace classifiers.

This evaluates record labels in the pinned CTH release. Because capture source
is perfectly aligned with the labels, the metrics must not be interpreted as
general execution-mode identification accuracy.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_mode_aware_association import FOLDS, SEED, load_rows  # noqa: E402
from attackagent.traces import ATOM_GROUP  # noqa: E402

OUT_JSON = ROOT / "results" / "local_model_comparison.json"
OUT_CSV = ROOT / "results" / "local_model_comparison_task_metrics.csv"
BOOTSTRAPS = 5000
MODELS = (
    "command_tfidf_logistic",
    "action_tfidf_logistic",
    "count_action_logistic",
    "count_action_extra_trees",
)


def task_key(row):
    return f"{row.benchmark}/{row.activity_id}"


def dense_behavior_features(records):
    """Small, fixed summaries; no task text, timestamps, model names, or paths."""
    actions = tuple(sorted(set(ATOM_GROUP.values())))
    matrix = np.zeros((len(records), len(actions) + 1), dtype=np.float64)
    for i, row in enumerate(records):
        n_actions = len(row.actions)
        matrix[i, 0] = np.log1p(len(row.commands))
        counts = collections.Counter(row.actions)
        if n_actions:
            for j, action in enumerate(actions, start=1):
                matrix[i, j] = counts[action] / n_actions
    return matrix


def build_features(records, train_indices):
    command = [" ".join(row.commands) for row in records]
    action = [" ".join(row.actions) for row in records]
    output = {}
    for name, texts in (("command_tfidf_logistic", command),
                        ("action_tfidf_logistic", action)):
        vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1,
        )
        try:
            vectorizer.fit([texts[i] for i in train_indices])
            output[name] = vectorizer.transform(texts)
        except ValueError:
            from scipy.sparse import csr_matrix
            output[name] = csr_matrix((len(records), 0), dtype=np.float64)
    stats = dense_behavior_features(records)
    output["count_action_logistic"] = stats
    output["count_action_extra_trees"] = stats
    return output


def classifier(name, seed):
    if name == "count_action_extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features=1.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=seed,
    )


def expected_calibration_error(y, p, weights, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(weights.sum())
    error = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (
            p <= edges[index + 1] if index == bins - 1 else p < edges[index + 1]
        )
        if not np.any(mask):
            continue
        mass = float(weights[mask].sum()) / total
        confidence = float(np.average(p[mask], weights=weights[mask]))
        frequency = float(np.average(y[mask], weights=weights[mask]))
        error += mass * abs(confidence - frequency)
    return float(error)


def grouped_predictions(records, probabilities):
    by_task = collections.defaultdict(list)
    for index, row in enumerate(records):
        by_task[task_key(row)].append(index)
    output = {}
    for task, indices in sorted(by_task.items()):
        labels = np.asarray([int(records[i].mode == "agent") for i in indices])
        scores = np.asarray([probabilities[i] for i in indices], dtype=float)
        if set(labels) != {0, 1}:
            continue
        benchmark = records[indices[0]].benchmark
        output[task] = {
            "benchmark": benchmark,
            "indices": indices,
            "labels": labels,
            "scores": scores,
            "metrics": {
                "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= 0.5)),
                "macro_f1": float(f1_score(labels, scores >= 0.5, average="macro", zero_division=0)),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "brier": float(brier_score_loss(labels, scores)),
            },
        }
    return output


def summarize_task_sample(task_sample):
    """Aggregate a task bootstrap draw, retaining repeated sampled clusters."""
    by_benchmark = collections.defaultdict(list)
    for task in task_sample:
        by_benchmark[task["benchmark"]].append(task)
    metric_names = ("balanced_accuracy", "macro_f1", "roc_auc", "brier")
    output = {
        metric: float(np.mean([
            np.mean([task["metrics"][metric] for task in tasks])
            for tasks in by_benchmark.values()
        ]))
        for metric in metric_names
    }
    y, p, weights = [], [], []
    number_benchmarks = len(by_benchmark)
    for tasks in by_benchmark.values():
        for task in tasks:
            per_record_weight = 1.0 / (
                number_benchmarks * len(tasks) * len(task["labels"])
            )
            y.extend(task["labels"].tolist())
            p.extend(task["scores"].tolist())
            weights.extend([per_record_weight] * len(task["labels"]))
    output["ece_10_bin"] = expected_calibration_error(
        np.asarray(y), np.asarray(p), np.asarray(weights), bins=10
    )
    output["task_count"] = len(task_sample)
    return output


def summarize_predictions(records, probabilities):
    task_map = grouped_predictions(records, probabilities)
    return summarize_task_sample(list(task_map.values())), task_map


def bootstrap_model(task_map, n_boot=BOOTSTRAPS, seed=SEED):
    by_benchmark = collections.defaultdict(list)
    for task, item in task_map.items():
        by_benchmark[item["benchmark"]].append(task)
    rng = np.random.default_rng(seed)
    draws = collections.defaultdict(list)
    for _ in range(n_boot):
        sampled = []
        for benchmark in sorted(by_benchmark):
            task_ids = by_benchmark[benchmark]
            picked = rng.choice(task_ids, size=len(task_ids), replace=True)
            sampled.extend(task_map[task] for task in picked)
        summary = summarize_task_sample(sampled)
        for metric in ("balanced_accuracy", "macro_f1", "roc_auc", "brier", "ece_10_bin"):
            draws[metric].append(summary[metric])
    return {
        metric: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for metric, values in draws.items()
    }


def main():
    _observations, records, _ = load_rows()
    y = np.asarray([int(row.mode == "agent") for row in records])
    groups = np.asarray([task_key(row) for row in records])
    splitter = GroupKFold(n_splits=FOLDS)
    predictions = {name: np.full(len(records), np.nan) for name in MODELS}
    fold_audit = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(records)), groups=groups), 1
    ):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise AssertionError("A task group crossed the outer split")
        features = build_features(records, train_idx)
        fold_audit.append({
            "fold": fold,
            "train_tasks": sorted(set(groups[train_idx])),
            "test_tasks": sorted(set(groups[test_idx])),
            "train_records": len(train_idx),
            "test_records": len(test_idx),
        })
        for name in MODELS:
            model = classifier(name, SEED + fold)
            model.fit(features[name][train_idx], y[train_idx])
            predictions[name][test_idx] = model.predict_proba(features[name][test_idx])[:, 1]

    if any(np.isnan(values).any() for values in predictions.values()):
        raise AssertionError("Every trace must receive exactly one out-of-fold score")
    task_outputs, aggregate = [], {}
    for number, name in enumerate(MODELS):
        summary, task_map = summarize_predictions(records, predictions[name])
        interval = bootstrap_model(task_map, seed=SEED + number * 73)
        aggregate[name] = {**summary, "95_task_cluster_ci": interval}
        task_outputs.extend({"model": name, "task": task, "benchmark": item["benchmark"],
                             "records": len(item["labels"]), **item["metrics"]}
                            for task, item in task_map.items())

    payload = {
        "status": "completed; exploratory trace-label comparison",
        "data": {
            "dataset": "Cyber Task Horizons public traces",
            "revision": "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae",
            "records": len(records),
            "task_groups": len(set(groups)),
            "labels": dict(collections.Counter(row.mode for row in records)),
            "source_label_mismatch_count": sum(
                (row.mode == "human") != (row.source == "expert") for row in records
            ),
        },
        "protocol": {
            "outer_split": "five-fold GroupKFold by benchmark/task_id; all records for one task remain in one fold",
            "unit": "benchmark/task_id; each task metric is weighted equally within benchmark, then benchmarks are equally weighted",
            "metrics": ["balanced accuracy", "macro-F1", "ROC-AUC", "Brier score", "10-bin expected calibration error"],
            "threshold": 0.5,
            "calibration": "No calibration method fitted; Brier and ECE assess raw model probabilities and do not imply calibrated confidence.",
            "features": "command names, coarse action categories, and fixed counts/proportions only; no task text, prompts, timestamps, paths, collection file name, or model source",
            "uncertainty": f"{BOOTSTRAPS} benchmark-stratified task-cluster bootstrap resamples on fixed out-of-fold predictions; excludes retraining uncertainty",
            "seed": SEED,
        },
        "models": aggregate,
        "folds": fold_audit,
        "interpretation_limits": [
            "The labels are perfectly aligned with collection interface/source in this release, so the experiment compares published trace-label predictability and not isolated actor identity.",
            "Task-held-out evaluation does not remove the terminal-input versus structured-tool-call acquisition difference.",
            "No pretrained semantic encoder was present in the local CPU environment; this run uses transparent lexical and fixed-statistic baselines, not a semantic model.",
            "Jev and OpenAI Decisions were not called because no API credential or verifiable free allowance was available; no paid calls were made.",
            "The 40 task groups were already used in exploratory work; all results remain exploratory, not independent confirmation.",
        ],
        "api_comparators": {
            "Jev": {"status": "not_run", "reason": "No credential or verifiable free allowance in this environment."},
            "OpenAI Decisions": {"status": "not_run", "reason": "No API credential or verifiable free allowance in this environment."},
            "local_pretrained_semantic_model": {"status": "not_run", "reason": "No local encoder/runtime or cached model weights; this CPU run uses explicit command/action features."},
        },
        "software": {"python": platform.python_version(), **{
            package: importlib.metadata.version(package)
            for package in ("numpy", "scikit-learn")
        }},
        "source_script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import csv
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_outputs[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(task_outputs)
    payload["task_metrics_csv"] = {
        "path": str(OUT_CSV.relative_to(ROOT)),
        "rows": len(task_outputs),
        "sha256": hashlib.sha256(OUT_CSV.read_bytes()).hexdigest(),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
