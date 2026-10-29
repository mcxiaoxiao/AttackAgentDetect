#!/usr/bin/env python3
"""Conditional mode-label permutation test for mode-aware trace association.

Labels are shuffled among records within each benchmark/task group, preserving
the human/Agent counts for every task. The task-level split and association
pair labels remain fixed. This tests whether trace-specific mode supervision
adds signal beyond task composition; it is not a causal test of APT activity.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import sys
import time
from dataclasses import replace

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attackagent.prototype import ModeAwareAssociationPrototype  # noqa: E402
from scripts.run_mode_aware_association import (  # noqa: E402
    FOLDS,
    SEED,
    load_rows,
    macro,
)
from scripts.run_specificity_controls import evaluate_gallery  # noqa: E402

DEFAULT_PERMUTATIONS = 999
PROTOCOLS = ("full_gallery", "heldout_gallery")


def task_group(row):
    return (row.benchmark, row.activity_id)


def mode_labels_by_group(records):
    groups = collections.defaultdict(list)
    for index, row in enumerate(records):
        groups[task_group(row)].append(index)
    return groups


def permute_modes_within_task(records, train_indices, rng):
    """Shuffle only training labels within task, retaining each task's counts."""
    labels = np.asarray([row.mode for row in records], dtype=object)
    train_indices = set(int(i) for i in train_indices)
    for indices in mode_labels_by_group(records).values():
        block = [i for i in indices if i in train_indices]
        if not block:
            continue
        shuffled = labels[block].copy()
        rng.shuffle(shuffled)
        labels[block] = shuffled
    return labels


def assert_group_counts_preserved(records, permuted_labels, train_indices):
    train_indices = set(int(i) for i in train_indices)
    for indices in mode_labels_by_group(records).values():
        block = [i for i in indices if i in train_indices]
        expected = collections.Counter(records[i].mode for i in block)
        observed = collections.Counter(str(permuted_labels[i]) for i in block)
        if observed != expected:
            raise AssertionError("Permutation changed the human/Agent count in a task group")


def build_fold_states(raw, records):
    """Fit fixed task folds and preserve association pairs across permutations."""
    grouped = np.asarray([row.task for row in raw])
    splitter = GroupKFold(n_splits=FOLDS)
    states = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(records)), groups=grouped), 1
    ):
        train = [records[i] for i in train_idx]
        train_tasks = {raw[i].task for i in train_idx}
        test_tasks = {raw[i].task for i in test_idx}
        if train_tasks & test_tasks:
            raise AssertionError("A task group crossed the outer split")

        model = ModeAwareAssociationPrototype(seed=SEED + fold).fit(train)
        train_pairs = model._training_pairs(train)
        train_pair_labels = np.asarray([pair[2] for pair in train_pairs])
        if len(np.unique(train_pair_labels)) != 2:
            raise AssertionError("Fixed association training pairs need both labels")

        command_matrix = model.command_vectorizer_.transform(
            model._text(records, "commands")
        )
        if hasattr(model.action_vectorizer_, "vocabulary_"):
            action_matrix = model.action_vectorizer_.transform(
                model._text(records, "actions")
            )
        else:
            action_matrix = csr_matrix((len(records), 0), dtype=float)

        test_set = set(test_idx)
        test_pairs = [
            (i, j)
            for i in range(len(records))
            for j in range(i + 1, len(records))
            if i in test_set or j in test_set
        ]
        pair_payload = [
            [train[a].trace_id, train[b].trace_id, int(label)]
            for a, b, label, *_ in train_pairs
        ]
        states.append({
            "fold": fold,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train": train,
            "model": model,
            "train_pairs": train_pairs,
            "train_pair_labels": train_pair_labels,
            "command_matrix": command_matrix,
            "action_matrix": action_matrix,
            "test_pairs": test_pairs,
            "test_tasks": test_tasks,
            "train_task_count": len(train_tasks),
            "training_pair_sha256": hashlib.sha256(
                json.dumps(pair_payload, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest(),
        })
    return states


def verify_fixed_evaluation(raw, records, states):
    """Recreate the saved real-label metrics before entering permutation runs."""
    reference = json.loads(
        (ROOT / "results/specificity_controls.json").read_text(encoding="utf-8")
    )
    expected = {
        (row["protocol"], row["fold"], row["task"], row["method"]): row
        for row in reference["task_metrics"]
        if row["direction"] == "human_agent"
        and row["method"] in ("similarity_only", "mode_aware")
    }
    actual = {}
    for state in states:
        model = state["model"]
        mode_prob = np.full(len(records), np.nan, dtype=float)
        mode_prob[state["train_idx"]] = model.training_mode_oof_
        mode_prob[state["test_idx"]] = [
            row["agent_like_score"]
            for row in model.predict_mode([records[i] for i in state["test_idx"]])
        ]
        scores = model.predict_pair_scores(
            records, state["test_pairs"], agent_probabilities=mode_prob
        )
        score_map = {
            pair: {method: float(values[index]) for method, values in scores.items()}
            for index, pair in enumerate(state["test_pairs"])
        }
        for protocol, heldout_only in (("full_gallery", False), ("heldout_gallery", True)):
            for row in evaluate_gallery(raw, score_map, state["test_tasks"], heldout_only):
                if row["direction"] != "human_agent" or row["method"] not in (
                    "similarity_only", "mode_aware"
                ):
                    continue
                key = (protocol, state["fold"], row["task"], row["method"])
                actual[key] = row
    if set(actual) != set(expected):
        raise AssertionError("Fixed-fold query/method rows differ from specificity controls")
    metrics = ("recall_at_1", "mrr", "query_auc", "random_recall_at_1", "candidate_tasks")
    max_difference = max(
        abs(actual[key][metric] - expected[key][metric])
        for key in actual
        for metric in metrics
    )
    if max_difference > 1e-12:
        raise AssertionError(f"Fixed evaluation differs from reference by {max_difference}")
    return {
        "status": "passed",
        "rows": len(actual),
        "max_absolute_metric_difference": float(max_difference),
    }


def predict_permuted_mode_scores(records, labels, state):
    """Produce task-OOF training scores and an outer-fold test score."""
    permuted_records = [
        replace(row, mode=str(labels[i])) for i, row in enumerate(records)
    ]
    train_idx = state["train_idx"]
    test_idx = state["test_idx"]
    fold = state["fold"]
    mode_model = ModeAwareAssociationPrototype(seed=SEED + fold)
    train_records = [permuted_records[i] for i in train_idx]
    oof = mode_model._crossfit_mode(train_records)
    test_scores, _, _ = ModeAwareAssociationPrototype._fit_mode(
        permuted_records, train_idx, test_idx, SEED + fold
    )
    scores = np.full(len(records), np.nan, dtype=float)
    scores[train_idx] = oof
    scores[test_idx] = test_scores
    if np.isnan(scores).any():
        raise AssertionError("Permutation mode scores are missing")
    return scores, oof


def evaluate_one_permutation(raw, records, labels, states):
    paired_rows, fold_audits = [], []
    for state in states:
        model = state["model"]
        train = state["train"]
        train_pairs = state["train_pairs"]
        mode_scores, mode_oof = predict_permuted_mode_scores(records, labels, state)

        train_x = model._pair_features(
            train,
            train_pairs,
            mode_oof,
            model.command_matrix_,
            model.action_matrix_,
            True,
        )
        ranker = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=3000,
            random_state=SEED + state["fold"],
        ).fit(train_x, state["train_pair_labels"])

        test_x = model._pair_features(
            records,
            state["test_pairs"],
            mode_scores,
            state["command_matrix"],
            state["action_matrix"],
            True,
        )
        test_scores = ranker.predict_proba(test_x)[:, 1]
        baseline_x = model._pair_features(
            records,
            state["test_pairs"],
            mode_scores,
            state["command_matrix"],
            state["action_matrix"],
            False,
        )
        baseline_scores = model.match_baseline_.predict_proba(baseline_x)[:, 1]
        score_map = {
            pair: {"permuted_mode": float(mode_score), "similarity_only": float(base_score)}
            for pair, mode_score, base_score in zip(
                state["test_pairs"], test_scores, baseline_scores
            )
        }
        for protocol, heldout_only in (("full_gallery", False), ("heldout_gallery", True)):
            evaluated = evaluate_gallery(
                raw, score_map, state["test_tasks"], heldout_only=heldout_only
            )
            metric_rows = [
                {"protocol": protocol, "fold": state["fold"], **item}
                for item in evaluated
                if item["direction"] == "human_agent"
                and item["method"] in ("permuted_mode", "similarity_only")
            ]
            by_task = collections.defaultdict(dict)
            for row in metric_rows:
                by_task[row["task"]][row["method"]] = row
            if not by_task or any(set(methods) != {"permuted_mode", "similarity_only"}
                                  for methods in by_task.values()):
                raise AssertionError("Permutation and baseline query sets differ")
            for task, methods in sorted(by_task.items()):
                mode_row = methods["permuted_mode"]
                base_row = methods["similarity_only"]
                paired_rows.append({
                    "protocol": protocol,
                    "fold": state["fold"],
                    "task": task,
                    "benchmark": mode_row["benchmark"],
                    "permuted_recall_at_1": mode_row["recall_at_1"],
                    "similarity_recall_at_1": base_row["recall_at_1"],
                    "delta_recall_at_1": mode_row["recall_at_1"] - base_row["recall_at_1"],
                    "permuted_mrr": mode_row["mrr"],
                    "similarity_mrr": base_row["mrr"],
                    "delta_mrr": mode_row["mrr"] - base_row["mrr"],
                    "permuted_query_auc": mode_row["query_auc"],
                    "similarity_query_auc": base_row["query_auc"],
                    "delta_query_auc": mode_row["query_auc"] - base_row["query_auc"],
                    "candidate_tasks": mode_row["candidate_tasks"],
                    "random_recall_at_1": mode_row["random_recall_at_1"],
                })
        train_idx = state["train_idx"]
        target_payload = sorted((records[i].trace_id, str(labels[i])) for i in train_idx)
        fold_audits.append({
            "fold": state["fold"],
            "training_tasks": state["train_task_count"],
            "training_records": len(train_idx),
            "training_mode_targets_sha256": hashlib.sha256(
                json.dumps(target_payload, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest(),
            "fixed_association_pairs_sha256": state["training_pair_sha256"],
        })
    return paired_rows, fold_audits


def aggregate_replicate(rows):
    result = {}
    for protocol in PROTOCOLS:
        subset = [row for row in rows if row["protocol"] == protocol]
        if not subset:
            raise AssertionError(f"No Human-to-Agent rows for {protocol}")
        result[protocol] = {
            "task_count": len({row["task"] for row in subset}),
            "benchmark_count": len({row["benchmark"] for row in subset}),
            **{
                metric: macro(subset, metric)
                for metric in (
                    "delta_recall_at_1",
                    "delta_mrr",
                    "delta_query_auc",
                )
            },
            "mean_permuted_recall_at_1": macro(subset, "permuted_recall_at_1"),
            "mean_similarity_recall_at_1": macro(subset, "similarity_recall_at_1"),
            "mean_candidate_tasks": macro(subset, "candidate_tasks"),
            "random_recall_at_1": macro(subset, "random_recall_at_1"),
        }
    return result


def observed_metrics():
    path = ROOT / "results/specificity_controls.json"
    controls = json.loads(path.read_text(encoding="utf-8"))
    expected = {"full_gallery": 40, "heldout_gallery": 36}
    observed = {}
    for protocol, task_count in expected.items():
        item = controls["retrieval"][f"{protocol}|human_agent|mode_aware"]
        delta = controls["paired_deltas"][
            f"{protocol}|human_agent|mode_aware_minus_similarity_only"
        ]
        if item["task_count"] != task_count:
            raise AssertionError(f"Unexpected observed task count for {protocol}")
        observed[protocol] = {
            "task_count": task_count,
            "recall_at_1": item["benchmark_macro"]["recall_at_1"],
            "task_cluster_95_ci": item["task_stratified_95_ci"]["recall_at_1"],
            "mode_minus_similarity_delta": delta["mean_delta"]["recall_at_1"],
            "paired_delta_task_cluster_95_ci": delta["task_stratified_95_ci"]["recall_at_1"],
            "random_recall_at_1": item["benchmark_macro"]["random_recall_at_1"],
            "mean_candidate_tasks": item["benchmark_macro"]["candidate_tasks"],
        }
    return observed


def summarize_permutations(replicates, observed):
    summary = {}
    for protocol in PROTOCOLS:
        values = np.asarray([row[protocol]["delta_recall_at_1"] for row in replicates])
        obs = observed[protocol]["mode_minus_similarity_delta"]
        summary[protocol] = {
            "permuted_delta_mean": float(values.mean()),
            "permuted_delta_median": float(np.median(values)),
            "permuted_delta_95_reference_interval": [
                float(v) for v in np.quantile(values, [0.025, 0.975])
            ],
            "observed_mode_minus_similarity_delta": float(obs),
            "observed_minus_permuted_delta_mean": float(obs - values.mean()),
            "empirical_upper_tail_fraction": float((1 + np.sum(values >= obs)) / (len(values) + 1)),
            "permutations_at_or_above_observed": int(np.sum(values >= obs)),
            "permutation_count": int(len(values)),
        }
    return summary


def main():
    wall_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED + 1701)
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")

    raw, records, _ = load_rows()
    groups = mode_labels_by_group(records)
    raw_groups = {(row.benchmark, row.task) for row in raw}
    record_groups = {(row.benchmark, row.activity_id) for row in records}
    if record_groups != raw_groups or len(groups) != len(raw_groups):
        raise AssertionError("Task-group audit count differs from loader audit")
    for key, indices in groups.items():
        if {records[i].mode for i in indices} != {"human", "agent"}:
            raise AssertionError(f"Permutation group is not paired: {key}")

    states = build_fold_states(raw, records)
    fixed_reproduction = verify_fixed_evaluation(raw, records, states)
    outer_folds = []
    for state in states:
        outer_folds.append({
            "fold": state["fold"],
            "training_task_ids": sorted({raw[i].task for i in state["train_idx"]}),
            "heldout_task_ids": sorted({raw[i].task for i in state["test_idx"]}),
            "training_trace_ids": sorted(records[i].trace_id for i in state["train_idx"]),
            "heldout_trace_ids": sorted(records[i].trace_id for i in state["test_idx"]),
            "fixed_association_pairs_sha256": state["training_pair_sha256"],
        })
    expected_task_counts = {"full_gallery": 40, "heldout_gallery": 36}
    observed = observed_metrics()
    rng = np.random.default_rng(args.seed)
    replicate_summaries, task_rows = [], []
    for permutation in range(args.permutations):
        started = time.perf_counter()
        permuted_rows, fold_audits = [], []
        for state in states:
            labels = permute_modes_within_task(records, state["train_idx"], rng)
            assert_group_counts_preserved(records, labels, state["train_idx"])
            fold_rows, fold_audit = evaluate_one_permutation(
                raw, records, labels, [state]
            )
            permuted_rows.extend(fold_rows)
            fold_audits.extend(fold_audit)
        aggregate = aggregate_replicate(permuted_rows)
        for protocol, count in expected_task_counts.items():
            if aggregate[protocol]["task_count"] != count:
                raise AssertionError(
                    f"Eligible tasks changed under permutation for {protocol}"
                )
        replicate_summaries.append({
            "permutation": permutation + 1,
            "elapsed_seconds": time.perf_counter() - started,
            "fold_audits": fold_audits,
            **aggregate,
        })
        task_rows.extend(
            {"permutation": permutation + 1, **row} for row in permuted_rows
        )
        if (permutation + 1) % 10 == 0 or permutation + 1 == args.permutations:
            print(f"Permutation {permutation + 1}/{args.permutations} complete", flush=True)

    summary = summarize_permutations(replicate_summaries, observed)
    dataset_manifest = sorted(
        (row.benchmark, row.task, record.trace_id, record.mode, record.source)
        for row, record in zip(raw, records)
    )
    package_versions = {}
    for package in ("numpy", "scipy", "scikit-learn"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"
    csv_path = ROOT / "results/mode_label_permutation_task_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(task_rows)
    csv_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    result = {
        "status": "post-hoc conditional permutation diagnostic; not a causal test",
        "data": {
            "dataset_revision": "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae",
            "records": len(records),
            "benchmark_task_groups": len(groups),
            "training_task_group_label_counts_preserved": True,
            "record_manifest_sha256": hashlib.sha256(
                json.dumps(dataset_manifest, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "protocol": {
            "permutation_unit": "for each outer fold, shuffle only training trace labels within each benchmark/task group; each group is wholly assigned to train or test; held-out labels remain untouched",
            "preserved": ["benchmark/task membership", "human/Agent count per training task", "association pair labels", "outer task folds", "candidate sets", "query/candidate grouping"],
            "refit_per_permutation": ["nested task-OOF mode classifier", "mode-aware association ranker"],
            "association_training_pairs": "fixed to the original mode-stratified pairs to isolate the mode-score feature; not resampled or relabeled",
            "gallery": "same two protocols as the specificity controls; Human-to-Agent queries only",
            "statistic": "primary: Human-to-Agent benchmark-macro Recall@1 paired delta (permuted mode-aware ranker minus similarity-only); compare this delta with the observed real-label delta",
            "primary_protocol": "full_gallery; heldout_gallery is a secondary robustness protocol",
            "target_mapping": "one shuffled mapping per outer-fold training set is reused for inner OOF and final mode-head fitting; no held-out mode label enters model fitting",
            "seed": args.seed,
            "permutations": args.permutations,
            "rng": "numpy.default_rng (PCG64); one seeded stream across permutations and folds",
        },
        "outer_folds": outer_folds,
        "observed": observed,
        "fixed_evaluation_reproduction": fixed_reproduction,
        "permutation_summary": summary,
        "permutation_compute_seconds": float(sum(row["elapsed_seconds"] for row in replicate_summaries)),
        "run_wall_seconds": float(time.perf_counter() - wall_started),
        "failed_permutations": 0,
        "software": {"python": platform.python_version(), **package_versions},
        "task_metrics_csv": {
            "path": "results/mode_label_permutation_task_metrics.csv",
            "rows": len(task_rows),
            "sha256": csv_sha256,
        },
        "interpretation_limits": [
            "The result tests trace-specific mode-label alignment conditional on task composition; it does not estimate a causal effect of Agent execution.",
            "Mode labels remain confounded with collection pathway, so a small permutation p-value cannot distinguish execution behavior from logger or interface artifacts.",
            "The reported upper-tail fraction is an exploratory training-label negative control, not a population-level p-value; it depends on within-task exchangeability.",
            f"This {args.permutations}-permutation run has empirical-tail resolution 1/{args.permutations + 1}; all 40 task groups have already been explored, so the result remains post-hoc rather than independent confirmation.",
            "Permutation reference intervals describe the shuffled-label distribution, not confidence intervals for a population effect.",
        ],
        "source_script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "replicates": replicate_summaries,
    }
    (ROOT / "results/mode_label_permutation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
