#!/usr/bin/env python3
"""Task-held-out feature ablation and matched-budget/order diagnostics.

The CTH labels are execution-source labels confounded with capture pathway.
All scores are therefore exploratory source-label separability on CTF traces.
Raw traces remain local; outputs contain aggregate metrics and trace IDs only.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
import sys
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, recall_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "scripts"))
import run_mixed_gallery as gallery  # noqa: E402
from attackagent import traces as trace_module  # noqa: E402

OUT = ROOT / "results" / "feature_ablation_v1.json"
TASK_CSV = ROOT / "results" / "feature_ablation_task_metrics_v1.csv"
PRED_CSV = ROOT / "results" / "feature_ablation_predictions_v1.csv"
FIG_DIR = ROOT / "results" / "figures"
BASE_SEED = 20261030
SPLIT_SEEDS = [BASE_SEED, *(BASE_SEED + 100 * i for i in range(1, 6))]
GROUPS = sorted(set(trace_module.ATOM_GROUP.values()))
FEATURES = ("F0_count_action", "F1_command_tfidf", "F2_sequence_structure", "F3_command_plus_structure")
CLASSIFIERS = ("logistic", "extra_trees")


def sequence_features(row):
    commands = list(row.names)
    actions = list(row.groups)
    n = len(commands)
    if n == 0:
        return np.zeros(10, dtype=float)
    cmd_pairs = list(zip(commands, commands[1:]))
    act_pairs = list(zip(actions, actions[1:]))

    def repeat_rate(pairs):
        return (len(pairs) - len(set(pairs))) / len(pairs) if pairs else 0.0

    def runs(values):
        if not values:
            return []
        lengths = []
        current = 1
        for left, right in zip(values, values[1:]):
            if left == right:
                current += 1
            else:
                lengths.append(current)
                current = 1
        lengths.append(current)
        return lengths

    action_runs = runs(actions)
    adjacent_action_repeat = sum(a == b for a, b in act_pairs) / len(act_pairs) if act_pairs else 0.0
    action_switch_rate = 1.0 - adjacent_action_repeat if act_pairs else 0.0
    return np.asarray([
        sum(a == b for a, b in zip(commands, commands[1:])) / max(1, n - 1),
        adjacent_action_repeat,
        action_switch_rate,
        len(set(commands)) / n,
        float(np.mean(action_runs)) / n,
        float(max(action_runs)) / n,
        repeat_rate(cmd_pairs),
        repeat_rate(act_pairs),
        float(n < 2),  # undefined transition statistics are explicitly marked
    ], dtype=float)


def f0_features(row):
    n = len(row.names)
    counts = collections.Counter(row.groups)
    return np.asarray([np.log1p(n), *[counts.get(group, 0) / max(1, n) for group in GROUPS]], dtype=float)


def grouped_folds(rows, seed, n_folds=3):
    by_benchmark = collections.defaultdict(set)
    for row in rows:
        by_benchmark[row.benchmark].add(row.task)
    fold_by_key = {}
    for bidx, benchmark in enumerate(sorted(by_benchmark)):
        tasks = sorted(by_benchmark[benchmark])
        rng = np.random.default_rng(seed + bidx)
        rng.shuffle(tasks)
        for index, task in enumerate(tasks):
            fold_by_key[(benchmark, task)] = index % n_folds
    folds = []
    for fold in range(n_folds):
        test_keys = {key for key, assigned in fold_by_key.items() if assigned == fold}
        train_keys = set(fold_by_key) - test_keys
        assert not train_keys & test_keys
        folds.append((train_keys, test_keys))
    return folds


def classifier(name, seed):
    if name == "logistic":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed)
    return ExtraTreesClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=seed, n_jobs=1,
    )


def make_design(train_rows, score_rows, feature, seed):
    if feature == "F0_count_action":
        train = np.vstack([f0_features(row) for row in train_rows])
        score = np.vstack([f0_features(row) for row in score_rows])
        return train, score, ["log1p_count", *[f"action:{g}" for g in GROUPS]]
    if feature == "F2_sequence_structure":
        train = np.vstack([sequence_features(row) for row in train_rows])
        score = np.vstack([sequence_features(row) for row in score_rows])
        return train, score, [
            "adjacent_command_repeat", "adjacent_action_repeat", "action_switch_rate",
            "unique_command_ratio", "mean_action_run_share", "max_action_run_share",
            "command_bigram_repeat", "action_bigram_repeat", "short_trace_flag",
        ]
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform([gallery.text_commands(row) for row in train_rows])
    X_score = vectorizer.transform([gallery.text_commands(row) for row in score_rows])
    if feature == "F1_command_tfidf":
        return X_train, X_score, [f"tfidf:{x}" for x in vectorizer.get_feature_names_out()]
    # F3 appends fixed, non-normalized sequence statistics to the vocabulary model.
    Z_train = np.vstack([sequence_features(row) for row in train_rows])
    Z_score = np.vstack([sequence_features(row) for row in score_rows])
    return hstack([X_train, csr_matrix(Z_train)], format="csr"), hstack([X_score, csr_matrix(Z_score)], format="csr"), [
        *[f"tfidf:{x}" for x in vectorizer.get_feature_names_out()],
        "adjacent_command_repeat", "adjacent_action_repeat", "action_switch_rate",
        "unique_command_ratio", "mean_action_run_share", "max_action_run_share",
        "command_bigram_repeat", "action_bigram_repeat", "short_trace_flag",
    ]


def calculate_task_metrics(rows, y_true, probabilities):
    by_task = collections.defaultdict(list)
    for idx, row in enumerate(rows):
        by_task[(row.benchmark, row.task)].append(idx)
    output = []
    for (benchmark, task), ids in sorted(by_task.items()):
        y = np.asarray([y_true[i] for i in ids])
        p = np.asarray([probabilities[i] for i in ids])
        pred = (p >= 0.5).astype(int)
        if len(np.unique(y)) != 2:
            continue
        output.append({
            "benchmark": benchmark, "task": task,
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "auroc": float(roc_auc_score(y, p)),
            "human_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
            "agent_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
            "n": len(ids),
        })
    return output


def aggregate_task_metrics(task_rows):
    grouped = collections.defaultdict(list)
    for row in task_rows:
        grouped[row["benchmark"]].append(row)
    metrics = ("balanced_accuracy", "macro_f1", "auroc", "human_recall", "agent_recall")
    return {
        metric: float(np.mean([np.mean([r[metric] for r in grouped[b]]) for b in sorted(grouped)]))
        for metric in metrics
    }


def task_bootstrap(task_rows, metric, draws=5000, seed=BASE_SEED):
    by_benchmark = collections.defaultdict(list)
    for row in task_rows:
        by_benchmark[row["benchmark"]].append(row[metric])
    if not by_benchmark:
        return None
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for i in range(draws):
        means = [np.mean(rng.choice(values, len(values), replace=True)) for values in by_benchmark.values()]
        samples[i] = np.mean(means)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def eligible_rows(rows, minimum=1):
    candidate = [row for row in rows if len(row.names) >= minimum]
    by_task = collections.defaultdict(set)
    for row in candidate:
        by_task[(row.benchmark, row.task)].add(row.label)
    matched = {key for key, labels in by_task.items() if labels == {"human", "agent"}}
    return [row for row in candidate if (row.benchmark, row.task) in matched]


def truncate(row, count):
    from dataclasses import replace
    return replace(row, names=tuple(row.names[:count]), groups=tuple(row.groups[:count]))


def evaluate_transfer(train_rows, test_rows, *, transfer_type, heldout_value, seed):
    """Fit only on non-held-out groups; score fixed features on the held-out domain."""
    records = []
    y_train = np.asarray([int(r.label == "agent") for r in train_rows])
    y_test = np.asarray([int(r.label == "agent") for r in test_rows])
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return [{"transfer_type": transfer_type, "heldout": heldout_value, "status": "not_evaluable_missing_class",
                 "n_train": len(train_rows), "n_test": len(test_rows)}]
    for feature in ("F1_command_tfidf", "F2_sequence_structure", "F3_command_plus_structure"):
        X_train, X_test, names = make_design(train_rows, test_rows, feature, seed)
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        per_task = calculate_task_metrics(test_rows, y_test, probabilities)
        aggregate = aggregate_task_metrics(per_task)
        records.append({
            "transfer_type": transfer_type, "heldout": heldout_value, "status": "evaluated",
            "feature": feature, "classifier": "logistic", "n_train": len(train_rows), "n_test": len(test_rows),
            "train_task_groups": len({(r.benchmark, r.task) for r in train_rows}),
            "test_task_groups": len({(r.benchmark, r.task) for r in test_rows}),
            "vocabulary_size": len(names), "metrics": aggregate,
            "test_benchmark_counts": dict(collections.Counter(r.benchmark for r in test_rows)),
            "task_metrics": per_task,
        })
    return records


def run_transfer_diagnostics(rows):
    records = []
    benchmarks = sorted({r.benchmark for r in rows})
    for index, heldout in enumerate(benchmarks):
        test = [r for r in rows if r.benchmark == heldout]
        train = [r for r in rows if r.benchmark != heldout]
        records.extend(evaluate_transfer(train, test, transfer_type="leave_one_benchmark",
                                         heldout_value=heldout, seed=BASE_SEED + 500 + index))
    models = sorted({r.source for r in rows if r.label == "agent"})
    for index, heldout_model in enumerate(models):
        agent_test = [r for r in rows if r.label == "agent" and r.source == heldout_model]
        candidate_tasks = {(r.benchmark, r.task) for r in agent_test}
        human_test = [r for r in rows if r.label == "human" and (r.benchmark, r.task) in candidate_tasks]
        matched_tasks = {(r.benchmark, r.task) for r in human_test} & candidate_tasks
        test = [r for r in agent_test + human_test if (r.benchmark, r.task) in matched_tasks]
        train = [r for r in rows if (r.benchmark, r.task) not in matched_tasks
                 and not (r.label == "agent" and r.source == heldout_model)]
        records.extend(evaluate_transfer(train, test, transfer_type="leave_one_agent_model_and_task",
                                         heldout_value=heldout_model, seed=BASE_SEED + 600 + index))
    return records


def evaluate_cohort(rows, *, cohort, seed, shuffle_seed=None, allowed_features=FEATURES):
    print(f"[cohort] {cohort}; n={len(rows)}; task_groups={len({(r.benchmark, r.task) for r in rows})}; seed={seed}", flush=True)
    results, predictions, task_records = [], [], []
    folds = grouped_folds(rows, seed)
    for fold_index, (train_keys, test_keys) in enumerate(folds, 1):
        train_rows = [r for r in rows if (r.benchmark, r.task) in train_keys]
        test_rows = [r for r in rows if (r.benchmark, r.task) in test_keys]
        if len({r.label for r in train_rows}) < 2 or len({r.label for r in test_rows}) < 2:
            raise ValueError(f"Fold {fold_index} lacks a class for {cohort}")
        if shuffle_seed is not None:
            from dataclasses import replace
            def shuffle_row(row):
                rng = np.random.default_rng(shuffle_seed + int(row.trace_id[:8], 16))
                pairs = list(zip(row.names, row.groups))
                rng.shuffle(pairs)
                return replace(row, names=tuple(x[0] for x in pairs), groups=tuple(x[1] for x in pairs))
            train_rows = [shuffle_row(row) for row in train_rows]
            test_rows = [shuffle_row(row) for row in test_rows]
        y_train = np.asarray([int(r.label == "agent") for r in train_rows])
        y_test = np.asarray([int(r.label == "agent") for r in test_rows])
        for feature in allowed_features:
            X_train, X_test, names = make_design(train_rows, test_rows, feature, seed + fold_index)
            if X_train.shape[0] != len(train_rows) or X_test.shape[0] != len(test_rows):
                raise AssertionError("Feature rows do not align")
            for model_name in CLASSIFIERS:
                model = classifier(model_name, seed + fold_index)
                model.fit(X_train, y_train)
                probabilities = model.predict_proba(X_test)[:, 1]
                task_metric_rows = calculate_task_metrics(test_rows, y_test, probabilities)
                task_records.extend({
                    "cohort": cohort, "seed": seed, "fold": fold_index,
                    "feature": feature, "classifier": model_name, **row,
                } for row in task_metric_rows)
                predictions.extend({
                    "cohort": cohort, "seed": seed, "fold": fold_index,
                    "feature": feature, "classifier": model_name,
                    "trace_id": row.trace_id, "benchmark": row.benchmark, "task": row.task,
                    "label_for_evaluation_only": row.label,
                    "probability": float(probabilities[i]),
                } for i, row in enumerate(test_rows))
                results.append({
                    "cohort": cohort, "seed": seed, "fold": fold_index,
                    "feature": feature, "classifier": model_name,
                    "n_train": len(train_rows), "n_test": len(test_rows),
                    "train_task_groups": len(train_keys), "test_task_groups": len(test_keys),
                    "vocabulary_size": len(names),
                })
    return results, predictions, task_records


def run():
    rows = gallery.load_rows()
    assert len(rows) == 177 and len({(r.benchmark, r.task) for r in rows}) == 40
    primary_rows = eligible_rows(rows)
    primary_results, predictions, task_metrics = [], [], []
    for seed in SPLIT_SEEDS:
        print(f"[primary split] seed={seed}", flush=True)
        current = evaluate_cohort(primary_rows, cohort="full_177", seed=seed)
        primary_results.extend(current[0]); predictions.extend(current[1]); task_metrics.extend(current[2])

    budget_results = []
    for min_count, label in ((2, "matched_ge2"), (4, "matched_ge4")):
        cohort_full = eligible_rows(rows, min_count)
        prefix_rows = [truncate(row, min_count) for row in cohort_full]
        for view, selected in (("full_same_cohort", cohort_full), (f"prefix_{min_count}", prefix_rows)):
            evaluated = evaluate_cohort(selected, cohort=f"{label}_{view}", seed=BASE_SEED,
                                        allowed_features=("F1_command_tfidf", "F3_command_plus_structure"))
            budget_results.extend(evaluated[0]); predictions.extend(evaluated[1]); task_metrics.extend(evaluated[2])

    # Five predeclared permutations on the fixed >=4-command matched cohort.
    shuffle_results = []
    shuffle_rows = [truncate(row, 4) for row in eligible_rows(rows, 4)]
    for permutation_seed in (17, 29, 43, 71, 101):
        evaluated = evaluate_cohort(
            shuffle_rows, cohort=f"prefix4_order_shuffle_{permutation_seed}", seed=BASE_SEED,
            shuffle_seed=permutation_seed,
            allowed_features=("F1_command_tfidf", "F3_command_plus_structure"),
        )
        shuffle_results.extend(evaluated[0]); predictions.extend(evaluated[1]); task_metrics.extend(evaluated[2])

    transfer_summary = run_transfer_diagnostics(primary_rows)

    # Aggregate repeated primary splits by task before task-cluster intervals.
    primary_summary = []
    metric_names = ("balanced_accuracy", "macro_f1", "auroc", "human_recall", "agent_recall")
    for feature in FEATURES:
        for model_name in CLASSIFIERS:
            task_seed_values = collections.defaultdict(list)
            for row in task_metrics:
                if row["cohort"] == "full_177" and row["feature"] == feature and row["classifier"] == model_name:
                    task_seed_values[(row["benchmark"], row["task"])].append(row)
            averaged_tasks = []
            for (benchmark, task), values in sorted(task_seed_values.items()):
                averaged_tasks.append({"benchmark": benchmark, "task": task, **{
                    metric: float(np.mean([v[metric] for v in values])) for metric in metric_names
                }})
            primary_summary.append({
                "cohort": "full_177_repeated_splits", "feature": feature, "classifier": model_name,
                "metrics": aggregate_task_metrics(averaged_tasks),
                "ci95_task_cluster": {metric: task_bootstrap(averaged_tasks, metric, seed=BASE_SEED + i)
                                      for i, metric in enumerate(metric_names)},
                "task_groups": len(averaged_tasks), "split_seeds": SPLIT_SEEDS,
            })

    # Point summaries for paired budget/permutation cohorts; task groups are the bootstrap unit.
    diagnostic_summary = []
    for cohort in sorted({r["cohort"] for r in task_metrics if r["cohort"] != "full_177"}):
        for feature in sorted({r["feature"] for r in task_metrics if r["cohort"] == cohort}):
            for model_name in CLASSIFIERS:
                rows_here = [r for r in task_metrics if r["cohort"] == cohort and r["feature"] == feature and r["classifier"] == model_name]
                if rows_here:
                    by_task = {}
                    for row in rows_here:
                        by_task.setdefault((row["benchmark"], row["task"]), []).append(row)
                    averaged = [{"benchmark": b, "task": t, **{
                        metric: float(np.mean([x[metric] for x in values])) for metric in metric_names
                    }} for (b, t), values in sorted(by_task.items())]
                    diagnostic_summary.append({
                        "cohort": cohort, "feature": feature, "classifier": model_name,
                        "metrics": aggregate_task_metrics(averaged), "task_groups": len(averaged),
                    })

    result = {
        "status": "completed exploratory CPU evaluation",
        "protocol_version": "feature-ablation-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "Cyber Task Horizons public CTF traces", "revision": "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae",
            "license": "CC BY 4.0", "records": len(rows), "benchmark_qualified_task_groups": 40,
            "mode_counts": dict(collections.Counter(r.label for r in rows)),
            "benchmark_counts": dict(collections.Counter(r.benchmark for r in rows)),
            "capture_confound": "Human terminal recordings and Agent tool-call logs are distinct capture paths, confounded with mode labels.",
        },
        "provenance": {
            "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "input_manifest_sha256": {
                "cth_intercode": hashlib.sha256((ROOT / "data" / "raw" / "cyber-task-horizons" / "download_manifest.json").read_bytes()).hexdigest(),
                "cth_crossbenchmark": hashlib.sha256((ROOT / "data" / "manifests" / "crossbenchmark_public_subset.json").read_bytes()).hexdigest(),
            },
            "code_sha256": {
                "runner": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                "feature_source": hashlib.sha256((ROOT.parent / "attackagent" / "traces.py").read_bytes()).hexdigest(),
                "baseline_source": hashlib.sha256((ROOT / "scripts" / "run_mixed_gallery.py").read_bytes()).hexdigest(),
            },
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        },
        "protocol": {
            "split": "3 benchmark-stratified folds, grouped by (benchmark, task); all feature transforms fit within training fold",
            "split_seeds": SPLIT_SEEDS,
            "primary_estimand": "task-macro, benchmark-equal-weighted source-label separability; not operator or actor identification",
            "features": {
                "F0_count_action": "log command count plus eight action-category proportions",
                "F1_command_tfidf": "command-name TF-IDF unigram and bigram, min_df=1, sublinear_tf=True",
                "F2_sequence_structure": "fixed sequence statistics independent of specific command vocabulary, plus short-trace flag; command count is excluded",
                "F3_command_plus_structure": "F1 concatenated with fixed F2 values",
            },
            "classifiers": "balanced logistic regression C=1; ExtraTrees 300 estimators, depth 8, min leaf 3",
            "uncertainty": "5,000 task-cluster bootstrap draws, stratified within benchmark after averaging each task across six split seeds; repeated splits are sensitivity checks, not independent replications",
            "budget": "same eligible rows compared at full sequence and prefix k; prefix 2 uses >=2-command cohort, prefix 4 uses >=4-command cohort",
            "order_control": "five fixed random permutations of aligned command/action pairs on the >=4-command cohort; refit and score under the same grouped folds",
        },
        "primary_summary": primary_summary,
        "diagnostic_summary": diagnostic_summary,
        "transfer_summary": transfer_summary,
        "outputs": {"task_metrics": str(TASK_CSV.relative_to(ROOT)), "predictions": str(PRED_CSV.relative_to(ROOT)),
                    "figure_script": "scripts/plot_feature_ablation.py"},
        "interpretation_guard": "The labels are confounded with logging pathway and the same-task cohort is a CTF proxy; these results do not establish human-versus-Agent identification in matched audit telemetry or real APT attribution.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import csv
    for path, records in ((TASK_CSV, task_metrics), (PRED_CSV, predictions)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if records:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))
