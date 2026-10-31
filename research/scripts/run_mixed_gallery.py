#!/usr/bin/env python3
"""Evaluate behavior and execution-mode signals in a mixed trace gallery.

All model fitting is task-grouped. At inference, query/candidate mode labels,
task names, model names, and source filenames are not used as features or filters.
Task identity is used only by the evaluator as a CTF same-challenge proxy label.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import math
import hashlib
import itertools
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "vendor" / "procgrep" / "src"))

from attackagent.traces import load_crossbenchmark_observations, load_observations  # noqa: E402
from procgrep import Trace as ProcTrace, encode as proc_encode, fit_bpe as proc_fit_bpe, jsd as proc_jsd  # noqa: E402

BASE_SEED = 20261030
BOOTSTRAPS = 5000
METHODS = (
    "command_cosine",
    "action_bigram",
    "procgrep_jsd",
    "length_only",
    "supervised_similarity",
    "mode_only",
    "mode_aware",
)
METRICS = ("hit_at_1", "mrr", "recall_at_5", "ndcg_at_5")
OUT_JSON = ROOT / "results" / "mixed_gallery.json"
OUT_TASK_CSV = ROOT / "results" / "mixed_gallery_task_metrics.csv"
OUT_QUERY_CSV = ROOT / "results" / "mixed_gallery_query_metrics.csv"
OUT_MODE_CSV = ROOT / "results" / "mixed_gallery_mode_predictions.csv"
FIG_DIR = ROOT / "results" / "figures"


def load_rows():
    rows = [row for row in load_observations() + load_crossbenchmark_observations() if row.names]
    by_task = collections.defaultdict(set)
    for row in rows:
        by_task[row.task].add(row.label)
    matched = {task for task, labels in by_task.items() if labels == {"human", "agent"}}
    return [row for row in rows if row.task in matched]


def split_tasks(rows, seed=BASE_SEED, n_folds=3):
    by_benchmark = collections.defaultdict(set)
    for row in rows:
        by_benchmark[row.benchmark].add(row.task)
    fold_by_task = {}
    for benchmark_index, benchmark in enumerate(sorted(by_benchmark)):
        tasks = sorted(by_benchmark[benchmark])
        rng = np.random.default_rng(seed + benchmark_index)
        rng.shuffle(tasks)
        for index, task in enumerate(tasks):
            fold_by_task[task] = index % n_folds
    folds = []
    for fold in range(n_folds):
        test = sorted(task for task, assignment in fold_by_task.items() if assignment == fold)
        train = sorted(task for task, assignment in fold_by_task.items() if assignment != fold)
        if set(train) & set(test):
            raise AssertionError("task leakage across outer folds")
        folds.append({"fold": fold + 1, "train_tasks": train, "test_tasks": test})
    return folds


def text_commands(row):
    return " ".join(row.names)


def action_bigram_text(row):
    return " ".join(f"{left}__{right}" for left, right in zip(row.groups, row.groups[1:]))


def cosine_matrix(train_rows, all_rows, field, ngram_range=(1, 1)):
    train_text = [field(row) for row in train_rows]
    all_text = [field(row) for row in all_rows]
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
    )
    try:
        vectorizer.fit(train_text)
        matrix = vectorizer.transform(all_text)
        return (matrix @ matrix.T).toarray()
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        return np.zeros((len(all_rows), len(all_rows)), dtype=float)


def procgrep_matrix(train_rows, all_rows, *, vocab_size_extra=32):
    train_sequences = [list(row.groups or row.names) for row in train_rows]
    all_sequences = [list(row.groups or row.names) for row in all_rows]
    alphabet = {atom for sequence in train_sequences for atom in sequence}
    if not alphabet:
        return np.zeros((len(all_rows), len(all_rows)), dtype=float), "empty"
    vocab = proc_fit_bpe(
        train_sequences,
        vocab_size=len(alphabet) + min(vocab_size_extra, max(0, 128 - len(alphabet))),
        seed=BASE_SEED,
        min_pair_frequency=2,
        fit_corpus="training-tasks-only",
    )
    traces = [ProcTrace(trace_id=row.trace_id, agent="unknown", atoms=all_sequences[i], group=row.trace_id)
              for i, row in enumerate(all_rows)]
    fingerprints = proc_encode(traces, vocab=vocab)
    output = np.eye(len(all_rows), dtype=float)
    for left, right in itertools.combinations(range(len(all_rows)), 2):
        value = 1.0 - proc_jsd(fingerprints[left].distribution(), fingerprints[right].distribution())
        output[left, right] = output[right, left] = value
    return output, vocab.spec.compact()


def fit_mode_scores(train_rows, all_rows, seed):
    """Task-OOF scores for train pairs plus a train-fitted test scorer."""
    y_train = np.asarray([int(row.label == "agent") for row in train_rows])
    train_groups = np.asarray([row.task for row in train_rows])
    if len(set(y_train)) != 2:
        raise ValueError("mode classifier training fold lacks one class")
    oof = np.full(len(train_rows), np.nan)
    splitter = GroupKFold(n_splits=3)
    for inner_fold, (fit_pos, val_pos) in enumerate(splitter.split(np.zeros(len(train_rows)), y_train, train_groups)):
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        X_fit = vectorizer.fit_transform([text_commands(train_rows[i]) for i in fit_pos])
        X_val = vectorizer.transform([text_commands(train_rows[i]) for i in val_pos])
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed + inner_fold)
        model.fit(X_fit, y_train[fit_pos])
        oof[val_pos] = model.predict_proba(X_val)[:, 1]
    if np.isnan(oof).any():
        raise AssertionError("missing inner task-OOF mode score")

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_train = vectorizer.fit_transform([text_commands(row) for row in train_rows])
    X_all = vectorizer.transform([text_commands(row) for row in all_rows])
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed)
    classifier.fit(X_train, y_train)
    all_probs = classifier.predict_proba(X_all)[:, 1]
    train_positions = {row.trace_id: index for index, row in enumerate(train_rows)}
    # Inner OOF values, not resubstitution scores, feed the pair-training data.
    for index, row in enumerate(all_rows):
        if row.trace_id in train_positions:
            all_probs[index] = oof[train_positions[row.trace_id]]

    # Platt calibration uses only train-task OOF predictions.
    logits = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / np.clip(1 - oof, 1e-6, 1))[:, None]
    calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    calibrator.fit(logits, y_train)
    calibrated = calibrator.predict_proba(
        np.log(np.clip(all_probs, 1e-6, 1 - 1e-6) / np.clip(1 - all_probs, 1e-6, 1))[:, None]
    )[:, 1]
    return all_probs, calibrated, oof


def pair_features(left, right, base, probabilities, *, with_mode=False):
    command = base["command"][left, right]
    action = base["action"][left, right]
    proc = base["procgrep"][left, right]
    length_l = np.log1p(len(base["rows"][left].names))
    length_r = np.log1p(len(base["rows"][right].names))
    length_sim = float(np.exp(-abs(length_l - length_r)))
    features = [command, action, proc, length_sim]
    if with_mode:
        p_left, p_right = float(probabilities[left]), float(probabilities[right])
        same_mode = p_left * p_right + (1 - p_left) * (1 - p_right)
        features.extend([same_mode, abs(p_left - p_right)])
    return features


def training_pairs(train_rows, seed):
    """Sample pairs using only benchmark and task identity, never mode/source."""
    by_task = collections.defaultdict(list)
    by_benchmark = collections.defaultdict(list)
    for index, row in enumerate(train_rows):
        by_task[(row.benchmark, row.task)].append(index)
    for task_key in sorted(by_task):
        by_benchmark[task_key[0]].append(task_key)
    rng = np.random.default_rng(seed)
    pairs, labels = [], []
    seen = set()
    for task_key, same_rows in sorted(by_task.items()):
        positive = list(itertools.combinations(same_rows, 2))
        if not positive:
            continue
        rng.shuffle(positive)
        positive = positive[:32]
        negative = []
        for other_key in by_benchmark[task_key[0]]:
            if other_key == task_key:
                continue
            negative.extend(itertools.product(same_rows, by_task[other_key]))
        if not negative:
            continue
        rng.shuffle(negative)
        negative = negative[:len(positive)]
        for left, right in [*positive, *negative]:
            key = (min(left, right), max(left, right))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            labels.append(int((train_rows[left].benchmark, train_rows[left].task) ==
                              (train_rows[right].benchmark, train_rows[right].task)))
    if set(labels) != {0, 1}:
        raise ValueError("task-only pair sampling did not produce both pair labels")
    return pairs, np.asarray(labels, dtype=int)


def expected_rank_metrics(scores, relevance, k=5):
    """Expected multi-positive ranking metrics under uniform tie breaking."""
    scores = np.asarray(scores, dtype=float)
    relevance = np.asarray(relevance, dtype=int)
    n = len(scores)
    positive_count = int(relevance.sum())
    if n == 0 or positive_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_rel = relevance[order]
    blocks = []
    start = 0
    while start < n:
        end = start + 1
        while end < n and np.isclose(ordered_scores[start], ordered_scores[end], atol=1e-12, rtol=1e-12):
            end += 1
        blocks.append((start, end, int(ordered_rel[start:end].sum())))
        start = end
    hit1 = (blocks[0][2] / (blocks[0][1] - blocks[0][0])) if blocks and blocks[0][0] == 0 else 0.0
    mrr = 0.0
    found_first_positive_block = False
    expected_relevant_discount = 0.0
    top_k = min(k, n)
    for start, end, positives in blocks:
        size = end - start
        if not positives:
            continue
        if positives and not found_first_positive_block:
            denominator = math.comb(size, positives)
            mrr = sum(
                (math.comb(size - offset, positives - 1) / denominator) / (start + offset)
                for offset in range(1, size - positives + 2)
            )
            found_first_positive_block = True
        overlap_start, overlap_end = start, min(end, top_k)
        if overlap_start < overlap_end:
            expected_relevant_discount += (positives / size) * sum(
                1.0 / np.log2(rank + 1) for rank in range(overlap_start + 1, overlap_end + 1)
            )
    ideal_count = min(positive_count, top_k)
    ideal_dcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_count + 1))
    recall_at_k = min(1.0, top_k / n)  # expected recall under a random ranking
    observed_recall = 0.0
    for start, end, positives in blocks:
        overlap = max(0, min(end, top_k) - start)
        if overlap:
            observed_recall += positives * overlap / (end - start)
    return {
        "hit_at_1": float(hit1),
        "mrr": float(mrr),
        "recall_at_5": float(observed_recall / positive_count),
        "ndcg_at_5": float(expected_relevant_discount / ideal_dcg) if ideal_dcg else 0.0,
        "query_count": 1,
        "candidate_count": n,
        "positive_count": positive_count,
        "random_hit_at_1": float(positive_count / n),
        "random_recall_at_5": float(recall_at_k),
    }


def classification_metrics(rows, probabilities):
    y = np.asarray([int(row.label == "agent") for row in rows])
    prediction = (probabilities >= 0.5).astype(int)
    matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y, prediction):
        matrix[actual, predicted] += 1
    class_metrics = []
    for label in (0, 1):
        tp = int(matrix[label, label])
        fp = int(matrix[1 - label, label])
        fn = int(matrix[label, 1 - label])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics.append({"precision": precision, "recall": recall, "f1": f1})
    brier = float(np.mean((probabilities - y) ** 2))
    ece = 0.0
    for lower in np.linspace(0, 0.9, 10):
        mask = (probabilities >= lower) & (probabilities < lower + 0.1 if lower < 0.9 else probabilities <= 1)
        if mask.any():
            ece += float(mask.mean() * abs(probabilities[mask].mean() - y[mask].mean()))
    coverage = {}
    for threshold in (0.55, 0.65, 0.75, 0.85, 0.95):
        keep = np.maximum(probabilities, 1 - probabilities) >= threshold
        coverage[str(threshold)] = {
            "coverage": float(keep.mean()),
            "selective_accuracy": float(np.mean(prediction[keep] == y[keep])) if keep.any() else None,
        }
    return {
        "n": int(len(y)),
        "confusion_matrix": matrix.tolist(),
        "human": class_metrics[0],
        "agent": class_metrics[1],
        "balanced_accuracy": float(np.mean([class_metrics[0]["recall"], class_metrics[1]["recall"]])),
        "macro_f1": float(np.mean([class_metrics[0]["f1"], class_metrics[1]["f1"]])),
        "brier": brier,
        "ece10": ece,
        "coverage_curve": coverage,
    }


def calibration_bins(rows, probabilities, bins=10):
    y = np.asarray([int(row.label == "agent") for row in rows])
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    output = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = ((probabilities >= lower) & (probabilities < upper))
        if index == bins - 1:
            mask |= probabilities == 1.0
        if mask.any():
            output.append({
                "lower": float(lower),
                "upper": float(upper),
                "mean_probability": float(probabilities[mask].mean()),
                "agent_fraction": float(y[mask].mean()),
                "count": int(mask.sum()),
            })
    return output


def evaluate_fold(rows, fold_spec, seed):
    train_tasks, test_tasks = set(fold_spec["train_tasks"]), set(fold_spec["test_tasks"])
    train_rows = [row for row in rows if row.task in train_tasks]
    test_rows = [row for row in rows if row.task in test_tasks]
    all_rows = [*train_rows, *test_rows]
    if train_tasks & test_tasks:
        raise AssertionError("task leakage")
    n_train = len(train_rows)
    base = {
        "rows": all_rows,
        "command": cosine_matrix(train_rows, all_rows, text_commands, (1, 2)),
        "action": cosine_matrix(train_rows, all_rows, action_bigram_text, (1, 1)),
    }
    base["procgrep"], proc_vocab = procgrep_matrix(train_rows, all_rows)
    mode_raw, mode_calibrated, mode_oof = fit_mode_scores(train_rows, all_rows, seed)
    train_pairs, pair_y = training_pairs(train_rows, seed)
    X_base = np.asarray([pair_features(left, right, base, mode_raw, with_mode=False) for left, right in train_pairs])
    X_mode = np.asarray([pair_features(left, right, base, mode_raw, with_mode=True) for left, right in train_pairs])
    base_model = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=seed)
    mode_model = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=seed)
    base_model.fit(X_base, pair_y)
    mode_model.fit(X_mode, pair_y)

    index_by_id = {row.trace_id: index for index, row in enumerate(all_rows)}
    test_indices = [index_by_id[row.trace_id] for row in test_rows]
    by_task_benchmark = {(row.benchmark, row.task) for row in test_rows}
    query_rows = []
    task_query_metrics: dict[tuple[str, str, str], list[dict]] = collections.defaultdict(list)
    for query in test_rows:
        qidx = index_by_id[query.trace_id]
        gallery = [row for row in test_rows if row.benchmark == query.benchmark and row.trace_id != query.trace_id]
        if not gallery:
            continue
        candidate_indices = [index_by_id[row.trace_id] for row in gallery]
        relevance = np.asarray([int(row.task == query.task and row.benchmark == query.benchmark) for row in gallery])
        pair_x_base = np.asarray([pair_features(qidx, candidate, base, mode_raw, with_mode=False) for candidate in candidate_indices])
        pair_x_mode = np.asarray([pair_features(qidx, candidate, base, mode_raw, with_mode=True) for candidate in candidate_indices])
        score_map = {
            "command_cosine": np.asarray([base["command"][qidx, idx] for idx in candidate_indices]),
            "action_bigram": np.asarray([base["action"][qidx, idx] for idx in candidate_indices]),
            "procgrep_jsd": np.asarray([base["procgrep"][qidx, idx] for idx in candidate_indices]),
            "length_only": np.asarray([pair_features(qidx, idx, base, mode_raw)[3] for idx in candidate_indices]),
            "supervised_similarity": base_model.predict_proba(pair_x_base)[:, 1],
            "mode_only": np.asarray([
                mode_raw[qidx] * mode_raw[idx] + (1 - mode_raw[qidx]) * (1 - mode_raw[idx])
                for idx in candidate_indices
            ]),
            "mode_aware": mode_model.predict_proba(pair_x_mode)[:, 1],
        }
        for method, values in score_map.items():
            metric = expected_rank_metrics(values, relevance)
            if metric is None:
                continue
            task_query_metrics[(query.benchmark, query.task, method)].append(metric)
            query_rows.append({
                "seed": seed,
                "fold": fold_spec["fold"],
                "benchmark": query.benchmark,
                "query_task": query.task,
                "query_trace_id": query.trace_id,
                "query_label_for_evaluation_only": query.label,
                "method": method,
                **metric,
            })

    task_rows = []
    for (benchmark, task, method), metrics in sorted(task_query_metrics.items()):
        task_rows.append({
            "seed": seed,
            "fold": fold_spec["fold"],
            "benchmark": benchmark,
            "task": task,
            "method": method,
            "query_count": len(metrics),
            "candidate_count_mean": float(np.mean([row["candidate_count"] for row in metrics])),
            "positive_count_mean": float(np.mean([row["positive_count"] for row in metrics])),
            "random_hit_at_1_mean": float(np.mean([row["random_hit_at_1"] for row in metrics])),
            **{metric: float(np.mean([row[metric] for row in metrics])) for metric in METRICS},
        })
    raw_test = mode_raw[n_train:]
    calibrated_test = mode_calibrated[n_train:]
    classification = classification_metrics(test_rows, calibrated_test)
    mode_rows = []
    for row, raw_probability, calibrated_probability in zip(test_rows, raw_test, calibrated_test):
        mode_rows.append({
            "seed": seed,
            "fold": fold_spec["fold"],
            "benchmark": row.benchmark,
            "task_for_evaluation_only": row.task,
            "trace_id": row.trace_id,
            "label_for_evaluation_only": row.label,
            "agent_probability_raw": float(raw_probability),
            "agent_probability_calibrated": float(calibrated_probability),
            "predicted_mode_at_0_5": "agent" if calibrated_probability >= .5 else "human",
        })
    fold_details = {
        **fold_spec,
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "training_pairs": int(len(train_pairs)),
        "training_pair_positive": int(pair_y.sum()),
        "training_pair_negative": int(len(pair_y) - pair_y.sum()),
        "pair_sampling_uses": ["benchmark", "activity_id"],
        "pair_sampling_excludes": ["mode", "source", "filename"],
        "procgrep_vocab": proc_vocab,
        "mode_classifier": classification,
        "candidate_rule": "same benchmark, all test-fold traces, exclude query by trace_id; mode/source labels not used",
    }
    return task_rows, query_rows, fold_details, mode_rows


def summarize_task_rows(task_rows, n_boot=BOOTSTRAPS, seed=BASE_SEED):
    by_method = collections.defaultdict(list)
    for row in task_rows:
        by_method[row["method"]].append(row)
    summary = {}
    by_method_task = {}
    for method, rows in by_method.items():
        by_benchmark = collections.defaultdict(list)
        for row in rows:
            by_benchmark[row["benchmark"]].append(row)
        task_values = {
            benchmark: {row["task"]: {metric: row[metric] for metric in METRICS} for row in values}
            for benchmark, values in by_benchmark.items()
        }
        by_method_task[method] = task_values
        point = {metric: float(np.mean([np.mean([row[metric] for row in values]) for values in by_benchmark.values()]))
                 for metric in METRICS}
        rng = np.random.default_rng(seed + sum(method.encode()))
        draws = {metric: [] for metric in METRICS}
        for _ in range(n_boot):
            for metric in METRICS:
                benchmark_means = []
                for values in by_benchmark.values():
                    sampled = rng.choice(len(values), size=len(values), replace=True)
                    benchmark_means.append(np.mean([values[index][metric] for index in sampled]))
                draws[metric].append(float(np.mean(benchmark_means)))
        summary[method] = {
            **point,
            "task_cluster_bootstrap_95_ci": {
                metric: [float(np.quantile(draws[metric], 0.025)), float(np.quantile(draws[metric], 0.975))]
                for metric in METRICS
            },
            "by_benchmark": {
                benchmark: {metric: float(np.mean([row[metric] for row in values])) for metric in METRICS}
                for benchmark, values in by_benchmark.items()
            },
            "task_count": sum(len(values) for values in by_benchmark.values()),
            "query_count": sum(row["query_count"] for row in rows),
            "candidate_count_mean": float(np.mean([row["candidate_count_mean"] for row in rows])),
            "random_hit_at_1_mean": float(np.mean([row["random_hit_at_1_mean"] for row in rows])),
        }
    deltas = {}
    pairwise = {}
    if "mode_aware" in by_method_task:
        for baseline in sorted(name for name in by_method_task if name != "mode_aware"):
            baseline_metrics = {}
            for metric in METRICS:
                paired = []
                for benchmark, tasks in by_method_task["mode_aware"].items():
                    control = by_method_task[baseline].get(benchmark, {})
                    paired.extend((tasks[task][metric] - control[task][metric], benchmark, task)
                                  for task in tasks.keys() & control.keys())
                point_by_bench = collections.defaultdict(list)
                rows_by_bench = collections.defaultdict(list)
                for delta, benchmark, task in paired:
                    point_by_bench[benchmark].append(delta)
                    rows_by_bench[benchmark].append((task, delta))
                point = float(np.mean([np.mean(values) for values in point_by_bench.values()])) if point_by_bench else None
                rng = np.random.default_rng(seed + 8000 + sum(metric.encode()) + sum(baseline.encode()))
                draws = []
                for _ in range(n_boot):
                    means = []
                    for values in rows_by_bench.values():
                        picked = rng.integers(0, len(values), size=len(values))
                        means.append(np.mean([values[index][1] for index in picked]))
                    if means:
                        draws.append(float(np.mean(means)))
                baseline_metrics[metric] = {
                    "mode_aware_minus_baseline": point,
                    "task_cluster_bootstrap_95_ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))] if draws else None,
                    "paired_tasks": len(paired),
                }
            pairwise[baseline] = baseline_metrics
        if "supervised_similarity" in pairwise:
            deltas = {
                metric: {
                    "mode_aware_minus_supervised_similarity": pairwise["supervised_similarity"][metric]["mode_aware_minus_baseline"],
                    "task_cluster_bootstrap_95_ci": pairwise["supervised_similarity"][metric]["task_cluster_bootstrap_95_ci"],
                    "paired_tasks": pairwise["supervised_similarity"][metric]["paired_tasks"],
                }
                for metric in METRICS
            }
    deltas["pairwise"] = pairwise
    return summary, deltas


def figure_retrieval(summary, deltas, robust_deltas=None, name="mixed_gallery_retrieval"):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks = ["cybench", "intercode-ctf", "nyuctf"]
    methods = list(METHODS)
    matrix = np.asarray([[summary[method]["by_benchmark"].get(benchmark, {}).get("hit_at_1", np.nan)
                          for benchmark in benchmarks] for method in methods])
    fig = plt.figure(figsize=(11.2, 5.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 1))
    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(benchmarks)), ["CyBench", "InterCode-CTF", "NYU CTF"])
    ax.set_yticks(range(len(methods)), [method.replace("_", " ") for method in methods])
    ax.set_title("Task-macro Hit@1 by benchmark")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        color="white" if value > 0.58 else "#17324d", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=.045, pad=.035)
    colorbar.set_label("Hit@1 (higher is better)")

    ax2 = fig.add_subplot(grid[0, 1])
    value = deltas["hit_at_1"]["mode_aware_minus_supervised_similarity"]
    lo, hi = deltas["hit_at_1"]["task_cluster_bootstrap_95_ci"]
    ax2.axvline(0, color="#68798b", linewidth=1, linestyle="--")
    ax2.errorbar([value], [1], xerr=[[value - lo], [hi - value]], fmt="o", color="#2369a8",
                 ecolor="#2369a8", capsize=5, markersize=7, linewidth=2, label="Primary split")
    y_ticks, y_labels = [1], ["Primary split"]
    if robust_deltas:
        robust = robust_deltas["hit_at_1"]["mode_aware_minus_supervised_similarity"]
        robust_lo, robust_hi = robust_deltas["hit_at_1"]["task_cluster_bootstrap_95_ci"]
        ax2.errorbar([robust], [0], xerr=[[robust - robust_lo], [robust_hi - robust]], fmt="o",
                     color="#78a9d1", ecolor="#78a9d1", capsize=5, markersize=7,
                     linewidth=2, label="Mean across 5 alternate splits")
        y_ticks.append(0)
        y_labels.append("Mean across 5 splits")
    ax2.set_yticks(y_ticks, y_labels)
    ax2.set_xlim(-.05, .12)
    ax2.set_xticks([-.05, 0, .05, .10], ["−5", "0", "+5", "+10"])
    ax2.set_ylim(-.6, 1.6)
    ax2.set_xlabel("Paired Hit@1 change (percentage points)")
    ax2.tick_params(axis="x", labelsize=9)
    ax2.set_title("Increment over task-only ranker")
    ax2.grid(axis="x", color="#d9e4ed", linewidth=.8)
    ax2.legend(frameon=False, loc="lower right", fontsize=8)
    for extension in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{extension}", dpi=300 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_confusion(fold_details):
    matrices = [np.asarray(row["mode_classifier"]["confusion_matrix"], dtype=float) for row in fold_details]
    matrix = sum(matrices)
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(4.6, 3.8), constrained_layout=True)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1], ["Human", "Agent"])
    ax.set_yticks([0, 1], ["Human", "Agent"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Evaluation-only label")
    ax.set_title("Task-held-out mode scores")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{int(matrix[row, col])}\n{normalized[row, col]:.1%}",
                    ha="center", va="center", color="white" if normalized[row, col] > .55 else "#17324d")
    bar = fig.colorbar(image, ax=ax, fraction=.05, pad=.04)
    bar.set_label("Within-class proportion")
    for extension in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"mixed_gallery_confusion.{extension}", dpi=300 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_calibration(mode_rows, name="mixed_gallery_calibration"):
    if not mode_rows:
        return
    from types import SimpleNamespace
    rows = [SimpleNamespace(label=row["label_for_evaluation_only"]) for row in mode_rows]
    raw = np.asarray([row["agent_probability_raw"] for row in mode_rows])
    calibrated = np.asarray([row["agent_probability_calibrated"] for row in mode_rows])
    labels = np.asarray([int(row.label == "agent") for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#8495a5", linestyle="--", linewidth=1.2, label="Ideal")
    for probabilities, color, label in ((raw, "#74a9d8", "Raw"), (calibrated, "#2369a8", "Platt calibrated")):
        bins = calibration_bins(rows, probabilities)
        if bins:
            ax.plot([item["mean_probability"] for item in bins], [item["agent_fraction"] for item in bins],
                    marker="o", markersize=4, linewidth=1.8, color=color, label=label)
            for item in bins:
                ax.annotate(str(item["count"]), (item["mean_probability"], item["agent_fraction"]),
                            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7, color="#435569")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted Agent probability",
           ylabel="Observed Agent fraction", title="Held-out calibration")
    ax.grid(color="#e2eaf1", linewidth=.7)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    thresholds = np.linspace(.5, .95, 10)
    for probabilities, color, label in ((raw, "#74a9d8", "Raw"), (calibrated, "#2369a8", "Platt calibrated")):
        confidence = np.maximum(probabilities, 1 - probabilities)
        prediction = (probabilities >= .5).astype(int)
        coverage, accuracy = [], []
        for threshold in thresholds:
            keep = confidence >= threshold
            coverage.append(float(keep.mean()))
            accuracy.append(float((prediction[keep] == labels[keep]).mean()) if keep.any() else np.nan)
        ax.plot(coverage, accuracy, marker="o", markersize=3.5, linewidth=1.8, color=color, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Coverage (retained traces)",
           ylabel="Accuracy among retained", title="Selective prediction")
    ax.grid(color="#e2eaf1", linewidth=.7)
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle("Task-held-out mode scores · capture path is confounded with label", fontsize=10, color="#31475b")
    for extension in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{extension}", dpi=300 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_protocol(rows, seed, *, name, bootstraps=BOOTSTRAPS, n_folds=3):
    folds = split_tasks(rows, seed=seed, n_folds=n_folds)
    task_rows, query_rows, fold_details, mode_rows = [], [], [], []
    for fold in folds:
        task, query, details, predictions = evaluate_fold(rows, fold, seed + fold["fold"])
        task_rows.extend(task)
        query_rows.extend(query)
        fold_details.append(details)
        mode_rows.extend(predictions)
    summary, deltas = summarize_task_rows(task_rows, n_boot=bootstraps, seed=seed)
    return {"name": name, "seed": seed, "folds": fold_details, "task_rows": task_rows,
            "query_rows": query_rows, "mode_rows": mode_rows, "summary": summary, "deltas": deltas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--robustness-seeds", type=int, default=5)
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()
    if args.bootstraps < 100:
        parser.error("--bootstraps must be at least 100 for reported intervals")
    rows = load_rows()
    if len(rows) != 177 or len({row.task for row in rows}) != 40:
        raise SystemExit(f"Pinned trace audit changed: expected 177 rows / 40 tasks, got {len(rows)} / {len({row.task for row in rows})}")
    primary = run_protocol(rows, BASE_SEED, name="primary_mixed_gallery", bootstraps=args.bootstraps)
    all_task_rows = list(primary["task_rows"])
    all_query_rows = list(primary["query_rows"])
    write_csv(OUT_MODE_CSV, primary["mode_rows"])
    robust_task_rows = []
    robustness = []
    if not args.skip_robustness:
        for offset in range(1, args.robustness_seeds + 1):
            seed = BASE_SEED + 100 * offset
            result = run_protocol(rows, seed, name=f"task_split_seed_{seed}", bootstraps=args.bootstraps)
            robustness.append({"seed": seed, "summary": result["summary"], "deltas": result["deltas"],
                               "folds": result["folds"]})
            all_task_rows.extend(result["task_rows"])
            all_query_rows.extend(result["query_rows"])
            robust_task_rows.extend(result["task_rows"])

    robust_summary = None
    robust_deltas = None
    if robust_task_rows:
        grouped = collections.defaultdict(list)
        for row in robust_task_rows:
            grouped[(row["benchmark"], row["task"], row["method"])].append(row)
        robust_task_means = []
        for (benchmark, task, method), values in sorted(grouped.items()):
            averaged = {metric: float(np.mean([row[metric] for row in values])) for metric in METRICS}
            robust_task_means.append({
                "seed": "robustness_mean",
                "fold": "repeated_splits",
                "benchmark": benchmark,
                "task": task,
                "method": method,
                "query_count": int(round(np.mean([row["query_count"] for row in values]))),
                "candidate_count_mean": float(np.mean([row["candidate_count_mean"] for row in values])),
                "positive_count_mean": float(np.mean([row["positive_count_mean"] for row in values])),
                "random_hit_at_1_mean": float(np.mean([row["random_hit_at_1_mean"] for row in values])),
                **averaged,
            })
        robust_summary, robust_deltas = summarize_task_rows(
            robust_task_means, n_boot=args.bootstraps, seed=BASE_SEED + 9000
        )

    source_manifest = json.loads((ROOT / "data" / "manifests" / "crossbenchmark_public_subset.json").read_text())
    script_hash = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    result_doc = {
        "status": "exploratory, task-held-out mixed trace-gallery evaluation",
        "protocol_version": "mixed-gallery-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "Cyber Task Horizons public traces",
            "revision": source_manifest["revision"],
            "license": source_manifest["data_license"],
            "records": len(rows),
            "tasks": len({row.task for row in rows}),
            "record_counts_by_benchmark_mode": dict(collections.Counter(f"{row.benchmark}|{row.label}" for row in rows)),
            "task_counts_by_benchmark": {
                benchmark: len({row.task for row in rows if row.benchmark == benchmark})
                for benchmark in sorted({row.benchmark for row in rows})
            },
            "source_mode_confound": "Human terminal-key capture vs Agent tool-call traces; labels and capture pathway are confounded.",
        },
        "evaluation": {
            "outer_folds": 3,
            "primary_seed": BASE_SEED,
            "unit_of_split_and_bootstrap": "benchmark-qualified CTF task group",
            "queries": "Every trace in the held-out fold",
            "gallery": "All other held-out traces from the same benchmark; no mode/source filtering or task aggregation; query itself excluded",
            "positive_proxy": "Same benchmark and CTF task; this is not shared actor or campaign ground truth",
            "metrics": list(METRICS),
            "tie_rule": "Expected metric under uniform random ordering inside score ties",
            "aggregation": "Average queries within task, then tasks within benchmark, then equal-weight benchmarks",
            "pair_training": "Balanced sampled pairs, labels derived only from same/different benchmark-task identity; no mode/source used to construct pairs",
            "pair_methods": "Behavior ranker uses command cosine, action-bigram cosine, ProcGrep similarity, and command-length similarity; mode ranker adds pairwise agreement/gap from task-OOF mode scores.",
            "uncertainty": "5,000 paired bootstrap resamples of task-level outcomes, conditional on fitted folds; not a new independent test set",
        },
        "primary": {key: value for key, value in primary.items() if key not in {"task_rows", "query_rows"}},
        "robustness": robustness,
        "robustness_mean_across_splits": {
            "summary": robust_summary,
            "deltas": robust_deltas,
            "interpretation": "For each benchmark-task, first average its outcomes across alternate split seeds; task-cluster bootstrap then summarizes across tasks. This is split sensitivity, not independent replication.",
        },
        "figures": {
            "mixed_gallery_retrieval": "scripts/run_mixed_gallery.py -> results/figures/mixed_gallery_retrieval.pdf/png",
            "mixed_gallery_confusion": "scripts/run_mixed_gallery.py -> results/figures/mixed_gallery_confusion.pdf/png",
            "mixed_gallery_calibration": "scripts/run_mixed_gallery.py -> results/figures/mixed_gallery_calibration.pdf/png",
        },
        "mode_predictions_csv": "results/mixed_gallery_mode_predictions.csv (primary split; labels are evaluation-only)",
        "source_sha256": script_hash,
        "interpretation_guard": "Task retrieval is a proxy feasibility study. It cannot establish APT campaign/actor linkage or causal impact of LLM Agents.",
    }
    OUT_JSON.write_text(json.dumps(result_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT_TASK_CSV, all_task_rows)
    write_csv(OUT_QUERY_CSV, all_query_rows)
    figure_retrieval(primary["summary"], primary["deltas"], robust_deltas)
    figure_confusion(primary["folds"])
    figure_calibration(primary["mode_rows"])
    print(json.dumps({
        "output": str(OUT_JSON),
        "primary_summary": primary["summary"],
        "primary_deltas": primary["deltas"],
        "robustness_mean_deltas": robust_deltas,
        "robustness_seed_count": len(robustness),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
