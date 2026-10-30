#!/usr/bin/env python3
"""Nested development selection and one-time locked external evaluation."""
from __future__ import annotations

import collections
import csv
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "multidimensional_rerun_v1"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_association_error_analysis as error_eval  # noqa: E402
import run_feature_ablation as split_utils  # noqa: E402
import run_mixed_gallery as gallery  # noqa: E402
from attackagent.profiles import (  # noqa: E402
    ACTION_LABELS, FEATURE_NAMES, TENSOR_CHANNELS, TENSOR_PHASES, PROFILE_VERSION,
    extract_behavior_features, extract_behavior_tensor,
)
from trace_adapter import (  # noqa: E402
    EXTERNAL, MIN_EXTERNAL_TASK_GROUPS, load_cohort,
)

BASELINE = "command_tfidf"
METHODS = (BASELINE, "profile_24d", "tensor_4x8x3", "tensor_tucker", "partial_sequence", "learned_fusion")
RATE_METRICS = (*error_eval.RATE_METRICS, "balanced_accuracy", "query_has_accepted_match")
RANK_METRICS = ("hit_at_1", "mrr", "recall_at_5", "ndcg_at_5")
LOWER_IS_BETTER = set(error_eval.RATE_METRICS)
JOINT_METRICS = (*RATE_METRICS, *RANK_METRICS)
TARGET_FSR = 0.20
TUCKER_RANKS = (2, 4, 2)
BOOTSTRAPS = 5000
THRESHOLD_GRID = 101
SEED = 20261031
SIMPLICITY = {name: index for index, name in enumerate(METHODS)}


def _profile_matrix(rows):
    output = []
    for row in rows:
        features = extract_behavior_features(row.names, row.groups)
        output.append([np.nan if features[name] is None else float(features[name])
                       for name in FEATURE_NAMES])
    return np.asarray(output, dtype=float)


def _tensor_matrix(rows):
    tensors = []
    for row in rows:
        if len(row.names) != len(row.groups):
            raise ValueError(f"unaligned command/action sequence: {row.trace_id}")
        tensors.append(np.asarray(extract_behavior_tensor(row.names, row.groups), dtype=float))
    return np.stack(tensors)


def _fit_hosvd(training_tensors):
    factors = []
    for mode, requested_rank in enumerate(TUCKER_RANKS):
        fibers = np.concatenate([
            np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1).T
            for tensor in training_tensors
        ], axis=0)
        _, _, right = np.linalg.svd(fibers, full_matrices=False)
        factors.append(right[:min(requested_rank, right.shape[0])].T.copy())
    return factors


def _project_tucker(tensor, factors):
    core = tensor
    for mode, factor in enumerate(factors):
        core = np.moveaxis(np.tensordot(core, factor, axes=(mode, 0)), -1, mode)
    return core


def _standardize(train_values, all_values):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train = scaler.fit_transform(imputer.fit_transform(train_values))
    transformed = scaler.transform(imputer.transform(all_values))
    return train, transformed


def _sequence_counters(rows, n_train):
    commands = [collections.Counter(row.names) for row in rows]
    bigrams = [collections.Counter(zip(row.names, row.names[1:])) for row in rows]

    def idf(values):
        document_frequency = collections.Counter()
        for counter in values[:n_train]:
            document_frequency.update(counter.keys())
        n = max(1, n_train)
        return {token: float(np.log((n + 1) / (count + 1)) + 1)
                for token, count in document_frequency.items()}

    return commands, bigrams, idf(commands), idf(bigrams)


def _weighted_jaccard(left, right, weights, unknown_weight):
    keys = left.keys() | right.keys()
    denominator = 0.0
    numerator = 0.0
    for token in keys:
        weight = weights.get(token, unknown_weight)
        a, b = left.get(token, 0), right.get(token, 0)
        numerator += min(a, b) * weight
        denominator += max(a, b) * weight
    return numerator / denominator if denominator else 0.0


def _pair_features(indices, all_rows, matrices, sequence_data):
    if not indices:
        return {name: np.asarray([], dtype=float) for name in METHODS if name != "learned_fusion"}
    index_by_id = {row.trace_id: index for index, row in enumerate(all_rows)}
    left = np.asarray([index_by_id[pair[0].trace_id] for pair in indices], dtype=int)
    right = np.asarray([index_by_id[pair[1].trace_id] for pair in indices], dtype=int)
    output = {
        "command_tfidf": matrices["command"][left, right],
        "profile_24d": np.exp(-np.mean(np.abs(matrices["profile"][left] - matrices["profile"][right]), axis=1)),
        "tensor_4x8x3": np.exp(-np.mean(np.abs(matrices["tensor"][left] - matrices["tensor"][right]), axis=1)),
        "tensor_tucker": np.exp(-np.mean(np.abs(matrices["tucker"][left] - matrices["tucker"][right]), axis=1)),
    }
    command_counts, bigram_counts, command_idf, bigram_idf, max_command_idf, max_bigram_idf = sequence_data
    partial = []
    for i, j in zip(left, right):
        token_overlap = _weighted_jaccard(command_counts[i], command_counts[j], command_idf, max_command_idf)
        local_overlap = _weighted_jaccard(bigram_counts[i], bigram_counts[j], bigram_idf, max_bigram_idf)
        partial.append(0.5 * token_overlap + 0.5 * local_overlap)
    output["partial_sequence"] = np.asarray(partial, dtype=float)
    return output


def fit_methods(train_rows, score_rows, seed):
    all_rows = [*train_rows, *score_rows]
    n_train = len(train_rows)
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
    )
    try:
        vectorizer.fit([gallery.text_commands(row) for row in train_rows])
        command_vectors = vectorizer.transform([gallery.text_commands(row) for row in all_rows])
        command_sim = cosine_similarity(command_vectors)
        vocabulary_size = len(vectorizer.vocabulary_)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        command_sim = np.zeros((len(all_rows), len(all_rows)), dtype=float)
        vocabulary_size = 0

    profile_raw = _profile_matrix(all_rows)
    _, profile_scaled = _standardize(profile_raw[:n_train], profile_raw)
    tensor_raw = _tensor_matrix(all_rows)
    _, tensor_scaled = _standardize(
        tensor_raw[:n_train].reshape(n_train, -1), tensor_raw.reshape(len(all_rows), -1)
    )
    factors = _fit_hosvd(tensor_raw[:n_train])
    tucker_raw = np.stack([_project_tucker(tensor, factors) for tensor in tensor_raw])
    _, tucker_scaled = _standardize(
        tucker_raw[:n_train].reshape(n_train, -1), tucker_raw.reshape(len(all_rows), -1)
    )
    matrices = {
        "command": command_sim,
        "profile": profile_scaled,
        "tensor": tensor_scaled,
        "tucker": tucker_scaled,
    }
    command_counts, bigram_counts, command_idf, bigram_idf = _sequence_counters(all_rows, n_train)
    sequence_data = (
        command_counts, bigram_counts, command_idf, bigram_idf,
        float(np.log(n_train + 1) + 1), float(np.log(n_train + 1) + 1),
    )

    train_pairs, pair_y = gallery.training_pairs(train_rows, seed)
    train_row_pairs = [(train_rows[left], train_rows[right]) for left, right in train_pairs]
    train_features = _pair_features(train_row_pairs, all_rows, matrices, sequence_data)
    design = np.column_stack([train_features[name] for name in METHODS if name != "learned_fusion"])
    fusion = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=seed)
    fusion.fit(design, pair_y)
    index_by_id = {row.trace_id: i for i, row in enumerate(all_rows)}

    def score_many(row_pairs):
        if not row_pairs:
            return {name: np.asarray([], dtype=float) for name in METHODS}
        # Convert IDs to rows drawn from all_rows, so there is one stable index path.
        pair_rows = [(all_rows[index_by_id[left.trace_id]], all_rows[index_by_id[right.trace_id]])
                     for left, right in row_pairs]
        direct = _pair_features(pair_rows, all_rows, matrices, sequence_data)
        features = np.column_stack([direct[name] for name in METHODS if name != "learned_fusion"])
        direct["learned_fusion"] = fusion.predict_proba(features)[:, 1]
        return direct

    dimensions = {
        "profile_dimensions": len(FEATURE_NAMES),
        "tensor_shape": list(tensor_raw.shape[1:]),
        "tensor_channels": list(TENSOR_CHANNELS),
        "tucker_ranks": list(TUCKER_RANKS),
        "tucker_core_shape": list(tucker_raw.shape[1:]),
        "command_vocabulary_size": int(vocabulary_size),
        "training_pair_count": int(len(train_pairs)),
        "pair_labels_use": "same versus different benchmark-qualified task only",
    }
    return score_many, dimensions


def make_pairs(rows, scorer, *, fold, split):
    edge_rows, row_pairs = [], []
    by_benchmark = collections.defaultdict(list)
    for row in rows:
        by_benchmark[row.benchmark].append(row)
    for benchmark in sorted(by_benchmark):
        traces = sorted(by_benchmark[benchmark], key=lambda row: row.trace_id)
        for query in traces:
            for candidate in traces:
                if query.trace_id == candidate.trace_id:
                    continue
                labels = {query.label, candidate.label}
                pair_type = "H-A" if labels == {"human", "agent"} else ("H-H" if labels == {"human"} else "A-A")
                edge_rows.append({
                    "split": split, "fold": fold, "benchmark": benchmark,
                    "query_task": query.task, "candidate_task": candidate.task,
                    "query_trace_id": query.trace_id, "candidate_trace_id": candidate.trace_id,
                    "query_mode_for_evaluation_only": query.label,
                    "candidate_mode_for_evaluation_only": candidate.label,
                    "same_task_proxy_for_evaluation_only": int(query.task == candidate.task),
                    "pair_type_for_evaluation_only": pair_type,
                })
                row_pairs.append((query, candidate))
    score_arrays = scorer(row_pairs)
    for index, row in enumerate(edge_rows):
        row.update({name: float(values[index]) for name, values in score_arrays.items()})
    return edge_rows


def rank_task_metrics(pair_rows):
    by_query = collections.defaultdict(list)
    for row in pair_rows:
        by_query[(row["benchmark"], row["query_task"], row["query_trace_id"])].append(row)
    task_methods = collections.defaultdict(list)
    for (benchmark, task, _query_id), rows in by_query.items():
        relevance = np.asarray([int(row["same_task_proxy_for_evaluation_only"]) for row in rows])
        if not relevance.any():
            continue
        for method in METHODS:
            metrics = gallery.expected_rank_metrics([row[method] for row in rows], relevance, k=5)
            if metrics is not None:
                task_methods[(benchmark, task, method)].append(metrics)
    output = []
    for (benchmark, task, method), values in sorted(task_methods.items()):
        output.append({
            "benchmark": benchmark, "task": task, "method": method,
            **{metric: float(np.mean([row[metric] for row in values])) for metric in RANK_METRICS},
            "query_count": len(values),
        })
    return output


def query_coverage(pair_rows, method, threshold):
    by_query = collections.defaultdict(list)
    for row in pair_rows:
        if row["same_task_proxy_for_evaluation_only"]:
            by_query[(row["benchmark"], row["query_task"], row["query_trace_id"])].append(
                row[method] >= threshold
            )
    by_task = collections.defaultdict(list)
    for (benchmark, task, _query_id), accepted in by_query.items():
        by_task[(benchmark, task)].append(float(any(accepted)))
    return [
        {"benchmark": benchmark, "task": task,
         "query_has_accepted_match": float(np.mean(values)), "method": method}
        for (benchmark, task), values in sorted(by_task.items())
    ]


def _macro(rates, metric):
    value, _benchmark_count, _task_count = error_eval._macro_metric(rates, metric)
    return value


def _directional_value(metric, value):
    return -float(value) if metric in LOWER_IS_BETTER else float(value)


def _joint_metric_values(rate_rows, rank_rows, method):
    values = {}
    for metric in RATE_METRICS:
        values[metric] = _macro([row for row in rate_rows if row["method"] == method], metric)
    for metric in RANK_METRICS:
        values[metric] = _macro([row for row in rank_rows if row["method"] == method], metric)
    return values


def select_thresholds(pair_rows, rank_rows, target_fsr=TARGET_FSR):
    thresholds = {}
    for method in METHODS:
        scores = np.asarray([row[method] for row in pair_rows], dtype=float)
        candidates = np.unique(np.quantile(scores, np.linspace(0, 1, THRESHOLD_GRID)))
        candidates = np.unique(np.r_[np.nextafter(scores.min(), -np.inf), candidates,
                                     np.nextafter(scores.max(), np.inf)])
        best = None
        for threshold in candidates:
            rates = error_eval._query_rates(pair_rows, method, float(threshold))
            fsr = _macro(rates, "false_split_rate")
            coverage = {(row["benchmark"], row["task"]): row["query_has_accepted_match"]
                        for row in query_coverage(pair_rows, method, float(threshold))}
            for row in rates:
                row["query_has_accepted_match"] = coverage.get((row["benchmark"], row["task"]))
            fmr = _macro(rates, "false_merge_rate")
            if fsr is None or fmr is None or fsr > target_fsr + 1e-12:
                continue
            utility_values = [
                _directional_value(metric, value)
                for metric in RATE_METRICS
                if (value := _macro(rates, metric)) is not None
            ]
            utility = float(np.mean(utility_values)) if utility_values else -np.inf
            score = (-utility, float(fmr), -float(threshold))
            if best is None or score < best[0]:
                best = (score, float(threshold), rates, utility)
        if best is None:
            raise RuntimeError(f"No threshold meets the training FSR <= {target_fsr:.0%} guard for {method}")
        _score, threshold, rates, utility = best
        thresholds[method] = {
            "threshold": threshold,
            "training_false_merge_rate": _macro(rates, "false_merge_rate"),
            "training_false_split_rate": _macro(rates, "false_split_rate"),
            "training_false_merge_human_agent": _macro(rates, "false_merge_human_agent"),
            "training_joint_classification_utility": utility,
            "fsr_ceiling": target_fsr,
            "selection": "highest mean direction-adjusted classification utility across 10 rates/coverage measures subject to macro FSR <= 20%; lower FMR then higher threshold breaks ties",
        }

    rate_rows, rank_rows = add_task_metrics(pair_rows, rank_rows, thresholds)
    metric_values = {method: _joint_metric_values(rate_rows, rank_rows, method) for method in METHODS}
    baseline_values = metric_values[BASELINE]
    comparable_metrics = [metric for metric in JOINT_METRICS
                          if all(metric_values[method][metric] is not None for method in METHODS)]
    directional_deltas = {
        method: {
            metric: (_directional_value(metric, value) - _directional_value(metric, baseline_values[metric]))
            if value is not None and baseline_values[metric] is not None else None
            for metric, value in values.items()
        }
        for method, values in metric_values.items()
    }

    def dominates(left, right):
        comparable = comparable_metrics
        if not comparable:
            return False
        left_values = [directional_deltas[left][metric] for metric in comparable]
        right_values = [directional_deltas[right][metric] for metric in comparable]
        return all(a >= b - 1e-12 for a, b in zip(left_values, right_values)) and any(
            a > b + 1e-12 for a, b in zip(left_values, right_values)
        )

    pareto_front = [method for method in METHODS
                    if not any(other != method and dominates(other, method) for other in METHODS)]
    candidates = []
    for method in METHODS:
        deltas = [directional_deltas[method][metric] for metric in JOINT_METRICS]
        available = [value for value in deltas if value is not None]
        joint_delta = float(np.mean(available)) if available else -np.inf
        worse_count = sum(value < -1e-12 for value in available)
        rank = metric_values[method]["ndcg_at_5"]
        fmr = metric_values[method]["false_merge_rate"]
        candidates.append((method not in pareto_front, -joint_delta, worse_count,
                           float(fmr if fmr is not None else 1.0),
                           -float(rank if rank is not None else -1.0), SIMPLICITY[method], method))
    selected = min(candidates)[-1]
    selected_deltas = directional_deltas[selected]
    selected_delta_values = [value for value in selected_deltas.values() if value is not None]
    selection = {
        "method": selected,
        "training_false_merge_rate": thresholds[selected]["training_false_merge_rate"],
        "training_false_split_rate": thresholds[selected]["training_false_split_rate"],
        "training_false_merge_human_agent": thresholds[selected]["training_false_merge_human_agent"],
        "training_ndcg_at_5": metric_values[selected]["ndcg_at_5"],
        "development_joint_mean_directional_delta_vs_baseline": (
            float(np.mean(selected_delta_values)) if selected_delta_values else None
        ),
        "development_compared_metric_count": len(selected_delta_values),
        "development_compared_metrics": comparable_metrics,
        "development_improved_metric_count": sum(value > 1e-12 for value in selected_delta_values),
        "development_worsened_metric_count": sum(value < -1e-12 for value in selected_delta_values),
        "development_pareto_front": pareto_front,
        "development_metric_values": metric_values,
        "development_directional_deltas_vs_baseline": directional_deltas,
        "selection_rule": "threshold maximizes mean direction-adjusted classification utility subject to FSR <= 20%; choose a Pareto method with the largest mean direction-adjusted change across 14 prespecified metrics, tie fewer worse metrics, lower FMR, higher NDCG@5, then simpler method",
    }
    return thresholds, selection


def add_task_metrics(pair_rows, rank_rows, thresholds):
    rate_rows = []
    for method in METHODS:
        threshold = thresholds[method]["threshold"]
        rates = error_eval._query_rates(pair_rows, method, threshold)
        coverage = {(row["benchmark"], row["task"]): row["query_has_accepted_match"]
                    for row in query_coverage(pair_rows, method, threshold)}
        for row in rates:
            row = dict(row)
            row["method"] = method
            row["threshold"] = threshold
            row["query_has_accepted_match"] = coverage.get((row["benchmark"], row["task"]))
            rate_rows.append(row)
    return rate_rows, rank_rows


def _paired_delta(rows, metric, method, baseline, draws=BOOTSTRAPS, seed=SEED):
    by_key = {(row["benchmark"], row["task"], row["method"]): row.get(metric) for row in rows}
    by_benchmark = collections.defaultdict(list)
    for (benchmark, task, name), value in by_key.items():
        if name != method or value is None:
            continue
        base = by_key.get((benchmark, task, baseline))
        if base is not None:
            by_benchmark[benchmark].append(float(value) - float(base))
    if not by_benchmark:
        return {"estimate": None, "ci_95": None, "paired_task_count": 0}
    point = float(np.mean([np.mean(values) for values in by_benchmark.values()]))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for index in range(draws):
        samples[index] = np.mean([
            np.mean(rng.choice(values, size=len(values), replace=True))
            for values in by_benchmark.values()
        ])
    return {
        "estimate": point,
        "ci_95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "paired_task_count": sum(map(len, by_benchmark.values())),
        "direction": f"{method} minus {baseline}",
        "bootstrap_unit": "benchmark-qualified task resampled within benchmark",
    }


def _joint_task_delta_interval(rate_rows, rank_rows, method, baseline, draws=BOOTSTRAPS, seed=SEED):
    by_task = collections.defaultdict(dict)
    for rows, metrics in ((rate_rows, RATE_METRICS), (rank_rows, RANK_METRICS)):
        for row in rows:
            key = (row["benchmark"], row["task"])
            for metric in metrics:
                value = row.get(metric)
                if value is not None:
                    by_task[key][(row["method"], metric)] = float(value)

    task_deltas = collections.defaultdict(list)
    metric_counts = []
    for (benchmark, _task), values in sorted(by_task.items()):
        deltas = []
        for metric in JOINT_METRICS:
            left = values.get((method, metric))
            right = values.get((baseline, metric))
            if left is not None and right is not None:
                deltas.append(_directional_value(metric, left) - _directional_value(metric, right))
        if deltas:
            task_deltas[benchmark].append(float(np.mean(deltas)))
            metric_counts.append(len(deltas))
    if not task_deltas:
        return {"estimate": None, "ci_95": None, "task_count": 0, "metric_count": 0}
    point = float(np.mean([np.mean(values) for values in task_deltas.values()]))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for index in range(draws):
        samples[index] = np.mean([
            np.mean(rng.choice(values, size=len(values), replace=True))
            for values in task_deltas.values()
        ])
    return {
        "estimate": point,
        "ci_95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "task_count": sum(map(len, task_deltas.values())),
        "benchmark_count": len(task_deltas),
        "mean_metrics_per_task": float(np.mean(metric_counts)),
        "utility": "per-task mean of paired direction-adjusted deltas across available prespecified metrics; task resampled within benchmark",
        "bootstrap_draws": draws,
    }


def _summaries(rate_rows, rank_rows):
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in (*RATE_METRICS, *RANK_METRICS):
            rows = rate_rows if metric in RATE_METRICS else rank_rows
            subset = [row for row in rows if row["method"] == method]
            summary[method][metric] = error_eval.task_cluster_interval(
                subset, metric, n_boot=BOOTSTRAPS, seed=SEED + METHODS.index(method) * 43 + sum(metric.encode())
            )
    deltas = {}
    for method in METHODS:
        if method == BASELINE:
            continue
        deltas[method] = {}
        for metric in (*RATE_METRICS, *RANK_METRICS):
            rows = rate_rows if metric in RATE_METRICS else rank_rows
            deltas[method][metric] = _paired_delta(
                rows, metric, method, BASELINE,
                seed=SEED + METHODS.index(method) * 71 + sum(metric.encode()),
            )
    return summary, deltas


def _by_benchmark_summary(rate_rows, rank_rows):
    output = {}
    for benchmark in EXTERNAL:
        output[benchmark] = {}
        for method in METHODS:
            rates = [row for row in rate_rows if row["benchmark"] == benchmark and row["method"] == method]
            ranks = [row for row in rank_rows if row["benchmark"] == benchmark and row["method"] == method]
            output[benchmark][method] = {
                "task_count": len({row["task"] for row in rates} | {row["task"] for row in ranks}),
                **{metric: (float(np.mean([row[metric] for row in rates if row.get(metric) is not None]))
                            if any(row.get(metric) is not None for row in rates) else None)
                   for metric in RATE_METRICS},
                **{metric: (float(np.mean([row[metric] for row in ranks if row.get(metric) is not None]))
                            if any(row.get(metric) is not None for row in ranks) else None)
                   for metric in RANK_METRICS},
            }
    return output


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_profile_artifacts(development, external):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = OUT_DIR / "behavior_profiles.jsonl"
    label_rows = []
    all_rows = [*development, *external]
    with profile_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            features = extract_behavior_features(row.names, row.groups)
            tensor = extract_behavior_tensor(row.names, row.groups)
            split = "development" if row.benchmark in {"intercode-ctf", "cybench", "nyuctf"} else "locked_external"
            record = {
                "schema_version": PROFILE_VERSION,
                "trace_id": row.trace_id,
                "benchmark": row.benchmark,
                "feature_dimensions": len(FEATURE_NAMES),
                "features": features,
                "missing_features": [name for name, value in features.items() if value is None],
                "tensor_shape": [TENSOR_PHASES, len(ACTION_LABELS), len(TENSOR_CHANNELS)],
                "tensor_action_labels": list(ACTION_LABELS),
                "tensor_channels": list(TENSOR_CHANNELS),
                "tensor": tensor,
                "fit_scope": "features extracted without labels; model transforms fitted on development training tasks only",
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            label_rows.append({
                "trace_id": row.trace_id,
                "benchmark": row.benchmark,
                "task_for_evaluation_only": row.task,
                "split_for_evaluation_only": split,
                "actor_label_for_evaluation_only": row.label,
                "capture_source_for_evaluation_only": row.source,
                "label_use": "evaluation stratification only; excluded from profile values and pair features",
            })
    _write_csv(OUT_DIR / "evaluation_labels.csv", label_rows)


def run():
    development, external_all, cohort_audit = load_cohort()
    _write_profile_artifacts(development, external_all)
    external_eligibility = cohort_audit["external"]["by_benchmark"]
    eligible_benchmarks = [benchmark for benchmark in EXTERNAL
                           if external_eligibility[benchmark]["primary_external_eligible"]]
    external_primary = [row for row in external_all if row.benchmark in eligible_benchmarks]
    if not development or not external_primary:
        raise RuntimeError("No usable development or primary external rows after the frozen cohort audit")

    action_names = {name for row in [*development, *external_all] for name in row.groups}
    tensor_actions = set(ACTION_LABELS)
    if not action_names <= tensor_actions:
        raise RuntimeError(f"Unrecognized action families: {sorted(action_names - tensor_actions)}")

    # Nested task-held-out assessment on the already-inspected development set.
    nested_folds = split_utils.grouped_folds(development, SEED, n_folds=3)
    nested_records = []
    for outer_index, (outer_train_keys, outer_test_keys) in enumerate(nested_folds, 1):
        outer_train = [row for row in development if (row.benchmark, row.task) in outer_train_keys]
        outer_test = [row for row in development if (row.benchmark, row.task) in outer_test_keys]
        inner_oof = []
        inner_folds = split_utils.grouped_folds(outer_train, SEED + outer_index * 100, n_folds=3)
        for inner_index, (inner_train_keys, inner_test_keys) in enumerate(inner_folds, 1):
            inner_train = [row for row in outer_train if (row.benchmark, row.task) in inner_train_keys]
            inner_test = [row for row in outer_train if (row.benchmark, row.task) in inner_test_keys]
            scorer, _dimensions = fit_methods(inner_train, inner_test, SEED + outer_index * 1000 + inner_index)
            inner_oof.extend(make_pairs(inner_test, scorer, fold=outer_index, split="development_inner_oof"))
        inner_ranks = rank_task_metrics(inner_oof)
        inner_thresholds, inner_selection = select_thresholds(inner_oof, inner_ranks)
        scorer, dimensions = fit_methods(outer_train, outer_test, SEED + outer_index)
        outer_pairs = make_pairs(outer_test, scorer, fold=outer_index, split="development_outer_test")
        calibrated, _ = add_task_metrics(outer_pairs, rank_task_metrics(outer_pairs), inner_thresholds)
        selected = inner_selection["method"]
        for record in calibrated:
            if record["method"] == selected:
                nested_records.append({
                    "outer_fold": outer_index, "selected_method": selected,
                    "benchmark": record["benchmark"], "task": record["task"],
                    **{metric: record.get(metric) for metric in (*RATE_METRICS, "threshold")},
                })
            elif record["method"] == BASELINE:
                nested_records.append({
                    "outer_fold": outer_index, "selected_method": BASELINE,
                    "benchmark": record["benchmark"], "task": record["task"],
                    **{metric: record.get(metric) for metric in (*RATE_METRICS, "threshold")},
                })
        print(f"[development fold {outer_index}] selected={selected}; outer_tasks={len(outer_test_keys)}", flush=True)

    # Cross-fitted development scores calibrate the final method and thresholds.
    development_oof = []
    dev_folds = split_utils.grouped_folds(development, SEED + 9000, n_folds=3)
    for fold_index, (train_keys, test_keys) in enumerate(dev_folds, 1):
        train_rows = [row for row in development if (row.benchmark, row.task) in train_keys]
        test_rows = [row for row in development if (row.benchmark, row.task) in test_keys]
        scorer, _dimensions = fit_methods(train_rows, test_rows, SEED + 9000 + fold_index)
        development_oof.extend(make_pairs(test_rows, scorer, fold=fold_index, split="development_oof"))
    development_ranks = rank_task_metrics(development_oof)
    thresholds, selection = select_thresholds(development_oof, development_ranks)

    # Fit all transforms and the fusion model on development tasks only. External
    # task labels are used only after frozen scores are written to evaluation rows.
    final_scorer, final_dimensions = fit_methods(development, external_all, SEED + 9900)
    external_pairs = make_pairs(external_all, final_scorer, fold=0, split="locked_external")
    external_ranks = rank_task_metrics(external_pairs)
    all_external_rate_rows, _ = add_task_metrics(external_pairs, external_ranks, thresholds)
    primary_pairs = [row for row in external_pairs if row["benchmark"] in eligible_benchmarks]
    primary_rate_rows = [row for row in all_external_rate_rows if row["benchmark"] in eligible_benchmarks]
    primary_rank_rows = [row for row in external_ranks if row["benchmark"] in eligible_benchmarks]
    summary, deltas = _summaries(primary_rate_rows, primary_rank_rows)
    selected_method = selection["method"]
    selected_deltas = deltas.get(selected_method, {})
    fmr_delta = selected_deltas.get("false_merge_rate", {})
    fsr_delta = selected_deltas.get("false_split_rate", {})
    ranking_deltas = {metric: selected_deltas.get(metric, {}) for metric in RANK_METRICS}
    external_directional_deltas = {
        metric: (_directional_value(metric, item["estimate"]) if item.get("estimate") is not None else None)
        for metric, item in selected_deltas.items()
    }
    if selected_method == BASELINE:
        baseline_metrics = _joint_metric_values(primary_rate_rows, primary_rank_rows, BASELINE)
        external_directional_deltas = {metric: 0.0 for metric, value in baseline_metrics.items()
                                       if value is not None}
    external_joint_values = [value for value in external_directional_deltas.values() if value is not None]
    joint_interval = _joint_task_delta_interval(
        primary_rate_rows, primary_rank_rows, selected_method, BASELINE,
        seed=SEED + 777,
    )
    multi_metric_summary = {
        "selected_method": selected_method,
        "compared_metric_count": len(external_joint_values),
        "improved_metric_count": sum(value > 1e-12 for value in external_joint_values),
        "tied_metric_count": sum(abs(value) <= 1e-12 for value in external_joint_values),
        "worsened_metric_count": sum(value < -1e-12 for value in external_joint_values),
        "mean_direction_adjusted_delta": joint_interval["estimate"],
        "joint_task_bootstrap": joint_interval,
        "direction_adjusted_deltas": external_directional_deltas,
        "interpretation": "descriptive point-estimate summary across available prespecified metrics; individual paired intervals remain authoritative",
    }
    fmr_supported = bool(fmr_delta.get("ci_95") and fmr_delta["ci_95"][1] < 0)
    fsr_guard = bool(fsr_delta.get("ci_95") and fsr_delta["ci_95"][1] <= 0.02)
    joint_improvement_supported = bool(joint_interval.get("ci_95") and joint_interval["ci_95"][0] > 0)
    no_clear_rank_regression = selected_method == BASELINE or all(
        item.get("ci_95") is not None and item["ci_95"][1] >= 0
        for item in ranking_deltas.values()
    )
    effective_improvement = (selected_method != BASELINE and fmr_supported and fsr_guard
                             and no_clear_rank_regression and joint_improvement_supported)

    nested_selected = [row for row in nested_records if row["selected_method"] != BASELINE]
    result = {
        "status": "completed nested development evaluation and one-time locked external evaluation",
        "protocol_version": "multidimensional-rerun-v2-joint-objective",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "git_head_at_run": _git_head(),
            "protocol_sha256": _hash_file(EXP / "program.md"),
            "adapter_sha256": _hash_file(EXP / "trace_adapter.py"),
            "runner_sha256": _hash_file(pathlib.Path(__file__).resolve()),
            "source_manifest_sha256": _hash_file(EXP / "source_manifest.json"),
            "development_manifest_sha256": _hash_file(ROOT / "data/manifests/crossbenchmark_public_subset.json"),
        },
        "cohort": cohort_audit,
        "development_nested": {
            "outer_fold_count": 3,
            "task_group_count": len({(row.benchmark, row.task) for row in development}),
            "selected_method_by_fold": [
                {"fold": index + 1, "method": next((row["selected_method"] for row in nested_records
                                                    if row["outer_fold"] == index + 1), None)}
                for index in range(3)
            ],
            "selected_method_task_rows": nested_selected,
            "baseline_task_rows": [row for row in nested_records if row["selected_method"] == BASELINE],
        },
        "selection": {
            **selection,
            "thresholds_by_method": thresholds,
            "development_oof_pair_count": len(development_oof),
            "development_oof_task_count": len({(row["benchmark"], row["query_task"]) for row in development_oof}),
        },
        "metrics": {
            "primary_external_benchmarks": eligible_benchmarks,
            "primary_error_and_ranking_summaries": summary,
            "paired_deltas_vs_command_baseline": deltas,
            "multi_metric_summary": multi_metric_summary,
            "by_benchmark_descriptive": _by_benchmark_summary(all_external_rate_rows, external_ranks),
            "all_external_benchmarks": list(EXTERNAL),
            "primary_metric": "false_merge_rate; lower is better",
            "false_split_guard": "external paired 95% CI upper bound for selected method minus command baseline <= +2 percentage points",
            "effectiveness_assessment": {
                "selected_method": selected_method,
                "false_merge_reduction_supported": fmr_supported,
                "false_split_guard_passed": fsr_guard,
                "joint_multi_metric_improvement_supported": joint_improvement_supported,
                "no_clear_ranking_regression": no_clear_rank_regression,
                "effective_improvement": effective_improvement,
                "interpretation": "meets the preregistered external FMR, FSR, ranking and joint-metric criteria" if effective_improvement else "does not meet all preregistered external criteria",
            },
            "bootstrap": "5,000 benchmark-stratified task-cluster paired resamples",
        },
        "model": {
            "methods": list(METHODS),
            "baseline": BASELINE,
            "dimensions": final_dimensions,
            "selection_constraints": {"training_false_split_rate_max": TARGET_FSR},
            "features": {
                "profile_24d": list(FEATURE_NAMES),
                "tensor": {"shape": [TENSOR_PHASES, 8, len(TENSOR_CHANNELS)], "channels": list(TENSOR_CHANNELS)},
                "partial_sequence": "0.5 * train-IDF weighted command-name Jaccard + 0.5 * train-IDF weighted ordered command-bigram Jaccard",
                "fusion": "logistic regression on five trace-pair similarities, fitted to task-only training pairs",
            },
            "fit_boundary": "vectorizer, IDF, imputation, scaling, Tucker factors and fusion are fitted on development training tasks only",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cohort_audit.json").write_text(json.dumps(cohort_audit, ensure_ascii=False, indent=2) + "\n")
    (OUT_DIR / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    _write_csv(OUT_DIR / "development_oof_pairs.csv", development_oof)
    _write_csv(OUT_DIR / "external_pair_scores.csv", external_pairs)
    _write_csv(OUT_DIR / "external_rate_metrics.csv", all_external_rate_rows)
    _write_csv(OUT_DIR / "external_primary_rate_metrics.csv", primary_rate_rows)
    _write_csv(OUT_DIR / "external_primary_rank_metrics.csv", primary_rank_rows)
    _write_csv(OUT_DIR / "development_nested_metrics.csv", nested_records)
    print(json.dumps({
        "status": result["status"],
        "external_eligible_benchmarks": eligible_benchmarks,
        "selected_method": selection["method"],
        "primary_false_merge_delta": deltas.get(selection["method"], {}).get("false_merge_rate"),
        "false_split_guard_delta": deltas.get(selection["method"], {}).get("false_split_rate"),
        "output": str(OUT_DIR / "results.json"),
    }, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
