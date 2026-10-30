#!/usr/bin/env python3
"""Nested task-held-out association evaluation for scalar and tensor profiles."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_mixed_gallery as gallery  # noqa: E402
import run_association_error_analysis as old  # noqa: E402
from run_feature_ablation import grouped_folds, sequence_features  # noqa: E402
from run_association_operating_points import (  # noqa: E402
    TARGETS, make_pairs, paired_delta, retrieval_task_metrics,
    summary_with_ci,
)
from attackagent.profiles import (  # noqa: E402
    ACTION_LABELS, FEATURE_NAMES, TENSOR_CHANNELS, TENSOR_PHASES,
    build_behavior_profile, extract_behavior_features, extract_behavior_tensor,
)
from attackagent import ModeScore, TraceEvidence  # noqa: E402

OUT_DIR = ROOT / "results" / "multidimensional_profile_association_v1"
OUT = OUT_DIR / "results.json"
PAIR_CSV = OUT_DIR / "pair_scores.csv"
TASK_CSV = OUT_DIR / "task_metrics.csv"
THRESHOLD_CSV = OUT_DIR / "thresholds.csv"
PROFILE_JSONL = OUT_DIR / "behavior_profiles_v3.jsonl"
LABELS_CSV = OUT_DIR / "evaluation_labels.csv"
METHODS = (
    "command_cosine", "behavior", "structure_aware", "mode_aware", "length_only",
    "profile_only", "tensor_flat_only", "tensor_tucker_only",
    "behavior_profile", "behavior_tensor_flat", "behavior_tensor_tucker",
    "behavior_mode_tucker",
)
RATE_METRICS = (*old.RATE_METRICS, "balanced_accuracy", "query_has_accepted_match")
RANK_METRICS = ("hit_at_1", "mrr", "recall_at_5", "ndcg_at_5")
TUCKER_RANKS = (2, 4, 2)


def _scalar_matrix(rows):
    return np.asarray([
        [np.nan if features[name] is None else float(features[name])
         for name in FEATURE_NAMES]
        for features in (extract_behavior_features(row.names, row.groups) for row in rows)
    ], dtype=float)


def _tensor_matrix(rows):
    tensors = []
    for row in rows:
        if len(row.names) != len(row.groups):
            raise ValueError(f"unaligned command/action sequence: {row.trace_id}")
        tensors.append(np.asarray(extract_behavior_tensor(row.names, row.groups), dtype=float))
    return np.stack(tensors)


def _fit_hosvd_factors(training_tensors):
    factors = []
    for mode, requested_rank in enumerate(TUCKER_RANKS):
        fibers = np.concatenate([
            np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1).T
            for tensor in training_tensors
        ], axis=0)
        _, _, right = np.linalg.svd(fibers, full_matrices=False)
        rank = min(requested_rank, right.shape[0])
        factors.append(right[:rank].T.copy())
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
    all_transformed = scaler.transform(imputer.transform(all_values))
    return train, all_transformed


def _cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def _profile_blocks(train_rows, all_rows):
    n_train = len(train_rows)
    scalar_raw = _scalar_matrix(all_rows)
    scalar_train, scalar_all = _standardize(scalar_raw[:n_train], scalar_raw)

    tensor_raw = _tensor_matrix(all_rows)
    tensor_train_flat, tensor_all_flat = _standardize(
        tensor_raw[:n_train].reshape(n_train, -1), tensor_raw.reshape(len(all_rows), -1)
    )
    factors = _fit_hosvd_factors(tensor_raw[:n_train])
    projected = np.stack([_project_tucker(tensor, factors) for tensor in tensor_raw])
    tensor_train_tucker, tensor_all_tucker = _standardize(
        projected[:n_train].reshape(n_train, -1), projected.reshape(len(all_rows), -1)
    )
    return {
        "profile": scalar_all,
        "tensor_flat": tensor_all_flat,
        "tensor_tucker": tensor_all_tucker,
        "tensor_raw": tensor_raw,
        "hosvd_factors": factors,
    }


def _pair_design(left, right, all_rows, base, probabilities, sequence_vectors, blocks):
    common = gallery.pair_features(left, right, base, probabilities, with_mode=False)
    mode = gallery.pair_features(left, right, base, probabilities, with_mode=True)[-2:]
    structural = []
    u, v = sequence_vectors[left], sequence_vectors[right]
    structural.extend([_cosine(u, v), float(np.exp(-np.mean(np.abs(u - v))))])
    profile_delta = np.abs(blocks["profile"][left] - blocks["profile"][right])
    flat_delta = np.abs(blocks["tensor_flat"][left] - blocks["tensor_flat"][right])
    tucker_delta = np.abs(blocks["tensor_tucker"][left] - blocks["tensor_tucker"][right])
    return {
        "behavior": np.asarray(common),
        "structure_aware": np.asarray([*common, *structural]),
        "mode_aware": np.asarray([*common, *mode]),
        "profile_only": profile_delta,
        "tensor_flat_only": flat_delta,
        "tensor_tucker_only": tucker_delta,
        "behavior_profile": np.concatenate([common, profile_delta]),
        "behavior_tensor_flat": np.concatenate([common, flat_delta]),
        "behavior_tensor_tucker": np.concatenate([common, tucker_delta]),
        "behavior_mode_tucker": np.concatenate([common, mode, tucker_delta]),
    }


def fit_score_model(train_rows, score_rows, seed):
    all_rows = [*train_rows, *score_rows]
    n_train = len(train_rows)
    base = {
        "rows": all_rows,
        "command": gallery.cosine_matrix(train_rows, all_rows, gallery.text_commands, (1, 2)),
        "action": gallery.cosine_matrix(train_rows, all_rows, gallery.action_bigram_text, (1, 1)),
    }
    base["procgrep"], proc_vocabulary = gallery.procgrep_matrix(train_rows, all_rows)
    probabilities, _, _ = gallery.fit_mode_scores(train_rows, all_rows, seed)
    sequence_raw = np.vstack([sequence_features(row) for row in all_rows])
    sequence_train, sequence_all = _standardize(sequence_raw[:n_train], sequence_raw)
    blocks = _profile_blocks(train_rows, all_rows)
    train_pairs, pair_y = gallery.training_pairs(train_rows, seed)

    indices = {row.trace_id: index for index, row in enumerate(all_rows)}
    train_designs = collections.defaultdict(list)
    for left, right in train_pairs:
        for name, features in _pair_design(left, right, all_rows, base, probabilities,
                                           sequence_all, blocks).items():
            train_designs[name].append(features)
    models = {}
    for name in train_designs:
        model = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000,
                                   random_state=seed)
        model.fit(np.asarray(train_designs[name]), pair_y)
        models[name] = model

    def score_many(row_pairs):
        scores = {name: [] for name in models}
        command_scores = []
        length_scores = []
        for left_row, right_row in row_pairs:
            left, right = indices[left_row.trace_id], indices[right_row.trace_id]
            features = _pair_design(left, right, all_rows, base, probabilities,
                                    sequence_all, blocks)
            for name, vector in features.items():
                scores[name].append(vector)
            command_scores.append(base["command"][left, right])
            length_scores.append(gallery.pair_features(left, right, base, probabilities,
                                                        with_mode=False)[3])
        result = {name: model.predict_proba(np.asarray(scores[name]))[:, 1]
                  for name, model in models.items()}
        result["command_cosine"] = np.asarray(command_scores, dtype=float)
        result["length_only"] = np.asarray(length_scores, dtype=float)
        return result

    mode_predictions = [
        {"trace_id": row.trace_id,
         "agent_like_score": float(probabilities[n_train + index]),
         "prediction": "agent" if probabilities[n_train + index] >= 0.5 else "human"}
        for index, row in enumerate(score_rows)
    ]
    return score_many, proc_vocabulary, len(train_pairs), {
        "scalar_profile_dimensions": len(FEATURE_NAMES),
        "tensor_shape": list(blocks["tensor_raw"].shape[1:]),
        "tensor_flat_dimensions": int(np.prod(blocks["tensor_raw"].shape[1:])),
        "tucker_core_shape": list(blocks["tensor_tucker"].shape[1:]),
        "tucker_ranks_requested": list(TUCKER_RANKS),
        "tucker_factors_fit_on": "outer or inner training traces only",
    }, mode_predictions


def _query_match_coverage(pair_rows, method, threshold):
    by_query = collections.defaultdict(list)
    for row in pair_rows:
        if row["same_task_proxy_for_evaluation_only"]:
            by_query[(row["benchmark"], row["query_task"], row["query_trace_id"])].append(
                row[method] >= threshold
            )
    task_values = collections.defaultdict(list)
    for (benchmark, task, _query_id), accepted in by_query.items():
        task_values[(benchmark, task)].append(float(any(accepted)))
    return [
        {"benchmark": benchmark, "task": task,
         "query_has_accepted_match": float(np.mean(values)),
         "query_count": len(values)}
        for (benchmark, task), values in sorted(task_values.items())
    ]


def choose_threshold_fast(pair_rows, method, target, grid_size=101):
    """Same macro objective as the baseline, vectorized over sorted query scores."""
    scores = np.asarray([row[method] for row in pair_rows], dtype=float)
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, grid_size)))
    candidates = np.unique(np.r_[np.nextafter(scores.min(), -np.inf), candidates,
                                 np.nextafter(scores.max(), np.inf)])
    grouped = collections.defaultdict(lambda: {"benchmark": "", "task": "", "positive": [], "negative": []})
    for row in pair_rows:
        key = (row["benchmark"], row["query_task"], row["query_trace_id"])
        group = grouped[key]
        group["benchmark"], group["task"] = row["benchmark"], row["query_task"]
        side = "positive" if row["same_task_proxy_for_evaluation_only"] else "negative"
        group[side].append(float(row[method]))

    task_query_rates = collections.defaultdict(lambda: {"benchmark": "", "split": [], "merge": []})
    for (benchmark, task, _query_id), group in grouped.items():
        task = task_query_rates[(benchmark, task)]
        task["benchmark"] = benchmark
        if group["positive"]:
            positive = np.sort(np.asarray(group["positive"], dtype=float))
            task["split"].append(np.searchsorted(positive, candidates, side="left") / len(positive))
        if group["negative"]:
            negative = np.sort(np.asarray(group["negative"], dtype=float))
            accepted = len(negative) - np.searchsorted(negative, candidates, side="left")
            task["merge"].append(accepted / len(negative))

    task_split, task_merge = {}, {}
    benchmark_split = collections.defaultdict(list)
    benchmark_merge = collections.defaultdict(list)
    for key, values in task_query_rates.items():
        benchmark, _task = key
        if values["split"]:
            task_split[key] = np.mean(values["split"], axis=0)
            benchmark_split[benchmark].append(task_split[key])
        if values["merge"]:
            task_merge[key] = np.mean(values["merge"], axis=0)
            benchmark_merge[benchmark].append(task_merge[key])
    benchmark_ids = sorted(set(benchmark_split) & set(benchmark_merge))
    if not benchmark_ids:
        raise RuntimeError("No validation tasks have both positive and negative pairs")
    false_split = np.mean([np.mean(benchmark_split[name], axis=0) for name in benchmark_ids], axis=0)
    false_merge = np.mean([np.mean(benchmark_merge[name], axis=0) for name in benchmark_ids], axis=0)
    feasible = np.flatnonzero(false_merge <= target + 1e-12)
    if not len(feasible):
        raise RuntimeError(f"No feasible threshold for {method} at target {target}")
    best_index = min(feasible, key=lambda index: (false_split[index], -candidates[index]))
    return {
        "threshold": float(candidates[best_index]),
        "training_macro_false_merge_rate": float(false_merge[best_index]),
        "training_macro_false_split_rate": float(false_split[best_index]),
        "validation_task_count": len(task_query_rates),
        "validation_benchmark_count": len(benchmark_ids),
        "target_false_merge_rate": target,
        "selection": "lowest validation false-split rate subject to macro false-merge rate <= target; highest threshold tie-break",
        "candidate_count": len(candidates),
        "search_implementation": "pre-sort pair scores within each query; vectorized candidate thresholds",
    }


def _descriptive_tensor_means(rows):
    """Label-stratified visualization summaries; never passed into a model."""
    tensors = _tensor_matrix(rows)
    output = {}
    labels = np.asarray([row.label for row in rows])
    for label in ("human", "agent"):
        selected = tensors[labels == label]
        output[label] = {
            "trace_count": int(len(selected)),
            "mean_tensor": selected.mean(axis=0).tolist(),
            "scope": "descriptive only; label is capture-path confounded and excluded from model features",
        }
    return output


def run(bootstraps=5000, threshold_grid=101):
    rows = gallery.load_rows()
    task_groups = {(row.benchmark, row.task) for row in rows}
    if len(rows) != 177 or len(task_groups) != 40:
        raise SystemExit("Pinned cohort changed; re-audit before running the multidimensional protocol")
    action_set = set(ACTION_LABELS)
    bad = [row.trace_id for row in rows if len(row.names) != len(row.groups)
           or not set(row.groups) <= action_set]
    if bad:
        raise SystemExit(f"Tensor schema is unavailable for {len(bad)} pinned traces; first={bad[0]}")

    outer_folds = grouped_folds(rows, 20261030, n_folds=3)
    print(f"[dataset] rows={len(rows)} task_groups={len(task_groups)}; outer_folds={len(outer_folds)}", flush=True)
    pair_rows, rate_rows, ranking_rows, threshold_rows, fold_details = [], [], [], [], []
    heldout_mode_predictions = []
    for fold_index, (outer_train_keys, outer_test_keys) in enumerate(outer_folds, 1):
        print(f"[outer fold {fold_index}] train_tasks={len(outer_train_keys)} test_tasks={len(outer_test_keys)}", flush=True)
        outer_train = [row for row in rows if (row.benchmark, row.task) in outer_train_keys]
        outer_test = [row for row in rows if (row.benchmark, row.task) in outer_test_keys]
        calibration_pairs = []
        inner_folds = grouped_folds(outer_train, 20261730 + fold_index, n_folds=3)
        for inner_index, (train_keys, valid_keys) in enumerate(inner_folds, 1):
            inner_train = [row for row in outer_train if (row.benchmark, row.task) in train_keys]
            inner_valid = [row for row in outer_train if (row.benchmark, row.task) in valid_keys]
            scorer, _vocabulary, _pair_count, _dimensions, _mode_predictions = fit_score_model(
                inner_train, inner_valid, 20261030 + fold_index * 100 + inner_index
            )
            calibration_pairs.extend(make_pairs(inner_valid, scorer, fold_index, "threshold_oof"))
            print(f"  [threshold OOF {inner_index}] train={len(train_keys)} valid={len(valid_keys)}", flush=True)
        if not calibration_pairs:
            raise RuntimeError("Nested threshold calibration produced no pairs")

        thresholds = {}
        for method in METHODS:
            thresholds[method] = {}
            for target in TARGETS:
                fitted = choose_threshold_fast(calibration_pairs, method, target, threshold_grid)
                thresholds[method][target] = fitted
                threshold_rows.append({"outer_fold": fold_index, "method": method, **fitted})

        scorer, vocabulary, fit_pair_count, dimensions, mode_predictions = fit_score_model(
            outer_train, outer_test, 20261030 + fold_index
        )
        heldout_mode_predictions.extend(
            {**row, "outer_fold": fold_index} for row in mode_predictions
        )
        current_pairs = make_pairs(outer_test, scorer, fold_index, "outer_test")
        for row in current_pairs:
            for method in METHODS:
                for target in TARGETS:
                    threshold = thresholds[method][target]["threshold"]
                    row[f"threshold_{method}_{int(target * 100):02d}"] = threshold
                    row[f"associated_{method}_{int(target * 100):02d}"] = int(row[method] >= threshold)
        pair_rows.extend(current_pairs)
        for method in METHODS:
            for target in TARGETS:
                task_rates = old._query_rates(
                    current_pairs, method, thresholds[method][target]["threshold"]
                )
                coverage = _query_match_coverage(
                    current_pairs, method, thresholds[method][target]["threshold"]
                )
                coverage_by_task = {(row["benchmark"], row["task"]): row
                                    for row in coverage}
                for record in task_rates:
                    record.update(coverage_by_task.get((record["benchmark"], record["task"]), {}))
                    rate_rows.append({"outer_fold": fold_index, "method": method,
                                      "target": target, **record})
        ranking_rows.extend(retrieval_task_metrics(outer_test, current_pairs, METHODS))
        fold_details.append({
            "outer_fold": fold_index,
            "train_task_count": len(outer_train_keys),
            "test_task_count": len(outer_test_keys),
            "inner_threshold_oof_pairs": len(calibration_pairs),
            "outer_test_pairs": len(current_pairs),
            "outer_training_pairs_used_for_model": fit_pair_count,
            "procgrep_vocabulary": vocabulary,
            "profile_dimensions": dimensions,
            "thresholds": thresholds,
        })

    rate_summaries = {}
    for target in TARGETS:
        target_rows = [row for row in rate_rows if row["target"] == target]
        rate_summaries[f"{int(target * 100):02d}"] = summary_with_ci(
            target_rows, METHODS, RATE_METRICS, bootstraps,
            20261030 + int(target * 1000)
        )
    rank_summaries = summary_with_ci(ranking_rows, METHODS, RANK_METRICS,
                                     bootstraps, 20264030)
    comparisons = {}
    primary_rate_rows = [row for row in rate_rows if row["target"] == 0.20]
    for metric_index, metric in enumerate((*old.RATE_METRICS, "hit_at_1", "mrr", "recall_at_5", "ndcg_at_5")):
        source = primary_rate_rows if metric in RATE_METRICS else ranking_rows
        comparisons[metric] = {
            method: paired_delta(source, metric, method, "behavior", bootstraps,
                                 20265000 + metric_index * 100 + METHODS.index(method))
            for method in METHODS if method not in {"behavior", "command_cosine", "length_only"}
        }

    manifest_path = ROOT / "data" / "manifests" / "crossbenchmark_public_subset.json"
    manifest = json.loads(manifest_path.read_text())
    result = {
        "status": "completed exploratory nested task-held-out multidimensional profile evaluation",
        "protocol_version": "multidimensional-profile-association-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "Cyber Task Horizons public CTF traces",
            "revision": manifest["revision"],
            "license": manifest.get("data_license", "CC BY 4.0"),
            "records": len(rows),
            "benchmark_qualified_task_groups": len(task_groups),
            "positive_proxy": "same benchmark-qualified CTF task, not same attack activity",
            "mode_capture_confound": "Human terminal recordings and Agent tool-call records are distinct, source-confounded capture pathways.",
        },
        "provenance": {
            "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "input_manifest_sha256": {
                "cth_crossbenchmark": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "cth_download": hashlib.sha256((ROOT / "data/raw/cyber-task-horizons/download_manifest.json").read_bytes()).hexdigest(),
            },
            "code_sha256": {
                "runner": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                "baseline_runner": hashlib.sha256((ROOT / "scripts/run_association_operating_points.py").read_bytes()).hexdigest(),
                "profile_source": hashlib.sha256((ROOT / "attackagent/profiles.py").read_bytes()).hexdigest(),
                "gallery_source": hashlib.sha256((ROOT / "scripts/run_mixed_gallery.py").read_bytes()).hexdigest(),
                "threshold_source": hashlib.sha256((ROOT / "scripts/run_association_error_analysis.py").read_bytes()).hexdigest(),
                "feature_source": hashlib.sha256((ROOT / "scripts/run_feature_ablation.py").read_bytes()).hexdigest(),
            },
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        },
        "evaluation": {
            "outer": "3 benchmark-stratified folds grouped by (benchmark, task), seed 20261030",
            "threshold_fit": "3 nested task-grouped folds inside outer training only; choose minimum false split subject to training macro false merge <= 10%, 20%, or 30%",
            "candidate_set": "all held-out traces in the same benchmark; query itself excluded; execution mode never filters candidates",
            "pair_sampling": "training pairs sampled by benchmark/task only; mode labels do not construct pairs",
            "profile": {
                "scalar_dimensions": list(FEATURE_NAMES),
                "tensor_axes": {"phase": TENSOR_PHASES, "action_family": list(ACTION_LABELS), "channels": list(TENSOR_CHANNELS)},
                "tensor_shape": [TENSOR_PHASES, len(ACTION_LABELS), len(TENSOR_CHANNELS)],
                "tucker_ranks": list(TUCKER_RANKS),
                "tucker_fit": "per-fold training traces only, via mode-unfolding SVD and Tucker core projection",
            },
            "methods": {method: method for method in METHODS},
            "error_aggregation": "candidate pairs within query, queries within task, tasks within benchmark, benchmarks equally weighted",
            "ranking_aggregation": "queries within task, tasks within benchmark, benchmarks equally weighted; expected tie handling",
            "intervals": f"{bootstraps:,} paired benchmark-stratified task-cluster bootstrap draws, conditional on fitted folds",
            "threshold_grid": threshold_grid,
        },
        "folds": fold_details,
        "false_merge_targets": {
            f"{int(target * 100):02d}": {"summary": rate_summaries[f"{int(target * 100):02d}"]}
            for target in TARGETS
        },
        "ranking_summary": rank_summaries,
        "paired_deltas_at_20_percent_target_vs_behavior": comparisons,
        "descriptive_tensor_means_by_record_label": _descriptive_tensor_means(rows),
        "outputs": {"pair_scores": str(PAIR_CSV.relative_to(ROOT)),
                    "task_metrics": str(TASK_CSV.relative_to(ROOT)),
                    "threshold_audit": str(THRESHOLD_CSV.relative_to(ROOT)),
                    "saved_profiles": str(PROFILE_JSONL.relative_to(ROOT)),
                    "label_sidecar": str(LABELS_CSV.relative_to(ROOT))},
        "interpretation_guard": "This is CTF same-task proxy association. Results do not demonstrate linkage among real attacks, attribution to a person or organization, or that Agent participation caused association errors.",
    }
    import csv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, records in ((PAIR_CSV, pair_rows),
                          (TASK_CSV, [*rate_rows, *ranking_rows]),
                          (THRESHOLD_CSV, threshold_rows)):
        if records:
            fields = list(dict.fromkeys(key for record in records for key in record))
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)
    mode_by_id = {row["trace_id"]: row for row in heldout_mode_predictions}
    row_by_id = {row.trace_id: row for row in rows}
    with PROFILE_JSONL.open("w", encoding="utf-8") as handle:
        for trace_id in sorted(row_by_id):
            row = row_by_id[trace_id]
            mode = mode_by_id[trace_id]
            source_format = "cth-comparable-commands-v1"
            evidence = TraceEvidence(trace_id, source_format, row.benchmark,
                                     row.names, row.groups)
            score = ModeScore(trace_id, mode["prediction"], mode["agent_like_score"],
                              False, True, None, len(row.names), source_format)
            profile = build_behavior_profile(evidence, score)
            profile["mode_estimate"]["provenance"] = (
                "outer task-held-out command TF-IDF score; uncalibrated, not actor identity"
            )
            handle.write(json.dumps(profile, ensure_ascii=False, allow_nan=False) + "\n")
    with LABELS_CSV.open("w", newline="", encoding="utf-8") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=("trace_id", "benchmark", "task",
                                                    "record_label_for_evaluation_only", "outer_fold"))
        writer.writeheader()
        for row in rows:
            mode = mode_by_id[row.trace_id]
            writer.writerow({"trace_id": row.trace_id, "benchmark": row.benchmark,
                             "task": row.task, "record_label_for_evaluation_only": row.label,
                             "outer_fold": mode["outer_fold"]})
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--threshold-grid", type=int, default=101)
    args = parser.parse_args()
    if args.bootstraps < 100 or args.threshold_grid < 21:
        parser.error("Use >=100 bootstraps and >=21 threshold grid points")
    print(json.dumps(run(args.bootstraps, args.threshold_grid), ensure_ascii=False, indent=2))
