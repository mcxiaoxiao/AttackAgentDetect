#!/usr/bin/env python3
"""Nested task-held-out association comparison at fixed training false-merge targets."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import itertools
import json
import pathlib
import sys
import subprocess

import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_mixed_gallery as gallery  # noqa: E402
import run_association_error_analysis as old  # noqa: E402
from run_feature_ablation import sequence_features, grouped_folds  # noqa: E402

OUT = ROOT / "results" / "association_operating_points_v1.json"
PAIR_CSV = ROOT / "results" / "association_operating_points_pairs_v1.csv"
TASK_CSV = ROOT / "results" / "association_operating_points_tasks_v1.csv"
THRESHOLD_CSV = ROOT / "results" / "association_operating_points_thresholds_v1.csv"
FIGURE = ROOT / "results" / "figures" / "association_operating_points_v1"
TARGETS = (0.10, 0.20, 0.30)
METHODS = ("command_cosine", "behavior", "structure_aware", "mode_aware", "length_only")
RATE_METRICS = (*old.RATE_METRICS, "balanced_accuracy")
RANK_METRICS = ("hit_at_1", "mrr", "recall_at_5", "ndcg_at_5")


def vector_pair_features(left_idx, right_idx, all_rows, base, probabilities, structure_vectors):
    left, right = all_rows[left_idx], all_rows[right_idx]
    common = gallery.pair_features(left_idx, right_idx, base, probabilities, with_mode=False)
    p_left, p_right = float(probabilities[left_idx]), float(probabilities[right_idx])
    same_probability = p_left * p_right + (1.0 - p_left) * (1.0 - p_right)
    mode = [same_probability, abs(p_left - p_right)]
    u, v = structure_vectors[left_idx], structure_vectors[right_idx]
    norm = float(np.linalg.norm(u) * np.linalg.norm(v))
    cosine = float(np.dot(u, v) / norm) if norm > 1e-12 else 0.0
    structural = [cosine, float(np.exp(-np.mean(np.abs(u - v))))]
    return common, structural, mode


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
    raw_structures = np.vstack([sequence_features(row) for row in all_rows])
    scaler = StandardScaler().fit(raw_structures[:n_train])
    structure_vectors = scaler.transform(raw_structures)

    train_pairs, pair_y = gallery.training_pairs(train_rows, seed)
    feature_sets = {name: [] for name in ("behavior", "structure_aware", "mode_aware")}
    for left, right in train_pairs:
        common, structural, mode = vector_pair_features(left, right, all_rows, base, probabilities, structure_vectors)
        feature_sets["behavior"].append(common)
        feature_sets["structure_aware"].append([*common, *structural])
        feature_sets["mode_aware"].append([*common, *mode])
    models = {}
    for name, values in feature_sets.items():
        model = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=seed)
        model.fit(np.asarray(values), pair_y)
        models[name] = model
    index_by_id = {row.trace_id: i for i, row in enumerate(all_rows)}

    def score_pairs(row_pairs):
        indices = [(index_by_id[left.trace_id], index_by_id[right.trace_id]) for left, right in row_pairs]
        arrays = {name: [] for name in models}
        command_scores, length_scores = [], []
        for left_idx, right_idx in indices:
            common, structural, mode = vector_pair_features(left_idx, right_idx, all_rows, base, probabilities, structure_vectors)
            arrays["behavior"].append(common)
            arrays["structure_aware"].append([*common, *structural])
            arrays["mode_aware"].append([*common, *mode])
            command_scores.append(base["command"][left_idx, right_idx])
            length_scores.append(common[3])
        output = {name: model.predict_proba(np.asarray(arrays[name]))[:, 1] for name, model in models.items()}
        output["command_cosine"] = np.asarray(command_scores, dtype=float)
        output["length_only"] = np.asarray(length_scores, dtype=float)
        return output

    return score_pairs, proc_vocabulary, len(train_pairs)


def make_pairs(rows, scorer, fold, split):
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
                same_task = int(query.task == candidate.task)
                modes = {query.label, candidate.label}
                pair_type = "H-A" if modes == {"human", "agent"} else ("H-H" if modes == {"human"} else "A-A")
                edge_rows.append({
                    "split": split, "outer_fold": fold, "benchmark": benchmark,
                    "query_task": query.task, "candidate_task": candidate.task,
                    "query_trace_id": query.trace_id, "candidate_trace_id": candidate.trace_id,
                    "query_mode_for_evaluation_only": query.label,
                    "candidate_mode_for_evaluation_only": candidate.label,
                    "same_task_proxy_for_evaluation_only": same_task,
                    "pair_type_for_evaluation_only": pair_type,
                })
                row_pairs.append((query, candidate))
    score_arrays = scorer(row_pairs)
    for i, row in enumerate(edge_rows):
        row.update({key: float(values[i]) for key, values in score_arrays.items()})
    return edge_rows


def choose_threshold(pair_rows, method, target, grid_size=101):
    scores = np.asarray([r[method] for r in pair_rows], dtype=float)
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, grid_size)))
    candidates = np.unique(np.r_[np.nextafter(scores.min(), -np.inf), candidates,
                                 np.nextafter(scores.max(), np.inf)])
    feasible = []
    for threshold in candidates:
        task_rates = old._query_rates(pair_rows, method, float(threshold))
        fmr, benchmark_count, task_count = old._macro_metric(task_rates, "false_merge_rate")
        fsr, _, _ = old._macro_metric(task_rates, "false_split_rate")
        if fmr is not None and fsr is not None and fmr <= target + 1e-12:
            feasible.append((fsr, -float(threshold), float(threshold), fmr, task_count, benchmark_count))
    if not feasible:
        raise RuntimeError(f"No feasible threshold for {method} at target {target}")
    # Minimize false splits subject to the prespecified training false-merge ceiling.
    best = min(feasible)
    return {
        "threshold": best[2], "training_macro_false_merge_rate": best[3],
        "training_macro_false_split_rate": best[0], "validation_task_count": best[4],
        "validation_benchmark_count": best[5], "target_false_merge_rate": target,
        "selection": "lowest validation false-split rate subject to macro false-merge rate <= target; highest threshold tie-break",
        "candidate_count": len(candidates),
    }


def retrieval_task_metrics(rows, pair_rows, methods):
    by_pair = {(r["query_trace_id"], r["candidate_trace_id"]): r for r in pair_rows}
    query_groups = collections.defaultdict(list)
    for row in rows:
        query_groups[(row.benchmark, row.task)].append(row)
    output = []
    for (benchmark, task), traces in sorted(query_groups.items()):
        per_method = collections.defaultdict(list)
        for query in traces:
            candidates = [r for r in rows if r.benchmark == benchmark and r.trace_id != query.trace_id]
            if not candidates:
                continue
            relevance = np.asarray([int(candidate.task == task) for candidate in candidates])
            for method in methods:
                scores = [by_pair[(query.trace_id, candidate.trace_id)][method] for candidate in candidates]
                metric = gallery.expected_rank_metrics(scores, relevance, k=5)
                if metric is not None:
                    per_method[method].append(metric)
        for method, records in per_method.items():
            output.append({
                "benchmark": benchmark, "task": task, "method": method,
                **{name: float(np.mean([row[name] for row in records])) for name in RANK_METRICS},
                "query_count": len(records),
                "candidate_count_mean": float(np.mean([row["candidate_count"] for row in records])),
                "positive_count_mean": float(np.mean([row["positive_count"] for row in records])),
            })
    return output


def summary_with_ci(task_rows, methods, metrics, bootstraps, seed):
    summary = {}
    for method in methods:
        rows = [row for row in task_rows if row["method"] == method]
        summary[method] = {}
        for index, metric in enumerate(metrics):
            summary[method][metric] = old.task_cluster_interval(
                rows, metric, n_boot=bootstraps, seed=seed + index + 43 * methods.index(method)
            )
    return summary


def paired_delta(task_rows, metric, left, right, bootstraps, seed):
    values = collections.defaultdict(list)
    by_key = {(r["benchmark"], r["task"], r["method"]): r.get(metric) for r in task_rows}
    for (benchmark, task, method), value in by_key.items():
        if method != left or value is None:
            continue
        other = by_key.get((benchmark, task, right))
        if other is not None:
            values[benchmark].append(float(value) - float(other))
    if not values:
        return {"estimate": None, "ci_95": None, "paired_task_count": 0}
    estimate = float(np.mean([np.mean(v) for v in values.values()]))
    rng = np.random.default_rng(seed)
    draws = np.empty(bootstraps)
    for index in range(bootstraps):
        draws[index] = np.mean([np.mean(rng.choice(v, size=len(v), replace=True)) for v in values.values()])
    return {"estimate": estimate, "ci_95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "paired_task_count": sum(len(v) for v in values.values()),
            "direction": f"{left} minus {right}", "bootstrap_unit": "benchmark-qualified task within benchmark"}


def run(bootstraps=5000, threshold_grid=101):
    rows = gallery.load_rows()
    if len(rows) != 177 or len({(r.benchmark, r.task) for r in rows}) != 40:
        raise SystemExit("Pinned cohort changed; re-audit before running the association protocol")
    outer_folds = grouped_folds(rows, 20261030, n_folds=3)
    print(f"[dataset] rows={len(rows)} task_groups=40; outer_folds={len(outer_folds)}", flush=True)
    pair_rows, threshold_rows, rate_rows, ranking_rows, fold_details = [], [], [], [], []
    for fold_index, (outer_train_keys, outer_test_keys) in enumerate(outer_folds, 1):
        print(f"[outer fold {fold_index}] train_tasks={len(outer_train_keys)} test_tasks={len(outer_test_keys)}", flush=True)
        outer_train = [r for r in rows if (r.benchmark, r.task) in outer_train_keys]
        outer_test = [r for r in rows if (r.benchmark, r.task) in outer_test_keys]
        calibration_pairs = []
        inner_folds = grouped_folds(outer_train, 20261730 + fold_index, n_folds=3)
        for inner_index, (train_keys, valid_keys) in enumerate(inner_folds, 1):
            print(f"  [threshold OOF {inner_index}] train={len(train_keys)} valid={len(valid_keys)}", flush=True)
            inner_train = [r for r in outer_train if (r.benchmark, r.task) in train_keys]
            inner_valid = [r for r in outer_train if (r.benchmark, r.task) in valid_keys]
            scorer, _, _ = fit_score_model(inner_train, inner_valid, 20261030 + fold_index * 100 + inner_index)
            calibration_pairs.extend(make_pairs(inner_valid, scorer, fold_index, "threshold_oof"))
        if not calibration_pairs:
            raise RuntimeError("Nested threshold calibration produced no pairs")
        thresholds = {}
        for method in METHODS:
            thresholds[method] = {}
            for target in TARGETS:
                fitted = choose_threshold(calibration_pairs, method, target, threshold_grid)
                thresholds[method][target] = fitted
                threshold_rows.append({"outer_fold": fold_index, "method": method, **fitted})

        final_scorer, vocabulary, fit_pair_count = fit_score_model(outer_train, outer_test, 20261030 + fold_index)
        current_pairs = make_pairs(outer_test, final_scorer, fold_index, "outer_test")
        for row in current_pairs:
            for method in METHODS:
                for target in TARGETS:
                    threshold = thresholds[method][target]["threshold"]
                    tag = f"{int(target * 100):02d}"
                    row[f"threshold_{method}_{tag}"] = threshold
                    row[f"associated_{method}_{tag}"] = int(row[method] >= threshold)
        pair_rows.extend(current_pairs)
        for method in METHODS:
            for target in TARGETS:
                tag = f"{int(target * 100):02d}"
                task_rates = old._query_rates(current_pairs, method, thresholds[method][target]["threshold"])
                for record in task_rates:
                    rate_rows.append({"outer_fold": fold_index, "method": method, "target": target, **record})
        ranking_rows.extend(retrieval_task_metrics(outer_test, current_pairs, METHODS))
        fold_details.append({
            "outer_fold": fold_index, "train_task_count": len(outer_train_keys), "test_task_count": len(outer_test_keys),
            "inner_threshold_oof_pairs": len(calibration_pairs), "outer_test_pairs": len(current_pairs),
            "outer_training_pairs_used_for_model": fit_pair_count, "procgrep_vocabulary": vocabulary,
            "thresholds": thresholds,
        })

    rate_summaries = {}
    for target in TARGETS:
        target_rows = [r for r in rate_rows if r["target"] == target]
        rate_summaries[f"{int(target*100):02d}"] = summary_with_ci(
            target_rows, METHODS, RATE_METRICS, bootstraps, 20261030 + int(target * 1000)
        )
    rank_summaries = summary_with_ci(ranking_rows, METHODS, RANK_METRICS, bootstraps, 20264030)
    main_deltas = {}
    for metric_index, metric in enumerate(("false_split_rate", "false_merge_rate", "false_split_human_agent", "false_merge_human_agent", "hit_at_1", "mrr", "recall_at_5")):
        if metric in RATE_METRICS:
            main_deltas[metric] = {
                "mode_vs_behavior": paired_delta([r for r in rate_rows if r["target"] == 0.20], metric,
                                                    "mode_aware", "behavior", bootstraps, 20265000 + metric_index),
                "structure_vs_behavior": paired_delta([r for r in rate_rows if r["target"] == 0.20], metric,
                                                        "structure_aware", "behavior", bootstraps, 20266000 + metric_index),
            }
        else:
            main_deltas[metric] = {
                "mode_vs_behavior": paired_delta(ranking_rows, metric, "mode_aware", "behavior", bootstraps, 20267000 + metric_index),
                "structure_vs_behavior": paired_delta(ranking_rows, metric, "structure_aware", "behavior", bootstraps, 20268000 + metric_index),
            }

    manifest_path = ROOT / "data" / "manifests" / "crossbenchmark_public_subset.json"
    manifest = json.loads(manifest_path.read_text())
    result = {
        "status": "completed exploratory nested task-held-out association evaluation",
        "protocol_version": "association-operating-points-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"name": "Cyber Task Horizons public CTF traces", "revision": manifest["revision"],
                    "license": manifest.get("data_license", "CC BY 4.0"), "records": len(rows),
                    "benchmark_qualified_task_groups": 40,
                    "mode_capture_confound": "Human terminal recordings and Agent tool-call records are distinct, source-confounded capture pathways."},
        "provenance": {
            "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "input_manifest_sha256": {
                "cth_intercode": hashlib.sha256((ROOT / "data" / "raw" / "cyber-task-horizons" / "download_manifest.json").read_bytes()).hexdigest(),
                "cth_crossbenchmark": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            },
            "code_sha256": {
                "runner": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                "threshold_helpers": hashlib.sha256((ROOT / "scripts" / "run_association_error_analysis.py").read_bytes()).hexdigest(),
                "feature_source": hashlib.sha256((ROOT / "scripts" / "run_feature_ablation.py").read_bytes()).hexdigest(),
                "baseline_source": hashlib.sha256((ROOT / "scripts" / "run_mixed_gallery.py").read_bytes()).hexdigest(),
                "trace_parser": hashlib.sha256((ROOT / "attackagent" / "traces.py").read_bytes()).hexdigest(),
            },
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        },
        "evaluation": {
            "outer": "3 benchmark-stratified folds grouped by (benchmark, task), seed 20261030",
            "threshold_fit": "3 nested task-grouped folds inside outer training only; choose minimum false split subject to training macro false merge <= 10%, 20%, or 30%",
            "candidate_set": "all held-out traces in the same benchmark; query itself excluded; execution mode never filters candidates",
            "positive_proxy": "same benchmark-qualified CTF task, not same attack activity",
            "pair_sampling": "training pairs sampled using benchmark/task identity only; mode labels never construct pairs",
            "methods": {
                "command_cosine": "command-name TF-IDF unigram/bigram cosine",
                "behavior": "logistic pair ranker over command cosine, action-bigram cosine, ProcGrep similarity, and length similarity",
                "structure_aware": "behavior ranker plus cosine and exp(-mean absolute difference) over standardized fixed sequence statistics",
                "mode_aware": "behavior ranker plus task-OOF execution-mode agreement probability and absolute probability gap",
                "length_only": "negative-control length similarity",
            },
            "error_aggregation": "candidate pairs within query, queries within task, tasks within benchmark, benchmarks equally weighted",
            "ranking_aggregation": "queries within task, tasks within benchmark, benchmarks equally weighted; expected tie handling",
            "intervals": f"{bootstraps:,} paired benchmark-stratified task-cluster bootstrap draws, conditional on fitted folds",
            "threshold_grid": threshold_grid,
        },
        "folds": fold_details,
        "false_merge_targets": {f"{int(t*100):02d}": {"summary": rate_summaries[f"{int(t*100):02d}"]} for t in TARGETS},
        "ranking_summary": rank_summaries,
        "paired_deltas_at_20_percent_target": main_deltas,
        "outputs": {"pair_scores": str(PAIR_CSV.relative_to(ROOT)), "task_metrics": str(TASK_CSV.relative_to(ROOT)),
                    "threshold_audit": str(THRESHOLD_CSV.relative_to(ROOT)),
                    "figure_script": "scripts/plot_association_operating_points.py"},
        "interpretation_guard": "These are CTF same-task proxy errors. They do not demonstrate false merges or splits among real attack activities, Agent-caused failure of APT correlation, or APT actor attribution. Mode scores inherit the release's capture-path confound.",
    }
    import csv
    for path, records in ((PAIR_CSV, pair_rows), (TASK_CSV, [*rate_rows, *ranking_rows]), (THRESHOLD_CSV, threshold_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if records:
            fields = list(dict.fromkeys(key for record in records for key in record))
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader(); writer.writerows(records)
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
