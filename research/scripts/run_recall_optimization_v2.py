#!/usr/bin/env python3
"""Nested evaluation of task-weighted and cross-mode association strategies.

This is a fixed second exploratory batch motivated by the v1 error audit. All
fitting, method selection and threshold selection happen inside outer training
tasks. The target remains same-task retrieval, not actor attribution.
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_recall_optimization_v1 as v1
import run_association_error_analysis as old
import run_association_operating_points as op
import run_mixed_gallery as gallery
from run_feature_ablation import BASE_SEED, grouped_folds

OUT = ROOT / "results" / "recall_optimization_v2"
TARGETS = (0.10, 0.20, 0.30)
MAIN_TARGET = 0.20
BOOTSTRAPS = 5000
GRID = 101
BASE_METHODS = ("behavior", "behavior_align", "behavior_multiscale", "behavior_align_profile")
NEW_METHODS = ("behavior_align_taskweighted", "behavior_align_ha")
METHODS = (*BASE_METHODS, *NEW_METHODS)
STRATIFIED_METHODS = ("behavior", "behavior_align", *NEW_METHODS)
RATE_METRICS = ("false_split_rate", "false_merge_rate", "false_split_human_agent",
                "false_merge_human_agent", "balanced_accuracy")
JOINT_METRICS = (*RATE_METRICS, "coverage", *op.RANK_METRICS)


def pair_kind(row):
    return "ha" if row["pair_type_for_evaluation_only"] == "H-A" else "same"


def weighted_pairs(rows, *, cross_mode_only=False):
    """All positives plus all in-benchmark negatives, with equal task mass."""
    by_task: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i, row in enumerate(rows):
        by_task[(row.benchmark, row.task)].append(i)
    by_benchmark: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for key in sorted(by_task):
        by_benchmark[key[0]].append(key)

    positives: list[tuple[int, int, tuple[str, str]]] = []
    for key, members in sorted(by_task.items()):
        if cross_mode_only:
            members = [i for i in members if rows[i].label == "human"], [i for i in members if rows[i].label == "agent"]
            pos = [(min(h, a), max(h, a), key) for h in members[0] for a in members[1]]
        else:
            pos = [(a, b, key) for a, b in itertools.combinations(members, 2)]
        positives.extend(pos)

    negatives: list[tuple[int, int, tuple[str, str], tuple[str, str]]] = []
    incident_counts: collections.Counter = collections.Counter()
    for benchmark, keys in sorted(by_benchmark.items()):
        for left_key, right_key in itertools.combinations(keys, 2):
            left, right = by_task[left_key], by_task[right_key]
            for a in left:
                for b in right:
                    if cross_mode_only and rows[a].label == rows[b].label:
                        continue
                    incident_counts[left_key] += 1
                    incident_counts[right_key] += 1
                    negatives.append((min(a, b), max(a, b), left_key, right_key))
    pos_tasks = {key for _, _, key in positives}
    neg_tasks = {key for _, _, left, right in negatives for key in (left, right)}
    if not positives or not negatives or not pos_tasks or not neg_tasks:
        raise ValueError("Task-weighted sampling requires both pair classes")

    records = []
    pos_per_task = collections.Counter(key for _, _, key in positives)
    for a, b, key in positives:
        weight = 0.5 / len(pos_tasks) / pos_per_task[key]
        records.append(((a, b), 1, weight))
    for a, b, left_key, right_key in negatives:
        weight = 0.5 / len(neg_tasks) * 0.5 * (
            1 / incident_counts[left_key] + 1 / incident_counts[right_key]
        )
        records.append(((a, b), 0, weight))
    return ([x[0] for x in records], np.asarray([x[1] for x in records], dtype=int),
            np.asarray([x[2] for x in records], dtype=float))


def fit_taskweighted_scorers(train_rows, score_rows, seed):
    """Fit one task-weighted ranker and one H-A-only ranker."""
    rows = [*train_rows, *score_rows]
    n_train = len(train_rows)
    base = {
        "rows": rows,
        "command": gallery.cosine_matrix(train_rows, rows, gallery.text_commands, (1, 2)),
        "action": gallery.cosine_matrix(train_rows, rows, gallery.action_bigram_text, (1, 1)),
    }
    base["procgrep"], _ = gallery.procgrep_matrix(train_rows, rows)
    counts = collections.Counter(token for row in train_rows for token in set(row.names))
    idf = {token: math.log((n_train + 1) / (count + 1)) + 1 for token, count in counts.items()}
    action_bins = {bins: v1.phase_action(rows, bins) for bins in (2, 4, 8)}
    raw = np.asarray([[np.nan if vals[name] is None else vals[name]
                       for name in v1.FEATURE_NAMES]
                      for vals in (v1.extract_behavior_features(r.names, r.groups) for r in rows)], dtype=float)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    profiles = scaler.fit_transform(imputer.fit_transform(raw[:n_train]))
    profiles = scaler.transform(imputer.transform(raw))

    index = {row.trace_id: i for i, row in enumerate(rows)}
    arrays = {}
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            common, align, _multi, _profile = v1.pair_vector(
                left, right, base, rows, index, idf, action_bins, profiles
            )
            arrays[(left, right)] = np.r_[common, align]

    def build_model(pair_data, model_seed):
        pairs, labels, weights = pair_data
        X = np.asarray([arrays[(min(a, b), max(a, b))] for a, b in pairs])
        model = LogisticRegression(C=0.5, max_iter=3000, random_state=model_seed)
        model.fit(X, labels, sample_weight=weights)
        return model

    weighted_data = weighted_pairs(train_rows)
    ha_data = weighted_pairs(train_rows, cross_mode_only=True)
    taskweighted = build_model(weighted_data, seed)
    ha_specialist = build_model(ha_data, seed + 1)

    def score_pairs(row_pairs):
        if not row_pairs:
            return {name: np.asarray([], dtype=float) for name in NEW_METHODS}
        features = np.asarray([arrays[(min(index[a.trace_id], index[b.trace_id]),
                                       max(index[a.trace_id], index[b.trace_id]))]
                               for a, b in row_pairs])
        labels = np.asarray([a.label != b.label for a, b in row_pairs], dtype=bool)
        scores = np.empty(len(row_pairs), dtype=float)
        scores[~labels] = taskweighted.predict_proba(features[~labels])[:, 1]
        scores[labels] = ha_specialist.predict_proba(features[labels])[:, 1]
        return {
            "behavior_align_taskweighted": taskweighted.predict_proba(features)[:, 1],
            "behavior_align_ha": scores,
        }

    return score_pairs, {
        "taskweighted_pair_count": len(weighted_data[0]),
        "ha_pair_count": len(ha_data[0]),
    }


def choose_stratified_threshold(pair_rows, method, target, kind):
    subset = [row for row in pair_rows if pair_kind(row) == kind]
    if not subset:
        raise RuntimeError(f"No calibration pairs for stratum={kind}")
    return op.choose_threshold(subset, method, target, GRID)


def apply_stratified(pair_rows, method, thresholds, output_name):
    result = []
    for row in pair_rows:
        row = dict(row)
        threshold = thresholds[pair_kind(row)]
        if isinstance(threshold, dict):
            threshold = threshold["threshold"]
        row[output_name] = int(row[method] >= threshold)
        result.append(row)
    return result


def macro_value(task_rows, method, metric):
    by_benchmark = collections.defaultdict(list)
    for row in task_rows:
        if row["method"] == method and row.get(metric) is not None:
            by_benchmark[row["benchmark"]].append(float(row[metric]))
    return float(np.mean([np.mean(values) for values in by_benchmark.values()])) if by_benchmark else None


def coverage_task_rows(pair_rows, method, threshold=0.5):
    by_query = collections.defaultdict(list)
    for row in pair_rows:
        by_query[(row["benchmark"], row["query_task"], row["query_trace_id"])].append(row)
    by_task = collections.defaultdict(list)
    for (benchmark, task, _query_id), pairs in by_query.items():
        by_task[(benchmark, task)].append(float(any(pair[method] >= threshold for pair in pairs)))
    return [{"benchmark": b, "task": t, "method": method, "coverage": float(np.mean(values))}
            for (b, t), values in sorted(by_task.items())]


def summarize_policy(pair_rows, rank_rows, method, threshold_policy):
    if threshold_policy["kind"] == "global":
        accepted_rows = [dict(row, policy_accept=int(row[method] >= threshold_policy["threshold"]))
                         for row in pair_rows]
    else:
        accepted_rows = apply_stratified(pair_rows, method, threshold_policy["thresholds"], "policy_accept")
    rates = old._query_rates(accepted_rows, "policy_accept", 0.5)
    summary = {metric: old._macro_metric(rates, metric)[0] for metric in RATE_METRICS}
    summary["coverage"] = macro_value(coverage_task_rows(accepted_rows, "policy_accept"),
                                      "policy_accept", "coverage")
    for metric in op.RANK_METRICS:
        summary[metric] = macro_value(rank_rows, method, metric)
    return summary, rates, accepted_rows


def best_all_metric_policy(calibration_pairs, calibration_rank_rows, methods, thresholds):
    baseline_policy = {"method": "behavior", "target": MAIN_TARGET, "kind": "global",
                       "threshold": thresholds["behavior"][MAIN_TARGET]["threshold"]}
    baseline, _, _ = summarize_policy(calibration_pairs, calibration_rank_rows,
                                      "behavior", baseline_policy)
    candidate_policies = [baseline_policy]
    for target in TARGETS:
        for method in methods:
            if method != "behavior" or target != MAIN_TARGET:
                candidate_policies.append({"method": method, "target": target, "kind": "global",
                                           "threshold": thresholds[method][target]["threshold"]})
            if method in STRATIFIED_METHODS:
                candidate_policies.append({"method": method, "target": target, "kind": "stratified",
                                           "thresholds": thresholds[f"{method}__strat"][target]})
    candidates = []
    for policy in candidate_policies[1:]:
        method = policy["method"]
        values, _, _ = summarize_policy(calibration_pairs, calibration_rank_rows, method, policy)
        oriented = []
        for metric in JOINT_METRICS:
            if values.get(metric) is None or baseline.get(metric) is None:
                oriented = []
                break
            delta = values[metric] - baseline[metric]
            oriented.append(-delta if metric in RATE_METRICS[:4] else delta)
        if not oriented:
            continue
        if min(oriented) >= -1e-12 and max(oriented) > 1e-12:
            # The maximin margin prefers a balanced improvement across the
            # prespecified scorecard; no test-fold label enters this choice.
            candidates.append((min(oriented), sum(oriented), policy))
    if candidates:
        selected = max(candidates, key=lambda item: (item[0], item[1], item[2]["method"]))[2]
        return {**selected, "selection": "inner OOF all-metric non-dominance guard", "fallback": False}
    return {**baseline_policy, "selection": "inner OOF all-metric non-dominance guard; baseline fallback",
            "fallback": True}


def run():
    rows = gallery.load_rows()
    tasks = {(r.benchmark, r.task) for r in rows}
    if len(rows) != 177 or len(tasks) != 40:
        raise SystemExit("Pinned cohort changed; re-audit before evaluation")
    OUT.mkdir(parents=True, exist_ok=True)
    pair_rows_all, threshold_rows, rate_rows, rank_rows, fold_rows = [], [], [], [], []
    folds = grouped_folds(rows, BASE_SEED, n_folds=3)
    for fold, (train_keys, test_keys) in enumerate(folds, 1):
        train = [r for r in rows if (r.benchmark, r.task) in train_keys]
        test = [r for r in rows if (r.benchmark, r.task) in test_keys]
        calibration_pairs, calibration_rank_rows = [], []
        for inner, (fit_keys, valid_keys) in enumerate(grouped_folds(train, 20261730 + fold, 3), 1):
            fit = [r for r in train if (r.benchmark, r.task) in fit_keys]
            valid = [r for r in train if (r.benchmark, r.task) in valid_keys]
            base_scorer, _ = v1.fit_scorer(fit, valid, BASE_SEED + fold * 100 + inner, hardneg=False)
            weighted_scorer, _counts = fit_taskweighted_scorers(fit, valid, BASE_SEED + fold * 100 + inner)

            def scorer(pairs, base_scorer=base_scorer, weighted_scorer=weighted_scorer):
                output = base_scorer(pairs)
                output.update(weighted_scorer(pairs))
                return output

            current = op.make_pairs(valid, scorer, fold, f"threshold_oof_{inner}")
            calibration_pairs.extend(current)
            calibration_rank_rows.extend(op.retrieval_task_metrics(valid, current, METHODS))

        thresholds = {method: {} for method in METHODS}
        for method in METHODS:
            for target in TARGETS:
                thresholds[method][target] = op.choose_threshold(calibration_pairs, method, target, GRID)
                threshold_rows.append({"outer_fold": fold, "method": method, "target": target,
                                       "kind": "global", **thresholds[method][target]})
        strat_thresholds = {method: {} for method in STRATIFIED_METHODS}
        for method in STRATIFIED_METHODS:
            thresholds[f"{method}__strat"] = {}
            for target in TARGETS:
                strat_thresholds[method][target] = {
                    kind: choose_stratified_threshold(calibration_pairs, method, target, kind)
                    for kind in ("ha", "same")
                }
                threshold_rows.extend({"outer_fold": fold, "method": method, "target": target,
                                       "kind": kind, **fitted}
                                      for kind, fitted in strat_thresholds[method][target].items())
                thresholds[f"{method}__strat"][target] = strat_thresholds[method][target]

        selected = best_all_metric_policy(calibration_pairs, calibration_rank_rows, METHODS, thresholds)
        final_base, _ = v1.fit_scorer(train, test, BASE_SEED + fold, hardneg=False)
        final_weighted, pair_counts = fit_taskweighted_scorers(train, test, BASE_SEED + fold)

        def final_scorer(pairs):
            output = final_base(pairs)
            output.update(final_weighted(pairs))
            return output

        outer = op.make_pairs(test, final_scorer, fold, "outer_test")
        for row in outer:
            for method in METHODS:
                for target in TARGETS:
                    tag = int(target * 100)
                    row[f"threshold_{method}_{tag}"] = thresholds[method][target]["threshold"]
                    row[f"accepted_{method}_{tag}"] = int(row[method] >= thresholds[method][target]["threshold"])
            for method in STRATIFIED_METHODS:
                for target in TARGETS:
                    tag = int(target * 100)
                    th = strat_thresholds[method][target]
                    chosen = th[pair_kind(row)]["threshold"]
                    row[f"threshold_{method}_strat_{tag}"] = chosen
                    row[f"accepted_{method}_strat_{tag}"] = int(row[method] >= chosen)

            if selected["kind"] == "global":
                joint_threshold = thresholds[selected["method"]][selected["target"]]["threshold"]
            else:
                joint_threshold = strat_thresholds[selected["method"]][selected["target"]][pair_kind(row)]["threshold"]
            row["joint_accepted"] = int(row[selected["method"]] >= joint_threshold)
            row["joint_method"] = selected["method"]
            row["joint_kind"] = selected["kind"]
            row["joint_threshold"] = joint_threshold
        pair_rows_all.extend(outer)

        ranking = op.retrieval_task_metrics(test, outer, METHODS)
        rank_rows.extend({"outer_fold": fold, **r} for r in ranking)
        joint_rank = [{**r, "joint_selected": r[selected["method"]]} for r in outer]
        rank_rows.extend({"outer_fold": fold, **r}
                         for r in op.retrieval_task_metrics(test, joint_rank, ("joint_selected",)))

        for target in TARGETS:
            for method in METHODS:
                rates = old._query_rates(outer, method, thresholds[method][target]["threshold"])
                rate_rows.extend({"outer_fold": fold, "target": target, "method": method,
                                  "policy": "global", **rate} for rate in rates)
            for method in STRATIFIED_METHODS:
                threshold_map = {kind: strat_thresholds[method][target][kind]["threshold"]
                                 for kind in ("ha", "same")}
                accepted = apply_stratified(outer, method, threshold_map, "policy_accept")
                rates = old._query_rates(accepted, "policy_accept", 0.5)
                rate_rows.extend({"outer_fold": fold, "target": target, "method": f"{method}_stratified",
                                  "policy": "stratified", **rate} for rate in rates)
        joint_rate = old._query_rates(outer, "joint_accepted", 0.5)
        rate_rows.extend({"outer_fold": fold, "target": MAIN_TARGET,
                          "method": "joint_selected", "policy": selected["kind"], **rate}
                         for rate in joint_rate)
        joint_cov = coverage_task_rows(outer, "joint_accepted")
        rate_rows.extend({"outer_fold": fold, **r, "target": MAIN_TARGET,
                          "method": "joint_selected", "policy": "coverage"}
                         for r in joint_cov)
        selected["outer_fold"] = fold
        selected["train_task_groups"] = len(train_keys)
        selected["test_task_groups"] = len(test_keys)
        selected["taskweighted_pair_count"] = pair_counts["taskweighted_pair_count"]
        selected["ha_pair_count"] = pair_counts["ha_pair_count"]
        fold_rows.append(selected)
        print(f"outer fold {fold}/3 complete; all-metric policy={selected['method']} / {selected['kind']} / {selected['target']}", flush=True)

    summary_methods = (*METHODS, *(f"{m}_stratified" for m in STRATIFIED_METHODS))
    summary = {}
    paired_deltas = {}
    for target in TARGETS:
        subset = [r for r in rate_rows if r.get("target") == target and r.get("policy") != "coverage"]
        current_methods = (*summary_methods, "joint_selected") if target == MAIN_TARGET else summary_methods
        summary[str(int(target * 100))] = op.summary_with_ci(
            subset, current_methods, RATE_METRICS, BOOTSTRAPS, BASE_SEED + int(target * 1000) + 11
        )
        paired_deltas[str(int(target * 100))] = {
            method: {metric: op.paired_delta(subset, metric, method, "behavior", BOOTSTRAPS,
                                             BASE_SEED + 5000 + ix * 17 + j)
                     for j, metric in enumerate(RATE_METRICS)}
            for ix, method in enumerate((*METHODS, *(f"{m}_stratified" for m in STRATIFIED_METHODS),
                                         *( ("joint_selected",) if target == MAIN_TARGET else ())))
            if method != "behavior"
        }
    rank_summary = op.summary_with_ci(rank_rows, (*METHODS, "joint_selected"), op.RANK_METRICS,
                                      BOOTSTRAPS, BASE_SEED + 930)

    for name, records in (("pair_scores.csv", pair_rows_all), ("thresholds.csv", threshold_rows),
                          ("task_metrics.csv", [*rate_rows, *rank_rows])):
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            fields = list(dict.fromkeys(key for record in records for key in record))
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    manifest = ROOT / "data/manifests/crossbenchmark_public_subset.json"
    result = {
        "status": "exploratory nested task-held-out evaluation",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": "plans/14-漏关联优化实验计划.md; second fixed batch",
        "dataset": {"records": len(rows), "task_groups": len(tasks),
                    "revision": json.loads(manifest.read_text())['revision'],
                    "label_scope": "same CTF task proxy; not attack campaign or actor identity"},
        "methods": list(METHODS), "targets": list(TARGETS), "outer_folds": fold_rows,
        "thresholds": threshold_rows, "false_merge_targets": summary,
        "paired_deltas_vs_behavior": paired_deltas, "ranking": rank_summary,
        "coverage": {"joint_selected": op.summary_with_ci(
            [r for r in rate_rows if r.get("method") == "joint_selected" and r.get("policy") == "coverage"],
            ("joint_selected",), ("coverage",), BOOTSTRAPS, BASE_SEED + 970)["joint_selected"]},
        "python_signal_audit": {"matched_records": len(rows),
                                "human_records": sum(r.label == "human" for r in rows),
                                "agent_records": sum(r.label == "agent" for r in rows),
                                "human_python_tool_calls": sum(r.skipped_python_calls for r in rows if r.label == "human"),
                                "agent_python_tool_calls_skipped": sum(r.skipped_python_calls for r in rows if r.label == "agent"),
                                "agent_traces_with_python_calls": sum(bool(r.skipped_python_calls) for r in rows if r.label == "agent"),
                                "decision": "excluded from shared features because coverage and source representation are asymmetric"},
        "provenance": {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "runner_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                       "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "interpretation_guard": "Exploratory reuse of previously inspected tasks; same-task proxy only; no actor attribution claim.",
    }
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"outputs": str(OUT.relative_to(ROOT)),
                      "selected": [{k: row[k] for k in ("outer_fold", "method", "kind", "target", "fallback")}
                                   for row in fold_rows]}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
