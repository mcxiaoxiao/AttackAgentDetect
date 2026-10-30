#!/usr/bin/env python3
"""Search a fixed inner-OOF Pareto grid for an all-metric association point."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import itertools
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_recall_optimization_v1 as v1
import run_recall_optimization_v2 as v2
import run_association_error_analysis as old
import run_association_operating_points as op
import run_mixed_gallery as gallery
from run_feature_ablation import BASE_SEED, grouped_folds

OUT = ROOT / "results" / "recall_joint_pareto_v3"
MAIN_TARGET = 0.20
BOOTSTRAPS = 5000
GLOBAL_GRID = 31
STRAT_GRID = 11
METHODS = v2.METHODS
STRATIFIED_METHODS = v2.STRATIFIED_METHODS
METRICS = (*v2.RATE_METRICS, "coverage", *op.RANK_METRICS)
ERROR_METRICS = v2.RATE_METRICS[:4]


def threshold_grid(rows, method, grid_size, kind=None):
    subset = rows if kind is None else [r for r in rows if v2.pair_kind(r) == kind]
    scores = np.asarray([r[method] for r in subset], dtype=float)
    candidates = np.quantile(scores, np.linspace(0, 1, grid_size))
    return np.unique(np.r_[np.nextafter(scores.min(), -np.inf), candidates,
                           np.nextafter(scores.max(), np.inf)])


def all_metric_selection(calibration_pairs, calibration_rank_rows, baseline_threshold):
    baseline_policy = {"method": "behavior", "kind": "global", "threshold": baseline_threshold,
                       "selection": "baseline 20% training FMR target"}
    baseline, _, _ = v2.summarize_policy(calibration_pairs, calibration_rank_rows,
                                        "behavior", baseline_policy)
    policies = []
    for method in METHODS:
        for threshold in threshold_grid(calibration_pairs, method, GLOBAL_GRID):
            if method == "behavior" and np.isclose(threshold, baseline_threshold):
                continue
            policies.append({"method": method, "kind": "global", "threshold": float(threshold)})
        if method not in STRATIFIED_METHODS:
            continue
        ha_thresholds = threshold_grid(calibration_pairs, method, STRAT_GRID, "ha")
        same_thresholds = threshold_grid(calibration_pairs, method, STRAT_GRID, "same")
        for ha_threshold, same_threshold in itertools.product(ha_thresholds, same_thresholds):
            policies.append({"method": method, "kind": "stratified",
                             "thresholds": {"ha": float(ha_threshold), "same": float(same_threshold)}})

    eligible = []
    for policy in policies:
        method = policy["method"]
        values, _, _ = v2.summarize_policy(calibration_pairs, calibration_rank_rows, method, policy)
        oriented = []
        for metric in METRICS:
            left, right = values.get(metric), baseline.get(metric)
            if left is None or right is None:
                oriented = []
                break
            delta = left - right
            oriented.append(-delta if metric in ERROR_METRICS else delta)
        if oriented and min(oriented) >= -1e-12 and max(oriented) > 1e-12:
            eligible.append((min(oriented), sum(oriented), policy))
    if eligible:
        score, margin, selected = max(eligible, key=lambda x: (x[0], x[1], x[2]["method"], x[2]["kind"]))
        return {**selected, "selection": "inner OOF Pareto grid; all-metric non-dominance",
                "fallback": False, "minimum_oriented_gain": score, "sum_oriented_gain": margin,
                "eligible_policy_count": len(eligible)}
    return {**baseline_policy, "fallback": True, "minimum_oriented_gain": 0.0,
            "sum_oriented_gain": 0.0, "eligible_policy_count": 0}


def apply_policy(pair_rows, policy):
    method = policy["method"]
    if policy["kind"] == "global":
        return [dict(row, joint_accept=int(row[method] >= policy["threshold"])) for row in pair_rows]
    return v2.apply_stratified(pair_rows, method, policy["thresholds"], "joint_accept")


def run():
    rows = gallery.load_rows()
    task_keys = {(r.benchmark, r.task) for r in rows}
    if len(rows) != 177 or len(task_keys) != 40:
        raise SystemExit("Pinned cohort changed; re-audit before evaluation")
    OUT.mkdir(parents=True, exist_ok=True)
    pair_rows_all, rate_rows, rank_rows, fold_rows, threshold_rows = [], [], [], [], []
    for fold, (train_keys, test_keys) in enumerate(grouped_folds(rows, BASE_SEED, n_folds=3), 1):
        train = [r for r in rows if (r.benchmark, r.task) in train_keys]
        test = [r for r in rows if (r.benchmark, r.task) in test_keys]
        calibration_pairs, calibration_rank_rows = [], []
        for inner, (fit_keys, valid_keys) in enumerate(grouped_folds(train, 20261730 + fold, 3), 1):
            fit = [r for r in train if (r.benchmark, r.task) in fit_keys]
            valid = [r for r in train if (r.benchmark, r.task) in valid_keys]
            base_scorer, _ = v1.fit_scorer(fit, valid, BASE_SEED + fold * 100 + inner, hardneg=False)
            weighted_scorer, _ = v2.fit_taskweighted_scorers(fit, valid, BASE_SEED + fold * 100 + inner)

            def scorer(pairs, base_scorer=base_scorer, weighted_scorer=weighted_scorer):
                output = base_scorer(pairs)
                output.update(weighted_scorer(pairs))
                return output

            current = op.make_pairs(valid, scorer, fold, f"threshold_oof_{inner}")
            calibration_pairs.extend(current)
            calibration_rank_rows.extend(op.retrieval_task_metrics(valid, current, METHODS))

        baseline_threshold = op.choose_threshold(calibration_pairs, "behavior", MAIN_TARGET,
                                                 v1.GRID)["threshold"]
        selected = all_metric_selection(calibration_pairs, calibration_rank_rows, baseline_threshold)
        selected["outer_fold"] = fold
        selected["train_task_groups"] = len(train_keys)
        selected["test_task_groups"] = len(test_keys)
        threshold_rows.append({"outer_fold": fold, "method": "behavior", "kind": "global",
                               "role": "baseline", "target": MAIN_TARGET,
                               "threshold": baseline_threshold})
        threshold_rows.append({"outer_fold": fold, "method": selected["method"], "kind": selected["kind"],
                               "role": "selected", "target": MAIN_TARGET,
                               **({"threshold": selected["threshold"]} if selected["kind"] == "global"
                                  else {"threshold_ha": selected["thresholds"]["ha"],
                                        "threshold_same": selected["thresholds"]["same"]})})

        final_base, _ = v1.fit_scorer(train, test, BASE_SEED + fold, hardneg=False)
        final_weighted, _ = v2.fit_taskweighted_scorers(train, test, BASE_SEED + fold)

        def final_scorer(pairs):
            output = final_base(pairs)
            output.update(final_weighted(pairs))
            return output

        outer = op.make_pairs(test, final_scorer, fold, "outer_test")
        selected_rows = apply_policy(outer, selected)
        for row in selected_rows:
            row["baseline_threshold"] = baseline_threshold
            row["baseline_accept"] = int(row["behavior"] >= baseline_threshold)
            row["joint_method"] = selected["method"]
            row["joint_kind"] = selected["kind"]
            row["joint_threshold"] = (selected["threshold"] if selected["kind"] == "global"
                                       else selected["thresholds"][v2.pair_kind(row)])
        pair_rows_all.extend(selected_rows)

        base_rates = old._query_rates(selected_rows, "behavior", baseline_threshold)
        joint_rates = old._query_rates(selected_rows, "joint_accept", 0.5)
        rate_rows.extend({"outer_fold": fold, "target": MAIN_TARGET, "method": "behavior",
                          "policy": "baseline", **r} for r in base_rates)
        rate_rows.extend({"outer_fold": fold, "target": MAIN_TARGET, "method": "joint_selected",
                          "policy": selected["kind"], **r} for r in joint_rates)

        base_rank = op.retrieval_task_metrics(test, selected_rows, ("behavior",))
        joint_rank_rows = [{**r, "joint_selected": r[selected["method"]]} for r in selected_rows]
        joint_rank = op.retrieval_task_metrics(test, joint_rank_rows, ("joint_selected",))
        rank_rows.extend({"outer_fold": fold, **r} for r in base_rank)
        rank_rows.extend({"outer_fold": fold, **r} for r in joint_rank)
        fold_rows.append(selected)
        print(f"outer fold {fold}/3 complete; selected={selected['method']} / {selected['kind']} / fallback={selected['fallback']}",
              flush=True)

    rate_only_rows = [r for r in rate_rows if r["policy"] != "coverage"]
    rate_summary = op.summary_with_ci(rate_only_rows, ("behavior", "joint_selected"), v2.RATE_METRICS,
                                      BOOTSTRAPS, BASE_SEED + 321)
    paired = {metric: op.paired_delta(rate_only_rows, metric, "joint_selected", "behavior", BOOTSTRAPS,
                                     BASE_SEED + 400 + index)
              for index, metric in enumerate(v2.RATE_METRICS)}
    rank_summary = op.summary_with_ci(rank_rows, ("behavior", "joint_selected"), op.RANK_METRICS,
                                      BOOTSTRAPS, BASE_SEED + 501)
    rank_deltas = {metric: op.paired_delta(rank_rows, metric, "joint_selected", "behavior", BOOTSTRAPS,
                                          BASE_SEED + 600 + index)
                   for index, metric in enumerate(op.RANK_METRICS)}
    coverage_rows = v2.coverage_task_rows(pair_rows_all, "joint_accept")
    coverage_summary = old.task_cluster_interval(coverage_rows, "coverage", n_boot=BOOTSTRAPS,
                                                 seed=BASE_SEED + 700)
    baseline_coverage = v2.coverage_task_rows(pair_rows_all, "baseline_accept")
    # baseline_accept is already binary, so the default 0.5 gate yields query coverage.
    baseline_coverage_summary = old.task_cluster_interval(baseline_coverage, "coverage", n_boot=BOOTSTRAPS,
                                                          seed=BASE_SEED + 701)
    coverage_delta_rows = [{**r, "method": "behavior"} for r in baseline_coverage]
    coverage_delta_rows += [{**r, "method": "joint_selected"} for r in coverage_rows]
    coverage_delta = op.paired_delta(coverage_delta_rows, "coverage", "joint_selected", "behavior",
                                     BOOTSTRAPS, BASE_SEED + 702)

    for filename, records in (("pair_scores.csv", pair_rows_all), ("task_metrics.csv", [*rate_rows, *rank_rows]),
                              ("thresholds.csv", threshold_rows)):
        with (OUT / filename).open("w", encoding="utf-8", newline="") as handle:
            fields = list(dict.fromkeys(key for record in records for key in record))
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    manifest = ROOT / "data/manifests/crossbenchmark_public_subset.json"
    result = {
        "status": "exploratory nested task-held-out all-metric Pareto-grid evaluation",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": "14-漏关联优化实验计划.md; fixed inner-OOF Pareto grid",
        "dataset": {"records": len(rows), "task_groups": len(task_keys),
                    "revision": json.loads(manifest.read_text())["revision"],
                    "label_scope": "same CTF task proxy; not attack campaign or actor identity"},
        "grid": {"global_quantiles": GLOBAL_GRID, "per_stratum_quantiles": STRAT_GRID,
                 "strata": ["H-A", "same-mode"], "all_metric_non_dominance": list(METRICS)},
        "outer_folds": fold_rows, "false_split_merge": rate_summary, "paired_rate_deltas": paired,
        "ranking": rank_summary, "paired_ranking_deltas": rank_deltas,
        "coverage": {"baseline": baseline_coverage_summary, "joint_selected": coverage_summary,
                     "paired_delta": coverage_delta},
        "provenance": {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "runner_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                       "v1_runner_sha256": hashlib.sha256((ROOT / "scripts/run_recall_optimization_v1.py").read_bytes()).hexdigest(),
                       "v2_runner_sha256": hashlib.sha256((ROOT / "scripts/run_recall_optimization_v2.py").read_bytes()).hexdigest(),
                       "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "interpretation_guard": "Exploratory reuse of inspected tasks; confidence intervals do not replace independent validation.",
    }
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": str(OUT.relative_to(ROOT)),
                      "selected": [{k: row.get(k) for k in ("outer_fold", "method", "kind", "fallback",
                                                                 "eligible_policy_count", "threshold", "thresholds")}
                                   for row in fold_rows],
                      "fsr": rate_summary["joint_selected"]["false_split_rate"],
                      "fmr": rate_summary["joint_selected"]["false_merge_rate"]}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
