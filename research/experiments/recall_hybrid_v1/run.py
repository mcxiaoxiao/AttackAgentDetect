#!/usr/bin/env python3
"""Fixed nested task-held-out query-normalized score-fusion experiment."""
from __future__ import annotations

import collections
import csv
import datetime as dt
import hashlib
import json
import pathlib
import sys

import numpy as np
import scipy
import sklearn
from scipy.stats import rankdata

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_recall_optimization_v1 as v1
import run_recall_optimization_v2 as v2
import run_association_error_analysis as old
import run_association_operating_points as op
import run_mixed_gallery as gallery
from run_feature_ablation import BASE_SEED, grouped_folds

OUT = pathlib.Path(__file__).resolve().parent / "output"
TARGET = 0.20
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
GRID_SIZE = 31
BOOTSTRAPS = 5000
RATE_METRICS = (*v2.RATE_METRICS, "balanced_accuracy")
METRICS = (*RATE_METRICS, "coverage", *op.RANK_METRICS)
ERROR_METRICS = set(v2.RATE_METRICS[:4])


def add_hybrid_scores(pair_rows: list[dict], alpha: float) -> list[dict]:
    """Fuse within-query score percentiles without reading pair labels."""
    by_query: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i, row in enumerate(pair_rows):
        by_query[(row["benchmark"], row["query_trace_id"])].append(i)
    output = [dict(row) for row in pair_rows]
    for indices in by_query.values():
        denom = max(1, len(indices) - 1)
        for key in ("behavior", "command_cosine"):
            scores = np.asarray([pair_rows[i][key] for i in indices], dtype=float)
            percentiles = (rankdata(scores, method="average") - 1.0) / denom
            for i, percentile in zip(indices, percentiles, strict=True):
                output[i][f"{key}_percentile"] = float(percentile)
        for i in indices:
            output[i]["hybrid"] = float(
                alpha * output[i]["behavior_percentile"]
                + (1.0 - alpha) * output[i]["command_cosine_percentile"]
            )
    return output


def percentile_grid(rows: list[dict]) -> np.ndarray:
    scores = np.asarray([row["hybrid"] for row in rows], dtype=float)
    if len(scores) == 0:
        raise ValueError("No calibration scores")
    return np.unique(np.r_[np.nextafter(scores.min(), -np.inf),
                           np.quantile(scores, np.linspace(0, 1, GRID_SIZE)),
                           np.nextafter(scores.max(), np.inf)])


def policy_metrics(pair_rows: list[dict], threshold: float,
                   rank_rows: list[dict], method: str) -> dict[str, float]:
    rates = old._query_rates(pair_rows, method, float(threshold))
    values = {metric: old._macro_metric(rates, metric)[0] for metric in RATE_METRICS}
    accepted = [dict(row, accepted=int(row[method] >= threshold)) for row in pair_rows]
    coverage = v2.coverage_task_rows(accepted, "accepted")
    values["coverage"] = v2.macro_value(coverage, "accepted", "coverage")
    for metric in op.RANK_METRICS:
        values[metric] = v2.macro_value(rank_rows, method, metric)
    return values


def select_policy(calibration_pairs: list[dict], calibration_ranks: list[dict],
                  calibration_hybrid_ranks: dict[float, list[dict]]) -> dict:
    baseline_threshold = op.choose_threshold(calibration_pairs, "behavior", TARGET, v1.GRID)["threshold"]
    # Baseline ranking rows are already computed on the same inner-OOF gallery.
    baseline = policy_metrics(calibration_pairs, baseline_threshold, calibration_ranks, "behavior")
    eligible = []
    candidate_audit = []
    for alpha in WEIGHTS:
        scored = add_hybrid_scores(calibration_pairs, alpha)
        ranking = calibration_hybrid_ranks[alpha]
        for threshold in percentile_grid(scored):
            current = policy_metrics(scored, float(threshold), ranking, "hybrid")
            oriented = []
            for metric in METRICS:
                left, right = current.get(metric), baseline.get(metric)
                if left is None or right is None or not np.isfinite(left) or not np.isfinite(right):
                    oriented = []
                    break
                delta = float(left - right)
                oriented.append(-delta if metric in ERROR_METRICS else delta)
            if not oriented:
                continue
            gain = {metric: float(value) for metric, value in zip(METRICS, oriented, strict=True)}
            record = {"alpha": alpha, "threshold": float(threshold),
                      "minimum_oriented_gain": min(oriented), "sum_oriented_gain": sum(oriented),
                      "dominates_baseline": min(oriented) >= -1e-12 and max(oriented) > 1e-12,
                      "metric_gains": gain}
            candidate_audit.append(record)
            if record["dominates_baseline"]:
                eligible.append(record)
    if eligible:
        selected = max(eligible, key=lambda x: (x["minimum_oriented_gain"],
                                                x["sum_oriented_gain"], -x["alpha"]))
        return {"method": "hybrid", "alpha": selected["alpha"],
                "threshold": selected["threshold"], "baseline_threshold": float(baseline_threshold),
                "fallback": False, "eligible_policy_count": len(eligible),
                "minimum_oriented_gain": selected["minimum_oriented_gain"],
                "sum_oriented_gain": selected["sum_oriented_gain"],
                "metric_gains_inner_oof": selected["metric_gains"],
                "candidate_policy_count": len(candidate_audit)}
    return {"method": "behavior", "alpha": None,
            "threshold": float(baseline_threshold), "baseline_threshold": float(baseline_threshold),
            "fallback": True, "eligible_policy_count": 0,
            "minimum_oriented_gain": 0.0, "sum_oriented_gain": 0.0,
            "metric_gains_inner_oof": {metric: 0.0 for metric in METRICS},
            "candidate_policy_count": len(candidate_audit)}


def run() -> dict:
    rows = gallery.load_rows()
    task_keys = {(row.benchmark, row.task) for row in rows}
    if len(rows) != 177 or len(task_keys) != 40:
        raise SystemExit("Pinned cohort changed; re-audit before evaluation")
    OUT.mkdir(parents=True, exist_ok=True)
    all_pairs, rate_rows, rank_rows, coverage_rows, folds = [], [], [], [], []

    for fold, (train_keys, test_keys) in enumerate(grouped_folds(rows, BASE_SEED, n_folds=3), 1):
        train = [row for row in rows if (row.benchmark, row.task) in train_keys]
        test = [row for row in rows if (row.benchmark, row.task) in test_keys]
        calibration_pairs, calibration_rank_rows = [], []
        calibration_hybrid_rank_rows = {alpha: [] for alpha in WEIGHTS}
        for inner, (fit_keys, valid_keys) in enumerate(grouped_folds(train, 20261730 + fold, 3), 1):
            fit = [row for row in train if (row.benchmark, row.task) in fit_keys]
            valid = [row for row in train if (row.benchmark, row.task) in valid_keys]
            scorer, _ = v1.fit_scorer(fit, valid, BASE_SEED + fold * 100 + inner, hardneg=False)
            current = op.make_pairs(valid, scorer, fold, f"inner_oof_{inner}")
            calibration_pairs.extend(current)
            calibration_rank_rows.extend(op.retrieval_task_metrics(valid, current, ("behavior",)))
            for alpha in WEIGHTS:
                scored = add_hybrid_scores(current, alpha)
                calibration_hybrid_rank_rows[alpha].extend(
                    op.retrieval_task_metrics(valid, scored, ("hybrid",))
                )
        # A fixed hand-off lets selection calculate hybrid rank metrics on the same OOF tasks.
        policy = select_policy(calibration_pairs, calibration_rank_rows, calibration_hybrid_rank_rows)
        policy["outer_fold"] = fold
        policy["train_task_groups"] = len(train_keys)
        policy["test_task_groups"] = len(test_keys)

        scorer, _ = v1.fit_scorer(train, test, BASE_SEED + fold, hardneg=False)
        outer = op.make_pairs(test, scorer, fold, "outer_test")
        baseline_threshold = policy["baseline_threshold"]
        baseline_rows = [dict(row, baseline_accept=int(row["behavior"] >= baseline_threshold))
                         for row in outer]
        if policy["fallback"]:
            selected_rows = [dict(row, selected_accept=row["baseline_accept"],
                                  selected_score=row["behavior"])
                             for row in baseline_rows]
            selected_method = "behavior"
        else:
            hybrid = add_hybrid_scores(outer, policy["alpha"])
            selected_rows = [dict(row, selected_accept=int(row["hybrid"] >= policy["threshold"]),
                                  selected_score=row["hybrid"])
                             for row in hybrid]
            selected_method = "hybrid"
        for row in selected_rows:
            row.update({"selected_method": selected_method, "selected_alpha": policy["alpha"],
                        "selected_threshold": policy["threshold"],
                        "baseline_threshold": baseline_threshold})
        all_pairs.extend(selected_rows)
        base_rates = old._query_rates(baseline_rows, "behavior", baseline_threshold)
        selected_rates = old._query_rates(selected_rows, "selected_score", policy["threshold"])
        rate_rows.extend({"outer_fold": fold, "method": "behavior", **r} for r in base_rates)
        rate_rows.extend({"outer_fold": fold, "method": "selected", **r} for r in selected_rates)
        base_rank = op.retrieval_task_metrics(test, outer, ("behavior",))
        if policy["fallback"]:
            selected_rank = base_rank
        else:
            selected_rank_rows = add_hybrid_scores(outer, policy["alpha"])
            selected_rank = op.retrieval_task_metrics(test, selected_rank_rows, ("hybrid",))
        rank_rows.extend({"outer_fold": fold, **r} for r in base_rank)
        rank_rows.extend({"outer_fold": fold, "method": "selected", **{
            k: r[k] for k in ("benchmark", "task", *op.RANK_METRICS, "query_count",
                             "candidate_count_mean", "positive_count_mean")
        }} for r in selected_rank)
        base_cov = v2.coverage_task_rows(baseline_rows, "baseline_accept")
        selected_cov = v2.coverage_task_rows(selected_rows, "selected_accept")
        coverage_rows.extend({**r, "outer_fold": fold, "method": "behavior"} for r in base_cov)
        coverage_rows.extend({**r, "outer_fold": fold, "method": "selected"} for r in selected_cov)
        folds.append(policy)
        print(f"outer {fold}/3: selected={selected_method}, alpha={policy['alpha']}, fallback={policy['fallback']}",
              flush=True)

    rates_summary = op.summary_with_ci(rate_rows, ("behavior", "selected"), RATE_METRICS,
                                       BOOTSTRAPS, BASE_SEED + 1800)
    rate_deltas = {metric: op.paired_delta(rate_rows, metric, "selected", "behavior",
                                           BOOTSTRAPS, BASE_SEED + 1900 + i)
                   for i, metric in enumerate(RATE_METRICS)}
    ranking_summary = op.summary_with_ci(rank_rows, ("behavior", "selected"), op.RANK_METRICS,
                                         BOOTSTRAPS, BASE_SEED + 2000)
    rank_deltas = {metric: op.paired_delta(rank_rows, metric, "selected", "behavior",
                                           BOOTSTRAPS, BASE_SEED + 2100 + i)
                   for i, metric in enumerate(op.RANK_METRICS)}
    coverage_summary = {method: old.task_cluster_interval(
        [r for r in coverage_rows if r["method"] == method], "coverage",
        n_boot=BOOTSTRAPS, seed=BASE_SEED + 2200 + i)
        for i, method in enumerate(("behavior", "selected"))}
    coverage_delta_rows = [{**r, "method": r["method"]} for r in coverage_rows]
    coverage_delta = op.paired_delta(coverage_delta_rows, "coverage", "selected", "behavior",
                                     BOOTSTRAPS, BASE_SEED + 2300)
    manifest_path = ROOT / "data/manifests/crossbenchmark_public_subset.json"
    manifest = json.loads(manifest_path.read_text())
    result = {
        "status": "completed exploratory nested query-normalized score-fusion evaluation",
        "protocol": "experiments/recall_hybrid_v1/program.md",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"records": len(rows), "task_groups": len(task_keys),
                    "revision": manifest["revision"],
                    "label_scope": "same CTF task proxy; not actor or campaign attribution"},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "selection": {"weights": WEIGHTS, "threshold_quantiles": GRID_SIZE,
                      "outer_seed": BASE_SEED, "inner_seed_rule": "20261730 + outer-fold index",
                      "rule": "inner-OOF all-metric dominance over nested behavior baseline; fallback otherwise",
                      "metrics": METRICS},
        "outer_folds": folds,
        "false_split_merge": rates_summary,
        "paired_rate_deltas": rate_deltas,
        "ranking": ranking_summary,
        "paired_ranking_deltas": rank_deltas,
        "coverage": {"summary": coverage_summary, "paired_delta": coverage_delta},
        "provenance": {
            "dataset_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            "v1_runner_sha256": hashlib.sha256((ROOT / "scripts/run_recall_optimization_v1.py").read_bytes()).hexdigest(),
            "v2_runner_sha256": hashlib.sha256((ROOT / "scripts/run_recall_optimization_v2.py").read_bytes()).hexdigest(),
        },
        "interpretation_guard": "Exploratory reuse of inspected tasks; confidence intervals condition on fitted folds and do not replace independent validation.",
        "implementation_notes": [
            "Attempt 1 stopped in outer fold 1 because the fallback ranking summary has no raw behavior-score field; corrected before a complete run.",
            "Attempt 2 finished all outer folds but stopped during final coverage interval summarization because task_cluster_interval requires keyword-only bootstrap arguments; corrected before a complete recorded run.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, records in (("pair_scores.csv", all_pairs), ("task_metrics.csv", [*rate_rows, *rank_rows, *coverage_rows])):
        with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
            fields = list(dict.fromkeys(k for record in records for k in record))
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps({"result": str((OUT / "results.json").relative_to(ROOT)),
                      "selected": [{k: fold.get(k) for k in ("outer_fold", "method", "alpha", "fallback",
                                                               "eligible_policy_count")}
                                   for fold in folds],
                      "rate_deltas": rate_deltas, "rank_deltas": rank_deltas,
                      "coverage_delta": coverage_delta}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
