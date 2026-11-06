#!/usr/bin/env python3
"""Threshold-free diagnostics for the frozen matched-association predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT / "research/results/matched_fingerprint_v1/pair_predictions.csv.gz"
NYU = ROOT / "research/results/matched_fingerprint_v2/independent_nyu_agent/predictions.csv.gz"
DIMENSION = ROOT / "research/results/matched_fingerprint_v2/dimension_selection_20261104/outer_fold_predictions.csv.gz"
OUT = ROOT / "research/results/matched_fingerprint_v2/ranking_diagnostics_20261104"
N_BOOT = 10_000
SEED = 20261104


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["fold", "fraction", "replicate", "query_id", "query_task"]
    rows = []
    for identity, group in frame.groupby(keys, sort=False, observed=True):
        labels = group.same_task.to_numpy(dtype=int)
        scores = group.score.to_numpy(dtype=float)
        predicted = group.predicted_link.to_numpy(dtype=int)
        if len(group) != 4 or labels.sum() != 1:
            raise AssertionError(f"expected one positive among four candidates: {identity}")
        positive = scores[labels == 1][0]
        negatives = scores[labels == 0]
        greater = int(np.sum(negatives > positive))
        tied = 1 + int(np.sum(negatives == positive))
        rank = 1 + greater
        rows.append({
            "fold": identity[0], "fraction": str(identity[1]),
            "replicate": identity[2], "query_id": identity[3],
            "query_task": identity[4],
            "hit1": float(max(0, min(1, (1 - greater) / tied))),
            "hit2": float(max(0, min(1, (2 - greater) / tied))),
            "mrr": float(np.mean([1 / (rank + j) for j in range(tied)])),
            "gallery_auc": float(np.mean((positive > negatives) +
                                           0.5 * (positive == negatives))),
            "fmr": float(predicted[labels == 0].mean()),
            "fsr": float(1 - predicted[labels == 1].mean()),
        })
    return pd.DataFrame(rows)


def task_values(frame: pd.DataFrame) -> pd.DataFrame:
    query = query_metrics(frame)
    task_share = query.groupby(["query_task", "fraction"], observed=True).agg(
        hit1=("hit1", "mean"), hit2=("hit2", "mean"), mrr=("mrr", "mean"),
        gallery_auc=("gallery_auc", "mean"), fmr=("fmr", "mean"),
        fsr=("fsr", "mean"), query_count=("query_id", "size")).reset_index()

    ap_rows = []
    for (task, fraction), group in frame.groupby(
            ["query_task", "fraction"], sort=False, observed=True):
        ap_rows.append({"query_task": task, "fraction": str(fraction),
                        "pair_ap": float(average_precision_score(
                            group.same_task.astype(int), group.score.astype(float)))})
    task_share = task_share.merge(pd.DataFrame(ap_rows), on=["query_task", "fraction"],
                                  validate="one_to_one")
    metric_names = ("hit1", "hit2", "mrr", "gallery_auc", "pair_ap", "fmr", "fsr")
    task = task_share.groupby("query_task", observed=True)[list(metric_names)].mean()
    task["ba"] = 1 - (task.fmr + task.fsr) / 2
    task["query_count"] = task_share.groupby("query_task", observed=True).query_count.sum()
    task["positive_prevalence"] = 0.25
    return task.sort_index()


def summarize(values: pd.DataFrame, indexes: np.ndarray) -> dict:
    point = values.mean(axis=0)
    boot = values.to_numpy()[indexes].mean(axis=1)
    bounds = np.quantile(boot, [0.025, 0.975], axis=0).T
    return {name: {"mean": float(point[name]),
                   "ci95": [float(bounds[i, 0]), float(bounds[i, 1])]}
            for i, name in enumerate(values.columns)}


def verify_pair_identity(left: pd.DataFrame, right: pd.DataFrame) -> None:
    keys = ["fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
            "query_task", "candidate_task", "same_task", "candidate_position"]
    a = left[keys].sort_values(keys).reset_index(drop=True)
    b = right[keys].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def append_comparison(result: dict, frame: pd.DataFrame, dataset: str, learner: str,
                      baseline: str, augmented: str, rng: np.random.Generator) -> None:
    left = frame[(frame.learner == learner) & (frame.variant == baseline)].copy()
    right = frame[(frame.learner == learner) & (frame.variant == augmented)].copy()
    if left.empty or right.empty:
        return
    verify_pair_identity(left, right)
    a, b = task_values(left), task_values(right)
    common = sorted(set(a.index) & set(b.index))
    if a.index.tolist() != common or b.index.tolist() != common:
        raise AssertionError("paired task support differs between feature variants")
    if len(common) < 4:
        raise AssertionError(f"too few task clusters for {dataset}/{learner}")
    indexes = rng.integers(0, len(common), size=(N_BOOT, len(common)))
    delta = b.drop(columns=["query_count", "positive_prevalence"]) - a.drop(
        columns=["query_count", "positive_prevalence"])
    key = f"{dataset}:{learner}:{baseline}->{augmented}"
    result[key] = {
        "baseline": baseline, "augmented": augmented,
        "task_count": len(common),
        "pair_decisions_per_arm": int(len(left)),
        "query_galleries_per_arm": int(len(left) // 4),
        "positive_pair_prevalence": float(left.same_task.mean()),
        "task_macro": {
            "baseline": summarize(a, indexes),
            "augmented": summarize(b, indexes),
            "paired_delta": summarize(delta, indexes),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = pd.read_csv(V1, compression="gzip")
    nyu = pd.read_csv(NYU, compression="gzip")
    dimension = pd.read_csv(DIMENSION, compression="gzip")
    v1["fraction"] = v1.fraction.astype(str)
    nyu["fraction"] = nyu.fraction.astype(str)
    for frame in (v1, nyu, dimension):
        frame["same_task"] = frame.same_task.astype(int)
        frame["predicted_link"] = frame.predicted_link.astype(int)
        frame["score"] = frame.score.astype(float)

    rng = np.random.default_rng(SEED)
    results: dict[str, dict] = {}
    cth = v1[v1.dataset == "cth"]
    ait = v1[v1.dataset == "ait"]
    for learner in ("svm", "tree", "xgb"):
        append_comparison(results, cth, "cth_five_share", learner, "evidence", "BS", rng)
    append_comparison(results, ait, "ait", "xgb", "evidence", "BST", rng)
    append_comparison(results, ait, "ait", "xgb", "BS", "BST", rng)
    append_comparison(results, dimension, "cth_nested_dimension", "xgb",
                      "command_only", "full", rng)
    append_comparison(results, dimension, "cth_nested_dimension", "xgb",
                      "evidence", "full", rng)
    append_comparison(results, dimension, "cth_nested_dimension", "xgb",
                      "full", "selected_compact", rng)
    for learner in ("svm", "tree", "xgb"):
        append_comparison(results, nyu, "nyu_aa", learner, "E", "E_BS", rng)

    metadata = {
        "protocol": "post hoc metrics from frozen paired predictions; no model or threshold refit",
        "bootstrap": {"unit": "query task", "draws": N_BOOT, "seed": SEED,
                      "paired_versions": True},
        "rank_metrics": {
            "hit1_hit2": "tie-aware probability that the positive is in the top k",
            "mrr": "mean reciprocal rank with uniform tie averaging",
            "gallery_auc": "within each four-candidate query, fraction of negatives below the positive, half credit for ties",
        },
        "pair_ap": "average precision pooled within each query task and role-share cell; candidate prevalence is fixed at one positive in four; repeated candidate pairs are summarized by task-cluster bootstrap",
        "cth": "each query task contributes five role-share cells with equal weight",
        "threshold_metrics": "use stored predictions from the original frozen thresholds",
        "inputs_sha256": {str(V1.relative_to(ROOT)): sha256(V1),
                          str(NYU.relative_to(ROOT)): sha256(NYU),
                          str(DIMENSION.relative_to(ROOT)): sha256(DIMENSION)},
        "comparisons": results,
    }
    out_json = OUT / "frozen_ranking_metrics.json"
    out_json.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    rows = []
    for comparison, item in results.items():
        for metric, values in item["task_macro"]["paired_delta"].items():
            rows.append({"comparison": comparison, "metric": metric,
                         "delta": values["mean"], "ci95_low": values["ci95"][0],
                         "ci95_high": values["ci95"][1],
                         "tasks": item["task_count"],
                         "pair_decisions_per_arm": item["pair_decisions_per_arm"]})
    pd.DataFrame(rows).to_csv(OUT / "frozen_ranking_metric_deltas.csv", index=False)
    print(out_json)
    for name, item in results.items():
        d = item["task_macro"]["paired_delta"]
        print(name, "tasks", item["task_count"], "pairs", item["pair_decisions_per_arm"])
        print("  " + " ".join(f"{m}={100*v['mean']:+.2f}pp [{100*v['ci95'][0]:+.2f},{100*v['ci95'][1]:+.2f}]"
                                  for m, v in d.items()))


if __name__ == "__main__":
    main()
