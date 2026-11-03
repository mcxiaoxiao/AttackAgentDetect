#!/usr/bin/env python3
"""Paired task-cluster analysis for the frozen NYU AA confirmation."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research/results/matched_fingerprint_v2/independent_nyu_agent"
PREDICTIONS = OUT_DIR / "predictions.csv.gz"
RUN = OUT_DIR / "run_manifest.json"
OUT = OUT_DIR / "analysis.json"
LEARNERS = ("svm", "tree", "xgb")
VERSIONS = ("E", "E_BS")
NBOOT, SEED = 10_000, 20261106


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_task_matrix(frame, tasks):
    values = []
    counts = {}
    for task in tasks:
        x = frame[frame.query_task == task]
        pos, neg = x[x.same_task == 1], x[x.same_task == 0]
        assert len(pos) and len(neg)
        fmr = float(neg.predicted_link.mean())
        fsr = float(1 - pos.predicted_link.mean())
        values.append((fmr, fsr, 1 - (fmr + fsr) / 2))
        counts[task] = {"positive_pairs": int(len(pos)), "negative_pairs": int(len(neg)),
                        "queries": int(x.query_id.nunique())}
    return np.asarray(values, dtype=float), counts


def summary(values, indexes):
    point = values.mean(axis=0)
    boots = values[indexes].mean(axis=1)
    bounds = np.quantile(boots, [0.025, 0.975], axis=0).T
    return {"metrics": dict(zip(("fmr", "fsr", "ba"), map(float, point))),
            "ci95": {name: list(map(float, bounds[i]))
                     for i, name in enumerate(("fmr", "fsr", "ba"))}}


def rank_rows(frame):
    rows = []
    keys = ["fold", "replicate", "query_id", "query_task"]
    for identity, group in frame.groupby(keys, sort=False, observed=True):
        if len(group) != 4 or int(group.same_task.sum()) != 1:
            raise AssertionError(f"invalid gallery {identity}: rows={len(group)}")
        scores = group.score.astype(float).to_numpy()
        labels = group.same_task.astype(int).to_numpy()
        positive = float(scores[labels == 1][0])
        negatives = scores[labels == 0]
        greater = int(np.sum(negatives > positive))
        ties = 1 + int(np.sum(negatives == positive))
        rank = 1 + greater
        rows.append({"fold": identity[0], "replicate": identity[1],
                     "query_id": identity[2], "query_task": identity[3],
                     "hit1": (1 / ties if greater == 0 else 0.0),
                     "mrr": float(np.mean([1 / (rank + j) for j in range(ties)])),
                     "auc": float(np.mean((positive > negatives) +
                                          0.5 * (positive == negatives)))})
    return pd.DataFrame(rows)


def rank_task_values(frame, tasks, name):
    values = frame.groupby("query_task", observed=True)[name].mean()
    return np.asarray([float(values.get(task, np.nan)) for task in tasks])


def rank_summary(before, after, tasks, indexes):
    result = {}
    for name in ("hit1", "mrr", "auc"):
        b = rank_task_values(before, tasks, name)
        a = rank_task_values(after, tasks, name)
        assert np.isfinite(b).all() and np.isfinite(a).all()
        delta = a - b
        boot = delta[indexes].mean(axis=1)
        result[name] = {"delta": float(delta.mean()),
                        "ci95": list(map(float, np.quantile(boot, [0.025, 0.975])))}
    return result


def transitions(before, after):
    keys = ["fold", "pair_id", "replicate", "query_task", "pair_type", "same_task"]
    paired = before[keys + ["predicted_link"]].merge(
        after[keys + ["predicted_link"]], on=keys,
        suffixes=("_before", "_after"), validate="one_to_one")
    result = {"pairs": int(len(paired)), "by_relation": {}}
    for relation, label in (("different_task", 0), ("same_task", 1)):
        x = paired[paired.same_task == label]
        y = x.same_task.astype(int).to_numpy()
        b = x.predicted_link_before.astype(int).to_numpy()
        a = x.predicted_link_after.astype(int).to_numpy()
        per_task = []
        for task, group in x.groupby("query_task", observed=True):
            yy = group.same_task.astype(int).to_numpy()
            bb = group.predicted_link_before.astype(int).to_numpy()
            aa = group.predicted_link_after.astype(int).to_numpy()
            corrected = int(((bb != yy) & (aa == yy)).sum())
            introduced = int(((bb == yy) & (aa != yy)).sum())
            per_task.append((task, corrected, introduced))
        result["by_relation"][relation] = {
            "pairs": int(len(x)),
            "errors_corrected": int(((b != y) & (a == y)).sum()),
            "errors_introduced": int(((b == y) & (a != y)).sum()),
            "tasks_net_error_reduction": int(sum(corrected > introduced
                                                  for _, corrected, introduced in per_task)),
            "tasks_net_error_increase": int(sum(corrected < introduced
                                                 for _, corrected, introduced in per_task))}
    return result


def main():
    frame = pd.read_csv(PREDICTIONS, compression="gzip")
    run = json.loads(RUN.read_text())
    tasks = sorted(frame.query_task.unique())
    assert len(tasks) == run["confirmation_tasks"] == 160
    assert set(frame.dataset) == {"nyu_ctf_independent_aa"}
    assert set(frame.pair_type) == {"AA"}
    assert set(frame.learner) == set(LEARNERS)
    assert set(frame.variant) == set(VERSIONS)
    assert len(frame) == run["prediction_rows"] == 1_404_000
    assert frame.groupby(["learner", "variant"]).size().eq(234_000).all()
    assert frame.fallback.astype(int).sum() == 0

    pair_keys = ["fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
                 "query_task", "candidate_task", "pair_type", "same_task", "candidate_position"]
    ref = frame[(frame.learner == "xgb") & (frame.variant == "E")][pair_keys]
    for learner in LEARNERS:
        for version in VERSIONS:
            current = frame[(frame.learner == learner) & (frame.variant == version)][pair_keys]
            pd.testing.assert_frame_equal(ref.reset_index(drop=True),
                                          current.reset_index(drop=True), check_dtype=False)

    indexes = np.random.default_rng(SEED).integers(0, len(tasks), size=(NBOOT, len(tasks)))
    matrices, pair_counts = {}, {}
    absolute = {}
    ranking = {}
    for learner in LEARNERS:
        absolute[learner], matrices[learner], ranking[learner] = {}, {}, {}
        model_frame = frame[frame.learner == learner]
        rank_frames = {}
        for version in VERSIONS:
            current = model_frame[model_frame.variant == version].copy()
            task_matrix, counts = metric_task_matrix(current, tasks)
            matrices[learner][version] = task_matrix
            pair_counts[f"{learner}:{version}"] = counts
            absolute[learner][version] = summary(task_matrix, indexes)
            rank_frames[version] = rank_rows(current)
        ranking[learner] = {"delta_E_BS_minus_E": rank_summary(
            rank_frames["E"], rank_frames["E_BS"], tasks, indexes),
            "galleries_per_version": {k: len(v) for k, v in rank_frames.items()}}

    result = {"protocol": "frozen cross-source AA confirmation on NYU CTF challenge groups",
              "scope": "same challenge is a task-retrieval proxy, not activity/actor/campaign identity",
              "test_tasks": len(tasks), "test_traces": run["confirmation_traces"],
              "query_galleries_per_repetition": run["query_galleries_per_repetition"],
              "repetitions": run["repetitions"], "candidate_count": run["candidate_count"],
              "pair_count_per_model_version": 234_000,
              "bootstrap_draws": NBOOT, "bootstrap_seed": SEED,
              "bootstrap_unit": "challenge task; repeated galleries averaged within each of 160 clusters",
              "pair_keys_identical_across_learners_and_versions": True,
              "all_pairs_have_fallback_false": True,
              "absolute_metrics": absolute,
              "paired_E_BS_minus_E": {}, "ranking": ranking,
              "error_transitions": {}, "pair_counts_by_task": pair_counts,
              "leave_one_task_out": {}}
    for learner in LEARNERS:
        before, after = matrices[learner]["E"], matrices[learner]["E_BS"]
        result["paired_E_BS_minus_E"][learner] = summary(after - before, indexes)
        model_frame = frame[frame.learner == learner]
        result["error_transitions"][learner] = transitions(
            model_frame[model_frame.variant == "E"],
            model_frame[model_frame.variant == "E_BS"])
        task_deltas = after[:, 2] - before[:, 2]
        order = np.argsort(task_deltas)
        result["leave_one_task_out"][learner] = {
            "full_delta_ba": float(task_deltas.mean()),
            "minimum_delta_after_deleting_one_task": float(min(
                (task_deltas.sum() - task_deltas[i]) / (len(tasks) - 1)
                for i in range(len(tasks)))),
            "maximum_delta_after_deleting_one_task": float(max(
                (task_deltas.sum() - task_deltas[i]) / (len(tasks) - 1)
                for i in range(len(tasks)))),
            "most_negative_contributing_task": tasks[int(order[0])],
            "most_positive_contributing_task": tasks[int(order[-1])]}
    result["training_parameters"] = run["parameters"]
    result["representation"] = run["representation"]
    result["thresholds"] = run["thresholds"]
    result["training_pair_count"] = run["training_pair_count"]
    result["source_revision"] = run["source_revision"]
    result["provenance_sha256"] = {
        "predictions": sha256(PREDICTIONS),
        "pair_manifest": sha256(OUT_DIR / "pair_manifest.csv.gz"),
        "run_manifest": sha256(RUN),
        "analysis_script": sha256(Path(__file__))}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n", encoding="utf8")
    print("wrote", OUT)
    for learner in LEARNERS:
        for version in VERSIONS:
            x = result["absolute_metrics"][learner][version]
            print(learner, version,
                  "BA/FMR/FSR %", *[round(100 * x["metrics"][k], 2)
                                    for k in ("ba", "fmr", "fsr")],
                  "BA CI %", [round(100 * z, 2) for z in x["ci95"]["ba"]])
        x = result["paired_E_BS_minus_E"][learner]
        print(learner, "E+B/S minus E ΔFMR/FSR/BA pp",
              *[round(100 * x["metrics"][k], 2) for k in ("fmr", "fsr", "ba")],
              "BA CI pp", [round(100 * z, 2) for z in x["ci95"]["ba"]],
              "ranking", {k: [round(100 * v["delta"], 2),
                             [round(100 * z, 2) for z in v["ci95"]]]
                          for k, v in result["ranking"][learner]["delta_E_BS_minus_E"].items()})


if __name__ == "__main__":
    main()
