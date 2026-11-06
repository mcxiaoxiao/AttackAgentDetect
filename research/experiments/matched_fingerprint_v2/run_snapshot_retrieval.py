#!/usr/bin/env python3
"""Hold out NYU challenges for matched retrieval of four known snapshots."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_gallery_ranking import load

ROOT = Path(__file__).resolve().parents[3]
SEED, DRAWS, PREFIX = 20261105, 10_000, 8
SCORE_VIEWS = ("length", "e_length", "bs", "e_length_bs")
METRICS = ("gallery_auc", "hit1", "mrr")


def fold_order(task):
    return hashlib.sha256(f"snapshot-v1:{SEED}:{task}".encode()).hexdigest()


def matrices(features, lengths, by_task, tasks, snapshots):
    i, j = np.where(~np.eye(len(tasks), dtype=bool))
    grid = np.array([[by_task[t][s] for s in snapshots] for t in tasks])
    q = np.repeat(grid[i], 4, axis=1).ravel()
    c = np.tile(grid[j], (1, 4)).ravel()
    e = features.evidence(q, c)
    bs, missing = features.full(e, q, c, ("B", "S"), "direct")
    assert not missing.any()
    log_q, log_c = np.log1p(lengths[q]), np.log1p(lengths[c])
    length = np.column_stack([np.abs(log_q - log_c), (log_q + log_c) / 2])
    return {"length": length, "e_length": np.column_stack([e, length]),
            "bs": bs[:, 2:], "e_length_bs": np.column_stack([e, length, bs[:, 2:]])}


def rank_fit(train, test, view, fit_task_count):
    scaler = StandardScaler().fit(train)
    x, v = scaler.transform(train), scaler.transform(test)
    if view in {"bs", "e_length_bs"}:
        start = 4 if view == "e_length_bs" else 0
        x[:, start:] *= 0.5
        v[:, start:] *= 0.5
    x = x.reshape(-1, 4, 4, x.shape[-1])
    pos = x[:, np.eye(4, dtype=bool), :].reshape(-1, 4, 1, x.shape[-1])
    neg = x[:, ~np.eye(4, dtype=bool), :].reshape(-1, 4, 3, x.shape[-1])
    diff = (pos - neg).reshape(-1, x.shape[-1])
    y = np.repeat([True, False], len(diff))
    model = LogisticRegression(C=0.1, fit_intercept=False, max_iter=1000)
    model.fit(np.concatenate([diff, -diff]), y,
              sample_weight=np.full(len(y), fit_task_count * 4 / len(y)))
    assert model.n_iter_.max() < model.max_iter
    return model.decision_function(v).reshape(-1, 4, 4)


def score_metrics(scores):
    identity = np.eye(4, dtype=bool)
    positive = scores[:, identity, None].reshape(-1, 4, 1)
    negative = scores[:, ~identity].reshape(-1, 4, 3)
    higher = (negative > positive).sum(axis=-1)
    tied = (negative == positive).sum(axis=-1)
    mrr = np.zeros_like(higher, dtype=float)
    for k in range(4):
        mrr += np.where(tied >= k, 1 / (1 + higher + k), 0)
    return np.column_stack([
        (1 - (higher + 0.5 * tied) / 3).mean(axis=1),
        ((higher == 0) / (1 + tied)).mean(axis=1),
        (mrr / (1 + tied)).mean(axis=1),
    ])


def evaluate(runner, rows, task_fold, snapshots, mode):
    by_task = {}
    for i, row in enumerate(rows):
        by_task.setdefault(row["task"], {})[row["source_submission"]] = i
    assert all(set(v) == set(snapshots) for v in by_task.values())
    profiles = runner.profile_cache(rows, minimum=2)
    lengths = np.array([len(row["commands"]) for row in rows])
    frames, points, draws = [], {}, {}
    for fold in range(3):
        fit = sorted(task for task in by_task if task_fold[task] != fold)
        test = sorted(task for task in by_task if task_fold[task] == fold)
        fit_idx = [idx for task in fit for idx in by_task[task].values()]
        feat = runner.Features(rows, fit_idx, profiles, 4, "mean_std")
        train = matrices(feat, lengths, by_task, fit, snapshots)
        outer = matrices(feat, lengths, by_task, test, snapshots)
        size = len(test)
        i, j = np.where(~np.eye(size, dtype=bool))
        rng = np.random.default_rng(SEED + fold)
        weights = rng.multinomial(size, np.full(size, 1 / size), size=DRAWS)
        denominator = size**2 - (weights**2).sum(axis=1)
        fold_mass = size / len(by_task)
        for view in SCORE_VIEWS:
            values = score_metrics(rank_fit(train[view], outer[view], view, len(fit)))
            for k, metric in enumerate(METRICS):
                matrix = np.zeros((size, size))
                matrix[i, j] = values[:, k]
                bootstrap = np.einsum("bi,ij,bj->b", weights, matrix, weights,
                                      optimize=True) / denominator
                points[view, metric] = points.get((view, metric), 0.0) + fold_mass * values[:, k].mean()
                draws[view, metric] = draws.get((view, metric), 0.0) + fold_mass * bootstrap
            frame = pd.DataFrame(values, columns=METRICS)
            frame["fold"], frame["view"], frame["mode"] = fold, view, mode
            frame["query_task"], frame["candidate_task"] = np.asarray(test)[i], np.asarray(test)[j]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), points, draws


def interval(mean, samples):
    return {"mean": float(mean), "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}


def prefix_cohort(rows, minimum):
    excluded = {row["task"] for row in rows if len(row["commands"]) < minimum}
    return [row for row in rows if row["task"] not in excluded]


def clip(rows, size):
    return [{**row, "commands": row["commands"][:size], "actions": row["actions"][:size]}
            for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.out or repo / "research/results/matched_fingerprint_v2/snapshot_retrieval_20261105"
    experiment = repo / "research/experiments/matched_fingerprint_v2"
    nyu_path = experiment / "run_independent_nyu_agent.py"
    nyu = load(nyu_path, "nyu")
    runner_path = experiment.parent / "matched_fingerprint_v1/run.py"
    runner = load(runner_path, "runner")
    loaded, manifest = nyu.load_external(args.source)
    snapshots = sorted({row["source_submission"] for row in loaded})
    assert len(snapshots) == 4
    complete = {task["task"] for task in manifest["eligible_tasks"]
                if len(task["traces"]) == 4
                and {r["submission"] for r in task["traces"]} == set(snapshots)}
    ordered = sorted(complete, key=fold_order)
    task_fold = {task: i % 3 for i, task in enumerate(ordered)}
    rows = [row for row in loaded if row["task"] in complete]
    common = prefix_cohort(rows, PREFIX)
    long = prefix_cohort(rows, 16)
    conditions = (("full", rows), ("common_full", common),
                  ("common_prefix8", clip(common, PREFIX)),
                  ("long_full", long), ("long_prefix8", clip(long, PREFIX)),
                  ("long_prefix16", clip(long, 16)))
    report, frames = {}, []
    for mode, cohort in conditions:
        frame, points, draws = evaluate(runner, cohort, task_fold, snapshots, mode)
        frames.append(frame)
        baseline, method = "e_length", "e_length_bs"
        delta = {m: interval(points[method, m] - points[baseline, m],
                             draws[method, m] - draws[baseline, m]) for m in METRICS}
        report[mode] = {
            "tasks": len(cohort) // len(snapshots),
            "galleries_per_view": int(frame.view.eq(SCORE_VIEWS[0]).sum()) * len(snapshots),
            "absolute": {v: {m: interval(points[v, m], draws[v, m]) for m in METRICS}
                         for v in SCORE_VIEWS},
            "delta_e_length_bs_minus_e_length": delta,
        }
        print(mode, "tasks", len(cohort) // 4, "delta", delta, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    values_path = out / "task_pair_metrics.csv.gz"
    pd.concat(frames).to_csv(values_path, index=False, compression={"method": "gzip", "mtime": 0})
    runner.write_json(out / "analysis.json", {
        "label": "same submission snapshot across different NYU challenges",
        "source_revision": manifest["source_revision"], "snapshots": snapshots,
        "seed": SEED, "bootstrap_draws": DRAWS, "task_folds": task_fold,
        "excluded_short_tasks": {
            "prefix8": sorted(complete - {row["task"] for row in common}),
            "prefix16": sorted(complete - {row["task"] for row in long}),
        },
        "inputs_sha256": {str(path.relative_to(repo)): runner.sha(path)
                          for path in (runner_path, nyu_path, nyu.MANIFEST_PATH)},
        "runner_sha256": runner.sha(Path(__file__)), "metrics_sha256": runner.sha(values_path),
        "results": report,
    })


if __name__ == "__main__":
    main()
