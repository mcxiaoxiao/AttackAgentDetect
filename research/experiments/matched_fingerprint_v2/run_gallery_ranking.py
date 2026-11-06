#!/usr/bin/env python3
"""Compare pair classification and same-query ranking on frozen CTH galleries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else None
EXPERIMENTS = Path("research/experiments")
RESULTS = Path("research/results")
SEED = 20261105
REP = (4, "mean_std", "conditional")
C_VALUES = (0.01, 0.1, 1.0)
BS_SCALES = (0.25, 1.0)
LOSSES = ("pointwise", "rank")
VIEWS = ("E", "BS")
IDENTITY = ["fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
            "query_task", "candidate_task", "same_task", "candidate_position"]


def training_pairs(rows, tasks):
    """Give each task equal weight, then each query and each class within it."""
    indices = np.asarray([i for i, row in enumerate(rows) if row["task"] in tasks])
    a, b = np.repeat(indices, len(indices)), np.tile(indices, len(indices))
    a, b = a[a != b], b[a != b]
    trace_tasks = np.asarray([row["task"] for row in rows])
    y = trace_tasks[a] == trace_tasks[b]
    weights = np.empty(len(y))
    query_pairs = []
    for q in indices:
        pos = np.flatnonzero((a == q) & y)
        neg = np.flatnonzero((a == q) & ~y)
        n_queries = np.sum(trace_tasks[indices] == trace_tasks[q])
        mass = len(indices) / (len(tasks) * n_queries)
        weights[pos] = mass / (2 * len(pos))
        weights[neg] = mass / (2 * len(neg))
        query_pairs.append((pos, neg, mass))
    return a, b, y, weights, query_pairs


def rank_samples(x, query_pairs):
    differences, weights = [], []
    for pos, neg, mass in query_pairs:
        delta = (x[pos, None, :] - x[neg, :]).reshape(-1, x.shape[1])
        differences.append(delta)
        weights.append(np.full(len(delta), mass / (2 * len(delta))))
    delta, weight = np.concatenate(differences), np.concatenate(weights)
    return (np.concatenate([delta, -delta]),
            np.repeat([True, False], len(delta)), np.tile(weight, 2))


def fit_scores(runner, rows, profiles, train_tasks, gallery, wanted=None):
    window, pooling, fusion = REP
    indices = {row["id"]: i for i, row in enumerate(rows)}
    train_idx = [i for i, row in enumerate(rows) if row["task"] in train_tasks]
    assert not train_tasks.intersection(gallery.query_task)
    assert not train_tasks.intersection(gallery.candidate_task)
    features = runner.Features(rows, train_idx, profiles, window, pooling)
    a, b, y, weight, query_pairs = training_pairs(rows, train_tasks)
    q = gallery.query_id.map(indices).to_numpy(dtype=int)
    c = gallery.candidate_id.map(indices).to_numpy(dtype=int)
    e_train, e_eval = features.evidence(a, b), features.evidence(q, c)
    bs_train, _ = features.full(e_train, a, b, ("B", "S"), fusion)
    bs_eval, missing = features.full(e_eval, q, c, ("B", "S"), fusion)
    gallery_fallback = np.repeat(missing.reshape(-1, 4).any(axis=1), 4)
    scores = {}
    for view, train, evaluate, scales in (("E", e_train, e_eval, (1.0,)),
                                          ("BS", bs_train, bs_eval, BS_SCALES)):
        scaler = StandardScaler().fit(train, sample_weight=weight)
        train, evaluate = scaler.transform(train), scaler.transform(evaluate)
        for scale in scales:
            multiplier = np.r_[np.ones(2), np.full(train.shape[1] - 2, scale)]
            x, v = train * multiplier, evaluate * multiplier
            for loss in LOSSES:
                cs = [reg for reg in C_VALUES
                      if wanted is None or (loss, view, scale, reg) in wanted]
                if not cs:
                    continue
                xx, yy, w = rank_samples(x, query_pairs) if loss == "rank" else (x, y, weight)
                for reg in cs:
                    model = LogisticRegression(C=reg, fit_intercept=(loss == "pointwise"),
                                               max_iter=1000)
                    model.fit(xx, yy, sample_weight=w)
                    assert model.n_iter_.max() < model.max_iter
                    scores[loss, view, scale, reg] = model.decision_function(v)
    return scores, gallery_fallback


def check_gallery(gallery):
    groups = ["fold", "fraction", "replicate", "query_id", "query_task"]
    y = gallery.same_task.to_numpy(dtype=bool).reshape(-1, 4)
    if not (y.sum(axis=1) == 1).all() or len(gallery.drop_duplicates(groups)) != len(y):
        raise ValueError("Expected one positive in each four-candidate gallery")
    for key in groups:
        values = gallery[key].to_numpy().reshape(-1, 4)
        if not (values == values[:, :1]).all():
            raise ValueError(f"Interleaved gallery rows for {key}")


def validation_rank(gallery, scores):
    y = gallery.same_task.to_numpy(dtype=bool).reshape(-1, 4)
    s = scores.reshape(-1, 4)
    pos, neg = s[y, None], s[~y].reshape(-1, 3)
    greater, tied = (neg > pos).sum(axis=1), (neg == pos).sum(axis=1)
    auc = ((neg < pos).sum(axis=1) + 0.5 * tied) / 3
    hit = (greater == 0) / (1 + tied)
    per_query = pd.DataFrame({"task": gallery.query_task.to_numpy()[::4],
                              "auc": auc, "hit1": hit})
    return per_query.groupby("task")[["auc", "hit1"]].mean().mean().to_list()


def select(runner, gallery, scores, fallback):
    picked, audit = {}, {}
    y = gallery.same_task.to_numpy(dtype=int)
    tasks = gallery.query_task.to_numpy()
    for loss in LOSSES:
        for view in VIEWS:
            name = f"{loss}_{view}"
            base = picked.get(f"{loss}_E")
            candidates = []
            for key, raw in scores.items():
                if key[:2] != (loss, view):
                    continue
                effective = np.where(fallback, scores[base["key"]], raw) if view == "BS" else raw
                auc, hit = validation_rank(gallery, effective)
                candidates.append({"key": key, "auc": auc, "hit1": hit})
            best = max(candidates, key=lambda item: (round(item["auc"], 12),
                       round(item["hit1"], 12), -item["key"][3], -item["key"][2]))
            raw = scores[best["key"]]
            miss = fallback if view == "BS" else None
            base_pred = scores[base["key"]] >= base["threshold"] if view == "BS" else None
            threshold = runner.threshold(y, raw, miss, base_pred)
            prediction = runner.decision(raw, threshold, miss, base_pred)
            fmr, fsr, ba = runner.macro(y, prediction, tasks)
            pooled_fmr = prediction[y == 0].mean()
            assert pooled_fmr <= 0.1 + 1e-12
            picked[name] = {**best, "threshold": threshold}
            audit[name] = {"candidates": candidates, "selected": picked[name],
                           "macro": {"fmr": fmr, "fsr": fsr, "ba": ba},
                           "pooled_fmr": pooled_fmr}
    return picked, audit


def evaluate_fold(runner, rows, tasks, folds, profiles, fold, original):
    gallery = original["frozen_E"].copy()
    check_gallery(gallery)
    test_tasks = {task for task in tasks if folds[task] == fold}
    assert set(gallery.query_task) == set(gallery.candidate_task) == test_tasks
    train_tasks = set(tasks) - test_tasks
    inner_chunks, inner_scores, inner_masks, split_audit = [], [], [], []
    for inner in range(2):
        val_tasks = {task for i, task in enumerate(sorted(train_tasks)) if i % 2 == inner}
        fit_tasks = train_tasks - val_tasks
        validation = pd.DataFrame(runner.gallery(rows, val_tasks, inner, 20,
                                                  SEED + 100 * fold + 20 * inner))
        check_gallery(validation)
        scores, missing = fit_scores(runner, rows, profiles, fit_tasks, validation)
        inner_chunks.append(validation)
        inner_scores.append(scores)
        inner_masks.append(missing)
        split_audit.append({"fit_tasks": sorted(fit_tasks), "validation_tasks": sorted(val_tasks)})
    validation = pd.concat(inner_chunks, ignore_index=True)
    joined_scores = {k: np.concatenate([s[k] for s in inner_scores]) for k in inner_scores[0]}
    picked, audit = select(runner, validation, joined_scores, np.concatenate(inner_masks))
    raw, missing = fit_scores(runner, rows, profiles, train_tasks, gallery,
                              wanted={item["key"] for item in picked.values()})
    frames = {name: df[IDENTITY + ["score", "threshold", "predicted_link", "fallback"]].copy()
              for name, df in original.items()}
    for name, item in picked.items():
        loss, view, _, _ = item["key"]
        frame = gallery[IDENTITY].copy()
        score, threshold = raw[item["key"]], item["threshold"]
        fallback = missing if view == "BS" else np.zeros(len(frame), dtype=bool)
        base = picked[f"{loss}_E"]
        frame["score"] = np.where(fallback, raw[base["key"]], score)
        frame["threshold"] = np.where(fallback, base["threshold"], threshold)
        frame["predicted_link"] = (frame.score >= frame.threshold).astype(int)
        frame["fallback"] = fallback.astype(int)
        frames[name] = frame
    for name, frame in frames.items():
        frame["arm"] = name
    return frames, {"outer_train": sorted(train_tasks), "outer_test": sorted(test_tasks),
                    "inner_splits": split_audit, "selection": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.out or repo / RESULTS / "matched_fingerprint_v2/gallery_ranking_20261105"
    out.mkdir(parents=True, exist_ok=True)
    runner_path = repo / EXPERIMENTS / "matched_fingerprint_v1/run.py"
    runner = load(runner_path, "runner")
    metrics_path = repo / EXPERIMENTS / "matched_fingerprint_v2/analyze_frozen_ranking_metrics.py"
    metrics = load(metrics_path, "metrics")
    rows, tasks, folds = runner.build_cth()
    profiles = runner.profile_cache(rows, minimum=2)
    parts, audit = [], {}
    sources = [runner_path, metrics_path, runner.FOLDS_CSV]
    for fold in sorted(set(folds.values())):
        original = {}
        for name, variant in (("frozen_E", "evidence"), ("frozen_BS", "BS")):
            source = repo / RESULTS / f"matched_fingerprint_v1/cth/fold-{fold}-xgb-{variant}.csv.gz"
            sources.append(source)
            original[name] = pd.read_csv(source)
        pd.testing.assert_frame_equal(original["frozen_E"][IDENTITY],
                                      original["frozen_BS"][IDENTITY])
        result, audit[fold] = evaluate_fold(runner, rows, tasks, folds, profiles, fold, original)
        parts.extend(result.values())
        print("fold", fold, "selected",
              {k: v["selected"]["key"][2:] for k, v in audit[fold]["selection"].items()},
              flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    arms = {k: metrics.task_values(df).drop(columns=["query_count", "positive_prevalence"])
            for k, df in predictions.groupby("arm")}
    assert set(predictions.groupby("arm").size()) == {54_400}
    index = np.random.default_rng(SEED).integers(0, len(tasks), size=(10_000, len(tasks)))
    comparisons = (("frozen_E", "pointwise_E"), ("frozen_E", "rank_E"),
                   ("frozen_BS", "pointwise_BS"), ("frozen_BS", "rank_BS"),
                   ("pointwise_E", "rank_E"), ("pointwise_BS", "rank_BS"),
                   ("pointwise_E", "pointwise_BS"), ("rank_E", "rank_BS"),
                   ("frozen_E", "rank_BS"), ("frozen_E", "frozen_BS"))
    deltas = {}
    for baseline, candidate in comparisons:
        task_delta = arms[candidate] - arms[baseline]
        delta = metrics.summarize(task_delta, index)
        deltas[f"{candidate} - {baseline}"] = {"aggregate": delta, "by_outer_fold": {
            fold: task_delta.loc[[task for task in tasks if folds[task] == fold]].mean().to_dict()
            for fold in sorted(set(folds.values()))}}
        print(candidate, "vs", baseline, " ".join(
            f"{key}={100 * delta[key]['mean']:+.2f}pp"
            for key in ("gallery_auc", "hit1", "ba", "fmr", "fsr")), flush=True)
    prediction_path = out / "predictions.csv.gz"
    predictions.to_csv(prediction_path, index=False, compression={"method": "gzip", "mtime": 0})
    runner.write_json(out / "analysis.json", runner.finite_json({
        "cohort": "CTH 17 repeatedly examined development tasks only",
        "representation": REP, "seed": SEED, "bootstrap_draws": len(index),
        "inputs_sha256": {str(path.relative_to(repo)): runner.sha(path) for path in sources},
        "runner_sha256": runner.sha(Path(__file__)), "predictions_sha256": runner.sha(prediction_path),
        "metrics": {k: metrics.summarize(v, index) for k, v in arms.items()},
        "comparisons": deltas, "folds": audit,
    }))


def load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
