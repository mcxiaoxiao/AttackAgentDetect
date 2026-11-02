#!/usr/bin/env python3
"""Cross-mode CTH retrieval on every mixed Human/Agent task group.

The held-out gallery is restricted to the outer test tasks so no gallery trace
is visible to training. A source-qualified CTH challenge is a task-level
retrieval proxy only; it is not an activity, operator, or campaign label.
"""
from __future__ import annotations

import collections
import csv
import importlib.util
import json
import pathlib
import random

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "research/results/behavioral_fingerprint_v2/cth_cross_mode"
SEED = 20261103
FOLDS = 5

spec = importlib.util.spec_from_file_location("cth_mixture_helpers", HERE / "run_cth_mixture.py")
mix = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mix)


def write_json(path: pathlib.Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cluster_ci(task_rows, metric, seed, draws=5000):
    tasks = sorted(task_rows)
    if len(tasks) < 2:
        return [None, None]
    values = np.asarray([task_rows[t][metric] for t in tasks], dtype=float)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(sample, [.025, .975])]


def rank_credit(scores, positive_index):
    scores = np.asarray(scores, dtype=float)
    top = np.nanmax(scores)
    top_tied = np.flatnonzero(np.isclose(scores, top, rtol=1e-10, atol=1e-12))
    hit = 1 / len(top_tied) if positive_index in top_tied else 0.0
    target = scores[positive_index]
    greater = int(np.sum(scores > target + 1e-12))
    target_tied = int(np.sum(np.isclose(scores, target, rtol=1e-10, atol=1e-12)))
    return hit, 1.0 / (greater + (target_tied + 1) / 2)


def main():
    all_rows = mix.load_rows()
    by_task_mode = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in all_rows:
        if row["profile_eligible"]:
            by_task_mode[row["task"]][row["mode"]].append(row)
    tasks = sorted(t for t, modes in by_task_mode.items()
                   if modes["human"] and modes["agent"])
    rows = [r for r in all_rows if r["profile_eligible"] and r["task"] in set(tasks)]
    if len(tasks) != 40 or len(rows) != 175:
        raise RuntimeError(f"Expected the frozen 40-task/175-trace mixed cohort; got {len(tasks)}/{len(rows)}")
    task_indices = collections.defaultdict(list)
    for i, row in enumerate(rows):
        task_indices[row["task"]].append(i)

    rng = random.Random(SEED)
    shuffled = tasks[:]
    rng.shuffle(shuffled)
    test_fold = {task: pos % FOLDS for pos, task in enumerate(shuffled)}
    fold_rows, query_rows, pair_rows = [], [], []
    methods = ["command_cosine", "action_bigram_cosine", *mix.METHOD_COLUMNS]

    for fold in range(FOLDS):
        test_tasks = sorted(t for t in tasks if test_fold[t] == fold)
        train_tasks = sorted(set(tasks) - set(test_tasks))
        train_idx = [i for t in train_tasks for i in task_indices[t]]
        test_idx = [i for t in test_tasks for i in task_indices[t]]
        train_pairs = mix.sample_training_pairs(rows, train_idx,
                                                random.Random(SEED + fold * 1009))
        pair_features, transform_audit = mix.prepare_features(rows, train_idx, list(range(len(rows))), SEED + fold)
        train_x = pair_features(train_pairs)
        train_y = np.asarray([p[2] for p in train_pairs], dtype=int)
        fitted = mix.fit_models(train_x, train_y)

        # Choose exactly one opposite-mode gallery trace per test task and
        # direction. The gallery never contains an outer-training task.
        galleries = {}
        for mode in ("human", "agent"):
            target_mode = "agent" if mode == "human" else "human"
            for task in test_tasks:
                choices = task_indices[task]
                eligible = [i for i in choices if rows[i]["mode"] == target_mode]
                if not eligible:
                    raise RuntimeError(f"Missing {target_mode} gallery for {task}")
                galleries[(mode, task)] = random.Random(SEED + fold * 10000 + len(task) + (0 if mode == "human" else 1)).choice(eligible)

        fold_queries = []
        fold_pairs = []
        for qi in test_idx:
            query = rows[qi]
            qmode = query["mode"]
            target_mode = "agent" if qmode == "human" else "human"
            candidates = [galleries[(qmode, task)] for task in test_tasks]
            candidate_tasks = test_tasks[:]
            positive_index = candidate_tasks.index(query["task"])
            pairs = [(qi, ci, int(rows[ci]["task"] == query["task"])) for ci in candidates]
            feat = pair_features(pairs)
            scores = {
                "command_cosine": feat[:, 0],
                "action_bigram_cosine": feat[:, 1],
                **{name: fitted[name].predict_proba(feat[:, cols])[:, 1]
                   for name, cols in mix.METHOD_COLUMNS.items()},
            }
            for method in methods:
                hit, mrr = rank_credit(scores[method], positive_index)
                qrow = {"fold": fold, "query_trace_id": query["trace_id"],
                        "query_task": query["task"], "query_mode": qmode,
                        "gallery_mode": target_mode, "method": method,
                        "hit_at_1": hit, "mrr": mrr,
                        "candidate_count": len(candidates),
                        "chance_hit_at_1": 1 / len(candidates)}
                fold_queries.append(qrow)
                for pos, (cid, task, pair, score) in enumerate(zip(candidates, candidate_tasks, pairs, scores[method])):
                    fold_pairs.append({"fold": fold, "query_trace_id": query["trace_id"],
                                       "candidate_trace_id": rows[cid]["trace_id"],
                                       "query_task": query["task"], "candidate_task": task,
                                       "query_mode": qmode, "candidate_mode": target_mode,
                                       "same_task_proxy": pair[2], "method": method,
                                       "score": float(score), "candidate_rank_slot": pos,
                                       "candidate_count": len(candidates)})
        query_rows.extend(fold_queries)
        pair_rows.extend(fold_pairs)
        fold_rows.append({"fold": fold, "train_tasks": len(train_tasks),
                          "test_tasks": len(test_tasks), "train_traces": len(train_idx),
                          "test_queries": len(test_idx), "train_pairs": len(train_pairs),
                          "train_positive_pairs": int(train_y.sum()),
                          "train_negative_pairs": int((train_y == 0).sum()),
                          "transform": transform_audit})
        print(f"CTH cross-mode fold {fold}: {len(train_tasks)} train tasks, {len(test_tasks)} test tasks, {len(test_idx)} queries", flush=True)

    # Reduce repeated queries within a task before estimating task-level
    # uncertainty. Report the two cross-mode directions as well as their macro.
    summary = []
    task_method = collections.defaultdict(dict)
    task_direction_method = collections.defaultdict(dict)
    for method in methods:
        for task in tasks:
            vals = [r for r in query_rows if r["method"] == method and r["query_task"] == task]
            if vals:
                task_method[method][task] = {k: float(np.mean([r[k] for r in vals])) for k in ("hit_at_1", "mrr")}
        for direction in ("human_to_agent", "agent_to_human"):
            direction_rows = [r for r in query_rows if r["method"] == method and
                              ((direction == "human_to_agent" and r["query_mode"] == "human") or
                               (direction == "agent_to_human" and r["query_mode"] == "agent"))]
            by_task = collections.defaultdict(list)
            for r in direction_rows:
                by_task[r["query_task"]].append(r)
            vals = {t: {k: float(np.mean([r[k] for r in rr])) for k in ("hit_at_1", "mrr")}
                    for t, rr in by_task.items()}
            task_direction_method[(method, direction)] = vals
        for direction in ("all", "human_to_agent", "agent_to_human"):
            values = task_method[method] if direction == "all" else task_direction_method[(method, direction)]
            record = {"method": method, "direction": direction,
                      "task_count": len(values),
                      "query_count": sum(1 for r in query_rows if r["method"] == method and
                          (direction == "all" or (direction == "human_to_agent" and r["query_mode"] == "human") or
                           (direction == "agent_to_human" and r["query_mode"] == "agent"))),
                      "candidate_count": FOLDS + 3, "chance_hit_at_1": 1 / (FOLDS + 3)}
            for metric in ("hit_at_1", "mrr"):
                record[metric] = float(np.mean([v[metric] for v in values.values()])) if values else None
                record[metric + "_task_cluster_95ci"] = cluster_ci(values, metric, SEED + len(method) + len(direction) + len(metric)) if values else [None, None]
            summary.append(record)

    paired = []
    for method in ("B_profile", "B_plus_S", "command_plus_BS"):
        base = task_method["command_cosine"]
        new = task_method[method]
        shared = sorted(set(base) & set(new))
        vals = {t: {"hit_at_1": new[t]["hit_at_1"] - base[t]["hit_at_1"],
                    "mrr": new[t]["mrr"] - base[t]["mrr"]} for t in shared}
        row = {"method": method, "baseline": "command_cosine", "task_count": len(shared)}
        for metric in ("hit_at_1", "mrr"):
            row["delta_" + metric] = float(np.mean([v[metric] for v in vals.values()]))
            arr = np.asarray([vals[t][metric] for t in shared], dtype=float)
            boot = np.random.default_rng(SEED + len(method) + len(metric)).choice(arr, (5000, len(arr)), replace=True).mean(axis=1)
            row["delta_" + metric + "_task_cluster_95ci"] = [float(x) for x in np.quantile(boot, [.025, .975])]
        paired.append(row)

    write_json(OUT / "metrics.json", {"cohort": {"tasks": len(tasks), "raw_traces": 177,
                "eligible_traces": len(rows), "human": sum(r["mode"] == "human" for r in rows),
                "agent": sum(r["mode"] == "agent" for r in rows), "outer_folds": FOLDS,
                "queries": len(set(r["query_trace_id"] for r in query_rows if r["method"] == "command_cosine")),
                "candidate_gallery": "one opposite-mode trace from each of the eight held-out tasks; one same-task proxy and seven different-task negatives",
                "chance_hit_at_1": 1 / (FOLDS + 3),
                "labels": "same source-qualified CTH challenge group; task retrieval proxy only",
                "timing": "not available in the normalized CTH records"}, "methods": summary,
                "paired_deltas_vs_command": paired,
                "uncertainty": "5,000 bootstrap draws over 40 task clusters after averaging repeated queries within task"})
    write_json(OUT / "folds.json", fold_rows)
    write_csv(OUT / "query_predictions.csv", query_rows)
    write_csv(OUT / "candidate_scores.csv", pair_rows)
    write_json(OUT / "provenance.json", {"protocol": "cth-cross-mode-retrieval-v1",
                "seed": SEED, "outer_folds": FOLDS,
                "feature_scaling": "training-fold traces only; task and mode are evaluation metadata, never pair features",
                "models": "command/action cosine; regularized class-balanced logistic regression for B, B+S, and command+B/S",
                "candidate_design": "fixed 8-task within-fold gallery; all query and gallery tasks are held out from model fitting",
                "claim_boundary": "retrieves same CTH challenge context across execution modes; does not establish a common actor, activity, attack, or campaign"})
    print(json.dumps({"cohort": {"tasks": len(tasks), "eligible_traces": len(rows)}, "methods": summary,
                      "paired_deltas": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
