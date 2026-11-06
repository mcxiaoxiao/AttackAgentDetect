#!/usr/bin/env python3
"""Frozen, task-disjoint cross-task retrieval of three CTH Agent snapshots."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research/experiments/matched_fingerprint_v2"
CATALOG = ROOT / "research/results/behavior_trace_dataset_v2/traces/trace-catalog-v2.json"
OUT = ROOT / "research/results/matched_fingerprint_v2/cth_model_snapshot_retrieval_20261106"
SEED = 20261106
N_BOOT = 10_000
SNAPSHOTS = ("claude-opus-4.6", "claude-sonnet-4.6", "gpt-5.5")
SHUFFLE_SEEDS = (20261106, 20261107, 20261108, 20261109, 20261110)
REAL_ARMS = ("E+length", "E+length+LCS", "E+length+B/S")
LEARNERS = ("svm", "tree", "xgb")
METRICS = ("hit1", "gallery_auc", "mrr")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


M = load(EXP / "run_arguments.py", "cth_snapshot_feature_runner")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def task_key(task: str) -> str:
    return hashlib.sha256(f"cth-snapshot-v1:{SEED}:{task}".encode()).hexdigest()


def load_rows(prefix_minimum: int | None = None):
    catalog = json.loads(CATALOG.read_text())
    relevant = [r for r in catalog["records"]
                if r["source_dataset"] == "cyber-task-horizons"
                and r["track"] == "ctf" and r["mode"] == "agent"
                and r["framework"] == "Inspect-ReAct"
                and r["observation_format"] == "inspect-shell-tool-call"
                and r["model"] in SNAPSHOTS]
    grouped = defaultdict(dict)
    for record in relevant:
        task, model = record["task_group"], record["model"]
        grouped[task].setdefault(model, []).append(record)
    complete = []
    for task, by_model in grouped.items():
        if set(by_model) != set(SNAPSHOTS):
            continue
        if any(len(by_model[m]) != 1 for m in SNAPSHOTS):
            continue
        if prefix_minimum and any(
                len(by_model[m][0]["commands"]) < prefix_minimum for m in SNAPSHOTS):
            continue
        complete.append(task)
    tasks = sorted(complete, key=task_key)
    rows = []
    revisions = set()
    for task in tasks:
        for model in SNAPSHOTS:
            r = grouped[task][model][0]
            commands, actions = r["commands"], r["actions"]
            if len(commands) != int(r["command_count"]) or len(commands) != len(actions):
                raise AssertionError(f"catalog count mismatch for {r['trace_id']}")
            revisions.add(r["source_revision"])
            rows.append({"id": r["trace_id"], "task": task, "model": model,
                         "mode": "agent", "commands": list(commands),
                         "actions": list(actions), "timestamps": None})
    if len(revisions) != 1:
        raise AssertionError(f"expected one pinned CTH revision; found {revisions}")
    if not tasks or len(rows) != len(tasks) * len(SNAPSHOTS):
        raise AssertionError("incomplete three-snapshot task groups")
    return rows, tasks, next(iter(revisions))


def permute_rows(rows, seed: int):
    rng = random.Random(seed)
    result = []
    for row in rows:
        order = list(range(len(row["commands"])))
        rng.shuffle(order)
        result.append({**row,
                       "commands": [row["commands"][i] for i in order],
                       "actions": [row["actions"][i] for i in order]})
    return result


def pair_lcs(rows, a, b):
    result = np.empty(len(a), dtype=float)
    for i, (ia, ib) in enumerate(zip(a, b)):
        x, y = rows[int(ia)]["commands"], rows[int(ib)]["commands"]
        if len(x) < len(y):
            x, y = y, x
        previous = [0] * (len(y) + 1)
        for token in x:
            current = [0]
            for j, other in enumerate(y, 1):
                current.append(previous[j - 1] + 1 if token == other
                               else max(previous[j], current[-1]))
            previous = current
        denom = len(x) + len(y)
        result[i] = 2 * previous[-1] / denom if denom else 0.0
    return result.reshape(-1, 1)


def length_features(rows, a, b, scaler=None, fit=False):
    values = np.log1p(np.asarray([len(r["commands"]) for r in rows], dtype=float))
    left, right = values[a], values[b]
    pair = np.column_stack([(left + right) / 2, np.abs(left - right)])
    if fit:
        scaler = StandardScaler().fit(pair)
    return scaler.transform(pair), scaler


def pair_rows(rows, tasks, fold_map, fold, evaluation=False):
    if evaluation:
        taskset = sorted(t for t in tasks if fold_map[t] == fold)
        task_to_rows = {t: [i for i, r in enumerate(rows) if r["task"] == t]
                        for t in taskset}
        pairs = []
        meta = []
        for query_task in taskset:
            for candidate_task in taskset:
                if query_task == candidate_task:
                    continue
                for q in task_to_rows[query_task]:
                    for c in task_to_rows[candidate_task]:
                        pairs.append((q, c, int(rows[q]["model"] == rows[c]["model"])))
                        meta.append({"fold": fold, "query_task": query_task,
                                     "candidate_task": candidate_task,
                                     "query_model": rows[q]["model"],
                                     "candidate_model": rows[c]["model"],
                                     "same_model": int(rows[q]["model"] == rows[c]["model"])})
        return pairs, meta
    taskset = {t for t in tasks if fold_map[t] != fold}
    indices = [i for i, row in enumerate(rows) if row["task"] in taskset]
    pairs = []
    for a, b in itertools.combinations(indices, 2):
        if rows[a]["task"] != rows[b]["task"]:
            pairs.append((a, b, int(rows[a]["model"] == rows[b]["model"])))
    return pairs, None


def feature_arms(rows, profiles, train_pairs, test_pairs, train_idx, shuffle_profiles=None,
                 shuffled_rows=None, fusion="conditional"):
    ta, tb = np.asarray([p[0] for p in train_pairs]), np.asarray([p[1] for p in train_pairs])
    va, vb = np.asarray([p[0] for p in test_pairs]), np.asarray([p[1] for p in test_pairs])
    y_train = np.asarray([p[2] for p in train_pairs], dtype=np.int8)
    features = M.Features(rows, train_idx, profiles, 4, "mean_std")
    e_train = features.evidence_head_only(ta, tb)
    e_test = features.evidence_head_only(va, vb)
    record_train, length_scaler = length_features(rows, ta, tb, fit=True)
    record_test, _ = length_features(rows, va, vb, scaler=length_scaler)
    lcs_train, lcs_test = pair_lcs(rows, ta, tb), pair_lcs(rows, va, vb)
    bs_train, missing_train = features.full(e_train, ta, tb, ("B", "S"), fusion)
    bs_test, missing_test = features.full(e_test, va, vb, ("B", "S"), fusion)
    if missing_train.any() or missing_test.any():
        raise AssertionError("the >=8-command cohort must have complete B/S features")
    matrices = {
        "E+length": (np.column_stack([e_train, record_train]),
                     np.column_stack([e_test, record_test])),
        "E+length+LCS": (np.column_stack([e_train, record_train, lcs_train]),
                         np.column_stack([e_test, record_test, lcs_test])),
        "E+length+B/S": (np.column_stack([e_train, record_train, bs_train[:, 2:]]),
                         np.column_stack([e_test, record_test, bs_test[:, 2:]])),
    }
    if shuffle_profiles is not None and shuffled_rows is not None:
        shuffled_features = M.Features(shuffled_rows, train_idx, shuffle_profiles, 4, "mean_std")
        sh_train, miss_tr = shuffled_features.full(e_train, ta, tb, ("B", "S"), fusion)
        sh_test, miss_te = shuffled_features.full(e_test, va, vb, ("B", "S"), fusion)
        if miss_tr.any() or miss_te.any():
            raise AssertionError("shuffled B/S unexpectedly missing")
        matrices["E+length+shuffled-B/S"] = (
            np.column_stack([e_train, record_train, sh_train[:, 2:]]),
            np.column_stack([e_test, record_test, sh_test[:, 2:]]))
    return matrices, y_train


def fit_scores(learner, x_train, y_train, x_test):
    param = {"svm": 1.0, "tree": 3, "xgb": 3}[learner]
    model = M.classifier(learner, param)
    if learner == "xgb":
        weights = np.where(y_train == 1,
                           len(y_train) / (2 * int(y_train.sum())),
                           len(y_train) / (2 * int((y_train == 0).sum())))
        model.fit(x_train, y_train, sample_weight=weights)
    else:
        model.fit(x_train, y_train)
    return model.decision_function(x_test) if learner == "svm" else model.predict_proba(x_test)[:, 1]


def galleries_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["view", "learner", "arm", "shuffle_seed", "fold", "query_task",
            "candidate_task", "query_model"]
    for key, group in predictions.groupby(keys, sort=False, dropna=False):
        if len(group) != 3 or int(group.same_model.sum()) != 1:
            raise AssertionError(f"bad 3-candidate model gallery {key}")
        score = group.score.to_numpy(dtype=float)
        positive = group.same_model.to_numpy(dtype=bool)
        ps = score[positive][0]
        neg = score[~positive]
        greater = int((neg > ps).sum())
        ties = int((neg == ps).sum())
        rows.append(dict(zip(keys, key)) | {
            "hit1": float(1 / (1 + ties)) if greater == 0 else 0.0,
            "gallery_auc": float(((neg < ps).sum() + 0.5 * ties) / 2),
            "mrr": float(np.mean([1 / rank for rank in range(greater + 1,
                                  greater + ties + 2)])),
        })
    return pd.DataFrame(rows)


def dyad_matrix(galleries, fold, metric, task_ids):
    current = galleries[galleries.fold == fold]
    dyad = current.groupby(["query_task", "candidate_task"])[metric].mean()
    matrix = np.zeros((len(task_ids), len(task_ids)), dtype=float)
    pos = {task: i for i, task in enumerate(task_ids)}
    for (q, c), value in dyad.items():
        matrix[pos[q], pos[c]] = float(value)
    assert len(dyad) == len(task_ids) * (len(task_ids) - 1)
    return matrix


def bootstrap_draws(matrix, draws, rng):
    n = matrix.shape[0]
    query_w = rng.multinomial(n, np.full(n, 1 / n), size=draws)
    gallery_w = rng.multinomial(n, np.full(n, 1 / n), size=draws)
    denominator = n * n - np.einsum("bi,bi->b", query_w, gallery_w)
    values = np.einsum("bi,ij,bj->b", query_w, matrix, gallery_w, optimize=True)
    return values / denominator


def summarize(gallery_df, tasks, fold_map):
    keys = ["view", "learner", "arm", "shuffle_seed"]
    combos = gallery_df[keys].drop_duplicates().itertuples(index=False, name=None)
    matrices = {}
    fold_tasks = {f: sorted(t for t in tasks if fold_map[t] == f) for f in range(3)}
    for view, learner, arm, seed in combos:
        key = (view, learner, arm, int(seed))
        matrices[key] = {metric: {fold: dyad_matrix(
            gallery_df[(gallery_df.view == view) & (gallery_df.learner == learner)
                       & (gallery_df.arm == arm) & (gallery_df.shuffle_seed == seed)],
            fold, metric, fold_tasks[fold]) for fold in range(3)} for metric in METRICS}
    draw_cache = {}
    result = {view: {learner: {} for learner in LEARNERS} for view in sorted(gallery_df.view.unique())}
    rng = np.random.default_rng(SEED + 1)
    for view in result:
        for learner in LEARNERS:
            order = {"E+length": 0, "E+length+LCS": 1,
                     "E+length+B/S": 2, "E+length+shuffled-B/S": 3}
            arms = sorted(gallery_df[(gallery_df.view == view) &
                                     (gallery_df.learner == learner)].arm.unique(),
                          key=lambda arm: order[arm])
            seeds_by_arm = {arm: sorted(gallery_df[(gallery_df.view == view) &
                                                   (gallery_df.learner == learner) &
                                                   (gallery_df.arm == arm)].shuffle_seed.unique())
                            for arm in arms}
            for arm in arms:
                result[view][learner][arm] = {}
                for seed in seeds_by_arm[arm]:
                    absolute, deltas = {}, {}
                    current_key = (view, learner, arm, int(seed))
                    for metric in METRICS:
                        point_folds, draw_folds = [], []
                        for fold in range(3):
                            ts = fold_tasks[fold]
                            mat = matrices[current_key][metric][fold]
                            point_folds.append(float(mat.sum() / (len(ts) * (len(ts) - 1))))
                            cache_key = (fold, len(ts))
                            # Reuse independent query/gallery task weights across every paired arm.
                            if cache_key not in draw_cache:
                                q_w = rng.multinomial(len(ts), np.full(len(ts), 1 / len(ts)), size=N_BOOT)
                                c_w = rng.multinomial(len(ts), np.full(len(ts), 1 / len(ts)), size=N_BOOT)
                                denominator = len(ts) ** 2 - np.einsum("bi,bi->b", q_w, c_w)
                                draw_cache[cache_key] = (q_w, c_w, denominator)
                            q_w, c_w, denominator = draw_cache[cache_key]
                            boot = np.einsum("bi,ij,bj->b", q_w, mat, c_w, optimize=True) / denominator
                            draw_folds.append(boot)
                        fold_weights = [len(fold_tasks[f]) / len(tasks) for f in range(3)]
                        point = float(sum(w * x for w, x in zip(fold_weights, point_folds)))
                        draws = sum(w * x for w, x in zip(fold_weights, draw_folds))
                        absolute[metric] = {"mean": point,
                                            "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}
                        draw_cache[(current_key, metric)] = draws
                    if arm == "E+length":
                        references = ()
                    elif arm == "E+length+LCS":
                        references = ("E+length",)
                    elif arm == "E+length+B/S":
                        references = ("E+length", "E+length+LCS")
                    else:
                        references = ("E+length", "E+length+B/S")
                    for reference in references:
                        reference_key = (view, learner, reference, 0)
                        if reference_key not in matrices:
                            continue
                        delta_metrics = {}
                        for metric in METRICS:
                            # Paired bootstrap: the task resamples are shared across arms.
                            target_draws = draw_cache[(current_key, metric)]
                            base_draws = draw_cache[(reference_key, metric)]
                            delta = target_draws - base_draws
                            # Point difference is calculated from the same fold-macro estimand.
                            base_point = result[view][learner][reference][0]["absolute"][metric]["mean"]
                            delta_metrics[metric] = {"mean": absolute[metric]["mean"] - base_point,
                                                     "ci95": np.quantile(delta, [0.025, 0.975]).tolist()}
                        deltas[reference] = delta_metrics
                    result[view][learner][arm][int(seed)] = {"absolute": absolute, "deltas_vs": deltas}
    return result


def run_view(base_rows, tasks, fold_map, view, shuffle=False, fusion="conditional"):
    rows = base_rows if view == "full" else [
        {**r, "commands": r["commands"][:8], "actions": r["actions"][:8]} for r in base_rows]
    profiles = M.profile_cache(rows, minimum=2)
    shuffled_cache = {}
    if shuffle:
        for seed in SHUFFLE_SEEDS:
            srows = permute_rows(rows, seed)
            shuffled_cache[seed] = (srows, M.profile_cache(srows, minimum=2))
    predictions = []
    fold_audit = {}
    for fold in range(3):
        train_idx = [i for i, r in enumerate(rows) if fold_map[r["task"]] != fold]
        train_pairs, _ = pair_rows(rows, tasks, fold_map, fold, evaluation=False)
        test_pairs, test_meta = pair_rows(rows, tasks, fold_map, fold, evaluation=True)
        if not train_pairs or not test_pairs:
            raise AssertionError(f"empty train/test pairs in fold {fold}")
        real_matrices, y_train = feature_arms(rows, profiles, train_pairs, test_pairs,
                                             train_idx, fusion=fusion)
        matrices_by_seed = {0: real_matrices}
        if shuffle:
            for seed, (srows, sprofiles) in shuffled_cache.items():
                sh_matrices, y_check = feature_arms(rows, profiles, train_pairs,
                    test_pairs, train_idx, sprofiles, srows, fusion=fusion)
                assert np.array_equal(y_train, y_check)
                matrices_by_seed[seed] = {"E+length+shuffled-B/S":
                    sh_matrices["E+length+shuffled-B/S"]}
        fold_audit[str(fold)] = {
            "train_tasks": sorted(t for t in tasks if fold_map[t] != fold),
            "test_tasks": sorted(t for t in tasks if fold_map[t] == fold),
            "train_pair_count": len(train_pairs),
            "train_positive_count": int(sum(p[2] for p in train_pairs)),
            "train_negative_count": int(sum(1 - p[2] for p in train_pairs)),
            "test_pair_count": len(test_pairs),
            "test_gallery_count": len(test_pairs) // 3,
            "train_test_task_intersection": sorted(
                {r["task"] for r in rows if fold_map[r["task"]] != fold} &
                {r["task"] for r in rows if fold_map[r["task"]] == fold}),
        }
        for learner in LEARNERS:
            for shuffle_seed, arm_matrices in matrices_by_seed.items():
                for arm, (x_train, x_test) in arm_matrices.items():
                    scores = fit_scores(learner, x_train, y_train, x_test)
                    for meta, score in zip(test_meta, scores):
                        predictions.append({"view": view, "learner": learner, "arm": arm,
                            "shuffle_seed": int(shuffle_seed), **meta, "score": float(score)})
        print(view, "fold", fold, "train_tasks", len(fold_audit[str(fold)]["train_tasks"]),
              "test_tasks", len(fold_audit[str(fold)]["test_tasks"]), flush=True)
    return predictions, fold_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--fusion", choices=("conditional", "direct"), default="conditional")
    args = parser.parse_args()
    start = time.monotonic()
    rows, tasks, source_revision = load_rows(prefix_minimum=8)
    fold_map = {task: i % 3 for i, task in enumerate(tasks)}
    if len(tasks) != 47:
        raise AssertionError(f"frozen common-prefix cohort changed: expected 47, got {len(tasks)}")
    predictions, audits = [], {}
    for view in ("equal_prefix8", "full"):
        values, audit = run_view(rows, tasks, fold_map, view, shuffle=True, fusion=args.fusion)
        predictions.extend(values)
        audits[view] = audit
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.csv.gz"
    pd.DataFrame(predictions).to_csv(pred_path, index=False,
        compression={"method": "gzip", "mtime": 0})
    galleries = galleries_from_predictions(pd.DataFrame(predictions))
    gallery_path = out / "gallery_metrics.csv.gz"
    galleries.to_csv(gallery_path, index=False, compression={"method": "gzip", "mtime": 0})
    result = summarize(galleries, tasks, fold_map)
    protocol_path = EXP / "cth_model_snapshot_retrieval_protocol.md"
    runner_path = Path(__file__)
    feature_path = EXP / "run_arguments.py"
    catalog = json.loads(CATALOG.read_text())
    model_counts = {model: sum(r["model"] == model for r in rows) for model in SNAPSHOTS}
    manifest = {
        "status": "completed_exploratory_task_disjoint_model_snapshot_retrieval",
        "target": "same recorded model snapshot across different CTF task groups",
        "source_revision": source_revision,
        "snapshot_labels": list(SNAPSHOTS), "snapshot_trace_counts": model_counts,
        "task_count": len(tasks), "trace_count": len(rows), "folds": fold_map,
        "fusion": args.fusion,
        "primary_view": "equal_prefix8", "secondary_view": "full",
        "outer_folds": audits, "bootstrap_draws": N_BOOT,
        "bootstrap_unit": "two-way query-task and gallery-task resampling within held-out fold",
        "pair_count_per_view_learner_real_arm": int(sum(a["test_pair_count"] for a in audits["full"].values())),
        "prediction_rows": len(predictions), "gallery_rows": len(galleries),
        "candidate_gallery_contract": "one same-snapshot and two different-snapshot candidates; query/gallery tasks differ and both are held out",
        "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in
                         (CATALOG, protocol_path, runner_path, feature_path)},
        "catalog_schema_version": catalog["schema_version"],
        "environment": {"python": __import__("sys").version, "numpy": np.__version__,
                        "pandas": pd.__version__},
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "results": result,
    }
    manifest_path = out / "analysis.json"
    manifest_path.write_text(json.dumps(M.finite_json(manifest), ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n")
    print("results", manifest_path, "elapsed", manifest["elapsed_seconds"], flush=True)
    for view in result:
        for learner in LEARNERS:
            baseline = result[view][learner]["E+length"][0]["absolute"]
            real = result[view][learner]["E+length+B/S"][0]
            print(view, learner,
                  "B/S delta Hit@1/AUC/MRR", {
                      m: real["deltas_vs"]["E+length"][m] for m in METRICS
                  }, "LCS Hit@1/AUC", {
                      m: result[view][learner]["E+length+LCS"][0]["absolute"][m]
                      for m in ("hit1", "gallery_auc")
                  }, flush=True)


if __name__ == "__main__":
    main()
