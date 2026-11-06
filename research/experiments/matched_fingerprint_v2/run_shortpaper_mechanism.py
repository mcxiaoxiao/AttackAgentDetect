#!/usr/bin/env python3
"""Run the single frozen five-arm CTH mechanism-control batch."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research/experiments/matched_fingerprint_v2"
RESULTS = ROOT / "research/results/matched_fingerprint_v2"
OUT = RESULTS / "shortpaper_mechanism_20261105"
SEED = 20261103
SHUFFLE_SEEDS = (20261105, 20261106, 20261107, 20261108, 20261109)
N_BOOT = 10_000
ARMS = ("E", "E+records", "E+records+LCS", "E+records+B/S",
        "E+records+shuffled-B/S")
SHARE_COUNTS = (0, 8, 16, 24, 32)
IDENTITY = ("fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
            "query_task", "candidate_task", "same_task", "candidate_position")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(obj)
    return obj


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def lcs_len(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def record_count_matrix(rows: list[dict], raw_audit: dict) -> np.ndarray:
    records = {r["id"]: r for r in raw_audit["per_trace"]}
    counts = []
    for row in rows:
        item = records[row["id"]]
        counts.append([math.log1p(float(item["head_events"])),
                       math.log1p(float(item["raw_shell_chunks"]))])
    return np.asarray(counts, dtype=float)


def pair_records(counts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xa, xb = counts[a], counts[b]
    return np.column_stack([(xa + xb) / 2, np.abs(xa - xb)])


def pair_lcs(rows: list[dict], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    values = np.empty(len(a), dtype=float)
    for i, (ia, ib) in enumerate(zip(a, b)):
        x, y = rows[int(ia)]["commands"], rows[int(ib)]["commands"]
        denom = len(x) + len(y)
        values[i] = 2 * lcs_len(x, y) / denom if denom else 0.0
    return values.reshape(-1, 1)


def permuted_rows(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    result = []
    for row in rows:
        order = list(range(len(row["commands"])))
        rng.shuffle(order)
        item = dict(row)
        item["commands"] = [row["commands"][i] for i in order]
        item["actions"] = [row["actions"][i] for i in order]
        result.append(item)
    return result


def as_features(features, e_train, e_eval, rows, counts, train_a, train_b,
                eval_a, eval_b, rep, fusion):
    rtrain0 = pair_records(counts, train_a, train_b)
    reval0 = pair_records(counts, eval_a, eval_b)
    rscale = StandardScaler().fit(rtrain0)
    rtrain, reval = rscale.transform(rtrain0), rscale.transform(reval0)
    base = {"E": (e_train, e_eval, np.zeros(len(eval_a), dtype=bool)),
            "E+records": (np.column_stack([e_train, rtrain]),
                          np.column_stack([e_eval, reval]),
                          np.zeros(len(eval_a), dtype=bool)),
            "E+records+LCS": (np.column_stack([e_train, rtrain,
                                                    pair_lcs(rows, train_a, train_b)]),
                              np.column_stack([e_eval, reval,
                                                    pair_lcs(rows, eval_a, eval_b)]),
                              np.zeros(len(eval_a), dtype=bool))}
    xtrain, _ = features.full(e_train, train_a, train_b, ("B", "S"), fusion)
    xeval, missing = features.full(e_eval, eval_a, eval_b, ("B", "S"), fusion)
    base["E+records+B/S"] = (np.column_stack([e_train, rtrain, xtrain[:, 2:]]),
                             np.column_stack([e_eval, reval, xeval[:, 2:]]), missing)
    return base


def train_arm(runner, train, labels, validation):
    model = runner.classifier("xgb", int(validation["parameter"]))
    model.fit(train, labels)
    return model.predict_proba(validation["x_eval"])[:, 1]


def fit_one(runner, x_train, y_train, x_eval, depth):
    model = runner.classifier("xgb", int(depth))
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval)[:, 1]


def fit_oof(runner, rows, train_tasks, profiles, counts, fold, rep, depth,
            shuffled_profiles=None):
    contexts = runner.inner_contexts(rows, train_tasks, profiles, fold, "cth")
    by_arm = {arm: {"scores": [], "labels": [], "missing": []} for arm in ARMS}
    for ctx in contexts:
        f, _, _ = ctx["features"][(rep[0], rep[1])]
        if shuffled_profiles is not None:
            f = runner.Features(rows, ctx["train_idx"], shuffled_profiles, rep[0], rep[1])
        ta, tb = ctx["train_a"], ctx["train_b"]
        va, vb = ctx["val_a"], ctx["val_b"]
        e_train, e_val = f.evidence(ta, tb), f.evidence(va, vb)
        matrices = as_features(f, e_train, e_val, rows, counts, ta, tb, va, vb,
                               rep, rep[2])
        y = ctx["train_y"]
        for arm in ("E", "E+records", "E+records+LCS", "E+records+B/S"):
            xt, xv, missing = matrices[arm]
            score = fit_one(runner, xt, y, xv, depth)
            by_arm[arm]["scores"].extend(score.tolist())
            by_arm[arm]["labels"].extend(ctx["val_y"].tolist())
            by_arm[arm]["missing"].extend(missing.tolist())
        if shuffled_profiles is not None:
            sf = runner.Features(rows, ctx["train_idx"], shuffled_profiles, rep[0], rep[1])
            se_train, se_val = sf.evidence(ta, tb), sf.evidence(va, vb)
            smatrices = as_features(sf, se_train, se_val, rows, counts, ta, tb, va, vb,
                                    rep, rep[2])
            xt, xv, missing = smatrices["E+records+B/S"]
            score = fit_one(runner, xt, y, xv, depth)
            by_arm["E+records+shuffled-B/S"]["scores"].extend(score.tolist())
            by_arm["E+records+shuffled-B/S"]["labels"].extend(ctx["val_y"].tolist())
            by_arm["E+records+shuffled-B/S"]["missing"].extend(missing.tolist())
    y_oof = np.asarray(by_arm["E"]["labels"], dtype=np.int8)
    e_threshold = runner.threshold(y_oof, np.asarray(by_arm["E"]["scores"]))
    e_pred = runner.decision(np.asarray(by_arm["E"]["scores"]), e_threshold)
    thresholds, audit = {"E": float(e_threshold)}, {}
    for arm in ARMS[1:4]:
        values = by_arm[arm]
        scores = np.asarray(values["scores"], dtype=float)
        labels = np.asarray(values["labels"], dtype=np.int8)
        miss = np.asarray(values["missing"], dtype=bool)
        threshold = runner.threshold(labels, scores, miss, e_pred)
        pred = runner.decision(scores, threshold, miss, e_pred)
        thresholds[arm] = float(threshold)
        fmr, fsr, ba = runner.macro(labels, pred, np.concatenate(
            [ctx["val_tasks"] for ctx in contexts]))
        audit[arm] = {"threshold": float(threshold), "oof_fmr": float(fmr),
                      "oof_fsr": float(fsr), "oof_ba": float(ba),
                      "fallback_pairs": int(miss.sum()), "oof_pairs": int(len(labels))}
    if shuffled_profiles is not None:
        arm = "E+records+shuffled-B/S"
        values = by_arm[arm]
        scores = np.asarray(values["scores"], dtype=float)
        labels = np.asarray(values["labels"], dtype=np.int8)
        miss = np.asarray(values["missing"], dtype=bool)
        threshold = runner.threshold(labels, scores, miss, e_pred)
        pred = runner.decision(scores, threshold, miss, e_pred)
        thresholds[arm] = float(threshold)
        fmr, fsr, ba = runner.macro(labels, pred, np.concatenate(
            [ctx["val_tasks"] for ctx in contexts]))
        audit[arm] = {"threshold": float(threshold), "oof_fmr": float(fmr),
                      "oof_fsr": float(fsr), "oof_ba": float(ba),
                      "fallback_pairs": int(miss.sum()), "oof_pairs": int(len(labels))}
    audit["E"] = {"threshold": float(e_threshold),
                   "oof_fmr": float(runner.macro(y_oof, e_pred,
                       np.concatenate([ctx["val_tasks"] for ctx in contexts]))[0]),
                   "oof_pairs": int(len(y_oof))}
    return thresholds, audit


def gallery_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for key, group in df.groupby(["arm", "shuffle_seed", "fold", "replicate", "fraction",
                                  "query_id", "query_task"], sort=False, dropna=False):
        y = group.same_task.to_numpy(dtype=bool)
        score = group.score.to_numpy(dtype=float)
        pred = group.predicted_link.to_numpy(dtype=bool)
        pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
        if len(pos) != 1 or len(neg) != 3:
            raise ValueError(f"bad frozen gallery {key}: positive={len(pos)} negative={len(neg)}")
        ps, ns = score[pos[0]], score[neg]
        gt, tie = int((ns > ps).sum()), int((ns == ps).sum())
        fmr, fsr = float(pred[neg].mean()), float(not pred[pos[0]])
        records.append({"arm": key[0], "shuffle_seed": int(key[1]), "fold": int(key[2]),
                        "fraction": float(key[4]), "query_id": key[5],
                        "query_task": key[6], "fmr": fmr, "fsr": fsr,
                        "ba": 1 - (fmr + fsr) / 2,
                        "hit1": 1 / (1 + gt + tie),
                        "gallery_auc": ((ns < ps).sum() + .5*tie) / 3})
    return pd.DataFrame(records)


def summarize_task_metrics(galleries: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    result = []
    for (arm, shuffle_seed, fraction), group in galleries.groupby(
            ["arm", "shuffle_seed", "fraction"], sort=True):
        values = group.groupby("query_task")[["fmr", "fsr", "ba", "hit1", "gallery_auc"]].mean()
        for metric in values:
            val = values[metric].to_numpy(dtype=float)
            ix = rng.integers(0, len(val), size=(N_BOOT, len(val)))
            draws = val[ix].mean(axis=1)
            result.append({"arm": arm, "shuffle_seed": int(shuffle_seed),
                           "fraction": float(fraction), "metric": metric,
                           "mean": float(val.mean()), "ci95_low": float(np.quantile(draws, .025)),
                           "ci95_high": float(np.quantile(draws, .975)),
                           "task_count": int(len(val)), "gallery_count": int(len(group))})
    return pd.DataFrame(result)


def contrast_table(galleries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in (0, *SHUFFLE_SEEDS):
        if seed == 0:
            slice_ = galleries[galleries.shuffle_seed == 0]
        else:
            deterministic = galleries[(galleries.shuffle_seed == 0) &
                                      (galleries.arm != "E+records+shuffled-B/S")]
            randomized = galleries[(galleries.shuffle_seed == seed) &
                                   (galleries.arm == "E+records+shuffled-B/S")]
            slice_ = pd.concat([deterministic, randomized], ignore_index=True)
        for fraction, block in slice_.groupby("fraction"):
            task_metrics = {arm: g.groupby("query_task")[["fmr", "fsr", "ba", "hit1",
                                                         "gallery_auc"]].mean()
                            for arm, g in block.groupby("arm")}
            comparisons = [("E", "E+records"),
                           ("E+records", "E+records+LCS"),
                           ("E+records", "E+records+B/S"),
                           ("E+records", "E+records+shuffled-B/S"),
                           ("E+records+shuffled-B/S", "E+records+B/S")]
            for baseline, candidate in comparisons:
                if baseline not in task_metrics or candidate not in task_metrics:
                    continue
                paired = task_metrics[candidate] - task_metrics[baseline]
                for metric in paired:
                    values = paired[metric].to_numpy(dtype=float)
                    ix = np.random.default_rng(SEED + int(fraction * 1000) + seed).integers(
                        0, len(values), size=(N_BOOT, len(values)))
                    draws = values[ix].mean(axis=1)
                    rows.append({"shuffle_seed": seed, "fraction": float(fraction),
                                 "baseline": baseline, "candidate": candidate,
                                 "metric": metric, "delta": float(values.mean()),
                                 "ci95_low": float(np.quantile(draws, .025)),
                                 "ci95_high": float(np.quantile(draws, .975)),
                                 "task_count": int(len(values))})
        # Equal-weight the five role shares within each task for one main contrast.
        block = slice_.groupby(["arm", "query_task"])[["fmr", "fsr", "ba", "hit1",
                                                       "gallery_auc"]].mean()
        for baseline, candidate in (("E+records", "E+records+LCS"),
                                    ("E+records", "E+records+B/S"),
                                    ("E+records", "E+records+shuffled-B/S"),
                                    ("E+records+shuffled-B/S", "E+records+B/S")):
            if baseline not in block.index.get_level_values(0) or candidate not in block.index.get_level_values(0):
                continue
            delta = block.loc[candidate] - block.loc[baseline]
            for metric, values in delta.items():
                arr = values.to_numpy(dtype=float)
                ix = np.random.default_rng(SEED + seed).integers(0, len(arr), size=(N_BOOT, len(arr)))
                draws = arr[ix].mean(axis=1)
                rows.append({"shuffle_seed": seed, "fraction": "equal_share_mean",
                             "baseline": baseline, "candidate": candidate,
                             "metric": metric, "delta": float(arr.mean()),
                             "ci95_low": float(np.quantile(draws, .025)),
                             "ci95_high": float(np.quantile(draws, .975)),
                             "task_count": int(len(arr))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    runner = load(ROOT / "research/experiments/matched_fingerprint_v1/run.py",
                  "matched_v1_mechanism")
    rows, tasks, folds = runner.build_cth()
    pairs, roles = runner.cth_pairs(rows, tasks, folds)
    assert len(pairs) == 54_400
    raw_path = RESULTS / "raw_command_coverage.json"
    raw_audit = json.loads(raw_path.read_text())
    counts = record_count_matrix(rows, raw_audit)
    profiles = runner.profile_cache(rows, minimum=2)
    profiles_by_seed = {seed: runner.profile_cache(permuted_rows(rows, seed), minimum=2)
                        for seed in SHUFFLE_SEEDS}
    fold_audits = {i: json.loads((ROOT / f"research/results/matched_fingerprint_v1/cth/fold-{i}-audit.json").read_text())
                   for i in sorted(set(folds.values()))}
    fold_splits, model_parameters, reps = {}, {}, {}
    for fold, audit in fold_audits.items():
        fold_splits[fold] = {"test": sorted(t for t in tasks if folds[t] == fold),
                             "train": sorted(t for t in tasks if folds[t] != fold)}
        model_parameters[fold] = int(audit["selection"]["xgb"]["parameter"])
        reps[fold] = tuple(audit["representation"])

    predictions = []
    threshold_audit = {}
    input_paths = [Path(__file__), ROOT / "research/experiments/matched_fingerprint_v1/run.py",
                   ROOT / "research/experiments/overnight_fingerprint_v1/task_folds.csv",
                   raw_path, ROOT / "research/experiments/matched_fingerprint_v2/shortpaper_review_protocol.md",
                   ROOT / "attackagent/fingerprint_v2.py", ROOT / "attackagent/traces.py",
                   ROOT / "research/results/behavior_corpus_v1/evidence.jsonl",
                   ROOT / "research/results/behavior_corpus_v1/trace_registry.csv"]
    input_paths.extend(ROOT / f"research/results/matched_fingerprint_v1/cth/fold-{i}-audit.json"
                       for i in range(3))
    for fold in sorted(fold_splits):
        train_tasks = set(fold_splits[fold]["train"])
        test_pairs = [p for p in pairs if p["fold"] == fold]
        train_pairs = runner.train_pairs(rows, train_tasks, runner.SEED + 900000 + fold)
        train_a, train_b, train_y = runner.train_arrays(train_pairs)
        test_a, test_b, _, _ = runner.as_arrays(test_pairs)
        train_idx = [i for i, r in enumerate(rows) if r["task"] in train_tasks]
        rep, depth = reps[fold], model_parameters[fold]
        threshold_by_seed = {}
        threshold_by_seed[0], audit_oof = fit_oof(runner, rows, train_tasks, profiles,
                                                   counts, fold, rep, depth)
        threshold_audit[f"{fold}:real"] = {"configuration": {"representation": list(rep),
                                            "xgb_depth": depth}, "thresholds": threshold_by_seed[0],
                                            "oof": audit_oof}
        for seed in SHUFFLE_SEEDS:
            shuffled_profiles = profiles_by_seed[seed]
            threshold_by_seed[seed], oof = fit_oof(runner, rows, train_tasks, profiles,
                                                    counts, fold, rep, depth, shuffled_profiles)
            # Keep only the shuffle OOF arm; the non-shuffle thresholds are identical by construction.
            threshold_audit[f"{fold}:shuffle:{seed}"] = {
                "configuration": {"representation": list(rep), "xgb_depth": depth},
                "shuffle_seed": seed,
                "threshold": threshold_by_seed[seed]["E+records+shuffled-B/S"],
                "oof": oof["E+records+shuffled-B/S"]}

        f = runner.Features(rows, train_idx, profiles, rep[0], rep[1])
        e_train, e_test = f.evidence(train_a, train_b), f.evidence(test_a, test_b)
        matrices = as_features(f, e_train, e_test, rows, counts, train_a, train_b,
                               test_a, test_b, rep, rep[2])
        base_score = fit_one(runner, matrices["E"][0], train_y, matrices["E"][1], depth)
        arm_scores = {"E": base_score}
        missing_by_arm = {"E": np.zeros(len(test_pairs), dtype=bool)}
        for arm in ("E+records", "E+records+LCS", "E+records+B/S"):
            xt, xv, missing = matrices[arm]
            arm_scores[arm] = fit_one(runner, xt, train_y, xv, depth)
            missing_by_arm[arm] = missing

        for seed in (0, *SHUFFLE_SEEDS):
            if seed == 0:
                continue
            shuffled_profiles = profiles_by_seed[seed]
            sf = runner.Features(rows, train_idx, shuffled_profiles, rep[0], rep[1])
            se_train, se_test = sf.evidence(train_a, train_b), sf.evidence(test_a, test_b)
            sm = as_features(sf, se_train, se_test, rows, counts, train_a, train_b,
                              test_a, test_b, rep, rep[2])
            xt, xv, missing = sm["E+records+B/S"]
            score = fit_one(runner, xt, train_y, xv, depth)
            arm_scores[("E+records+shuffled-B/S", seed)] = score
            missing_by_arm[("E+records+shuffled-B/S", seed)] = missing

        for arm in ("E", "E+records", "E+records+LCS", "E+records+B/S"):
            score = arm_scores[arm]
            missing = missing_by_arm[arm]
            thresholds = threshold_by_seed[0]
            for i, pair in enumerate(test_pairs):
                use_base = bool(arm.endswith("B/S") and missing[i])
                actual_score = base_score[i] if use_base else score[i]
                actual_t = thresholds["E"] if use_base else thresholds[arm]
                predictions.append({**{k: pair[k] for k in IDENTITY}, "arm": arm,
                                    "shuffle_seed": 0, "score": float(actual_score),
                                    "threshold": float(actual_t),
                                    "predicted_link": int(actual_score >= actual_t),
                                    "fallback": int(use_base)})
        for seed in SHUFFLE_SEEDS:
            arm_key = ("E+records+shuffled-B/S", seed)
            score, missing = arm_scores[arm_key], missing_by_arm[arm_key]
            thresholds = threshold_by_seed[seed]
            for i, pair in enumerate(test_pairs):
                use_base = bool(missing[i])
                actual_score = base_score[i] if use_base else score[i]
                actual_t = thresholds["E"] if use_base else thresholds["E+records+shuffled-B/S"]
                predictions.append({**{k: pair[k] for k in IDENTITY},
                                    "arm": "E+records+shuffled-B/S", "shuffle_seed": seed,
                                    "score": float(actual_score), "threshold": float(actual_t),
                                    "predicted_link": int(actual_score >= actual_t),
                                    "fallback": int(use_base)})
        print("completed fold", fold, "seed thresholds",
              {s: {k: round(v, 6) for k, v in x.items()}
               for s, x in threshold_by_seed.items()}, flush=True)

    pred = pd.DataFrame(predictions)
    assert len(pred) == len(pairs) * 9
    assert pred.groupby(["arm", "shuffle_seed"]).size().eq(len(pairs)).all()
    assert pred.groupby(["arm", "shuffle_seed", "pair_id"]).size().eq(1).all()
    pred.to_csv(out / "predictions.csv.gz", index=False,
                compression={"method": "gzip", "mtime": 0})
    galleries = gallery_metrics(pred)
    task_metrics = summarize_task_metrics(galleries, np.random.default_rng(SEED))
    contrasts = contrast_table(galleries)
    task_metrics.to_csv(out / "task_metrics.csv", index=False)
    contrasts.to_csv(out / "paired_contrasts.csv", index=False)
    dump(out / "threshold_audit.json", threshold_audit)
    meta = {"status": "completed_one_frozen_five_arm_mechanism_batch",
            "protocol": "shortpaper_review_protocol.md",
            "protocol_sha256": sha(ROOT / "research/experiments/matched_fingerprint_v2/shortpaper_review_protocol.md"),
            "test_tasks": len(tasks), "test_traces": len(rows), "outer_folds": 3,
            "pairs_per_arm": len(pairs), "prediction_rows": len(pred),
            "agent_role_shares": [c / 32 for c in SHARE_COUNTS],
            "training_pair_counts_by_fold": {str(f): len(runner.train_pairs(
                rows, set(fold_splits[f]["train"]), runner.SEED + 900000 + f))
                for f in sorted(fold_splits)},
            "representation_by_fold": {str(f): list(reps[f]) for f in reps},
            "xgb_depth_by_fold": {str(f): model_parameters[f] for f in model_parameters},
            "shuffle_seeds": list(SHUFFLE_SEEDS),
            "arm_pairings_identical": True,
            "fallback_counts": {f"{a}:{s}": int(g.fallback.sum()) for (a, s), g in pred.groupby(["arm", "shuffle_seed"])},
            "threshold_audit_sha256": sha(out / "threshold_audit.json"),
            "prediction_sha256": sha(out / "predictions.csv.gz"),
            "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in input_paths if p.exists()},
            "environment": {"python": __import__("sys").version,
                            "numpy": np.__version__, "pandas": pd.__version__},
            "elapsed_seconds": round(time.monotonic() - started, 2)}
    dump(out / "run_manifest.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
