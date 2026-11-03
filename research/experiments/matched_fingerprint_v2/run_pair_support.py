#!/usr/bin/env python3
"""Test whether broader pair coverage fixes the CTH fingerprint deficit.

The evaluation gallery, feature representation, learner and outer task folds
are frozen from the existing matched CTH protocol. Only training-pair coverage
changes: the original sampled set versus all distinct within-task positives and
cross-task negatives. Both sets use equal total training weight by label and
HH/HA/AA cell so the comparison isolates coverage more closely.
"""
from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import xgboost

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
RESULTS = ROOT / "research/results/matched_fingerprint_v2"
OUT = RESULTS / "pair_support"
INPUT = RESULTS / "argument_profile/cth/pairs.csv.gz"
AUDITS = RESULTS / "argument_profile/cth"
SEED = 20261105
CONDITIONS = ("sampled_8_per_task", "all_pairs")
VERSIONS = ("E", "E_args", "E_BS", "E_args_BS")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_module(EXP / "run_arguments.py", "run_arguments_pair_support")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exhaustive_pairs(rows, taskset):
    indices = [i for i, row in enumerate(rows) if row["task"] in taskset]
    result = []
    for a, b in itertools.combinations(indices, 2):
        same = int(rows[a]["task"] == rows[b]["task"])
        result.append((a, b, same))
    return result


def pair_type_weights(rows, pairs, total_mass):
    """Give each label equal mass and each observed HH/HA/AA cell equal mass."""
    kinds = [M.pair_type(rows[a]["mode"], rows[b]["mode"]) for a, b, _ in pairs]
    labels = np.asarray([label for _, _, label in pairs], dtype=np.int8)
    kinds = np.asarray(kinds, dtype=object)
    weights = np.zeros(len(pairs), dtype=float)
    for label in (0, 1):
        present = sorted(set(kinds[labels == label]))
        if not present:
            raise AssertionError(f"no training examples for label {label}")
        for kind in present:
            mask = (labels == label) & (kinds == kind)
            weights[mask] = (total_mass / 2) / len(present) / int(mask.sum())
    for label in (0, 1):
        assert abs(weights[labels == label].sum() - total_mass / 2) < 1e-9
        cell_masses = [weights[(labels == label) & (kinds == kind)].sum()
                       for kind in sorted(set(kinds[labels == label]))]
        assert max(cell_masses) - min(cell_masses) < 1e-12
    return weights


def training_pairs(rows, taskset, seed, condition):
    reference = M.train_pairs(rows, taskset, seed)
    if condition == "sampled_8_per_task":
        pairs = reference
    elif condition == "all_pairs":
        pairs = exhaustive_pairs(rows, taskset)
    else:
        raise ValueError(condition)
    return pairs, pair_type_weights(rows, pairs, total_mass=len(reference))


def fit_weighted(param, x_train, y_train, weights, x_score):
    model = M.classifier("xgb", param)
    model.fit(x_train, y_train, sample_weight=weights)
    return model.predict_proba(x_score)[:, 1]


def feature_matrices(features, train_pairs, score_pairs, fusion):
    ta, tb, ty = M.train_arrays(train_pairs)
    qa, qb, qy, qt = M.as_arrays(score_pairs)
    e_train, e_score = features.evidence(ta, tb), features.evidence(qa, qb)
    eh_train, eh_score = features.evidence_head_only(ta, tb), features.evidence_head_only(qa, qb)
    x_hbs_train, _ = features.full(eh_train, ta, tb, ("B", "S"), fusion)
    x_hbs_score, miss_hbs = features.full(eh_score, qa, qb, ("B", "S"), fusion)
    x_abs_train, _ = features.full(e_train, ta, tb, ("B", "S"), fusion)
    x_abs_score, miss_abs = features.full(e_score, qa, qb, ("B", "S"), fusion)
    matrices = {
        "E": (eh_train, eh_score, np.zeros(len(qy), dtype=bool), "E"),
        "E_args": (e_train, e_score, np.zeros(len(qy), dtype=bool), "E_args"),
        "E_BS": (x_hbs_train, x_hbs_score, miss_hbs, "E"),
        "E_args_BS": (x_abs_train, x_abs_score, miss_abs, "E_args"),
    }
    return matrices, qy, qt


def inner_contexts(rows, outer_train_tasks, folds, profiles, rep, outer_fold, condition):
    tasks = sorted(outer_train_tasks)
    inner_map = {task: i % 2 for i, task in enumerate(tasks)}
    contexts = []
    for inner in (0, 1):
        validation_tasks = {task for task in tasks if inner_map[task] == inner}
        fit_tasks = set(tasks) - validation_tasks
        assert not fit_tasks & validation_tasks
        fit_idx = [i for i, row in enumerate(rows) if row["task"] in fit_tasks]
        train, weights = training_pairs(
            rows, fit_tasks, M.SEED + 10000 * outer_fold + 257 * inner, condition)
        validation = M.gallery(rows, validation_tasks, inner, 1,
                               M.SEED + 50000 * outer_fold + 101 * inner)
        features = M.Features(rows, fit_idx, profiles, rep[0], rep[1])
        matrices, y, query_tasks = feature_matrices(features, train, validation, rep[2])
        a, b, labels = M.train_arrays(train)
        contexts.append({"inner": inner, "fit_tasks": sorted(fit_tasks),
                         "validation_tasks": sorted(validation_tasks),
                         "train_pairs": train, "train_weights": weights,
                         "train_arrays": (a, b, labels), "validation_pairs": validation,
                         "features": features, "matrices": matrices,
                         "y": y, "tasks": query_tasks})
    return contexts


def train_outer_fold(rows, tasks, folds, profiles, pairs, fold, condition):
    audit = json.loads((AUDITS / f"fold-{fold}-audit.json").read_text())
    rep = tuple(audit["representation"])
    param = float(audit["selection"]["xgb"]["parameter"])
    test_tasks = {task for task in tasks if folds[task] == fold}
    train_tasks = set(tasks) - test_tasks
    contexts = inner_contexts(rows, train_tasks, folds, profiles, rep, fold, condition)
    oof = {name: {"y": [], "scores": [], "missing": [], "tasks": []}
           for name in VERSIONS}
    inner_counts = []
    for context in contexts:
        weights = context["train_weights"]
        y_train = context["train_arrays"][2]
        inner_counts.append({"inner": context["inner"],
                             "fit_tasks": context["fit_tasks"],
                             "validation_tasks": context["validation_tasks"],
                             "training_pairs": len(y_train),
                             "positive_pairs": int(y_train.sum()),
                             "negative_pairs": int((y_train == 0).sum()),
                             "pair_type_counts": dict(collections.Counter(
                                 M.pair_type(rows[a]["mode"], rows[b]["mode"])
                                 for a, b, _ in context["train_pairs"])),
                             "weighted_mass_by_label": {
                                 str(label): float(weights[y_train == label].sum())
                                 for label in (0, 1)}})
        for name in VERSIONS:
            x_train, x_score, missing, _ = context["matrices"][name]
            score = fit_weighted(param, x_train, y_train, weights, x_score)
            oof[name]["y"].extend(context["y"].tolist())
            oof[name]["scores"].extend(score.tolist())
            oof[name]["missing"].extend(missing.tolist())
            oof[name]["tasks"].extend(context["tasks"].tolist())

    thresholds, oof_audit = {}, {}
    for name in VERSIONS:
        y = np.asarray(oof[name]["y"], dtype=np.int8)
        score = np.asarray(oof[name]["scores"], dtype=float)
        missing = np.asarray(oof[name]["missing"], dtype=bool)
        if name == "E_BS":
            fallback_name = "E"
        elif name == "E_args_BS":
            fallback_name = "E_args"
        else:
            fallback_name = None
        fallback_pred = None
        if fallback_name:
            fallback = np.asarray(oof[fallback_name]["scores"], dtype=float)
            fallback_threshold = thresholds[fallback_name]
            fallback_pred = M.decision(fallback, fallback_threshold)
            threshold = M.threshold(y, score, missing, fallback_pred)
            pred = M.decision(score, threshold, missing, fallback_pred)
        else:
            threshold = M.threshold(y, score)
            pred = M.decision(score, threshold)
        thresholds[name] = threshold
        metric = M.macro(y, pred, np.asarray(oof[name]["tasks"], dtype=object))
        oof_audit[name] = {"threshold": float(threshold),
                           "fmr": float(metric[0]), "fsr": float(metric[1]),
                           "ba": float(metric[2]), "pairs": int(len(y)),
                           "missing_fingerprint_pairs": int(missing.sum())}

    outer_train, weights = training_pairs(
        rows, train_tasks, M.SEED + 900000 + fold, condition)
    train_idx = [i for i, row in enumerate(rows) if row["task"] in train_tasks]
    features = M.Features(rows, train_idx, profiles, rep[0], rep[1])
    test_pairs = [pair for pair in pairs if int(pair["fold"]) == fold]
    assert test_pairs
    matrices, test_y, test_tasks = feature_matrices(features, outer_train, test_pairs, rep[2])
    _, _, train_y = M.train_arrays(outer_train)
    records = []
    raw_scores = {}
    for name in VERSIONS:
        x_train, x_test, missing, fallback_name = matrices[name]
        score = fit_weighted(param, x_train, train_y, weights, x_test)
        raw_scores[name] = (score, missing)
    predictions = {}
    for name in VERSIONS:
        score, missing = raw_scores[name]
        fallback_name = matrices[name][3]
        fallback_scores = None
        fallback_pred = None
        if fallback_name in ("E", "E_args"):
            fallback_scores = raw_scores[fallback_name][0]
            fallback_pred = M.decision(fallback_scores, thresholds[fallback_name])
        pred = M.decision(score, thresholds[name], missing if fallback_name in
                          ("E", "E_args") else None, fallback_pred)
        predictions[name] = pred
        for i, pair in enumerate(test_pairs):
            use_fallback = bool(missing[i]) and fallback_name in ("E", "E_args")
            actual_score = float(fallback_scores[i] if use_fallback else score[i])
            actual_threshold = float(thresholds[fallback_name] if use_fallback else thresholds[name])
            actual_pred = bool(fallback_pred[i] if use_fallback else pred[i])
            item = {key: pair[key] for key in (
                "fold", "pair_id", "replicate", "fraction", "query_idx", "candidate_idx",
                "query_id", "candidate_id", "query_task", "candidate_task", "query_mode",
                "candidate_mode", "pair_type", "same_task", "candidate_position")}
            item.update(condition=condition, learner="xgb", version=name,
                        representation="/".join(map(str, rep)), parameter=param,
                        score=f"{actual_score:.12g}", threshold=f"{actual_threshold:.12g}",
                        predicted_link=int(actual_pred), fallback=int(use_fallback))
            records.append(item)
    y_outer = np.asarray([pair["same_task"] for pair in test_pairs], dtype=np.int8)
    assert np.array_equal(y_outer, np.asarray(test_y, dtype=np.int8))
    return records, {"fold": fold, "train_tasks": sorted(train_tasks),
                     "test_tasks": sorted(test_tasks), "representation": list(rep),
                     "parameter": param, "thresholds": thresholds,
                     "inner_oof_metrics": oof_audit,
                     "inner_training_audits": inner_counts,
                     "outer_training_pairs": len(outer_train),
                     "outer_positive_pairs": int(train_y.sum()),
                     "outer_negative_pairs": int((train_y == 0).sum()),
                     "outer_weighted_mass_by_label": {
                         str(label): float(weights[train_y == label].sum())
                         for label in (0, 1)},
                     "outer_pair_type_counts": dict(collections.Counter(
                         M.pair_type(rows[a]["mode"], rows[b]["mode"])
                         for a, b, _ in outer_train)),
                     "test_pairs": len(test_pairs),
                     "test_fallback_pairs": {
                         name: int(raw_scores[name][1].sum()) for name in VERSIONS}}


def read_frozen_pairs():
    with gzip.open(INPUT, "rt", newline="", encoding="utf8") as handle:
        result = []
        for row in csv.DictReader(handle):
            for key in ("fold", "replicate", "query_idx", "candidate_idx",
                        "same_task", "candidate_position"):
                row[key] = int(row[key])
            row["fraction"] = float(row["fraction"])
            result.append(row)
    return result


def run():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, tasks, folds = M.build_cth()
    profiles = M.profile_cache(rows, minimum=2)
    generated_pairs, _ = M.cth_pairs(rows, tasks, folds)
    pairs = read_frozen_pairs()
    keys = ("fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
            "query_task", "candidate_task", "query_mode", "candidate_mode", "pair_type",
            "same_task", "candidate_position")
    assert len(generated_pairs) == len(pairs) == 54400
    assert [tuple(str(p[k]) for k in keys) for p in generated_pairs] == [
        tuple(str(p[k]) for k in keys) for p in pairs]

    output, fold_audits = [], {}
    for condition in CONDITIONS:
        fold_audits[condition] = {}
        for fold in sorted(set(folds.values())):
            records, audit = train_outer_fold(rows, tasks, folds, profiles, pairs, fold, condition)
            output.extend(records)
            fold_audits[condition][str(fold)] = audit
            print(condition, "fold", fold, "train pairs", audit["outer_training_pairs"],
                  "thresholds", audit["thresholds"], flush=True)

    out_predictions = OUT / "pair_predictions.csv.gz"
    with gzip.open(out_predictions, "wt", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    manifest = {"status": "completed_development_pair_support_diagnostic",
                "dataset": "CTH 17-task mixed-role development cohort",
                "hypothesis": "broader training-pair coverage reduces variance or recovers same-task signal",
                "comparison": "sampled 8 positives per training task versus all distinct within-task positives and cross-task negatives",
                "weighting": "both conditions assign equal total weight to labels and equal mass to each observed HH/HA/AA label cell; same weights apply to E and E+B/S feature versions",
                "primary_learner": "XGBoost, using each frozen outer-fold representation and parameter",
                "versions": list(VERSIONS), "outer_folds": fold_audits,
                "task_count": len(tasks), "trace_count": len(rows),
                "pair_manifest_rows": len(pairs), "prediction_rows": len(output),
                "pair_manifest_sha256": sha(INPUT),
                "inputs_sha256": {"evidence": sha(ROOT / "research/results/behavior_corpus_v1/evidence.jsonl"),
                                  "registry": sha(ROOT / "research/results/behavior_corpus_v1/trace_registry.csv"),
                                  "runner": sha(Path(__file__)),
                                  "features": sha(ROOT / "attackagent/fingerprint_v2.py")},
                "environment": {"python": sys.version, "numpy": np.__version__,
                                "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                                "seed": SEED},
                "elapsed_seconds": round(time.monotonic() - started, 2)}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                        indent=2, allow_nan=False) + "\n")
    print(json.dumps({"prediction_rows": len(output), "elapsed_seconds": manifest["elapsed_seconds"]},
                     indent=2))


if __name__ == "__main__":
    run()
