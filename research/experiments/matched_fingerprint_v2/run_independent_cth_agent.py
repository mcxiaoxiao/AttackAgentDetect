#!/usr/bin/env python3
"""Frozen AA-only confirmation on previously unused CTH CTF task groups."""
from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import xgboost

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
RESULTS = ROOT / "research/results/matched_fingerprint_v2"
MANIFEST = RESULTS / "independent_cth_task_manifest.json"
OUT = RESULTS / "independent_cth_agent"
SEED = 20261104


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


M = module(EXP / "run_arguments.py", "run_arguments_independent")
ARG = module(EXP / "argument_profile.py", "argument_profile_independent")
LEGACY = module(ROOT / "research/experiments/overnight_fingerprint_v1/run.py", "legacy_independent")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def build_rows():
    dev_rows, tasks, task_folds = M.build_cth()
    manifest = json.loads(MANIFEST.read_text())
    wanted = {trace["trace_id"] for task in manifest["eligible_tasks"] for trace in task["traces"]}
    all_cth = [r for r in LEGACY.load_corpus()
               if r.get("source_dataset") == "cyber-task-horizons" and r.get("track") == "ctf"]
    external = [{"id": r["trace_id"], "task": r["task_group"], "mode": r["mode"],
                 "commands": r["commands"], "actions": r["actions"], "timestamps": None,
                 "locator": r.get("record_locator", "")}
                for r in all_cth if r["trace_id"] in wanted]
    assert len(external) == len(wanted), (len(external), len(wanted))
    assert not ({r["task"] for r in external} & set(tasks))
    assert all(r["mode"] == "agent" for r in external)
    assert len({r["id"] for r in [*dev_rows, *external]}) == len(dev_rows) + len(external)
    profiles = ARG.cth_profiles(external)
    for row in external:
        profile = profiles[row["id"]]
        assert profile["recognized_heads"] == len(row["commands"])
        row["argument_profile"] = profile["features"]
    rows = [*dev_rows, *sorted(external, key=lambda r: (r["task"], r["id"]))]
    assert len({r["id"] for r in rows}) == len(rows)
    return rows, tasks, task_folds, manifest


def selected_configs():
    audits = [json.loads((RESULTS / "argument_profile" / "cth" / f"fold-{fold}-audit.json").read_text())
              for fold in range(3)]
    reps = collections.Counter(tuple(a["representation"]) for a in audits)
    representation = max(reps, key=lambda r: (reps[r], r == M.PRIOR, -r[0]))
    params = {}
    fold_parameters = {}
    for learner in ("svm", "tree", "xgb"):
        values = [a["selection"][learner]["parameter"] for a in audits]
        counts = collections.Counter(values)
        params[learner] = max(counts, key=lambda p: (counts[p], -float(p)))
        fold_parameters[learner] = values
    return representation, params, {
        "representation_by_outer_fold": [a["representation"] for a in audits],
        "parameter_by_outer_fold": fold_parameters,
        "selection_rule": "mode of the three frozen outer-fold selections; ties resolve to prior representation and smaller parameter",
    }


def crossfit_thresholds(rows, tasks, folds, profiles, representation, params):
    chunks = {learner: {name: {"y": [], "tasks": [], "score": [], "missing": []}
                        for name in ("E", "E_args", "E_BS", "E_args_BS")}
              for learner in ("svm", "tree", "xgb")}
    fold_audit = []
    for fold in sorted(set(folds.values())):
        test_tasks = {t for t in tasks if folds[t] == fold}
        train_tasks = set(tasks) - test_tasks
        train_idx = [i for i, r in enumerate(rows) if r["task"] in train_tasks]
        train_pairs = M.train_pairs(rows, train_tasks, M.SEED + 900000 + fold)
        ta, tb, ty = M.train_arrays(train_pairs)
        val_pairs = M.gallery(rows, test_tasks, fold, 1, M.SEED + 700000 + fold,
                              fraction=-1)
        va, vb, vy, val_tasks = M.as_arrays(val_pairs)
        features = M.Features(rows, train_idx, profiles, representation[0], representation[1])
        e_train, e_val = features.evidence(ta, tb), features.evidence(va, vb)
        eh_train, eh_val = features.evidence_head_only(ta, tb), features.evidence_head_only(va, vb)
        x_e_bs_train, _ = features.full(eh_train, ta, tb, ("B", "S"), representation[2])
        x_e_bs_val, missing_e_bs = features.full(eh_val, va, vb, ("B", "S"), representation[2])
        x_ea_bs_train, _ = features.full(e_train, ta, tb, ("B", "S"), representation[2])
        x_ea_bs_val, missing_ea_bs = features.full(e_val, va, vb, ("B", "S"), representation[2])
        matrices = {"E": (eh_train, eh_val, np.zeros(len(vy), dtype=bool)),
                    "E_args": (e_train, e_val, np.zeros(len(vy), dtype=bool)),
                    "E_BS": (x_e_bs_train, x_e_bs_val, missing_e_bs),
                    "E_args_BS": (x_ea_bs_train, x_ea_bs_val, missing_ea_bs)}
        for learner in ("svm", "tree", "xgb"):
            parameter = params[learner]
            for name, (x_train, x_val, missing) in matrices.items():
                scores = M.fit_scores(learner, parameter, x_train, ty, x_val)
                c = chunks[learner][name]
                c["y"].extend(vy.tolist())
                c["tasks"].extend(val_tasks.tolist())
                c["score"].extend(scores.tolist())
                c["missing"].extend(missing.tolist())
        fold_audit.append({"fold": fold, "fit_tasks": sorted(train_tasks),
                           "validation_tasks": sorted(test_tasks),
                           "training_pair_count": len(train_pairs),
                           "validation_query_count": len(val_pairs) // 4,
                           "validation_pair_count": len(val_pairs),
                           "validation_pair_types": dict(collections.Counter(p["pair_type"] for p in val_pairs))})
    thresholds = {}
    for learner in ("svm", "tree", "xgb"):
        result = {}
        threshold_values = {}
        for name in ("E", "E_args", "E_BS", "E_args_BS"):
            c = chunks[learner][name]
            y = np.asarray(c["y"], dtype=np.int8)
            scores = np.asarray(c["score"], dtype=float)
            missing = np.asarray(c["missing"], dtype=bool)
            if name == "E_BS":
                fallback_scores = np.asarray(chunks[learner]["E"]["score"], dtype=float)
                fallback_t = threshold_values["E"]
                fallback_pred = M.decision(fallback_scores, fallback_t)
                threshold = M.threshold(y, scores, missing, fallback_pred)
                pred = M.decision(scores, threshold, missing, fallback_pred)
            elif name == "E_args_BS":
                fallback_scores = np.asarray(chunks[learner]["E_args"]["score"], dtype=float)
                fallback_t = threshold_values["E_args"]
                fallback_pred = M.decision(fallback_scores, fallback_t)
                threshold = M.threshold(y, scores, missing, fallback_pred)
                pred = M.decision(scores, threshold, missing, fallback_pred)
            else:
                threshold = M.threshold(y, scores)
                pred = M.decision(scores, threshold)
            metric = M.macro(y, pred, np.asarray(c["tasks"], dtype=object))
            threshold_values[name] = threshold
            result[name] = {"threshold": finite(threshold),
                            "threshold_is_infinite": bool(np.isinf(threshold)),
                            "oof_macro_fmr": float(metric[0]), "oof_macro_fsr": float(metric[1]),
                            "oof_macro_ba": float(metric[2]),
                            "oof_pairs": int(len(y)),
                            "oof_query_task_count": int(len(set(c["tasks"]))) }
        thresholds[learner] = result
    return thresholds, fold_audit


def external_pairs(rows, manifest, repetitions=100):
    tasks = [r["task"] for r in manifest["eligible_tasks"]]
    by = collections.defaultdict(list)
    for i, row in enumerate(rows):
        if row["task"] in tasks and row["mode"] == "agent":
            by[row["task"]].append(i)
    task_fold = {task: int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) % 3
                 for task in sorted(tasks)}
    assert all(len(by[t]) >= 2 for t in tasks)
    assert all(sum(f == task_fold[t] for t in tasks) >= 4 for f in range(3))
    output = []
    for rep in range(repetitions):
        rng = random.Random(SEED + rep)
        qindex = 0
        for fold in range(3):
            pool_tasks = sorted(t for t in tasks if task_fold[t] == fold)
            for task in pool_tasks:
                for q in sorted(by[task], key=lambda i: rows[i]["id"]):
                    same = [i for i in by[task] if i != q]
                    positive = rng.choice(same)
                    neg_tasks = rng.sample([t for t in pool_tasks if t != task], 3)
                    candidates = [positive] + [rng.choice(by[t]) for t in neg_tasks]
                    for position, candidate in enumerate(candidates):
                        M.append_pair(output, rows, q, candidate, fold, rep, 1.0,
                                      qindex, position)
                    qindex += 1
    assert len(output) == repetitions * sum(len(v) for v in by.values()) * 4
    assert all(p["pair_type"] == "AA" for p in output)
    return output, task_fold


def run():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, tasks, folds, manifest = build_rows()
    profiles = M.profile_cache(rows, minimum=2)
    representation, params, selected_audit = selected_configs()
    thresholds, cv_audit = crossfit_thresholds(rows, tasks, folds, profiles,
                                               representation, params)
    pair_rows, external_fold_map = external_pairs(rows, manifest)
    M.write_gz_csv(OUT / "pair_manifest.csv.gz", pair_rows)

    dev_idx = [i for i, r in enumerate(rows) if r["task"] in set(tasks)]
    train_pairs = M.train_pairs(rows, set(tasks), M.SEED + 950001)
    ta, tb, ty = M.train_arrays(train_pairs)
    qa, qb, _, _ = M.as_arrays(pair_rows)
    features = M.Features(rows, dev_idx, profiles, representation[0], representation[1])
    e_train, e_test = features.evidence(ta, tb), features.evidence(qa, qb)
    eh_train, eh_test = features.evidence_head_only(ta, tb), features.evidence_head_only(qa, qb)
    x_e_bs_train, _ = features.full(eh_train, ta, tb, ("B", "S"), representation[2])
    x_e_bs_test, missing_e_bs = features.full(eh_test, qa, qb, ("B", "S"), representation[2])
    x_ea_bs_train, _ = features.full(e_train, ta, tb, ("B", "S"), representation[2])
    x_ea_bs_test, missing_ea_bs = features.full(e_test, qa, qb, ("B", "S"), representation[2])
    matrices = {"E": (eh_train, eh_test, np.zeros(len(pair_rows), dtype=bool), None),
                "E_args": (e_train, e_test, np.zeros(len(pair_rows), dtype=bool), "E"),
                "E_BS": (x_e_bs_train, x_e_bs_test, missing_e_bs, "E"),
                "E_args_BS": (x_ea_bs_train, x_ea_bs_test, missing_ea_bs, "E_args")}
    records = []
    for learner in ("svm", "tree", "xgb"):
        parameter = params[learner]
        scored = {}
        for name, (x_train, x_test, missing, fallback_name) in matrices.items():
            scores = M.fit_scores(learner, parameter, x_train, ty, x_test)
            scored[name] = (scores, missing)
        for name, (scores, missing) in scored.items():
            fallback_name = matrices[name][3]
            fallback_pred = None
            fallback_scores = None
            fallback_threshold = None
            if fallback_name:
                fallback_scores = scored[fallback_name][0]
                fallback_threshold = thresholds[learner][fallback_name]["threshold"]
                if thresholds[learner][fallback_name]["threshold_is_infinite"]:
                    fallback_threshold = np.inf
                fallback_pred = M.decision(fallback_scores, fallback_threshold)
            threshold_info = thresholds[learner][name]
            limit = np.inf if threshold_info["threshold_is_infinite"] else threshold_info["threshold"]
            pred = M.decision(scores, limit, missing if fallback_name else None, fallback_pred)
            for i, pair in enumerate(pair_rows):
                use_fallback = bool(missing[i]) if fallback_name else False
                actual_score = float(fallback_scores[i] if use_fallback else scores[i])
                actual_limit = float(fallback_threshold if use_fallback else limit)
                actual_pred = bool(fallback_pred[i] if use_fallback else pred[i])
                item = {k: pair[k] for k in (
                    "fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
                    "query_task", "candidate_task", "query_mode", "candidate_mode",
                    "pair_type", "same_task", "candidate_position")}
                item.update(dataset="cth_independent_aa", learner=learner, variant=name,
                            representation="/".join(map(str, representation)), parameter=parameter,
                            score=f"{actual_score:.12g}",
                            threshold="inf" if np.isinf(actual_limit) else f"{actual_limit:.12g}",
                            predicted_link=int(actual_pred), fallback=int(use_fallback))
                records.append(item)
    out_path = OUT / "predictions.csv.gz"
    with gzip.open(out_path, "wt", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metadata = {"status": "executed_frozen_external_task_confirmation",
                "source": "Cyber Task Horizons CTF; unseen source-qualified task groups only",
                "source_revision": "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae",
                "evaluation_scope": "Agent-only AA; same_task is the challenge identity proxy",
                "test_task_count": len(manifest["eligible_tasks"]),
                "test_trace_count": sum(t["eligible_trace_count"] for t in manifest["eligible_tasks"]),
                "test_query_count_per_repetition": len({p["query_id"] for p in pair_rows}),
                "repetitions": 100, "candidate_count": 4,
                "training_task_count": len(tasks), "training_trace_count": len(dev_idx),
                "training_pair_count": len(train_pairs),
                "external_candidate_task_partition": external_fold_map,
                "representation": list(representation), "parameters": params,
                "selection_audit": selected_audit,
                "thresholds": thresholds, "threshold_oof_folds": cv_audit,
                "pair_manifest_rows": len(pair_rows), "prediction_rows": len(records),
                "inputs_sha256": {"trace_evidence": sha(ROOT / "research/results/behavior_corpus_v1/evidence.jsonl"),
                                  "trace_registry": sha(ROOT / "research/results/behavior_corpus_v1/trace_registry.csv"),
                                  "task_manifest": sha(MANIFEST),
                                  "runner": sha(Path(__file__)),
                                  "fingerprint_v2": sha(ROOT / "attackagent/fingerprint_v2.py"),
                                  "argument_profile": sha(EXP / "argument_profile.py")},
                "environment": {"python": sys.version, "numpy": np.__version__,
                                "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                                "seed": SEED},
                "elapsed_seconds": round(time.monotonic() - started, 2)}
    write_json(OUT / "run_manifest.json", metadata)
    print(json.dumps({k: metadata[k] for k in (
        "test_task_count", "test_trace_count", "test_query_count_per_repetition",
        "training_task_count", "training_trace_count", "training_pair_count",
        "pair_manifest_rows", "prediction_rows", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    run()
