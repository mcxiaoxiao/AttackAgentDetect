#!/usr/bin/env python3
"""Evaluate B/S/T fingerprint readouts on the existing AIT audit logs.

Two cohorts are reported separately: the frozen v1 89-run cohort for paired
comparison, and all traces with at least one valid four-event fingerprint
window. Labels are the dataset's Human/Agent execution labels, not identities.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import importlib.util
import json
import math
import pathlib
import random
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from attackagent.fingerprint_v2 import combine_summary, fingerprint_matrix, summarize  # noqa: E402

PILOT_PATH = ROOT / "research/experiments/log_fingerprint_pilot_v1/run.py"
V1_PREDICTIONS = ROOT / "research/results/overnight_fingerprint_v1/recognition/predictions.csv"
OUT = ROOT / "research/results/behavioral_fingerprint_v2/ait_recognition"
SEED = 20261103
MIN_EVENTS = 4
MAX_EVENTS = 128
FEATURES = ("core8", "B", "B_plus_S", "B_plus_S_plus_T", "command_tfidf", "length")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_rows(pilot):
    raw = pilot.load_runs()
    rows = []
    for run in raw:
        events = [e for e in run["events"] if e["login"]][:MAX_EVENTS]
        commands = [e["command"] for e in events]
        if len(commands) < MIN_EVENTS:
            continue
        timestamps = [e["time"] for e in events]
        blocks, audit = fingerprint_matrix(commands, timestamps=timestamps,
                                           window=8, min_window_events=4,
                                           max_events=MAX_EVENTS)
        if blocks is None:
            continue
        core, quality = pilot.stats(commands)
        rows.append({"id": run["id"], "task": run["task"], "app": run["app"],
                     "source": run["source"], "actor": run["actor"],
                     "y": int(run["y"]), "commands": commands,
                     "timestamps": timestamps, "blocks": blocks,
                     "core8": np.asarray([np.nan if core[k] is None else core[k]
                                           for k in pilot.FEATURES], dtype=float),
                     "B": summarize(blocks["B"]),
                     "B_plus_S": combine_summary(blocks, ("B", "S")),
                     "B_plus_S_plus_T": combine_summary(blocks, ("B", "S", "T")),
                     "quality": quality, "audit": audit})
    return raw, rows


def cohort_ids_v1():
    ids = set()
    with V1_PREDICTIONS.open(newline="") as f:
        for row in csv.DictReader(f):
            if (row["variant"] == "primary" and row["panel"] == "task_holdout"
                    and row["model"] == "window_summary"):
                ids.add(row["run_id"])
    if len(ids) != 89:
        raise RuntimeError(f"Expected 89 frozen v1 IDs, found {len(ids)}")
    return ids


def source_folds(rows):
    humans = sorted({r["source"] for r in rows if r["y"] == 0})
    agents = sorted({r["source"] for r in rows if r["y"] == 1})
    mapping = {name: i % 3 for i, name in enumerate(humans)}
    mapping.update({name: i % 3 for i, name in enumerate(agents)})
    return mapping


def partitions(rows, panel, folds):
    if panel == "task_holdout":
        keys = sorted({r["task"] for r in rows})
        for key in keys:
            test = [i for i, r in enumerate(rows) if r["task"] == key]
            train = [i for i, r in enumerate(rows) if r["task"] != key]
            yield key, train, test
    elif panel == "source_holdout":
        for fold in sorted(set(folds[r["source"]] for r in rows)):
            test = [i for i, r in enumerate(rows) if folds[r["source"]] == fold]
            train = [i for i, r in enumerate(rows) if folds[r["source"]] != fold]
            yield str(fold), train, test
    else:
        raise ValueError(panel)


def feature_matrix(rows, indices, method):
    if method == "length":
        return np.asarray([[math.log1p(len(rows[i]["commands"]))] for i in indices], float)
    return np.asarray([rows[i][method] for i in indices], float)


def predict_fold(rows, train, test, method):
    ytrain = np.asarray([rows[i]["y"] for i in train], int)
    if len(set(ytrain)) < 2:
        return None
    if method == "command_tfidf":
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                              token_pattern=r"(?u)\b[\w.+-]+\b", lowercase=True)
        xtrain = vec.fit_transform([" ".join(rows[i]["commands"]) for i in train])
        xtest = vec.transform([" ".join(rows[i]["commands"]) for i in test])
        model = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear",
                                   max_iter=2000, random_state=SEED)
        model.fit(xtrain, ytrain)
        return model.predict_proba(xtest)[:, 1]
    xtrain, xtest = feature_matrix(rows, train, method), feature_matrix(rows, test, method)
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(),
                          LogisticRegression(C=1.0, class_weight="balanced",
                                             solver="liblinear", max_iter=2000,
                                             random_state=SEED))
    model.fit(xtrain, ytrain)
    return model.predict_proba(xtest)[:, 1]


def predict_extra_trees(rows, train, test):
    ytrain = [rows[i]["y"] for i in train]
    if len(set(ytrain)) < 2:
        return None
    xtrain, xtest = feature_matrix(rows, train, "B_plus_S_plus_T"), feature_matrix(rows, test, "B_plus_S_plus_T")
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          ExtraTreesClassifier(n_estimators=300, max_depth=4,
                                               min_samples_leaf=3, class_weight="balanced",
                                               random_state=SEED, n_jobs=1))
    model.fit(xtrain, ytrain)
    return model.predict_proba(xtest)[:, 1]


def task_bootstrap(values_by_task, seed, draws=5000):
    values = np.asarray([values_by_task[k] for k in sorted(values_by_task)], dtype=float)
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    return [float(x) for x in np.quantile(values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1), [.025, .975])]


def summarize_predictions(rows, predictions, dataset, panel, method):
    if not predictions:
        return {"dataset": dataset, "panel": panel, "method": method, "available": False}
    y = np.asarray([r["label"] for r in predictions], int)
    score = np.asarray([r["score"] for r in predictions], float)
    pred = score >= .5
    by_task = collections.defaultdict(list)
    for r in predictions:
        by_task[r["task"]].append(r)
    task_ba = {}
    for task, items in by_task.items():
        yy = np.asarray([x["label"] for x in items], int)
        pp = np.asarray([x["score"] >= .5 for x in items], bool)
        if len(set(yy)) == 2:
            task_ba[task] = float(balanced_accuracy_score(yy, pp))
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    return {"dataset": dataset, "panel": panel, "method": method, "available": len(set(y)) == 2,
            "n": int(len(y)), "human": int(np.sum(y == 0)), "agent": int(np.sum(y == 1)),
            "task_count": len(task_ba), "task_macro_ba": float(np.mean(list(task_ba.values()))) if task_ba else None,
            "task_cluster_95ci_ba": task_bootstrap(task_ba, SEED + len(method) + len(dataset)),
            "pooled_ba": float(balanced_accuracy_score(y, pred)) if len(set(y)) == 2 else None,
            "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
            "human_fpr": fp / (tn + fp) if tn + fp else None,
            "agent_fnr": fn / (fn + tp) if fn + tp else None,
            "auroc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
            "brier": float(brier_score_loss(y, score)),
            "confusion_human_agent": [[tn, fp], [fn, tp]]}


def main():
    pilot = load_module(PILOT_PATH, "ait_pilot_v2")
    raw, extended_rows = load_rows(pilot)
    frozen_ids = cohort_ids_v1()
    extended_by_id = {r["id"]: r for r in extended_rows}
    core_rows = [extended_by_id[i] for i in sorted(frozen_ids) if i in extended_by_id]
    if len(core_rows) != 89:
        raise RuntimeError(f"The v2 feature view only retained {len(core_rows)} of the frozen 89")
    cohorts = {"frozen_v1_89": core_rows,
               "extended_all_eligible_min4": extended_rows}
    predictions_out, fold_out, metrics_out = [], [], []
    cohort_audit = {"raw_runs": len(raw), "raw_human": sum(not r["y"] for r in raw),
                    "raw_agent": sum(bool(r["y"]) for r in raw),
                    "extended_min4_eligible": len(extended_rows),
                    "extended_mode_counts": dict(collections.Counter("agent" if r["y"] else "human" for r in extended_rows)),
                    "extended_tasks": len({r["task"] for r in extended_rows}),
                    "v1_frozen_89_ids_available": len(core_rows),
                    "v1_frozen_89_mode_counts": dict(collections.Counter("agent" if r["y"] else "human" for r in core_rows)),
                    "v1_frozen_89_tasks": len({r["task"] for r in core_rows}),
                    "time_fields": "auditd EXECVE launch timestamps; no execution durations were present; T block uses measured inter-launch intervals only"}
    all_pred_index = {}
    for dataset, rows in cohorts.items():
        mapping = source_folds(rows)
        cohort_audit[dataset] = {"n": len(rows), "task_count": len({r["task"] for r in rows}),
                                 "source_fold_map": mapping,
                                 "valid_time_gaps": sum(r["audit"]["time"].get("valid_gaps", 0) for r in rows),
                                 "missing_time_events": sum(r["audit"]["time"].get("timestamps", 0) == 0 for r in rows)}
        for panel in ("task_holdout", "source_holdout"):
            by_method = collections.defaultdict(dict)
            for fold, train, test in partitions(rows, panel, mapping):
                train_classes = sorted({rows[i]["y"] for i in train})
                available = len(train_classes) == 2
                fold_out.append({"dataset": dataset, "panel": panel, "fold": fold,
                                 "train_n": len(train), "test_n": len(test),
                                 "train_classes": train_classes, "available": available})
                if not available:
                    continue
                for method in FEATURES:
                    if method == "extra_trees_B_plus_S_plus_T":
                        continue
                    scores = predict_fold(rows, train, test, method)
                    if scores is None:
                        continue
                    for i, score in zip(test, scores):
                        by_method[method][i] = float(score)
            # The shallow tree model is a fixed comparator on the same folds.
            for fold, train, test in partitions(rows, panel, mapping):
                if len({rows[i]["y"] for i in train}) < 2:
                    continue
                scores = predict_extra_trees(rows, train, test)
                if scores is None:
                    continue
                for i, score in zip(test, scores):
                    by_method["extra_trees_B_plus_S_plus_T"][i] = float(score)

            for method, scores_by_index in by_method.items():
                records = []
                for i, score in sorted(scores_by_index.items()):
                    r = {"dataset": dataset, "panel": panel, "method": method,
                         "run_id": rows[i]["id"], "task": rows[i]["task"],
                         "source_group": rows[i]["source"], "label": rows[i]["y"],
                         "score": score, "prediction": int(score >= .5)}
                    records.append(r)
                    predictions_out.append(r)
                    all_pred_index[(dataset, panel, method, r["run_id"])] = r
                metrics_out.append(summarize_predictions(rows, records, dataset, panel, method))
            print(f"AIT {dataset} {panel}: {len(rows)} eligible; methods={len(by_method)}", flush=True)

    paired = []
    for dataset, rows in cohorts.items():
        for panel in ("task_holdout", "source_holdout"):
            comparisons = (
                ("B_plus_S", "core8"),
                ("B_plus_S_plus_T", "core8"),
                ("B_plus_S_plus_T", "B_plus_S"),
                ("extra_trees_B_plus_S_plus_T", "core8"),
            )
            for method, base_method in comparisons:
                per_task = collections.defaultdict(lambda: {"base": [], "comparison": []})
                common_ids = set()
                for r in predictions_out:
                    if r["dataset"] == dataset and r["panel"] == panel and r["method"] == base_method:
                        if (dataset, panel, method, r["run_id"]) in all_pred_index:
                            common_ids.add(r["run_id"])
                for rid in sorted(common_ids):
                    a = all_pred_index[(dataset, panel, base_method, rid)]
                    b = all_pred_index[(dataset, panel, method, rid)]
                    per_task[a["task"]]["base"].append(a)
                    per_task[a["task"]]["comparison"].append(b)
                task_delta = {}
                for task, groups in per_task.items():
                    ya = np.asarray([x["label"] for x in groups["base"]], int)
                    yb = np.asarray([x["label"] for x in groups["comparison"]], int)
                    if len(set(ya)) < 2 or not np.array_equal(ya, yb):
                        continue
                    ba = balanced_accuracy_score(ya, [x["score"] >= .5 for x in groups["base"]])
                    bb = balanced_accuracy_score(yb, [x["score"] >= .5 for x in groups["comparison"]])
                    task_delta[task] = float(bb - ba)
                paired.append({"dataset": dataset, "panel": panel, "comparison": method,
                               "baseline": base_method, "task_count": len(task_delta),
                               "task_macro_delta_ba": float(np.mean(list(task_delta.values()))) if task_delta else None,
                               "task_cluster_95ci": task_bootstrap(task_delta, SEED+len(method)+len(panel)) if task_delta else [None, None]})
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "metrics.json", metrics_out)
    write_json(OUT / "paired_deltas.json", paired)
    write_json(OUT / "cohorts.json", cohort_audit)
    write_csv(OUT / "predictions.csv", predictions_out)
    write_csv(OUT / "folds.csv", fold_out)
    write_json(OUT / "provenance.json", {"protocol": "ait-execution-fingerprint-v2",
                "seed": SEED, "min_events": MIN_EVENTS, "max_events": MAX_EVENTS,
                "feature_sets": {"core8": 8, "B": 24, "B_plus_S": 48, "B_plus_S_plus_T": 72,
                                 "command_tfidf": "unigrams+bigrams", "length": 1,
                                 "extra_trees_B_plus_S_plus_T": "max_depth=4, min_samples_leaf=3, n_estimators=300"},
                "classifier": "task/source grouped fold; StandardScaler and median imputation fit on training only; balanced logistic regression C=1; fixed probability cutoff=0.5",
                "input_exclusions": ["task", "app", "source group", "actor name", "label", "audit IDs", "collection format"],
                "cohorts": cohort_audit,
                "pilot_archive_sha256": sha256(ROOT / "research/data/external/ait-shell-pilot-v1/data.zip"),
                "pilot_parser_sha256": sha256(PILOT_PATH),
                "fingerprint_code_sha256": sha256(ROOT / "attackagent/fingerprint_v2.py"),
                "runner_sha256": sha256(pathlib.Path(__file__))})
    print(json.dumps({"cohorts": cohort_audit, "metrics": metrics_out, "paired": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
