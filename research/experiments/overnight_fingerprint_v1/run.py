#!/usr/bin/env python3
"""Execute the frozen fingerprint study; never writes over earlier experiments."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
import random
import sys
import time
import traceback

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from attackagent.fingerprints import (FEATURES, core8, dtw_distance, fit_column_scale,
                                     matrix, summary_features, transform_matrix)
from attackagent.traces import ATOM_GROUP

EXP = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "research/results/overnight_fingerprint_v1"
CORPUS = ROOT / "research/results/behavior_corpus_v1"
PILOT_PATH = ROOT / "research/experiments/log_fingerprint_pilot_v1/run.py"
SEED = 20261102
WRAPPERS = {"sudo", "su", "sh", "bash", "dash", "env", "sshd", "systemd-executor"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_frozen_inputs():
    protocol = json.loads((EXP / "protocol.json").read_text())
    checked = {}
    for rel, expected in protocol["inputs"].items():
        actual = sha((ROOT / rel).read_bytes())
        if actual != expected:
            raise RuntimeError(f"Frozen input changed: {rel}")
        checked[rel] = actual
    archive = protocol["ait_archive"]
    actual = sha((ROOT / archive["path"]).read_bytes())
    if actual != archive["sha256"]:
        raise RuntimeError("Frozen AIT archive differs from protocol")
    checked[archive["path"]] = actual
    return checked


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def write_json(path: pathlib.Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def ledger(stage: str, status: str, started: float, detail: str = ""):
    path = OUT / "run_ledger.csv"
    exists = path.exists()
    elapsed = time.monotonic() - started
    start_unix = time.time() - elapsed
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "status", "start_unix", "end_unix", "elapsed_seconds", "detail"])
        if not exists:
            w.writeheader()
        w.writerow({"stage": stage, "status": status, "start_unix": f"{start_unix:.3f}",
                    "end_unix": f"{time.time():.3f}", "elapsed_seconds": f"{elapsed:.3f}",
                    "detail": detail})


def vectorizer(ngram=(1, 2)):
    return TfidfVectorizer(ngram_range=ngram, sublinear_tf=True,
                           token_pattern=r"(?u)\b[\w.+-]+\b", lowercase=True)


def command_text(commands):
    return " ".join(commands) if commands else "EMPTYTOKEN"


def action_bigram_text(actions):
    pairs = [f"{a}__{b}" for a, b in zip(actions, actions[1:]) if a is not None and b is not None]
    return " ".join(pairs) if pairs else "EMPTYTOKEN"


def task_macro_ba(records):
    groups = collections.defaultdict(list)
    for row in records:
        groups[row["task"]].append(row)
    vals = []
    for rows in groups.values():
        y = np.asarray([x["y"] for x in rows], int)
        if len(set(y)) == 2:
            vals.append(float(balanced_accuracy_score(y, [x["pred"] for x in rows])))
    return float(np.mean(vals)) if vals else None, len(vals)


def task_cluster_ci(records, seed=SEED, draws=2000):
    groups = collections.defaultdict(list)
    for row in records:
        groups[row["task"]].append(row)
    per_task = []
    for task, rows in sorted(groups.items()):
        y = np.asarray([x["y"] for x in rows], int)
        if len(set(y)) == 2:
            per_task.append(float(balanced_accuracy_score(y, [x["pred"] for x in rows])))
    if len(per_task) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_task), size=(draws, len(per_task)))
    values = np.asarray(per_task)[idx].mean(axis=1)
    return [float(x) for x in np.quantile(values, [.025, .975])]


def recognition_metrics(records):
    if not records:
        return {"n": 0, "available": False}
    y = np.asarray([x["y"] for x in records], int)
    score = np.asarray([x["score"] for x in records], float)
    pred = score >= .5
    macro, n_tasks = task_macro_ba([{**r, "pred": int(p)} for r, p in zip(records, pred)])
    return {"available": len(set(y)) == 2, "n": len(y), "human": int((y == 0).sum()),
            "agent": int((y == 1).sum()), "task_macro_balanced_accuracy": macro,
            "tasks_with_both_classes": n_tasks,
            "task_macro_ba_cluster_bootstrap_95ci": task_cluster_ci(
                [{**r, "pred": int(p)} for r, p in zip(records, pred)]),
            "pooled_balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(set(y)) == 2 else None,
            "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
            "human_fpr": float(np.mean(pred[y == 0])) if np.any(y == 0) else None,
            "agent_fnr": float(np.mean(~pred[y == 1])) if np.any(y == 1) else None,
            "auroc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
            "brier": float(brier_score_loss(y, score)),
            "confusion_human_agent": [[int(np.sum((y == 0) & ~pred)), int(np.sum((y == 0) & pred))],
                                      [int(np.sum((y == 1) & ~pred)), int(np.sum((y == 1) & pred))]]}


def load_pilot():
    spec = importlib.util.spec_from_file_location("frozen_pilot_reference", PILOT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ait_rows(view="login_exec", budget=128, min_commands=16, wrappers=True, shuffle_commands=False, shuffle_windows=False):
    pilot = load_pilot()
    raw_runs = pilot.load_runs()
    base, admission = pilot.view_runs(raw_runs, view)
    rows = []
    for r in base:
        commands = list(r["commands"])
        if not wrappers:
            commands = [x for x in commands if x not in WRAPPERS]
        commands = commands[:budget]
        if len(commands) < min_commands:
            continue
        if shuffle_commands:
            rng = random.Random(SEED + int(r["id"][:8], 16))
            rng.shuffle(commands)
        mat, q = matrix(commands, max_commands=budget, min_commands=min_commands)
        core, whole_quality = core8(commands)
        if mat is None or whole_quality["eligible_pairs"] < 5:
            continue
        if shuffle_windows:
            rng = np.random.default_rng(SEED + int(r["id"][:8], 16))
            mat = mat[rng.permutation(len(mat))]
        rows.append({**r, "commands": commands, "core": core, "matrix": mat,
                     "summary": summary_features(mat), "quality": q})
    return rows, {**admission, "admitted_after_matrix_gate": len(rows), "run_count": len(raw_runs)}


def ait_partitions(rows, panel, source_folds):
    tasks = sorted({r["task"] for r in rows})
    if panel == "task_holdout":
        keys = tasks
    elif panel == "source_holdout":
        keys = range(3)
    elif panel == "application_holdout":
        keys = sorted({r["app"] for r in rows})
    elif panel == "crossed_task_source_holdout":
        keys = list(itertools.product(tasks, range(3)))
    else:
        raise ValueError(panel)
    for key in keys:
        if panel == "task_holdout":
            test = [i for i, r in enumerate(rows) if r["task"] == key]
            train = [i for i, r in enumerate(rows) if r["task"] != key]
        elif panel == "source_holdout":
            test = [i for i, r in enumerate(rows) if source_folds[r["source"]] == key]
            train = [i for i, r in enumerate(rows) if source_folds[r["source"]] != key]
        elif panel == "application_holdout":
            test = [i for i, r in enumerate(rows) if r["app"] == key]
            train = [i for i, r in enumerate(rows) if r["app"] != key]
        else:
            task, fold = key
            test = [i for i, r in enumerate(rows) if r["task"] == task and source_folds[r["source"]] == fold]
            train = [i for i, r in enumerate(rows) if r["task"] != task and source_folds[r["source"]] != fold]
        if not test:
            continue
        test_hashes = {rows[i]["seq_hash"] for i in test}
        train = [i for i in train if rows[i]["seq_hash"] not in test_hashes]
        if not train or len({rows[i]["y"] for i in train}) < 2:
            yield str(key), [], test, len(test_hashes)
            continue
        assert not {rows[i]["seq_hash"] for i in train} & test_hashes
        yield str(key), train, test, len(test_hashes)


def select_classifier_features(rows, idx, feature):
    if feature == "length":
        return np.asarray([[math.log1p(len(rows[i]["commands"]))] for i in idx])
    if feature == "core8":
        return np.asarray([rows[i]["core"] for i in idx])
    values = np.asarray([rows[i]["summary"] for i in idx])
    if feature == "window_summary":
        return values
    keep = {"ablate_tool_selection": list(range(9, 24)),
            "ablate_action_organization": list(range(0, 9)) + list(range(21, 24)),
            "ablate_local_repetition": list(range(0, 21))}[feature]
    return values[:, keep]


def fit_ait_fold(rows, train, test, feature):
    ytrain = [rows[i]["y"] for i in train]
    if feature == "command_tfidf":
        vec = vectorizer()
        xtrain = [command_text(rows[i]["commands"]) for i in train]
        xtest = [command_text(rows[i]["commands"]) for i in test]
        xtr, xte = vec.fit_transform(xtrain), vec.transform(xtest)
        clf = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED)
        clf.fit(xtr, ytrain)
        return clf.predict_proba(xte)[:, 1].astype(float)
    xtrain = select_classifier_features(rows, train, feature)
    xtest = select_classifier_features(rows, test, feature)
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear",
                                             max_iter=2000, random_state=SEED))
    model.fit(xtrain, ytrain)
    return model.predict_proba(xtest)[:, 1].astype(float)


def dtw_ait_fold(rows, train, test):
    center, scale = fit_column_scale([rows[i]["matrix"] for i in train])
    matrices = {i: transform_matrix(rows[i]["matrix"], center, scale) for i in set(train + test)}
    train_by_class = {c: [i for i in train if rows[i]["y"] == c] for c in [0, 1]}
    scores, coverage = {}, {}
    for i in test:
        class_distance = {}
        for c in [0, 1]:
            d = [dtw_distance(matrices[i], matrices[j]) for j in train_by_class[c]]
            d = sorted(x for x in d if np.isfinite(x))[:3]
            class_distance[c] = float(np.mean(d)) if d else math.nan
        if np.isfinite(class_distance[0]) and np.isfinite(class_distance[1]):
            logits = np.exp(-np.asarray([class_distance[0], class_distance[1]]) + min(class_distance.values()))
            scores[i] = float(logits[1] / logits.sum())
            coverage[i] = True
        else:
            coverage[i] = False
    return scores, coverage


def paired_task_delta(base, comparison, draws=2000):
    a = {r["run_id"]: r for r in base}
    b = {r["run_id"]: r for r in comparison}
    shared = sorted(set(a) & set(b))
    tasks = sorted({a[i]["task"] for i in shared})
    deltas, task_rows = [], {}
    for task in tasks:
        ids = [i for i in shared if a[i]["task"] == task]
        y = [a[i]["y"] for i in ids]
        if len(set(y)) < 2:
            continue
        ba = float(balanced_accuracy_score(y, [a[i]["pred"] for i in ids]))
        bb = float(balanced_accuracy_score(y, [b[i]["pred"] for i in ids]))
        task_rows[task] = {"base_ba": ba, "comparison_ba": bb, "delta": bb - ba, "n": len(ids)}
        deltas.append(bb - ba)
    if not deltas:
        return {"available": False, "shared_runs": len(shared), "tasks": 0}
    rng = np.random.default_rng(SEED)
    values = np.asarray(deltas)
    ci = np.quantile(values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1), [.025, .975])
    return {"available": True, "shared_runs": len(shared), "tasks": len(deltas),
            "task_macro_delta_ba": float(values.mean()), "paired_task_bootstrap_95ci": [float(ci[0]), float(ci[1])],
            "task_deltas": task_rows}


def oof_window_predictions(rows, source_folds):
    output = []
    for fold, train, test, purged in ait_partitions(rows, "task_holdout", source_folds):
        if not train or len({rows[i]["y"] for i in train}) < 2:
            continue
        scores = fit_ait_fold(rows, train, test, "window_summary")
        output.extend({"run_id": rows[i]["id"], "task": rows[i]["task"], "y": int(rows[i]["y"]),
                       "score": float(s), "pred": int(s >= .5)} for i, s in zip(test, scores))
    return output


def run_recognition():
    started = time.monotonic()
    all_predictions, fold_rows, cohort = [], [], {}
    models = ["length", "command_tfidf", "core8", "window_summary", "ablate_tool_selection",
              "ablate_action_organization", "ablate_local_repetition", "matrix_dtw_3nn"]
    controls = ["fixed_first32", "without_wrappers", "window_order_shuffle", "command_order_shuffle"]
    pilot = load_pilot()
    raw = pilot.load_runs()
    humans = sorted({r["source"] for r in raw if r["y"] == 0})
    agents = sorted({r["source"] for r in raw if r["y"] == 1})
    if len(humans) != 6 or agents != ["GPT4.1", "GPT4o", "GPT5"]:
        raise RuntimeError("AIT source groups differ from frozen source-fold protocol")
    source_folds = {name: i % 3 for i, name in enumerate(humans)}
    source_folds.update({name: i for i, name in enumerate(agents)})

    main, audit = ait_rows()
    if len(main) != 89 or len({r["task"] for r in main}) != 12:
        raise RuntimeError(f"AIT frozen cohort mismatch: {len(main)} runs, {len({r['task'] for r in main})} tasks")
    cohort["login_exec"] = audit
    variants = {"primary": main}
    first32, q32 = ait_rows(budget=32, min_commands=32)
    variants["fixed_first32"] = first32
    cohort["fixed_first32"] = q32
    no_wrap, qnw = ait_rows(wrappers=False)
    variants["without_wrappers"] = no_wrap
    cohort["without_wrappers"] = qnw
    shufwin, qsw = ait_rows(shuffle_windows=True)
    variants["window_order_shuffle"] = shufwin
    cohort["window_order_shuffle"] = qsw
    shufcmd, qsc = ait_rows(shuffle_commands=True)
    variants["command_order_shuffle"] = shufcmd
    cohort["command_order_shuffle"] = qsc

    for variant_name, rows in variants.items():
        active_models = models if variant_name == "primary" else ["window_summary"]
        for panel in ("task_holdout", "source_holdout", "application_holdout", "crossed_task_source_holdout"):
            panel_preds = collections.defaultdict(dict)
            for fold, train, test, purged in ait_partitions(rows, panel, source_folds):
                fold_rows.append({"variant": variant_name, "panel": panel, "fold": fold, "train_n": len(train),
                                  "test_n": len(test), "purged_exact_sequence_n": purged,
                                  "train_classes": len({rows[i]["y"] for i in train}) if train else 0})
                if not train or len({rows[i]["y"] for i in train}) < 2:
                    continue
                for model in active_models:
                    if model == "matrix_dtw_3nn":
                        pred, coverage = dtw_ait_fold(rows, train, test)
                        for i in test:
                            if coverage.get(i):
                                panel_preds[model][i] = pred[i]
                        continue
                    try:
                        values = fit_ait_fold(rows, train, test, model)
                        for i, score in zip(test, values):
                            panel_preds[model][i] = float(score)
                    except Exception as exc:
                        fold_rows.append({"variant": variant_name, "panel": panel, "fold": fold,
                                          "model": model, "error": f"{type(exc).__name__}: {exc}"})
            for model in active_models:
                recs = [{"run_id": rows[i]["id"], "task": rows[i]["task"], "source": rows[i]["source"],
                         "app": rows[i]["app"], "y": int(rows[i]["y"]), "score": score,
                         "pred": int(score >= .5)} for i, score in sorted(panel_preds[model].items())]
                if not recs:
                    continue
                metric = recognition_metrics(recs)
                metric.update({"variant": variant_name, "panel": panel, "model": model,
                               "coverage": len(recs) / len(rows), "cohort_n": len(rows)})
                all_predictions.extend([{**x, "variant": variant_name, "panel": panel, "model": model}
                                        for x in recs])
                write_json(OUT / "recognition" / f"{variant_name}__{panel}__{model}.json", metric)
                print("AIT", variant_name, panel, model, metric["n"], metric.get("task_macro_balanced_accuracy"), flush=True)

    write_csv(OUT / "recognition" / "predictions.csv", all_predictions)
    write_csv(OUT / "recognition" / "folds.csv", fold_rows)
    write_json(OUT / "recognition" / "cohort.json", cohort)
    metrics = []
    for path in sorted((OUT / "recognition").glob("*__*__*.json")):
        data = json.loads(path.read_text())
        metrics.append(data)
    write_json(OUT / "recognition" / "metrics.json", metrics)
    main_metrics = {m["model"]: m for m in metrics if m.get("variant") == "primary" and m.get("panel") == "task_holdout"}
    if "window_summary" in main_metrics and "core8" in main_metrics:
        rows_by_model = {m: [r for r in all_predictions if r["variant"] == "primary" and r["panel"] == "task_holdout" and r["model"] == m]
                         for m in ["window_summary", "core8"]}
        a, b = rows_by_model["window_summary"], rows_by_model["core8"]
        a_by, b_by = {r["run_id"]: r for r in a}, {r["run_id"]: r for r in b}
        tasks = sorted({r["task"] for r in a if r["run_id"] in b_by})
        per = {}
        for task in tasks:
            aa = [r for r in a if r["task"] == task and r["run_id"] in b_by]
            yy = [r["y"] for r in aa]
            if len(set(yy)) < 2:
                continue
            per[task] = [balanced_accuracy_score(yy, [r["pred"] for r in aa]),
                         balanced_accuracy_score(yy, [b_by[r["run_id"]]["pred"] for r in aa])]
        if per:
            delta = float(np.mean([x[0] - x[1] for x in per.values()]))
            rng = np.random.default_rng(SEED)
            vals = np.asarray([x[0] - x[1] for x in per.values()])
            ci = np.quantile(vals[rng.integers(0, len(vals), (2000, len(vals)))].mean(axis=1), [.025, .975])
            write_json(OUT / "recognition" / "primary_delta.json", {"comparison": "window_summary minus whole_session_core8",
                        "task_macro_delta_ba": delta, "paired_task_bootstrap_95ci": [float(ci[0]), float(ci[1])],
                        "tasks": len(per), "conditional_on_fixed_oof_predictions": True})
    main_task_preds = [r for r in all_predictions if r["variant"] == "primary" and r["panel"] == "task_holdout" and r["model"] == "window_summary"]
    paired = {"ablations": {}, "controls": {}}
    for model in ("ablate_tool_selection", "ablate_action_organization", "ablate_local_repetition"):
        compare = [r for r in all_predictions if r["variant"] == "primary" and r["panel"] == "task_holdout" and r["model"] == model]
        paired["ablations"][model] = paired_task_delta(main_task_preds, compare)
    paired_control_rows = []
    for name in controls:
        variant_rows = variants[name]
        common = {r["id"] for r in main} & {r["id"] for r in variant_rows}
        main_common = [r for r in main if r["id"] in common]
        variant_common = [r for r in variant_rows if r["id"] in common]
        base_pred = oof_window_predictions(main_common, source_folds)
        alt_pred = oof_window_predictions(variant_common, source_folds)
        paired["controls"][name] = paired_task_delta(base_pred, alt_pred)
        base_by, alt_by = {r["run_id"]: r for r in base_pred}, {r["run_id"]: r for r in alt_pred}
        for rid in sorted(set(base_by) & set(alt_by)):
            paired_control_rows.append({"control": name, "run_id": rid, "task": base_by[rid]["task"],
                "label": base_by[rid]["y"], "base_score": base_by[rid]["score"], "base_prediction": base_by[rid]["pred"],
                "control_score": alt_by[rid]["score"], "control_prediction": alt_by[rid]["pred"]})
    write_json(OUT / "recognition" / "paired_comparisons.json", paired)
    write_csv(OUT / "recognition" / "paired_control_predictions.csv", paired_control_rows)
    write_json(OUT / "recognition" / "provenance.json", {
        "status": "exploratory_frozen_protocol_execution", "seed": SEED,
        "pilot_archive_sha256": sha((ROOT / "research/data/external/ait-shell-pilot-v1/data.zip").read_bytes()),
        "pilot_parser_sha256": sha(PILOT_PATH.read_bytes()),
        "fingerprint_implementation_sha256": sha((ROOT / "attackagent/fingerprints.py").read_bytes()),
        "runner_sha256": sha(pathlib.Path(__file__).read_bytes()),
        "frozen_inputs": verify_frozen_inputs(),
        "protocol_sha256": sha((EXP / "protocol.json").read_bytes()),
        "raw_runs": audit["run_count"], "primary_eligible_runs": len(main), "primary_tasks": len({r["task"] for r in main}),
        "source_folds": source_folds, "elapsed_seconds": time.monotonic() - started,
        "feature_inputs": "ordered executable names only; labels, IDs, app, task, source, observation format excluded"})
    ledger("recognition", "complete", started, f"main={len(main)}")
    return metrics


def load_corpus():
    with (CORPUS / "trace_registry.csv").open() as f:
        registry = list(csv.DictReader(f))
    evidence = {}
    with (CORPUS / "evidence.jsonl").open() as f:
        for line in f:
            item = json.loads(line)
            evidence[item["trace_id"]] = item
    rows = []
    for meta in registry:
        item = evidence.get(meta["trace_id"])
        if not item:
            continue
        rows.append({**meta, **item})
    return rows


def cos_matrix(rows, train_indices, all_indices, kind):
    if kind == "command_cosine":
        docs = [command_text(rows[i]["commands"]) for i in train_indices]
        texts = [command_text(rows[i]["commands"]) for i in all_indices]
        vec = vectorizer()
    elif kind == "action_bigram_cosine":
        docs = [action_bigram_text(rows[i]["actions"]) for i in train_indices]
        texts = [action_bigram_text(rows[i]["actions"]) for i in all_indices]
        vec = TfidfVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b[\w.+-]+\b", sublinear_tf=True)
    else:
        raise ValueError(kind)
    train_mat = vec.fit_transform(docs)
    all_mat = vec.transform(texts)
    mapping = {index: pos for pos, index in enumerate(all_indices)}
    return all_mat, mapping


def cosine(a, b):
    return float(a.multiply(b).sum())


def select_by_apportionment(slot_counts, target):
    total = sum(slot_counts)
    quotas = [target * x / total for x in slot_counts]
    alloc = [int(math.floor(x)) for x in quotas]
    for i in sorted(range(len(alloc)), key=lambda j: (quotas[j] - alloc[j], -j), reverse=True)[:target - sum(alloc)]:
        alloc[i] += 1
    return alloc


def choose_modes_and_runs(tasks, sessions_by_task, agent_slots, rng):
    slots = [(task, role) for task in tasks for role in ("query", "gallery")]
    chosen = set(rng.choice(len(slots), size=agent_slots, replace=False).tolist()) if agent_slots else set()
    out = {}
    for pos, (task, role) in enumerate(slots):
        mode = "agent" if pos in chosen else "human"
        pool = sessions_by_task[task][mode]
        if len(pool) < sum(1 for p, (t, _) in enumerate(slots) if t == task and ("agent" if p in chosen else "human") == mode):
            raise RuntimeError("Insufficient distinct same-mode sessions for sampled CTF role slots")
        out[(task, role)] = mode
    for task in tasks:
        qmode, gmode = out[(task, "query")], out[(task, "gallery")]
        if qmode == gmode:
            picks = rng.choice(sessions_by_task[task][qmode], size=2, replace=False)
            out[(task, "query", "row")] = int(picks[0])
            out[(task, "gallery", "row")] = int(picks[1])
        else:
            out[(task, "query", "row")] = int(rng.choice(sessions_by_task[task][qmode]))
            out[(task, "gallery", "row")] = int(rng.choice(sessions_by_task[task][gmode]))
    return out


def inner_threshold(rows, train_idx, train_tasks, kind, source, outer_fold):
    # Task-held-out inner scores; the test concentration never changes this threshold.
    folds = {t: i % 3 for i, t in enumerate(sorted(train_tasks))}
    pos_scores, neg_scores = [], []
    for inner in range(3):
        val_tasks = {t for t in train_tasks if folds[t] == inner}
        fit_idx = [i for i in train_idx if rows[i]["task_group"] not in val_tasks]
        val_idx = [i for i in train_idx if rows[i]["task_group"] in val_tasks]
        if len(val_tasks) < 2 or not fit_idx:
            continue
        matrix_, mapping = cos_matrix(rows, fit_idx, val_idx, kind)
        by_task = collections.defaultdict(list)
        for i in val_idx:
            by_task[rows[i]["task_group"]].append(i)
        keys = sorted(by_task)
        for task in keys:
            for a, b in itertools.combinations(by_task[task], 2):
                s = cosine(matrix_[mapping[a]], matrix_[mapping[b]])
                pos_scores.append(s)
        for a, task_a in ((i, rows[i]["task_group"]) for i in val_idx):
            for task_b in keys:
                if task_a == task_b:
                    continue
                for b in by_task[task_b]:
                    neg_scores.append(cosine(matrix_[mapping[a]], matrix_[mapping[b]]))
    if not pos_scores or not neg_scores:
        return math.inf, {"reason": "insufficient_inner_task_pairs", "positive_pairs": len(pos_scores), "negative_pairs": len(neg_scores)}
    # Choose the training-only threshold with greatest positive recall under FMR <= 10%.
    candidates = sorted(set(neg_scores + pos_scores), reverse=True) + [math.inf]
    best = (0.0, math.inf)
    for threshold in candidates:
        fmr = float(np.mean(np.asarray(neg_scores) >= threshold))
        if fmr <= .10 + 1e-12:
            recall = float(np.mean(np.asarray(pos_scores) >= threshold))
            if recall > best[0] or (math.isclose(recall, best[0]) and threshold > best[1]):
                best = (recall, threshold)
    return float(best[1]), {"reason": None, "positive_pairs": len(pos_scores), "negative_pairs": len(neg_scores),
                            "training_recall_at_threshold": best[0],
                            "training_fmr_at_threshold": float(np.mean(np.asarray(neg_scores) >= best[1]))}


def mixture_task_metrics(pair_rows):
    by_task = collections.defaultdict(list)
    for row in pair_rows:
        by_task[row["query_task"]].append(row)
    task_values = {}
    for task, rows in by_task.items():
        pos = [r for r in rows if r["same_task"]]
        neg = [r for r in rows if not r["same_task"]]
        if pos and neg:
            task_values[task] = {"fsr": float(np.mean([r["pred_link"] == 0 for r in pos])),
                                 "fmr": float(np.mean([r["pred_link"] == 1 for r in neg])),
                                 "hit_at_1": float(np.mean([r["top_credit"] for r in pos]))}
    if not task_values:
        return {}
    fsr = float(np.mean([x["fsr"] for x in task_values.values()]))
    fmr = float(np.mean([x["fmr"] for x in task_values.values()]))
    hit = float(np.mean([x["hit_at_1"] for x in task_values.values()]))
    type_rows = {}
    for typ in ("HH", "HA", "AA"):
        pos = [r for r in pair_rows if r["same_task"] and r["pair_type"] == typ]
        neg = [r for r in pair_rows if not r["same_task"] and r["pair_type"] == typ]
        type_rows[typ] = {"positive_n": len(pos), "fsr": float(np.mean([r["pred_link"] == 0 for r in pos])) if pos else None,
                          "negative_n": len(neg), "fmr": float(np.mean([r["pred_link"] == 1 for r in neg])) if neg else None}
    return {"tasks": len(task_values), "task_macro_fsr": fsr, "task_macro_fmr": fmr,
            "task_macro_balanced_accuracy": 1 - (fsr + fmr) / 2, "task_macro_hit_at_1": hit,
            "pair_type_errors": type_rows, "task_metrics": task_values}


def bootstrap_task_mean(values_by_task, seed=SEED, draws=2000):
    values = [float(np.mean(v)) for _, v in sorted(values_by_task.items()) if v]
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    return [float(x) for x in np.quantile(np.asarray(values)[idx].mean(axis=1), [.025, .975])]


def run_mixture():
    started = time.monotonic()
    source = "cyber-task-horizons"
    rows = [r for r in load_corpus() if r["source_dataset"] == source and r["track"] == "ctf"]
    pools = collections.defaultdict(lambda: {"human": [], "agent": []})
    for i, row in enumerate(rows):
        pools[row["task_group"]][row["mode"]].append(i)
    eligible_tasks = sorted(t for t, p in pools.items() if len(p["human"]) >= 2 and len(p["agent"]) >= 2)
    eligible_rows = [r for r in rows if r["task_group"] in set(eligible_tasks)]
    fold_map = {}
    with (OUT / "task_folds.csv").open() as f:
        for r in csv.DictReader(f):
            if r["endpoint"] == "cth_mixture":
                fold_map[r["group"]] = int(r["fold"])
    if len(eligible_tasks) != 17 or not set(eligible_tasks) <= set(fold_map):
        raise RuntimeError(f"CTF mixture frozen cohort mismatch ({len(eligible_tasks)} tasks)")
    models = ["command_cosine", "action_bigram_cosine"]
    folds = sorted({fold_map[t] for t in eligible_tasks})
    task_by_fold = {fold: [t for t in eligible_tasks if fold_map[t] == fold] for fold in folds}
    slots = [2 * len(task_by_fold[f]) for f in folds]
    nominal_targets = {0.0: 0, .25: 8, .5: 17, .75: 26, 1.0: 34}
    target_by_fraction = {fraction: select_by_apportionment(slots, target)
                          for fraction, target in nominal_targets.items()}
    score_cache, thresholds, threshold_audit, all_records = {}, {}, {}, []
    for fold in folds:
        train_tasks = [t for t in eligible_tasks if fold_map[t] != fold]
        test_tasks = task_by_fold[fold]
        train_idx = [i for i, r in enumerate(rows) if r["task_group"] in set(train_tasks)]
        test_idx = [i for i, r in enumerate(rows) if r["task_group"] in set(test_tasks)]
        for model in models:
            thresholds[(fold, model)], threshold_audit[(fold, model)] = inner_threshold(
                rows, train_idx, set(train_tasks), model, source, fold)
            mat, mapping = cos_matrix(rows, train_idx, test_idx, model)
            score_cache[(fold, model)] = (mat, mapping)
    fold_metrics = []
    for fraction, total_target in nominal_targets.items():
        alloc = target_by_fraction[fraction]
        for rep in range(100):
            for fold_pos, fold in enumerate(folds):
                test_tasks = task_by_fold[fold]
                rng = np.random.default_rng(SEED + rep * 101 + fold * 10007 + int(fraction * 1000))
                selected = choose_modes_and_runs(test_tasks, pools, alloc[fold_pos], rng)
                for task in test_tasks:
                    qi = selected[(task, "query", "row")]
                    pi = selected[(task, "gallery", "row")]
                    negatives = [t for t in test_tasks if t != task]
                    negatives = rng.choice(negatives, size=4, replace=False).tolist()
                    candidate_ids = [pi] + [int(rng.choice(pools[t][selected[(t, "gallery")]])) for t in negatives]
                    qmode = selected[(task, "query")]
                    for model in models:
                        mat, mapping = score_cache[(fold, model)]
                        raw_scores = [cosine(mat[mapping[qi]], mat[mapping[c]]) for c in candidate_ids]
                        threshold = thresholds[(fold, model)]
                        ranked = sorted(range(len(candidate_ids)), key=lambda j: raw_scores[j], reverse=True)
                        top = ranked[0]
                        ties = [j for j in ranked if math.isclose(raw_scores[j], raw_scores[top], rel_tol=1e-10, abs_tol=1e-12)]
                        top_credit = (1.0 / len(ties)) if 0 in ties else 0.0
                        for pos, (cand, neg_task, score) in enumerate(zip(candidate_ids, [task] + negatives, raw_scores)):
                            gmode = selected[(neg_task, "gallery")]
                            pair_type = "HH" if qmode == gmode == "human" else ("AA" if qmode == gmode == "agent" else "HA")
                            all_records.append({"source": source, "fold": fold, "replicate": rep,
                                "nominal_agent_fraction": fraction, "agent_roles_in_fold": alloc[fold_pos],
                                "role_slots_in_fold": 2 * len(test_tasks), "query_task": task,
                                "candidate_task": neg_task, "query_trace_id": rows[qi]["trace_id"],
                                "candidate_trace_id": rows[cand]["trace_id"], "query_mode": qmode,
                                "candidate_mode": gmode, "pair_type": pair_type, "same_task": pos == 0,
                                "method": model, "score": score, "threshold": threshold,
                                "pred_link": int(score >= threshold), "top_credit": top_credit if pos == 0 else 0.0,
                                "actual_agent_fraction": None})
    # Exact overall concentration allocation is fixed by construction; annotate each replicate.
    for fraction in nominal_targets:
        for rep in range(100):
            matching = [r for r in all_records if r["nominal_agent_fraction"] == fraction and r["replicate"] == rep]
            roles = {}
            for r in matching:
                roles[(r["fold"], r["query_task"], "query")] = r["query_mode"]
                roles[(r["fold"], r["candidate_task"], "gallery")] = r["candidate_mode"]
            # Each query role and each gallery role is counted once, regardless of scorer.
            unique = {k: v for k, v in roles.items()}
            actual = sum(v == "agent" for v in unique.values()) / max(1, len(unique))
            for r in matching:
                r["actual_agent_fraction"] = actual
    summary = []
    for fraction in nominal_targets:
        for model in models:
            sub = [r for r in all_records if r["nominal_agent_fraction"] == fraction and r["method"] == model]
            by_rep = collections.defaultdict(list)
            for r in sub:
                by_rep[r["replicate"]].append(r)
            rep_metrics = [mixture_task_metrics(x) for _, x in sorted(by_rep.items())]
            valid = [x for x in rep_metrics if x]
            row = {"source": source, "nominal_agent_fraction": fraction, "method": model,
                   "actual_agent_fraction_mean": float(np.mean([r["actual_agent_fraction"] for r in sub])) if sub else None,
                   "replicates": len(valid), "task_macro_fsr_mean": float(np.mean([x["task_macro_fsr"] for x in valid])) if valid else None,
                   "task_macro_fmr_mean": float(np.mean([x["task_macro_fmr"] for x in valid])) if valid else None,
                   "task_macro_ba_mean": float(np.mean([x["task_macro_balanced_accuracy"] for x in valid])) if valid else None,
                   "task_macro_hit_at_1_mean": float(np.mean([x["task_macro_hit_at_1"] for x in valid])) if valid else None,
                   "sampling_spread_fsr_95pct": [float(x) for x in np.quantile([x["task_macro_fsr"] for x in valid], [.025, .975])] if valid else None,
                   "sampling_spread_fmr_95pct": [float(x) for x in np.quantile([x["task_macro_fmr"] for x in valid], [.025, .975])] if valid else None}
            task_metric_values = {"task_macro_fsr": collections.defaultdict(list),
                                  "task_macro_fmr": collections.defaultdict(list),
                                  "task_macro_hit_at_1": collections.defaultdict(list)}
            for rep_metric in valid:
                for task, vals in rep_metric["task_metrics"].items():
                    for metric_name in task_metric_values:
                        task_metric_values[metric_name][task].append(vals[metric_name.removeprefix("task_macro_")])
            for metric_name, task_values in task_metric_values.items():
                row[metric_name + "_task_cluster_95ci"] = bootstrap_task_mean(task_values)
                row[metric_name + "_tasks"] = len(task_values)
            pair_types = {}
            for typ in ("HH", "HA", "AA"):
                type_records = [r for r in sub if r["pair_type"] == typ]
                neg = [r for r in type_records if not r["same_task"]]
                pos = [r for r in type_records if r["same_task"]]
                pair_types[typ] = {"negative_pairs": len(neg), "fmr": float(np.mean([r["pred_link"] for r in neg])) if neg else None,
                                   "positive_pairs": len(pos), "fsr": float(np.mean([1-r["pred_link"] for r in pos])) if pos else None}
            row["pair_type_errors"] = pair_types
            summary.append(row)
    write_csv(OUT / "mixture" / "pair_predictions.csv", all_records)
    write_json(OUT / "mixture" / "thresholds.json", {f"fold{f}/{m}": {"threshold": thresholds[(f, m)], **threshold_audit[(f, m)]}
                                                        for f in folds for m in models})
    write_json(OUT / "mixture" / "metrics.json", summary)
    write_json(OUT / "mixture" / "eligibility.json", {"source": source, "eligible_tasks": len(eligible_tasks),
                "eligible_sessions_within_selected_tasks": len(eligible_rows), "minimum_commands": 1,
                "folds": {str(f): len(task_by_fold[f]) for f in folds},
                "matrix_increment": {"status": "not_run_insufficient_groups", "eligible_2H2A_tasks_at_16_commands": 1,
                                     "required": 15}})
    # Orientation-specific mixed-mode errors for the motivating Human -> Agent asymmetry.
    orient = []
    for fraction in nominal_targets:
        for model in models:
            for direction, qmode, cmode in [("Human-to-Agent", "human", "agent"), ("Agent-to-Human", "agent", "human")]:
                sub = [r for r in all_records if r["nominal_agent_fraction"] == fraction and r["method"] == model
                       and r["query_mode"] == qmode and r["candidate_mode"] == cmode]
                for positive in (True, False):
                    use = [r for r in sub if r["same_task"] == positive]
                    orient.append({"fraction": fraction, "method": model, "direction": direction,
                                   "pair_label": "positive_same_task" if positive else "negative_different_task",
                                   "n": len(use), "error_rate": float(np.mean([1-r["pred_link"] if positive else r["pred_link"] for r in use])) if use else None})
    write_csv(OUT / "mixture" / "orientation_errors.csv", orient)
    write_json(OUT / "mixture" / "provenance.json", {"status": "exploratory_frozen_protocol_execution", "seed": SEED,
                "source_rows": len(rows), "eligible_tasks": len(eligible_tasks), "replicates_per_fraction": 100,
                "runner_sha256": sha(pathlib.Path(__file__).read_bytes()), "frozen_inputs": verify_frozen_inputs(),
                "task_group_folds": fold_map, "threshold_fit": "3-fold task-held-out training-only pair scores",
                "threshold_target": "empirical training FMR <= 0.10, maximize training positive recall",
                "concentration_slots": {str(f): nominal_targets[f] for f in nominal_targets},
                "matrix_claim": "not evaluated because 1 eligible task < required 15", "elapsed_seconds": time.monotonic()-started})
    print("CTF MIXTURE", len(eligible_tasks), "tasks", len(all_records), "query-candidate records", flush=True)
    ledger("mixture", "complete", started, f"tasks={len(eligible_tasks)}")
    return summary


def make_retrieval_sessions(source, min_commands=16):
    raw = [r for r in load_corpus() if r["source_dataset"] == source and r["track"] == "ctf" and r["mode"] == "agent"]
    sessions = []
    for r in raw:
        commands = list(r["commands"][:128])
        if len(commands) < min_commands:
            continue
        mat, q = matrix(commands, max_commands=128, min_commands=min_commands)
        if mat is None:
            continue
        sessions.append({**r, "commands": commands, "matrix": mat, "quality": q})
    task_counts = collections.Counter(r["task_group"] for r in sessions)
    return [r for r in sessions if task_counts[r["task_group"]] >= 2]


def pair_indices(train_groups, group_by_index, indices, rng):
    pos, neg = [], []
    by_group = collections.defaultdict(list)
    for i in indices:
        by_group[group_by_index[i]].append(i)
    for group, members in by_group.items():
        pos.extend((a, b, 1) for a, b in itertools.combinations(members, 2))
    groups = sorted(by_group)
    for ga, gb in itertools.combinations(groups, 2):
        for a in by_group[ga]:
            for b in by_group[gb]:
                neg.append((a, b, 0))
    n = min(len(pos), len(neg), 10000)
    if n == 0:
        return []
    pos = list(pos) if len(pos) <= n else [pos[i] for i in rng.choice(len(pos), n, replace=False)]
    neg = list(neg) if len(neg) <= n else [neg[i] for i in rng.choice(len(neg), n, replace=False)]
    return pos + neg


def retrieval_query_sets(sessions, test_tasks, group_by_index, k, seed):
    rng = np.random.default_rng(seed)
    by_task = collections.defaultdict(list)
    for i in range(len(sessions)):
        if group_by_index[i] in test_tasks:
            by_task[group_by_index[i]].append(i)
    candidate_tasks = sorted(t for t, v in by_task.items() if len(v) >= 2)
    sets = []
    for task in candidate_tasks:
        negatives = [t for t in candidate_tasks if t != task]
        if len(negatives) < k - 1:
            continue
        for qi in by_task[task]:
            possible = [j for j in by_task[task] if j != qi]
            pi = int(rng.choice(possible))
            neg_tasks = rng.choice(negatives, size=k-1, replace=False).tolist()
            candidate = [pi] + [int(rng.choice(by_task[t])) for t in neg_tasks]
            sets.append({"query": qi, "task": task, "candidates": candidate})
    return sets


def evaluate_ranking(records):
    by_task = collections.defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    per_task = []
    for task, rows in by_task.items():
        per_task.append(float(np.mean([r["top_credit"] for r in rows])))
    return {"query_n": sum(len(v) for v in by_task.values()), "task_n": len(per_task),
            "task_macro_hit_at_1": float(np.mean(per_task)) if per_task else None,
            "task_cluster_bootstrap_95ci": [float(x) for x in np.quantile(
                np.asarray(per_task)[np.random.default_rng(SEED).integers(0, len(per_task), (2000, len(per_task)))].mean(axis=1), [.025, .975])]
                if len(per_task) > 1 else None}


def run_retrieval_source(source, endpoint, candidates):
    sessions = make_retrieval_sessions(source)
    fold_map = {}
    with (OUT / "task_folds.csv").open() as f:
        for r in csv.DictReader(f):
            if r["endpoint"] == endpoint:
                fold_map[r["group"]] = int(r["fold"])
    groups = sorted({s["task_group"] for s in sessions})
    if not set(groups) <= set(fold_map):
        raise RuntimeError(f"Missing frozen task fold for {source}")
    group_by_index = {i: s["task_group"] for i, s in enumerate(sessions)}
    all_predictions, summary, fold_info = [], [], []
    fusion_variants = ("full_matrix", "ablate_tool_selection", "ablate_action_organization",
                       "ablate_local_repetition", "window_order_shuffle")
    for fold in sorted({fold_map[t] for t in groups}):
        train_tasks = [t for t in groups if fold_map[t] != fold]
        test_tasks = [t for t in groups if fold_map[t] == fold]
        train_idx = [i for i in range(len(sessions)) if group_by_index[i] in set(train_tasks)]
        test_idx = [i for i in range(len(sessions)) if group_by_index[i] in set(test_tasks)]
        if not train_idx or not test_idx:
            continue
        docs = [command_text(sessions[i]["commands"]) for i in train_idx]
        vector = vectorizer()
        vector.fit(docs)
        all_docs = vector.transform([command_text(s["commands"]) for s in sessions])
        # Fit DTW scaling with training sessions only; task windows retain event order.
        center, scale = fit_column_scale([sessions[i]["matrix"] for i in train_idx])
        matrices = {i: transform_matrix(sessions[i]["matrix"], center, scale) for i in range(len(sessions))}
        variant_matrices = {}
        drop_cols = {"ablate_tool_selection": [0, 1, 2],
                     "ablate_action_organization": [3, 4, 5, 6],
                     "ablate_local_repetition": [7]}
        for variant in fusion_variants:
            values = {}
            for i, mat in matrices.items():
                value = mat.copy()
                if variant in drop_cols:
                    value[:, drop_cols[variant]] = np.nan
                elif variant == "window_order_shuffle":
                    rng = np.random.default_rng(SEED + int(sessions[i]["trace_id"][:8], 16))
                    value = value[rng.permutation(len(value))]
                values[i] = value
            variant_matrices[variant] = values
        rng = np.random.default_rng(SEED + fold * 9001)
        pairs = pair_indices(train_tasks, group_by_index, train_idx, rng)
        ytrain = [p[2] for p in pairs]
        train_models, train_coverage = {}, {}
        cmd_train = [cosine(all_docs[a], all_docs[b]) for a, b, _ in pairs]
        for variant in fusion_variants:
            Xtrain = []
            for a, b, _ in pairs:
                d = dtw_distance(variant_matrices[variant][a], variant_matrices[variant][b])
                Xtrain.append([cosine(all_docs[a], all_docs[b]), -d if np.isfinite(d) else np.nan])
            Xtrain = np.asarray(Xtrain, dtype=float)
            coverage = float(np.isfinite(Xtrain[:, 1]).mean()) if len(Xtrain) else 0.0
            train_coverage[variant] = coverage
            model = None
            if len(set(ytrain)) == 2 and np.isfinite(Xtrain[:, 1]).sum() >= 10:
                model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                      LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear",
                                                         max_iter=2000, random_state=SEED))
                model.fit(Xtrain, ytrain)
            train_models[variant] = model
        fold_queries = retrieval_query_sets(sessions, set(test_tasks), group_by_index, candidates,
                                            SEED + fold * 12345)
        fold_info.append({"fold": fold, "train_tasks": len(train_tasks), "test_tasks": len(test_tasks),
                          "train_sessions": len(train_idx), "test_sessions": len(test_idx),
                          "training_pairs": len(pairs), "training_matrix_pair_coverage": train_coverage,
                          "fused_available": {k: bool(v) for k, v in train_models.items()},
                          "test_queries": len(fold_queries)})
        for q in fold_queries:
            qi = q["query"]
            base_scores = []
            fused_scores = {variant: [] for variant in fusion_variants}
            matrix_ok = {variant: [] for variant in fusion_variants}
            for ci in q["candidates"]:
                cmd = cosine(all_docs[qi], all_docs[ci])
                base_scores.append(cmd)
                for variant in fusion_variants:
                    dist = dtw_distance(variant_matrices[variant][qi], variant_matrices[variant][ci])
                    matrix_ok[variant].append(bool(np.isfinite(dist)))
                    model = train_models[variant]
                    if model is not None:
                        pairx = np.asarray([[cmd, -dist if np.isfinite(dist) else np.nan]])
                        fused_scores[variant].append(float(model.predict_proba(pairx)[0, 1]))
                    else:
                        fused_scores[variant].append(math.nan)
            candidate_scores = [("command_cosine", base_scores)] + [(variant, fused_scores[variant]) for variant in fusion_variants]
            for model, values in candidate_scores:
                if not np.all(np.isfinite(values)):
                    continue
                top_value = max(values)
                ties = [j for j, value in enumerate(values) if math.isclose(value, top_value, rel_tol=1e-10, abs_tol=1e-12)]
                credit = 1 / len(ties) if 0 in ties else 0.0
                variant = model if model in fusion_variants else "full_matrix"
                all_predictions.append({"source": source, "fold": fold, "task": q["task"],
                    "query_trace_id": sessions[qi]["trace_id"], "method": model, "hit_credit": credit,
                    "matrix_query_candidate_coverage": float(np.mean(matrix_ok[variant])),
                    "query_has_any_matrix_candidate": any(matrix_ok[variant]), "chance": 1/candidates,
                    "candidate_count": candidates})
    model_names = ("command_cosine",) + fusion_variants
    for model in model_names:
        rows = [r for r in all_predictions if r["source"] == source and r["method"] == model]
        metric = evaluate_ranking([{"task": r["task"], "top_credit": r["hit_credit"]} for r in rows])
        metric.update({"source": source, "endpoint": endpoint, "method": model, "candidate_count": candidates,
                       "chance": 1/candidates, "matrix_candidate_coverage_mean":
                       float(np.mean([r["matrix_query_candidate_coverage"] for r in rows])) if rows else None,
                       "query_with_any_matrix_fraction": float(np.mean([r["query_has_any_matrix_candidate"] for r in rows])) if rows else None})
        summary.append(metric)
    # Paired task-level difference; one model never receives a different candidate set.
    by_method_task = {method: collections.defaultdict(list) for method in model_names}
    for r in all_predictions:
        by_method_task[r["method"]][r["task"]].append(r["hit_credit"])
    for model in fusion_variants:
        shared = sorted(set(by_method_task["command_cosine"]) & set(by_method_task[model]))
        deltas = [np.mean(by_method_task[model][t]) - np.mean(by_method_task["command_cosine"][t]) for t in shared]
        if deltas:
            rng = np.random.default_rng(SEED)
            ci = np.quantile(np.asarray(deltas)[rng.integers(0, len(deltas), (2000, len(deltas)))].mean(axis=1), [.025, .975]) if len(deltas) > 1 else [np.nan, np.nan]
            summary.append({"source": source, "endpoint": endpoint, "method": f"paired_delta_{model}_minus_command",
                           "candidate_count": candidates, "task_n": len(deltas), "task_macro_hit_at_1_delta": float(np.mean(deltas)),
                           "task_cluster_bootstrap_95ci": [float(ci[0]), float(ci[1])] if np.isfinite(ci).all() else None,
                           "same_queries_and_candidates": True})
    return all_predictions, summary, fold_info, len(sessions), len(groups)


def run_cross_source_retrieval(train_source, test_source, candidates):
    train_sessions = make_retrieval_sessions(train_source)
    test_sessions = make_retrieval_sessions(test_source)
    train_tasks = sorted({s["task_group"] for s in train_sessions})
    test_tasks = sorted({s["task_group"] for s in test_sessions})
    if not train_sessions or not test_sessions or len(test_tasks) < candidates:
        return [], {"available": False, "reason": "insufficient_eligible_source_tasks"}
    combined = train_sessions + test_sessions
    n_train = len(train_sessions)
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, len(combined)))
    group_by_index = {i: combined[i]["task_group"] for i in range(len(combined))}
    vec = vectorizer()
    vec.fit([command_text(s["commands"]) for s in train_sessions])
    all_docs = vec.transform([command_text(s["commands"]) for s in combined])
    center, scale = fit_column_scale([s["matrix"] for s in train_sessions])
    matrices = {i: transform_matrix(combined[i]["matrix"], center, scale) for i in range(len(combined))}
    rng = np.random.default_rng(SEED + sum(map(ord, train_source + test_source)))
    pairs = pair_indices(train_tasks, group_by_index, train_idx, rng)
    X, y = [], []
    for a, b, label in pairs:
        d = dtw_distance(matrices[a], matrices[b])
        X.append([cosine(all_docs[a], all_docs[b]), -d if np.isfinite(d) else np.nan])
        y.append(label)
    coverage = float(np.isfinite(np.asarray(X)[:, 1]).mean()) if X else 0.0
    if len(set(y)) < 2 or np.isfinite(np.asarray(X)[:, 1]).sum() < 10:
        return [], {"available": False, "reason": "insufficient_train_pairs", "training_pairs": len(pairs),
                    "training_matrix_pair_coverage": coverage}
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear",
                                             max_iter=2000, random_state=SEED))
    model.fit(np.asarray(X), y)
    query_sets = retrieval_query_sets(combined, set(test_tasks), group_by_index, candidates,
                                      SEED + 400019 + sum(map(ord, train_source + test_source)))
    records = []
    for q in query_sets:
        qi = q["query"]
        cmd_scores, matrix_scores, matrix_ok = [], [], []
        for ci in q["candidates"]:
            cmd = cosine(all_docs[qi], all_docs[ci])
            d = dtw_distance(matrices[qi], matrices[ci])
            cmd_scores.append(cmd)
            matrix_ok.append(bool(np.isfinite(d)))
            matrix_scores.append(float(model.predict_proba(np.asarray([[cmd, -d if np.isfinite(d) else np.nan]]))[0, 1]))
        for method, scores in (("command_cosine", cmd_scores), ("command_plus_matrix", matrix_scores)):
            best = max(scores)
            ties = [i for i, x in enumerate(scores) if math.isclose(x, best, rel_tol=1e-10, abs_tol=1e-12)]
            records.append({"train_source": train_source, "test_source": test_source, "task": q["task"],
                "query_trace_id": combined[qi]["trace_id"], "method": method,
                "hit_credit": 1.0 / len(ties) if 0 in ties else 0.0,
                "matrix_query_candidate_coverage": float(np.mean(matrix_ok)), "candidate_count": candidates,
                "chance": 1 / candidates})
    method_metrics = []
    per_task = {method: collections.defaultdict(list) for method in ("command_cosine", "command_plus_matrix")}
    for r in records:
        per_task[r["method"]][r["task"]].append(r["hit_credit"])
    for method in per_task:
        vals = {t: float(np.mean(v)) for t, v in per_task[method].items()}
        arr = np.asarray(list(vals.values()))
        rng = np.random.default_rng(SEED + len(method) + candidates)
        ci = np.quantile(arr[rng.integers(0, len(arr), (2000, len(arr)))].mean(axis=1), [.025, .975]) if len(arr) > 1 else [np.nan, np.nan]
        method_metrics.append({"train_source": train_source, "test_source": test_source, "method": method,
            "task_macro_hit_at_1": float(arr.mean()) if len(arr) else None,
            "task_cluster_bootstrap_95ci": [float(ci[0]), float(ci[1])] if np.isfinite(ci).all() else None,
            "task_n": len(vals), "query_n": sum(len(v) for v in per_task[method].values()),
            "candidate_count": candidates, "chance": 1 / candidates,
            "matrix_candidate_coverage_mean": float(np.mean([r["matrix_query_candidate_coverage"] for r in records
                if r["method"] == method])) if records else None})
    base, fused = per_task["command_cosine"], per_task["command_plus_matrix"]
    shared = sorted(set(base) & set(fused))
    deltas = [float(np.mean(fused[t]) - np.mean(base[t])) for t in shared]
    arr = np.asarray(deltas)
    rng = np.random.default_rng(SEED + candidates)
    ci = np.quantile(arr[rng.integers(0, len(arr), (2000, len(arr)))].mean(axis=1), [.025, .975]) if len(arr) > 1 else [np.nan, np.nan]
    method_metrics.append({"train_source": train_source, "test_source": test_source,
        "method": "paired_delta_matrix_minus_command", "task_macro_hit_at_1_delta": float(arr.mean()) if len(arr) else None,
        "task_cluster_bootstrap_95ci": [float(ci[0]), float(ci[1])] if np.isfinite(ci).all() else None,
        "task_n": len(shared), "same_queries_and_candidates": True})
    info = {"available": True, "train_source": train_source, "test_source": test_source,
        "train_tasks": len(train_tasks), "train_sessions": len(train_sessions), "test_tasks": len(test_tasks),
        "test_sessions": len(test_sessions), "training_pairs": len(pairs),
        "training_matrix_pair_coverage": coverage, "test_queries": len(query_sets),
        "task_relation": "same source-qualified challenge within the held-out target source"}
    return records, {"fold": info, "metrics": method_metrics}


def run_retrieval():
    started = time.monotonic()
    all_predictions, summaries, folds, transfer_predictions, transfer_results = [], [], [], [], []
    specs = [("cyber-task-horizons", "agent_retrieval/cyber-task-horizons", 5),
             ("antieval/cybench-trajectories", "agent_retrieval/antieval/cybench-trajectories", 3)]
    for source, endpoint, k in specs:
        pred, metrics, fold_info, n, tasks = run_retrieval_source(source, endpoint, k)
        all_predictions.extend(pred)
        summaries.extend(metrics)
        folds.extend({"source": source, **x} for x in fold_info)
        print("RETRIEVAL", source, "sessions", n, "tasks", tasks,
              [(m["method"], m.get("task_macro_hit_at_1"), m.get("task_macro_hit_at_1_delta")) for m in metrics], flush=True)
    for train_source, test_source, candidates in [
            ("cyber-task-horizons", "antieval/cybench-trajectories", 3),
            ("antieval/cybench-trajectories", "cyber-task-horizons", 5)]:
        pred, result = run_cross_source_retrieval(train_source, test_source, candidates)
        transfer_predictions.extend(pred)
        if result.get("available"):
            transfer_results.extend(result["metrics"])
            folds.append({"source": f"{train_source}->{test_source}", **result["fold"]})
        else:
            transfer_results.append({"train_source": train_source, "test_source": test_source, **result})
        print("CROSS-SOURCE", train_source, "->", test_source, result, flush=True)
    write_csv(OUT / "retrieval" / "predictions.csv", all_predictions)
    write_csv(OUT / "retrieval" / "folds.csv", folds)
    write_json(OUT / "retrieval" / "metrics.json", summaries)
    write_csv(OUT / "retrieval" / "cross_source_predictions.csv", transfer_predictions)
    write_json(OUT / "retrieval" / "cross_source_metrics.json", transfer_results)
    write_json(OUT / "retrieval" / "provenance.json", {"status": "exploratory_frozen_protocol_execution",
                "seed": SEED, "sources_kept_separate": True, "min_commands": 16,
                "runner_sha256": sha(pathlib.Path(__file__).read_bytes()), "frozen_inputs": verify_frozen_inputs(),
                "models": ["command TF-IDF cosine", "command cosine + negative masked-DTW"],
                "train_pair_cap": 10000, "pair_labels": "same source-qualified task group",
                "elapsed_seconds": time.monotonic() - started})
    ledger("retrieval", "complete", started, f"source_tracks={len(specs)}")
    return summaries


def run_all(stages):
    frozen = verify_frozen_inputs()
    print("FROZEN INPUTS VERIFIED", len(frozen), flush=True)
    for stage, fn in (("recognition", run_recognition), ("mixture", run_mixture), ("retrieval", run_retrieval)):
        if stages not in ("all", stage):
            continue
        t = time.monotonic()
        print(f"START {stage}", flush=True)
        try:
            fn()
        except Exception as exc:
            ledger(stage, "failed", t, f"{type(exc).__name__}: {exc}")
            write_json(OUT / "stage_failure.json", {"stage": stage, "error": str(exc),
                        "traceback": traceback.format_exc(), "time": time.time()})
            raise
        status = OUT / "stage_failure.json"
        if status.exists():
            status.unlink()
        write_json(OUT / "execution_status.json", {"last_completed_stage": stage, "time": time.time(),
                    "status": "in_progress" if stage != "retrieval" else "completed"})
    print("ALL REQUESTED STAGES COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", choices=["all", "recognition", "mixture", "retrieval"], default="all")
    args = parser.parse_args()
    run_all(args.stage)
