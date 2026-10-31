#!/usr/bin/env python3
"""Fixed, task-grouped exploratory feature expansion; never replaces v1 outputs."""
from __future__ import annotations
import collections
import csv
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "scripts"))
import run_mixed_gallery as gallery
from run_feature_ablation import (BASE_SEED, SPLIT_SEEDS, grouped_folds, make_design,
                                  calculate_task_metrics, aggregate_task_metrics, task_bootstrap)
from attackagent import ModeScore, TraceEvidence
from attackagent.profiles import (FEATURE_NAMES, FEATURE_SPECS, LENGTH_SENSITIVE,
                                  extract_behavior_features, build_behavior_profile)

OUT = ROOT / "results" / "behavior_profiles_v2"
METHODS = ("command_tfidf", "legacy_F3", "numeric24", "expanded24", "expanded21_control", "inner_selected")
WEIGHTS = (0.0, 0.1, 0.3, 1.0)


def numeric(rows, control=False):
    names = [name for name in FEATURE_NAMES if not control or name not in LENGTH_SENSITIVE]
    return np.asarray([[np.nan if values[name] is None else values[name] for name in names]
                       for values in [extract_behavior_features(r.names, r.groups) for r in rows]], dtype=float)


def matrices(train, score, *, control=False):
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), sublinear_tf=True)
    tc = vectorizer.fit_transform([" ".join(r.names) for r in train])
    sc = vectorizer.transform([" ".join(r.names) for r in score])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    tn = scaler.fit_transform(imputer.fit_transform(numeric(train, control)))
    sn = scaler.transform(imputer.transform(numeric(score, control)))
    return tc, sc, csr_matrix(tn), csr_matrix(sn)


def fit_score(x, z, rows, seed):
    model = LogisticRegression(C=1, class_weight="balanced", max_iter=3000, random_state=seed)
    model.fit(x, [int(r.label == "agent") for r in rows])
    return model.predict_proba(z)[:, 1]


def summary_metrics(rows, probabilities):
    labels = np.asarray([int(r.label == "agent") for r in rows])
    metrics = calculate_task_metrics(rows, labels, probabilities)
    for metric in metrics:
        ids = [i for i, r in enumerate(rows) if r.benchmark == metric["benchmark"] and r.task == metric["task"]]
        metric["brier"] = float(np.mean((probabilities[ids] - labels[ids]) ** 2))
    return metrics


def aggregate(metrics):
    result = aggregate_task_metrics(metrics)
    result["brier"] = float(np.mean([np.mean([r["brier"] for r in metrics if r["benchmark"] == b])
                                    for b in sorted({r["benchmark"] for r in metrics})]))
    return result


def choose_weight(train, seed):
    metrics = {weight: [] for weight in WEIGHTS}
    for fold, (train_keys, valid_keys) in enumerate(grouped_folds(train, seed + 1700)):
        fitting = [r for r in train if (r.benchmark, r.task) in train_keys]
        valid = [r for r in train if (r.benchmark, r.task) in valid_keys]
        tc, sc, tn, sn = matrices(fitting, valid)
        for weight in WEIGHTS:
            x, z = (tc, sc) if weight == 0 else (hstack([tc, tn * weight]), hstack([sc, sn * weight]))
            metrics[weight].extend(summary_metrics(valid, fit_score(x, z, fitting, seed + fold)))
    scores = {weight: aggregate(values)["balanced_accuracy"] for weight, values in metrics.items()}
    # The tuple ordering makes an exact tie prefer the baseline or a smaller weight.
    chosen = max(WEIGHTS, key=lambda weight: (scores[weight], -weight))
    return chosen, scores


def evaluate(train, score, seed, methods=METHODS):
    if {(r.benchmark, r.task) for r in train} & {(r.benchmark, r.task) for r in score}:
        raise AssertionError("Task leakage")
    tc, sc, tn, sn = matrices(train, score)
    predictions, audit = {}, {}
    for method in methods:
        if method == "command_tfidf":
            x, z = tc, sc
        elif method == "legacy_F3":
            x, z, _ = make_design(train, score, "F3_command_plus_structure", seed)
        elif method == "numeric24":
            x, z = tn, sn
        elif method == "expanded24":
            x, z = hstack([tc, tn]), hstack([sc, sn])
        elif method == "expanded21_control":
            _, _, ctl_train, ctl_score = matrices(train, score, control=True)
            x, z = hstack([tc, ctl_train]), hstack([sc, ctl_score])
        else:
            weight, inner = choose_weight(train, seed)
            audit = {"selected_weight": weight, "inner_task_macro_ba": inner}
            x, z = (tc, sc) if weight == 0 else (hstack([tc, tn * weight]), hstack([sc, sn * weight]))
        predictions[method] = fit_score(x, z, train, seed)
    return predictions, audit


def averaged_tasks(records):
    groups = collections.defaultdict(list)
    for row in records:
        groups[(row["benchmark"], row["task"])].append(row)
    keys = ("balanced_accuracy", "macro_f1", "auroc", "human_recall", "agent_recall", "brier")
    return [{"benchmark": b, "task": task, **{key: float(np.mean([r[key] for r in values])) for key in keys}}
            for (b, task), values in sorted(groups.items())]


def paired_delta(records, method, cohort):
    first = averaged_tasks([r for r in records if r["cohort"] == cohort and r["method"] == method])
    base = {(r["benchmark"], r["task"]): r for r in averaged_tasks(
        [r for r in records if r["cohort"] == cohort and r["method"] == "command_tfidf"])}
    differences = [{"benchmark": r["benchmark"], "task": r["task"],
                    "balanced_accuracy": r["balanced_accuracy"] - base[(r["benchmark"], r["task"])]["balanced_accuracy"]}
                   for r in first]
    benchmarks = sorted({r["benchmark"] for r in differences})
    delta = float(np.mean([np.mean([r["balanced_accuracy"] for r in differences if r["benchmark"] == b]) for b in benchmarks]))
    return {"delta_ba": delta, "ci95": task_bootstrap(differences, "balanced_accuracy", draws=5000), "task_groups": len(differences)}


def run():
    started = time.monotonic()
    rows = gallery.load_rows()
    if len(rows) != 177 or len({(r.benchmark, r.task) for r in rows}) != 40:
        raise ValueError("Pinned cohort changed; audit before running")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, predictions, fold_audit = [], [], []

    def record(cohort, train, test, seed, fold, methods=METHODS):
        if time.monotonic() - started > 1200:
            raise RuntimeError("Fixed 20-minute experiment budget reached")
        probs, audit = evaluate(train, test, seed + fold, methods)
        for method, values in probs.items():
            metrics.extend({"cohort": cohort, "method": method, "seed": seed, "fold": fold, **r}
                           for r in summary_metrics(test, values))
            predictions.extend({"cohort": cohort, "method": method, "seed": seed, "fold": fold,
                                "trace_id": r.trace_id, "benchmark": r.benchmark, "task": r.task,
                                "label_for_evaluation_only": r.label, "agent_like_score": float(value)}
                               for r, value in zip(test, values))
        fold_audit.append({"cohort": cohort, "seed": seed, "fold": fold,
                           "train_ids": [r.trace_id for r in train], "test_ids": [r.trace_id for r in test],
                           "train_task_groups": len({(r.benchmark, r.task) for r in train}),
                           "test_task_groups": len({(r.benchmark, r.task) for r in test}), **audit})
        print(f"{cohort}: seed={seed}, fold={fold}, train={len(train)}, test={len(test)}, weight={audit.get('selected_weight')}", flush=True)

    for seed in SPLIT_SEEDS:
        for fold, (train_keys, test_keys) in enumerate(grouped_folds(rows, seed), 1):
            record("full", [r for r in rows if (r.benchmark, r.task) in train_keys],
                   [r for r in rows if (r.benchmark, r.task) in test_keys], seed, fold)

    diagnostics = ("command_tfidf", "inner_selected")
    eligible = [r for r in rows if len(r.names) >= 4]
    # Pair the same task cohort, original vs prefix; retain paired command/action alignment.
    for cohort, current in (("eligible4_full", eligible), ("eligible4_prefix4", [replace(r, names=r.names[:4], groups=r.groups[:4]) for r in eligible])):
        for fold, (train_keys, test_keys) in enumerate(grouped_folds(current, BASE_SEED), 1):
            record(cohort, [r for r in current if (r.benchmark, r.task) in train_keys],
                   [r for r in current if (r.benchmark, r.task) in test_keys], BASE_SEED, fold, diagnostics)
    shuffled = []
    for r in rows:
        pairs = list(zip(r.names, r.groups))
        np.random.default_rng(BASE_SEED + int(r.trace_id[:8], 16)).shuffle(pairs)
        shuffled.append(replace(r, names=tuple(x[0] for x in pairs), groups=tuple(x[1] for x in pairs)))
    for fold, (train_keys, test_keys) in enumerate(grouped_folds(shuffled, BASE_SEED), 1):
        record("shuffled", [r for r in shuffled if (r.benchmark, r.task) in train_keys],
               [r for r in shuffled if (r.benchmark, r.task) in test_keys], BASE_SEED, fold, diagnostics)
    for fold, b in enumerate(sorted({r.benchmark for r in rows}), 1):
        record("leave_benchmark:" + b, [r for r in rows if r.benchmark != b],
               [r for r in rows if r.benchmark == b], BASE_SEED, fold, diagnostics)
    for model_index, source in enumerate(sorted({r.source for r in rows if r.label == "agent"})):
        seed = BASE_SEED + 900 + model_index
        for fold, (train_keys, test_keys) in enumerate(grouped_folds(rows, seed), 1):
            train = [r for r in rows if (r.benchmark, r.task) in train_keys and not (r.label == "agent" and r.source == source)]
            test = [r for r in rows if (r.benchmark, r.task) in test_keys and (r.label == "human" or r.source == source)]
            record("unseen_model:" + source, train, test, seed, fold, diagnostics)

    summaries = []
    for cohort in sorted({r["cohort"] for r in metrics}):
        for method in METHODS:
            selected = [r for r in metrics if r["cohort"] == cohort and r["method"] == method]
            if not selected:
                continue
            averaged = averaged_tasks(selected)
            summaries.append({"cohort": cohort, "method": method, "metrics": aggregate(averaged),
                              "ci95_ba": task_bootstrap(averaged, "balanced_accuracy", draws=5000),
                              "task_groups": len(averaged), "versus_command_tfidf": paired_delta(metrics, method, cohort)})
    for name, records in (("task_metrics.csv", metrics), ("predictions.csv", predictions)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    manifest_paths = ["data/raw/cyber-task-horizons/download_manifest.json", "data/manifests/crossbenchmark_public_subset.json"]
    code_paths = ["attackagent/profiles.py", "scripts/run_behavior_profiles_v2.py", "scripts/run_feature_ablation.py",
                  "scripts/run_mixed_gallery.py", "attackagent/traces.py", "experiments/feature_profiles_v2/program.md"]
    result = {
        "status": "completed exploratory reuse; not an independent confirmation",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": {"records": 177, "task_groups": 40, "capture_confound": "human terminal input versus Agent tool calls"},
        "protocol": "experiments/feature_profiles_v2/program.md",
        "feature_names": FEATURE_NAMES, "feature_count": len(FEATURE_NAMES), "seeds": SPLIT_SEEDS,
        "weights_selected_in_inner_training_only": WEIGHTS,
        "summaries": summaries, "fold_audit": fold_audit,
        "provenance": {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in manifest_paths + code_paths},
                       "python": sys.version, "numpy": np.__version__},
        "elapsed_seconds": time.monotonic() - started,
        "default_public_detector_changed": False,
    }
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    by_id = collections.defaultdict(list)
    for r in predictions:
        if r["cohort"] == "full" and r["method"] == "inner_selected":
            by_id[r["trace_id"]].append(r["agent_like_score"])
    with (OUT / "profiles.jsonl").open("w", encoding="utf-8") as handle:
        for r in rows:
            evidence = TraceEvidence(r.trace_id, "cth-comparable-commands-v1", r.benchmark, r.names, r.groups)
            value = float(np.mean(by_id[r.trace_id]))
            score = ModeScore(r.trace_id, "agent" if value >= .5 else "human", value, False, True, None, len(r.names), evidence.source_format)
            profile = build_behavior_profile(evidence, score)
            profile["mode_estimate"]["provenance"] = "mean of six task-held-out scores; inner-selected experimental model; not a deployable saved detector"
            handle.write(json.dumps(profile, ensure_ascii=False, allow_nan=False) + "\n")
    (OUT / "feature_catalog.json").write_text(json.dumps([
        {"id": f"P{i:02d}", "name": name, "label_zh": label, "selection_reason": reason,
         "origin": origin, "status": "implemented_evaluated", "used_for_training": True}
        for i, (name, label, reason, origin) in enumerate(FEATURE_SPECS, 1)], ensure_ascii=False, indent=2) + "\n")
    for row in summaries:
        if row["cohort"] == "full":
            print(row["method"], row["metrics"], row["versus_command_tfidf"], flush=True)


if __name__ == "__main__":
    run()
