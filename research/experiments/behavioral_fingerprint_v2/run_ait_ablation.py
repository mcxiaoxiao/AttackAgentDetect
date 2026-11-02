#!/usr/bin/env python3
"""Ablation controls for sequence order, observed gap order, and fixed prefixes."""
from __future__ import annotations

import collections
import importlib.util
import json
import pathlib
import random
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from attackagent.fingerprint_v2 import combine_summary, fingerprint_matrix, summarize, terminal_action

RUNNER = pathlib.Path(__file__).with_name("run_ait_recognition.py")
OUT = ROOT / "research/results/behavioral_fingerprint_v2/ait_ablation"
SEED = 20261103
PERMUTATIONS = 20


def column_nanmean(values):
    """Average each feature over perturbations without noisy all-NaN warnings."""
    stack = np.vstack(values)
    valid = np.isfinite(stack)
    counts = valid.sum(axis=0)
    sums = np.where(valid, stack, 0.0).sum(axis=0)
    return np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0)


def load_runner():
    spec = importlib.util.spec_from_file_location("ait_recognition_runner_v2", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def window_order_feature(row, seed):
    symbols = row["commands"][:128]
    actions = [terminal_action(x) for x in symbols]
    chunks = [symbols[i:i + 8] for i in range(0, len(symbols), 8)]
    if chunks and len(chunks[-1]) < 4:
        tail = chunks.pop()
    else:
        tail = []
    full_summaries = []
    for draw in range(PERMUTATIONS):
        permuted = [chunk[:] for chunk in chunks]
        random.Random(seed + draw * 7919).shuffle(permuted)
        seq = [x for chunk in permuted + ([tail] if tail else []) for x in chunk]
        permuted_actions = [terminal_action(x) for x in seq]
        blocks, _ = fingerprint_matrix(seq, actions=permuted_actions, window=8,
                                       min_window_events=4, max_events=128)
        if blocks is not None:
            full_summaries.append(combine_summary(blocks, ("B", "S")))
    return column_nanmean(full_summaries)


def time_gap_order_feature(row, seed):
    symbols = row["commands"][:128]
    actions = [terminal_action(x) for x in symbols]
    source_times = row["timestamps"][:len(symbols)]
    variants = []
    audit = {"eligible_windows": 0, "preserved_windows": 0}
    for draw in range(PERMUTATIONS):
        synthetic = [None] * len(symbols)
        rng = np.random.default_rng(seed + draw * 104729)
        for start in range(0, len(symbols), 8):
            end = min(start + 8, len(symbols))
            chunk = source_times[start:end]
            if len(chunk) < 4 or any(x is None for x in chunk):
                synthetic[start:end] = chunk
                audit["preserved_windows"] += 1
                continue
            times = np.asarray(chunk, dtype=float)
            gaps = np.diff(times)
            if not np.isfinite(gaps).all() or np.any(gaps < 0):
                synthetic[start:end] = chunk
                audit["preserved_windows"] += 1
                continue
            rng.shuffle(gaps)
            rebuilt = [0.0]
            for gap in gaps:
                rebuilt.append(rebuilt[-1] + float(gap))
            synthetic[start:end] = rebuilt
            audit["eligible_windows"] += 1
        blocks, _ = fingerprint_matrix(symbols, actions=actions, timestamps=synthetic,
                                       window=8, min_window_events=4, max_events=128)
        if blocks is not None:
            variants.append(np.concatenate([summarize(blocks["B"]),
                                            summarize(blocks["S"]),
                                            summarize(blocks["T"])]))
    if not variants:
        return None, audit
    return column_nanmean(variants), audit


def add_ablation_features(rows):
    time_audit = collections.defaultdict(lambda: {"eligible_windows": 0, "preserved_windows": 0})
    for i, row in enumerate(rows):
        row["first_window_BS"] = np.concatenate([
            summarize(row["blocks"]["B"][:1]), summarize(row["blocks"]["S"][:1])
        ])
        row["B_plus_S_window_order_shuffled"] = window_order_feature(
            row, SEED + i * 1009
        )
        row["B_plus_S_plus_T_gap_order_shuffled"], audit = time_gap_order_feature(
            row, SEED + i * 1013
        )
        for k, v in audit.items():
            time_audit[row["source"]][k] += v
    return {key: dict(value) for key, value in time_audit.items()}


def ci(values, seed, draws=5000):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sims = arr[rng.integers(0, len(arr), (draws, len(arr)))].mean(axis=1)
    return [float(x) for x in np.quantile(sims, [.025, .975])]


def task_ba(predictions):
    grouped = collections.defaultdict(list)
    for r in predictions:
        grouped[r["task"]].append(r)
    output = {}
    for task, items in grouped.items():
        y = [r["label"] for r in items]
        if len(set(y)) == 2:
            output[task] = balanced_accuracy_score(y, [r["score"] >= .5 for r in items])
    return output


def main():
    runner = load_runner()
    pilot = runner.load_module(runner.PILOT_PATH, "ait_pilot_ablation_v2")
    raw, extended = runner.load_rows(pilot)
    frozen = runner.cohort_ids_v1()
    by_id = {r["id"]: r for r in extended}
    cohorts = {
        "frozen_v1_89": [by_id[x] for x in sorted(frozen)],
        "extended_all_eligible_min4": extended,
    }
    methods = (
        "B_plus_S",
        "B_plus_S_window_order_shuffled",
        "first_window_BS",
        "B_plus_S_plus_T",
        "B_plus_S_plus_T_gap_order_shuffled",
    )
    all_metrics, all_predictions, all_folds, all_deltas = [], [], [], []
    audit_out = {"raw_runs": len(raw), "permutations_per_trace": PERMUTATIONS}
    for dataset, rows in cohorts.items():
        time_audit = add_ablation_features(rows)
        audit_out[dataset] = {"n": len(rows), "tasks": len({r["task"] for r in rows}),
                              "gap_shuffle_window_audit": time_audit}
        folds = runner.source_folds(rows)
        for panel in ("task_holdout", "source_holdout"):
            pred_by_method = collections.defaultdict(list)
            for fold, train, test in runner.partitions(rows, panel, folds):
                fold_available = len({rows[i]["y"] for i in train}) == 2
                all_folds.append({"dataset": dataset, "panel": panel, "fold": fold,
                                  "train_n": len(train), "test_n": len(test),
                                  "available": fold_available})
                if not fold_available:
                    continue
                for method in methods:
                    if method == "B_plus_S_plus_T_gap_order_shuffled":
                        eligible_test = [i for i in test if rows[i][method] is not None]
                        if not eligible_test:
                            continue
                        test_use = eligible_test
                    else:
                        test_use = test
                    scores = runner.predict_fold(rows, train, test_use, method)
                    if scores is None:
                        continue
                    for i, score in zip(test_use, scores):
                        pred_by_method[method].append({
                            "dataset": dataset, "panel": panel, "method": method,
                            "run_id": rows[i]["id"], "task": rows[i]["task"],
                            "source_group": rows[i]["source"], "label": rows[i]["y"],
                            "score": float(score), "prediction": int(score >= .5)
                        })
            for method in methods:
                preds = pred_by_method[method]
                if not preds:
                    continue
                summary = runner.summarize_predictions(rows, preds, dataset, panel, method)
                all_metrics.append(summary)
                all_predictions.extend(preds)
                base = task_ba(pred_by_method["B_plus_S"])
                current = task_ba(preds)
                shared = sorted(set(base) & set(current))
                if shared:
                    deltas = [current[t] - base[t] for t in shared]
                    all_deltas.append({
                        "dataset": dataset, "panel": panel, "method": method,
                        "baseline": "B_plus_S", "task_count": len(shared),
                        "delta_task_macro_ba": float(np.mean(deltas)),
                        "task_cluster_95ci": ci(deltas, SEED + len(dataset) + len(panel) + len(method))
                    })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps({
        "protocol": "AIT grouped holdout; window and time-order perturbation controls",
        "metrics": all_metrics, "paired_deltas_vs_B_plus_S": all_deltas,
        "uncertainty": "paired bootstrap over task-level balanced accuracy differences",
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (OUT / "provenance.json").write_text(json.dumps({
        "seed": SEED, "permutations_per_trace": PERMUTATIONS,
        "source_archive_sha256": runner.sha256(runner.PILOT_PATH.parent / "data.zip")
            if (runner.PILOT_PATH.parent / "data.zip").exists() else None,
        "controls": {
            "first_window_BS": "only first valid eight-event window; tests whether full length is required",
            "B_plus_S_window_order_shuffled": "randomly permuted eight-event chunks; B window contents are unchanged, S is recomputed",
            "B_plus_S_plus_T_gap_order_shuffled": "permutes observed gaps within each eight-event window while preserving that window's gap multiset",
        },
        "claim_boundary": "AIT troubleshooting labels and capture workflows are confounded; controls diagnose this cohort only",
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (OUT / "folds.json").write_text(json.dumps(all_folds, ensure_ascii=False, indent=2) + "\n")
    with (OUT / "predictions.csv").open("w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(all_predictions[0]))
        writer.writeheader()
        writer.writerows(all_predictions)
    (OUT / "time-gap-audit.json").write_text(json.dumps(audit_out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"metrics": all_metrics, "paired_deltas": all_deltas,
                      "time_gap_audit": audit_out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
