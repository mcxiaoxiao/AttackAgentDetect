#!/usr/bin/env python3
"""Compare global and HH/HA/AA-specific thresholds on the same CTH pairs."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
GLOBAL = ROOT / "research/results/matched_fingerprint_v2/argument_profile/pair_predictions.csv.gz"
TYPE_DIR = ROOT / "research/results/matched_fingerprint_v2/type_threshold"
OUT = ROOT / "research/results/matched_fingerprint_v2/type_threshold/threshold_comparison.json"
NBOOT, SEED = 10_000, 20261103
TASKS = None
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_type_predictions():
    frames = []
    for fold in range(3):
        for learner in ("svm", "tree", "xgb"):
            for variant in ("evidence", "BS"):
                path = TYPE_DIR / f"fold-{fold}-{learner}-{variant}.csv.gz"
                frames.append(pd.read_csv(path, compression="gzip"))
    return pd.concat(frames, ignore_index=True)


def validate_pairs(before, after):
    keys = ["dataset", "fold", "pair_id", "replicate", "fraction", "query_id",
            "candidate_id", "query_task", "candidate_task", "query_mode",
            "candidate_mode", "pair_type", "same_task", "candidate_position"]
    assert len(before) == len(after)
    a = before[keys].sort_values(keys[:3]).reset_index(drop=True)
    b = after[keys].sort_values(keys[:3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def task_cells(df, tasks, stratum):
    """Return task x share x metric cells; missing task/share cells remain NaN."""
    result = np.full((len(tasks), len(FRACTIONS), 3), np.nan, dtype=float)
    for ti, task in enumerate(tasks):
        for fi, fraction in enumerate(FRACTIONS):
            x = df[(df.query_task == task) & (df.fraction == fraction)]
            if stratum != "ALL":
                x = x[x.pair_type == stratum]
            pos, neg = x[x.same_task == 1], x[x.same_task == 0]
            if len(pos) and len(neg):
                fsr = float(1 - pos.predicted_link.mean())
                fmr = float(neg.predicted_link.mean())
                result[ti, fi] = (fmr, fsr, 1 - (fmr + fsr) / 2)
    return result


def aggregate_cells(cells, task_indices=None):
    if task_indices is None:
        task_indices = np.arange(cells.shape[0])
    selected = cells[task_indices]
    per_fraction = []
    for fi in range(len(FRACTIONS)):
        col = selected[:, fi, :]
        if np.isfinite(col[:, 0]).any():
            per_fraction.append(np.nanmean(col, axis=0))
    if not per_fraction:
        return np.full(3, np.nan)
    return np.mean(np.stack(per_fraction), axis=0)


def summarize_cells(cells, rng):
    point = aggregate_cells(cells)
    boots = np.empty((NBOOT, 3), dtype=float)
    n = cells.shape[0]
    for i in range(NBOOT):
        indexes = rng.integers(0, n, size=n)
        boots[i] = aggregate_cells(cells, indexes)
    ci = np.nanquantile(boots, [0.025, 0.975], axis=0).T
    return {"metrics": dict(zip(("fmr", "fsr", "ba"), map(float, point))),
            "ci95": {k: list(map(float, ci[i]))
                     for i, k in enumerate(("fmr", "fsr", "ba"))},
            "task_share_cells": int(np.isfinite(cells[:, :, 0]).sum()),
            "tasks_with_any_cell": int(np.isfinite(cells[:, :, 0]).any(axis=1).sum()),
            "shares_with_cells": [FRACTIONS[i] for i in range(len(FRACTIONS))
                                  if np.isfinite(cells[:, i, 0]).any()]}


def paired_delta(before, after, rng):
    delta = after - before
    point = aggregate_cells(delta)
    boots = np.empty((NBOOT, 3), dtype=float)
    n = delta.shape[0]
    for i in range(NBOOT):
        indexes = rng.integers(0, n, size=n)
        boots[i] = aggregate_cells(delta, indexes)
    ci = np.nanquantile(boots, [0.025, 0.975], axis=0).T
    return {"delta": dict(zip(("fmr", "fsr", "ba"), map(float, point))),
            "ci95": {k: list(map(float, ci[i]))
                     for i, k in enumerate(("fmr", "fsr", "ba"))}}


def per_share(before, after, stratum):
    output = {}
    for fraction in FRACTIONS:
        metrics = {}
        for name, frame in (("global", before), ("type_specific", after)):
            x = frame[frame.fraction == fraction]
            if stratum != "ALL":
                x = x[x.pair_type == stratum]
            pos, neg = x[x.same_task == 1], x[x.same_task == 0]
            if len(pos) and len(neg):
                fmr = float(neg.predicted_link.mean())
                fsr = float(1 - pos.predicted_link.mean())
                metrics[name] = {"positive": int(len(pos)), "negative": int(len(neg)),
                                 "fmr": fmr, "fsr": fsr, "ba": 1 - (fmr + fsr) / 2}
        if metrics:
            output[str(fraction)] = metrics
    return output


def transitions(before, after):
    keys = ["fold", "pair_id", "fraction", "pair_type", "same_task"]
    x = before[keys + ["predicted_link"]].merge(
        after[keys + ["predicted_link"]], on=keys, suffixes=("_global", "_type"),
        validate="one_to_one")
    result = []
    for (fraction, pair_type, label), g in x.groupby(
            ["fraction", "pair_type", "same_task"], observed=True):
        y = g.same_task.astype(int).to_numpy()
        b = g.predicted_link_global.astype(int).to_numpy()
        a = g.predicted_link_type.astype(int).to_numpy()
        result.append({"fraction": float(fraction), "pair_type": pair_type,
                       "relation": "same_task" if label else "different_task",
                       "records": int(len(g)),
                       "errors_corrected": int(((b != y) & (a == y)).sum()),
                       "errors_introduced": int(((b == y) & (a != y)).sum())})
    return result


def main():
    global_df = pd.read_csv(GLOBAL, compression="gzip")
    typed_df = load_type_predictions()
    global_df = global_df[(global_df.dataset == "cth") &
                          global_df.learner.isin(("svm", "tree", "xgb")) &
                          global_df.variant.isin(("evidence", "BS"))].copy()
    tasks = sorted(global_df.query_task.unique())
    assert len(tasks) == 17 and len(global_df) == 170 * 5 * 16 * 4 * 2 * 3
    result = {"protocol": "same frozen CTH pairs, features, model parameters, and outer task folds; only threshold calibration changes",
              "threshold_rule": "inner OOF threshold maximizing recall subject to FMR <= 10%, fitted separately within each pair type",
              "caveat": "pair-type thresholds require Human/Agent role labels at decision time; this diagnostic tests threshold/composition drift and is not a role-free deployable method",
              "task_count": len(tasks), "bootstrap_draws": NBOOT, "seed": SEED,
              "bootstrap_unit": "query task; paired models and all five shares kept together",
              "comparisons": {}}
    rng = np.random.default_rng(SEED)
    for learner in ("svm", "tree", "xgb"):
        result["comparisons"][learner] = {}
        for variant in ("evidence", "BS"):
            before = global_df[(global_df.learner == learner) &
                               (global_df.variant == variant)].copy()
            after = typed_df[(typed_df.learner == learner) &
                             (typed_df.variant == variant)].copy()
            validate_pairs(before, after)
            entry = {"pair_count": int(len(before)), "pair_keys_equal": True,
                     "strata": {}}
            for stratum in ("ALL", "HH", "HA", "AA"):
                b = task_cells(before, tasks, stratum)
                a = task_cells(after, tasks, stratum)
                entry["strata"][stratum] = {
                    "global": summarize_cells(b, rng),
                    "type_specific": summarize_cells(a, rng),
                    "paired_change_type_minus_global": paired_delta(b, a, rng),
                    "by_share": per_share(before, after, stratum)}
            entry["error_transitions"] = transitions(before, after)
            result["comparisons"][learner][variant] = entry
    result["pair_counts_by_share_and_type"] = {}
    for fraction in FRACTIONS:
        x = global_df[(global_df.learner == "xgb") & (global_df.variant == "BS") &
                      (global_df.fraction == fraction)]
        for stratum in ("ALL", "HH", "HA", "AA"):
            group = x if stratum == "ALL" else x[x.pair_type == stratum]
            result["pair_counts_by_share_and_type"][f"{fraction}:{stratum}"] = {
                "pairs": int(len(group)), "positive": int(group.same_task.sum()),
                "negative": int((1 - group.same_task).sum()),
                "query_tasks": int(group.query_task.nunique())}
    audit = json.loads((TYPE_DIR / "threshold_audit.json").read_text())
    result["validation_threshold_audit"] = audit["folds"]
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("wrote", OUT)
    for learner in ("svm", "tree", "xgb"):
        for variant in ("evidence", "BS"):
            x = result["comparisons"][learner][variant]["strata"]
            delta = x["ALL"]["paired_change_type_minus_global"]["delta"]
            ci = x["ALL"]["paired_change_type_minus_global"]["ci95"]
            print(learner, variant,
                  "ΔFMR/FSR/BA pp", *[round(delta[k] * 100, 2) for k in ("fmr", "fsr", "ba")],
                  "BA CI pp", [round(v * 100, 2) for v in ci["ba"]])


if __name__ == "__main__":
    main()
