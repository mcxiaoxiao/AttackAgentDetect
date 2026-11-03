#!/usr/bin/env python3
"""Paired task-cluster summary of B/S increments after role-balanced training."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PRED = ROOT / "research/results/matched_fingerprint_v2/role_balanced/pair_predictions.csv.gz"
OUT = ROOT / "research/results/matched_fingerprint_v2/role_balanced/fingerprint_increment.json"
SEED, NBOOT = 20261103, 10_000


def json_threshold(value):
    value = float(value)
    return None if math.isinf(value) else value


def task_metric(df, tasks, fraction, stratum):
    x = df[df.fraction == fraction]
    if stratum != "ALL":
        x = x[x.pair_type == stratum]
    out = {}
    for task in tasks:
        t = x[x.query_task == task]
        pos, neg = t[t.same_task == 1], t[t.same_task == 0]
        if len(pos) and len(neg):
            fmr = float(neg.predicted_link.mean())
            fsr = float(1 - pos.predicted_link.mean())
            out[task] = np.array([fmr, fsr, 1 - (fmr + fsr) / 2])
    return out


def paired(full, base, tasks, stratum):
    fractions = sorted(full.fraction.unique())
    per_task = defaultdict(list)
    by_fraction = {}
    for fraction in fractions:
        a = task_metric(base, tasks, fraction, stratum)
        b = task_metric(full, tasks, fraction, stratum)
        common = sorted(set(a) & set(b))
        for task in common:
            per_task[task].append(b[task] - a[task])
        by_fraction[str(fraction)] = summarize({t: [b[t] - a[t]] for t in common}, [fraction])
    return summarize(per_task, fractions) | {"by_fraction": by_fraction}


def summarize(per_task, fractions):
    tasks = sorted(per_task)
    if not tasks:
        return {"tasks": 0, "fractions_averaged": fractions, "delta": None, "ci95": None}
    values = np.stack([np.mean(per_task[t], axis=0) for t in tasks])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(tasks), size=(NBOOT, len(tasks)))
    boots = values[idx].mean(axis=1)
    bounds = np.quantile(boots, [.025, .975], axis=0).T
    return {"tasks": len(tasks), "fractions_averaged": fractions,
            "delta": dict(zip(("fmr", "fsr", "ba"), map(float, values.mean(axis=0)))),
            "ci95": {k: list(map(float, bounds[i])) for i, k in enumerate(("fmr", "fsr", "ba"))}}


def main():
    df = pd.read_csv(PRED, compression="gzip")
    assert set(df.dataset) == {"cth"} and len(df) == 435_200
    tasks = sorted(df.query_task.unique())
    result = {"protocol": "same frozen 170x5 CTH query-candidate pairs; fold-local global thresholds; task macro",
              "task_count": len(tasks), "bootstrap_draws": NBOOT, "seed": SEED,
              "role_balanced_training": True, "threshold_policy": "inner OOF pooled FMR <= 10%",
              "methods": {}}
    for learner in ("svm", "tree", "xgb"):
        base = df[(df.learner == learner) & (df.variant == "evidence")]
        full = df[(df.learner == learner) & (df.variant == "BS")]
        if base.empty or full.empty:
            continue
        keys = ["fold", "pair_id"]
        assert set(map(tuple, base[keys].to_numpy())) == set(map(tuple, full[keys].to_numpy()))
        result["methods"][learner] = {
            "increment": {kind: paired(full, base, tasks, kind) for kind in ("ALL", "HH", "HA", "AA")},
            "pair_counts": {},
            "thresholds": [],
        }
        for fraction in sorted(base.fraction.unique()):
            for kind in ("ALL", "HH", "HA", "AA"):
                part = base[base.fraction == fraction]
                if kind != "ALL":
                    part = part[part.pair_type == kind]
                result["methods"][learner]["pair_counts"][f"{fraction}:{kind}"] = {
                    "records": int(len(part)), "positive": int(part.same_task.sum()),
                    "negative": int((1 - part.same_task).sum())}
        for fold in sorted(base.fold.unique()):
            result["methods"][learner]["thresholds"].append({
                "fold": int(fold),
                "evidence": json_threshold(base[base.fold == fold].threshold.iloc[0]),
                "fingerprint": json_threshold(full[full.fold == fold].threshold.iloc[0])})

    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for learner, data in result["methods"].items():
        print(learner)
        for kind, entry in data["increment"].items():
            if not entry["tasks"]:
                continue
            fmt = lambda key: (round(100*entry["delta"][key], 1),
                               [round(100*x, 1) for x in entry["ci95"][key]])
            print(" ", kind, "ΔFMR", fmt("fmr"), "ΔFSR", fmt("fsr"), "ΔBA", fmt("ba"))


if __name__ == "__main__":
    main()
