#!/usr/bin/env python3
"""Matched task-cluster analysis of argument recovery and B/S increments."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PRED = ROOT / "research/results/matched_fingerprint_v2/argument_profile/pair_predictions.csv.gz"
OUT = ROOT / "research/results/matched_fingerprint_v2/argument_profile/increment_analysis.json"
SEED, NBOOT = 20261103, 10_000


def threshold_json(value):
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


def summarize(per_task, fractions):
    tasks = sorted(per_task)
    if not tasks:
        return {"tasks": 0, "fractions_averaged": fractions, "delta": None, "ci95": None}
    vals = np.stack([np.mean(per_task[t], axis=0) for t in tasks])
    rng = np.random.default_rng(SEED)
    indexes = rng.integers(0, len(tasks), size=(NBOOT, len(tasks)))
    boots = vals[indexes].mean(axis=1)
    bounds = np.quantile(boots, [.025, .975], axis=0).T
    return {"tasks": len(tasks), "fractions_averaged": fractions,
            "delta": dict(zip(("fmr", "fsr", "ba"), map(float, vals.mean(axis=0)))),
            "ci95": {k: list(map(float, bounds[i])) for i, k in enumerate(("fmr", "fsr", "ba"))}}


def effect(after, before, tasks, stratum):
    fractions = sorted(after.fraction.unique())
    by_task = defaultdict(list)
    by_fraction = {}
    for fraction in fractions:
        a = task_metric(before, tasks, fraction, stratum)
        b = task_metric(after, tasks, fraction, stratum)
        common = sorted(set(a) & set(b))
        for task in common:
            by_task[task].append(b[task] - a[task])
        by_fraction[str(fraction)] = summarize(
            {task: [b[task] - a[task]] for task in common}, [fraction])
    return summarize(by_task, fractions) | {"by_fraction": by_fraction}


def transition_counts(after, before):
    cols = ["fold", "pair_id", "fraction", "pair_type", "same_task", "predicted_link"]
    x = before[cols].merge(after[cols[:5] + ["predicted_link"]], on=cols[:5],
                           suffixes=("_before", "_after"), validate="one_to_one")
    output = []
    for (fraction, pair_type, label), rows in x.groupby(
            ["fraction", "pair_type", "same_task"], observed=True):
        y = rows.same_task.astype(int).to_numpy()
        b = rows.predicted_link_before.astype(int).to_numpy()
        a = rows.predicted_link_after.astype(int).to_numpy()
        output.append({"fraction": float(fraction), "pair_type": pair_type,
                       "relation": "same_task" if label else "different_task",
                       "records": len(rows), "errors_corrected": int(((b != y) & (a == y)).sum()),
                       "errors_introduced": int(((b == y) & (a != y)).sum())})
    return output


def main():
    df = pd.read_csv(PRED, compression="gzip")
    assert set(df.dataset) == {"cth"} and len(df) == 598_400
    tasks = sorted(df.query_task.unique())
    result = {"protocol": "same frozen CTH pairs; 17-task cluster; equal average over five exact role shares",
              "task_count": len(tasks), "bootstrap_draws": NBOOT, "seed": SEED,
              "bootstrap_note": "query/candidate reuse means task-cluster intervals are approximate",
              "comparisons": {}}
    for learner in ("svm", "tree", "xgb"):
        versions = {v: df[(df.learner == learner) & (df.variant == v)].copy()
                    for v in ("head_only", "evidence", "BS")}
        if any(x.empty for x in versions.values()):
            continue
        keys = ["fold", "pair_id"]
        sets = {v: set(map(tuple, x[keys].to_numpy())) for v, x in versions.items()}
        assert len({len(x) for x in sets.values()}) == 1 and sets["head_only"] == sets["evidence"] == sets["BS"]
        entry = {"pair_keys_equal": True, "pairs_per_version": len(sets["BS"]),
                 "arguments_vs_heads": {}, "fingerprint_vs_arguments": {},
                 "error_transitions": {}}
        for stratum in ("ALL", "HH", "HA", "AA"):
            entry["arguments_vs_heads"][stratum] = effect(
                versions["evidence"], versions["head_only"], tasks, stratum)
            entry["fingerprint_vs_arguments"][stratum] = effect(
                versions["BS"], versions["evidence"], tasks, stratum)
        entry["error_transitions"]["arguments_vs_heads"] = transition_counts(
            versions["evidence"], versions["head_only"])
        entry["error_transitions"]["fingerprint_vs_arguments"] = transition_counts(
            versions["BS"], versions["evidence"])
        entry["thresholds"] = []
        for fold in sorted(versions["BS"].fold.unique()):
            entry["thresholds"].append({"fold": int(fold), **{
                v: threshold_json(versions[v][versions[v].fold == fold].threshold.iloc[0])
                for v in ("head_only", "evidence", "BS")}})
        result["comparisons"][learner] = entry

    result["pair_counts"] = {}
    for fraction in sorted(df.fraction.unique()):
        for stratum in ("ALL", "HH", "HA", "AA"):
            x = df[(df.learner == "xgb") & (df.variant == "evidence") & (df.fraction == fraction)]
            if stratum != "ALL":
                x = x[x.pair_type == stratum]
            result["pair_counts"][f"{fraction}:{stratum}"] = {
                "records": int(len(x)), "positive": int(x.same_task.sum()),
                "negative": int((1 - x.same_task).sum())}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for learner, item in result["comparisons"].items():
        print(learner)
        for name in ("arguments_vs_heads", "fingerprint_vs_arguments"):
            x = item[name]["ALL"]
            def fmt(k):
                return (round(100*x["delta"][k], 2), [round(100*v, 2) for v in x["ci95"][k]])
            print(" ", name, "ΔFMR", fmt("fmr"), "ΔFSR", fmt("fsr"), "ΔBA", fmt("ba"))


if __name__ == "__main__":
    main()
