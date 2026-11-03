#!/usr/bin/env python3
"""Paired task-cluster comparison of ordinary and HH/HA/AA-balanced training."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "research/results/matched_fingerprint_v1/pair_predictions.csv.gz"
NEW = ROOT / "research/results/matched_fingerprint_v2/role_balanced/pair_predictions.csv.gz"
OUT = ROOT / "research/results/matched_fingerprint_v2/training_weighting_comparison.json"
SEED = 20261103
NBOOT = 10_000


def load(path):
    df = pd.read_csv(path, compression="gzip")
    df = df[df.dataset == "cth"].copy()
    df["error"] = np.where(df.same_task == 1, 1 - df.predicted_link, df.predicted_link)
    return df


def task_rates(df, tasks, fraction, pair_type):
    part = df[df.fraction == fraction]
    if pair_type != "ALL":
        part = part[part.pair_type == pair_type]
    out = {}
    for task in tasks:
        t = part[part.query_task == task]
        if not len(t):
            continue
        pos, neg = t[t.same_task == 1], t[t.same_task == 0]
        if not len(pos) or not len(neg):
            continue
        fmr = float(neg.predicted_link.mean())
        fsr = float(1 - pos.predicted_link.mean())
        out[task] = np.array([fmr, fsr, 1 - (fmr + fsr) / 2])
    return out


def per_task_average(df, tasks, pair_type):
    fractions = sorted(df.fraction.unique())
    vals = {t: [] for t in tasks}
    for fraction in fractions:
        for task, row in task_rates(df, tasks, fraction, pair_type).items():
            vals[task].append(row)
    return {t: np.mean(x, axis=0) for t, x in vals.items() if x}


def summarize_deltas(deltas, fractions):
    tasks = sorted(deltas)
    if not tasks:
        return {"tasks": 0, "delta": None, "ci95": None, "fractions_averaged": fractions}
    array = np.stack([np.mean(deltas[t], axis=0) for t in tasks])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(tasks), size=(NBOOT, len(tasks)))
    boots = array[idx].mean(axis=1)
    ci = np.quantile(boots, [.025, .975], axis=0).T
    return {
        "tasks": len(tasks),
        "delta": {k: float(v) for k, v in zip(("fmr", "fsr", "ba"), array.mean(axis=0))},
        "ci95": {k: [float(v[0]), float(v[1])] for k, v in zip(("fmr", "fsr", "ba"), ci)},
        "fractions_averaged": fractions,
    }


def paired_summary(old, new, tasks):
    result = {}
    for pair_type in ("ALL", "HH", "HA", "AA"):
        of = old if pair_type == "ALL" else old[old.pair_type == pair_type]
        nf = new if pair_type == "ALL" else new[new.pair_type == pair_type]
        fractions = sorted(set(of.fraction) & set(nf.fraction))
        by_task = defaultdict(list)
        by_fraction = {}
        for fraction in fractions:
            a = task_rates(old, tasks, fraction, pair_type)
            b = task_rates(new, tasks, fraction, pair_type)
            common = sorted(set(a) & set(b))
            for task in common:
                by_task[task].append(b[task] - a[task])
            by_fraction[str(fraction)] = summarize_deltas(
                {task: [b[task] - a[task]] for task in common}, [fraction])
        result[pair_type] = summarize_deltas(by_task, fractions)
        result[pair_type]["by_fraction"] = by_fraction
    return result


def error_transitions(old, new):
    keys = ["fold", "pair_id"]
    joined = old[keys + ["fraction", "pair_type", "same_task", "predicted_link"]].merge(
        new[keys + ["predicted_link"]], on=keys, suffixes=("_before", "_after"),
        validate="one_to_one")
    out = []
    for (fraction, pair_type, same), rows in joined.groupby(
            ["fraction", "pair_type", "same_task"], observed=True):
        y = rows.same_task.astype(int).to_numpy()
        before = rows.predicted_link_before.astype(int).to_numpy()
        after = rows.predicted_link_after.astype(int).to_numpy()
        out.append({"fraction": float(fraction), "pair_type": pair_type,
                    "relation": "same_task" if same else "different_task",
                    "records": len(rows), "old_error_corrected": int(((before != y) & (after == y)).sum()),
                    "new_error_introduced": int(((before == y) & (after != y)).sum())})
    return out


def main():
    old, new = load(OLD), load(NEW)
    tasks = sorted(set(old.query_task))
    assert len(tasks) == 17 == len(set(new.query_task))
    common_models = sorted(set(zip(old.learner, old.variant)) & set(zip(new.learner, new.variant)))
    checks = {}
    result = {"protocol": "same frozen CTH pairs; task macro; equal tier average; paired 10k task bootstrap",
              "seed": SEED, "bootstrap_draws": NBOOT, "task_count": len(tasks), "methods": {}}
    for learner, variant in common_models:
        a = old[(old.learner == learner) & (old.variant == variant)].copy()
        b = new[(new.learner == learner) & (new.variant == variant)].copy()
        ka = set(zip(a.fold, a.pair_id))
        kb = set(zip(b.fold, b.pair_id))
        assert ka == kb, (learner, variant, len(ka), len(kb))
        checks[f"{learner}:{variant}"] = {"same_pair_keys": True, "pairs_per_version": len(ka)}
        result["methods"][f"{learner}:{variant}"] = paired_summary(a, b, tasks)
        result["methods"][f"{learner}:{variant}"]["error_transitions"] = error_transitions(a, b)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for method, strata in result["methods"].items():
        x = strata["ALL"]
        print(method, "ΔBA", round(100*x["delta"]["ba"], 2),
              "CI", [round(100*v, 2) for v in x["ci95"]["ba"]],
              "ΔFMR", round(100*x["delta"]["fmr"], 2),
              "ΔFSR", round(100*x["delta"]["fsr"], 2))
    print("pair identities verified:", all(x["same_pair_keys"] for x in checks.values()))


if __name__ == "__main__":
    main()
