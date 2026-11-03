#!/usr/bin/env python3
"""Summarize frozen predictions by task; no training or candidate selection."""
import collections
import csv
import json
import pathlib
import warnings

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "research/results/matched_fingerprint_v2/role_balanced"
SEED = 20261103
NBOOT = 10000
warnings.filterwarnings("ignore", message="Mean of empty slice")


def safe(x):
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return safe(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def ci(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return None
    return [float(v) for v in np.quantile(vals, [0.025, 0.975])]


def task_values(df, tasks):
    if df.empty:
        return np.full((len(tasks), 3), np.nan)
    rates = df.groupby(["query_task", "same_task"], observed=True)["error"].mean().unstack()
    rates = rates.reindex(tasks)
    fmr = rates[0].to_numpy() if 0 in rates else np.full(len(tasks), np.nan)
    fsr = rates[1].to_numpy() if 1 in rates else np.full(len(tasks), np.nan)
    return np.column_stack([fmr, fsr, 1 - (fmr + fsr) / 2])


def mean_available(v, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(v, axis=axis)


def metrics(df, tasks, idx):
    v = task_values(df, tasks)
    points = mean_available(v, axis=0)
    boots = mean_available(v[idx], axis=1)
    result = {}
    for j, name in enumerate(("fmr", "fsr", "ba")):
        b = boots[:, j]
        result[name] = {"point": points[j], "ci95": ci(b),
                        "valid_tasks": int(np.isfinite(v[:, j]).sum()),
                        "valid_bootstraps": int(np.isfinite(b).sum())}
    return result, v, boots


def aggregate_fractions(df, tasks, fractions, idx):
    values = np.stack([task_values(df[df.fraction == p], tasks) for p in fractions], axis=1)
    # Five fractions have the same tasks under the frozen 17-task protocol.
    assert np.isfinite(values).all()
    v = values.mean(axis=1)
    boots = v[idx].mean(axis=1)
    result = {name: {"point": float(v[:, j].mean()), "ci95": ci(boots[:, j]),
                     "valid_tasks": len(tasks), "valid_bootstraps": NBOOT}
              for j, name in enumerate(("fmr", "fsr", "ba"))}
    return result, v, boots


def paired(current, reference):
    out = {}
    for metric in ("fmr", "fsr", "ba"):
        j = {"fmr": 0, "fsr": 1, "ba": 2}[metric]
        delta = current[1][:, j] - reference[1][:, j]
        boots = current[2][:, j] - reference[2][:, j]
        out[metric] = {"point": float(mean_available(delta, axis=0)),
                       "ci95": ci(boots),
                       "paired_tasks": int(np.isfinite(delta).sum()),
                       "valid_bootstraps": int(np.isfinite(boots).sum())}
    return out


def load_dataset(dataset):
    folders = OUT / dataset
    files = sorted(folders.glob("fold-*-*-*.csv.gz"))
    expected = 24 if dataset == "cth" else 15
    assert len(files) == expected, (dataset, len(files), files)
    frames = []
    cols = ["fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
            "query_task", "candidate_task", "query_mode", "candidate_mode", "pair_type",
            "same_task", "learner", "variant", "predicted_link", "fallback"]
    for p in files:
        df = pd.read_csv(p, usecols=cols)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["error"] = np.where(df.same_task == 1, 1 - df.predicted_link, df.predicted_link)
    return df


def summarize(dataset):
    df = load_dataset(dataset)
    tasks = sorted(df.query_task.unique())
    idx = np.random.default_rng(SEED).integers(0, len(tasks), size=(NBOOT, len(tasks)))
    fractions = sorted(df.fraction.unique())
    expected = 17 if dataset == "cth" else 12
    assert len(tasks) == expected, (dataset, len(tasks))
    methods = sorted(set(zip(df.learner, df.variant)))
    result = {"dataset": dataset, "tasks": tasks, "bootstrap_draws": NBOOT,
              "fractions": fractions, "methods": []}
    raw = {}
    for kind, variant in methods:
        source = df[(df.learner == kind) & (df.variant == variant)]
        key = f"{kind}:{variant}"
        raw[key] = {}
        for fraction in fractions:
            fracdf = source[source.fraction == fraction]
            for stratum in ("ALL", "HH", "HA", "AA"):
                selected = fracdf if stratum == "ALL" else fracdf[fracdf.pair_type == stratum]
                computed = metrics(selected, tasks, idx)
                unique = len(selected[["query_id", "candidate_id"]].drop_duplicates())
                meta = {"records": len(selected),
                        "positive_records": int((selected.same_task == 1).sum()),
                        "negative_records": int((selected.same_task == 0).sum()),
                        "unique_ordered_trace_pairs": unique,
                        "fallback_records": int(selected.fallback.sum())}
                raw[key][(str(fraction), stratum)] = (computed, meta)
        if dataset == "cth":
            raw[key][("average_5_fractions", "ALL")] = (
                aggregate_fractions(source, tasks, fractions, idx),
                {"records": len(source), "positive_records": int((source.same_task == 1).sum()),
                 "negative_records": int((source.same_task == 0).sum()),
                 "unique_ordered_trace_pairs": len(source[["query_id", "candidate_id"]].drop_duplicates()),
                 "fallback_records": int(source.fallback.sum())})
    for key, entries in raw.items():
        learner, variant = key.split(":")
        for (fraction, stratum), (computed, meta) in entries.items():
            refkey = f"{learner}:evidence"
            fullkey = "xgb:BS" if dataset == "cth" else "xgb:BST"
            row = {"learner": learner, "variant": variant, "fraction": fraction,
                   "pair_type": stratum, **meta, "metrics": computed[0],
                   "vs_evidence": paired(computed, raw[refkey][(fraction, stratum)][0]),
                   "vs_xgb_full": paired(computed, raw[fullkey][(fraction, stratum)][0])
                   if learner == "xgb" else None}
            result["methods"].append(row)
    return df, safe(result)


def transitions(df, dataset):
    out = []
    for learner in sorted(df.learner.unique()):
        baseline = df[(df.learner == learner) & (df.variant == "evidence")]
        variant = "BS" if dataset == "cth" else "BST"
        full = df[(df.learner == learner) & (df.variant == variant)]
        keys = ["fold", "pair_id"]
        assert not baseline.duplicated(keys).any() and not full.duplicated(keys).any()
        b = baseline[keys + ["fraction", "pair_type", "same_task", "predicted_link"]]
        f = full[keys + ["predicted_link"]]
        joined = b.merge(f, on=keys, suffixes=("_before", "_after"), validate="one_to_one")
        assert len(joined) == len(baseline) == len(full)
        for frac in sorted(joined.fraction.unique()):
            part = joined[joined.fraction == frac]
            for typ in ("ALL", "HH", "HA", "AA"):
                group = part if typ == "ALL" else part[part.pair_type == typ]
                for label, name in ((1, "same_task"), (0, "different_task")):
                    x = group[group.same_task == label]
                    before_correct = x.predicted_link_before == label
                    after_correct = x.predicted_link_after == label
                    out.append({"dataset": dataset, "learner": learner, "fraction": frac,
                                "pair_type": typ, "relation": name, "records": len(x),
                                "corrected": int((~before_correct & after_correct).sum()),
                                "new_errors": int((before_correct & ~after_correct).sum()),
                                "correct_both": int((before_correct & after_correct).sum()),
                                "wrong_both": int((~before_correct & ~after_correct).sum())})
    return out


def leave_one_task(df, dataset, tasks):
    fullvariant = "BS" if dataset == "cth" else "BST"
    x = df[(df.learner == "xgb") & df.variant.isin(["evidence", fullvariant])]
    results = []
    fractions = sorted(x.fraction.unique())
    for removed in tasks:
        reduced = x[(x.query_task != removed) & (x.candidate_task != removed)]
        per = {}
        for variant in ("evidence", fullvariant):
            values = []
            for frac in fractions:
                sel = reduced[(reduced.variant == variant) & (reduced.fraction == frac)]
                v = task_values(sel, [t for t in tasks if t != removed])
                means = mean_available(v, axis=0)
                assert np.isfinite(means).all()
                values.append(means)
            per[variant] = np.asarray(values).mean(axis=0)
        results.append({"removed_task": removed, "fractions": len(fractions),
                        "evidence_fmr": per["evidence"][0], "evidence_fsr": per["evidence"][1],
                        "evidence_ba": per["evidence"][2], "full_fmr": per[fullvariant][0],
                        "full_fsr": per[fullvariant][1], "full_ba": per[fullvariant][2],
                        "delta_ba": per[fullvariant][2] - per["evidence"][2]})
    return safe(results)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    allresults = {}
    transition_rows = []
    for dataset in ("cth",):
        df, result = summarize(dataset)
        result["leave_one_task_out"] = leave_one_task(df, dataset, result["tasks"])
        transition_rows.extend(transitions(df, dataset))
        allresults[dataset] = result
        print(dataset, "records", len(df), "methods", len(result["methods"]), flush=True)
    (OUT / "metrics.json").write_text(json.dumps(safe(allresults), indent=2, ensure_ascii=False,
                                                allow_nan=False) + "\n")
    with (OUT / "error_transitions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(transition_rows[0]))
        w.writeheader()
        w.writerows(transition_rows)
    rows = []
    for dataset, result in allresults.items():
        for item in result["methods"]:
            row = {k: item[k] for k in ("learner", "variant", "fraction", "pair_type",
                                        "records", "positive_records", "negative_records",
                                        "unique_ordered_trace_pairs", "fallback_records")}
            row["dataset"] = dataset
            for metric in ("fmr", "fsr", "ba"):
                v = item["metrics"][metric]
                row[metric] = v["point"]
                row[metric + "_ci_low"] = v["ci95"][0] if v["ci95"] else None
                row[metric + "_ci_high"] = v["ci95"][1] if v["ci95"] else None
                row[metric + "_task_count"] = v["valid_tasks"]
                for comp in ("vs_evidence", "vs_xgb_full"):
                    d = item[comp][metric] if item[comp] else None
                    row[f"{metric}_{comp}_delta"] = d["point"] if d else None
                    row[f"{metric}_{comp}_low"] = d["ci95"][0] if d and d["ci95"] else None
                    row[f"{metric}_{comp}_high"] = d["ci95"][1] if d and d["ci95"] else None
            rows.append(row)
    with (OUT / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
