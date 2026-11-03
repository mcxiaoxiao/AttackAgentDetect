#!/usr/bin/env python3
"""Paired task-cluster analysis for the CTH training-pair coverage diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "research/results/matched_fingerprint_v2/pair_support/pair_predictions.csv.gz"
RUN = ROOT / "research/results/matched_fingerprint_v2/pair_support/run_manifest.json"
OUT = ROOT / "research/results/matched_fingerprint_v2/pair_support/analysis.json"
CONDITIONS = ("sampled_8_per_task", "all_pairs")
VERSIONS = ("E", "E_args", "E_BS", "E_args_BS")
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
STRATA = ("ALL", "HH", "HA", "AA")
NBOOT, SEED = 10_000, 20261105


def metric_cells(frame, tasks, stratum):
    cells = np.full((len(tasks), len(FRACTIONS), 3), np.nan, dtype=float)
    counts = {}
    for ti, task in enumerate(tasks):
        for fi, fraction in enumerate(FRACTIONS):
            x = frame[(frame.query_task == task) & (frame.fraction == fraction)]
            if stratum != "ALL":
                x = x[x.pair_type == stratum]
            pos, neg = x[x.same_task == 1], x[x.same_task == 0]
            counts[f"{task}|{fraction}|{stratum}"] = {
                "positive_pairs": int(len(pos)), "negative_pairs": int(len(neg)),
                "query_count": int(x.query_id.nunique())}
            if len(pos) and len(neg):
                fmr = float(neg.predicted_link.mean())
                fsr = float(1 - pos.predicted_link.mean())
                cells[ti, fi] = (fmr, fsr, 1 - (fmr + fsr) / 2)
    return cells, counts


def mean_ignore_nan(values, axis):
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.nansum(values, axis=axis)
    out = np.full(total.shape, np.nan, dtype=float)
    np.divide(total, count, out=out, where=count > 0)
    return out


def aggregate(cells, task_indexes=None, only_fraction=None):
    if task_indexes is None:
        task_indexes = np.arange(cells.shape[0])
    selected = cells[task_indexes]
    frac_indexes = range(len(FRACTIONS)) if only_fraction is None else [only_fraction]
    across_shares = []
    for fi in frac_indexes:
        column = selected[:, fi, :]
        if np.isfinite(column[:, 2]).any():
            across_shares.append(mean_ignore_nan(column, axis=0))
    if not across_shares:
        return np.full(3, np.nan)
    return mean_ignore_nan(np.stack(across_shares), axis=0)


def summarize(cells, indexes, fraction=None):
    point = aggregate(cells, only_fraction=fraction)
    if not np.isfinite(point).all():
        return None
    boots = np.full((len(indexes), 3), np.nan, dtype=float)
    fractions = range(len(FRACTIONS)) if fraction is None else [fraction]
    for start in range(0, len(indexes), 500):
        stop = min(start + 500, len(indexes))
        sample = cells[indexes[start:stop]]
        per_share = mean_ignore_nan(sample, axis=1)
        selected = per_share[:, list(fractions), :]
        boots[start:stop] = mean_ignore_nan(selected, axis=1)
    valid = np.isfinite(boots).all(axis=1)
    if not valid.any():
        return None
    ci = np.quantile(boots[valid], [0.025, 0.975], axis=0).T
    return {"metrics": dict(zip(("fmr", "fsr", "ba"), map(float, point))),
            "ci95": {name: list(map(float, ci[i]))
                     for i, name in enumerate(("fmr", "fsr", "ba"))},
            "task_fraction_cells": int(np.isfinite(cells[:, :, 2]).sum()),
            "tasks_with_any_cell": int(np.isfinite(cells[:, :, 2]).any(axis=1).sum())}


def summarize_delta(before, after, indexes, fraction=None):
    assert np.array_equal(np.isfinite(before), np.isfinite(after))
    return summarize(after - before, indexes, fraction)


def transition_summary(before, after):
    keys = ["fold", "pair_id", "replicate", "fraction", "query_task", "pair_type", "same_task"]
    merged = before[keys + ["predicted_link"]].merge(
        after[keys + ["predicted_link"]], on=keys, suffixes=("_before", "_after"),
        validate="one_to_one")
    output = {"pairs": int(len(merged)), "by_class": {}}
    for relation, label in (("different_task", 0), ("same_task", 1)):
        x = merged[merged.same_task == label]
        y = x.same_task.astype(int).to_numpy()
        b = x.predicted_link_before.astype(int).to_numpy()
        a = x.predicted_link_after.astype(int).to_numpy()
        output["by_class"][relation] = {
            "pairs": int(len(x)),
            "errors_corrected": int(((b != y) & (a == y)).sum()),
            "errors_introduced": int(((b == y) & (a != y)).sum())}
    return output


def main():
    frame = pd.read_csv(DATA, compression="gzip")
    run = json.loads(RUN.read_text())
    tasks = sorted(frame.query_task.unique())
    assert len(tasks) == 17
    assert set(frame.condition) == set(CONDITIONS)
    assert set(frame.version) == set(VERSIONS)
    assert set(frame.learner) == {"xgb"}
    assert len(frame) == run["prediction_rows"] == 435_200
    assert frame.groupby(["condition", "version"]).size().eq(54_400).all()

    # A condition change may alter scores and decisions, never the query/candidate set.
    keys = ["fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
            "query_task", "candidate_task", "pair_type", "same_task", "candidate_position"]
    reference = frame[(frame.condition == CONDITIONS[0]) & (frame.version == "E")][keys]
    for condition in CONDITIONS:
        for version in VERSIONS:
            current = frame[(frame.condition == condition) & (frame.version == version)][keys]
            pd.testing.assert_frame_equal(reference.reset_index(drop=True),
                                          current.reset_index(drop=True), check_dtype=False)

    indexes = np.random.default_rng(SEED).integers(0, len(tasks), size=(NBOOT, len(tasks)))
    cells, counts = {}, {}
    for condition in CONDITIONS:
        for version in VERSIONS:
            selected = frame[(frame.condition == condition) & (frame.version == version)]
            cells[(condition, version)] = {}
            for stratum in STRATA:
                cell, count = metric_cells(selected, tasks, stratum)
                cells[(condition, version)][stratum] = cell
                counts[f"{condition}:{version}:{stratum}"] = count

    results = {"protocol": "fixed CTH 17-task, five-share gallery; XGBoost parameters and representation frozen per outer fold; threshold recalibrated in inner task OOF; only training pair coverage changes",
               "scope": "development diagnostic; same-task is the CTF challenge proxy",
               "training_weighting": "equal total mass by label and equal mass among observed HH/HA/AA cells; each condition has the same total effective training weight within each fold",
               "tasks": len(tasks), "task_ids": tasks,
               "bootstrap_draws": NBOOT, "bootstrap_seed": SEED,
               "bootstrap_unit": "query task, paired across the two pair-coverage conditions and all five shares",
               "pair_keys_identical_across_conditions_and_versions": True,
               "counts": counts, "comparisons": {}, "fingerprint_increment": {},
               "error_transitions": {},
               "per_share": {}, "per_pair_type": {}}
    for version in VERSIONS:
        results["comparisons"][version] = {}
        for stratum in STRATA:
            a = cells[(CONDITIONS[0], version)][stratum]
            b = cells[(CONDITIONS[1], version)][stratum]
            results["comparisons"][version][stratum] = {
                "sampled_8_per_task": summarize(a, indexes),
                "all_pairs": summarize(b, indexes),
                "all_pairs_minus_sampled": summarize_delta(a, b, indexes),
                "by_share": {str(fraction): {
                    "sampled_8_per_task": summarize(a, indexes, fi),
                    "all_pairs": summarize(b, indexes, fi),
                    "all_pairs_minus_sampled": summarize_delta(a, b, indexes, fi)}
                    for fi, fraction in enumerate(FRACTIONS)}}
    for condition in CONDITIONS:
        results["fingerprint_increment"][condition] = {}
        for stratum in STRATA:
            results["fingerprint_increment"][condition][stratum] = {}
            for after, before in (("E_BS", "E"), ("E_args_BS", "E_args")):
                a = cells[(condition, after)][stratum]
                b = cells[(condition, before)][stratum]
                results["fingerprint_increment"][condition][stratum][f"{after}_minus_{before}"] = {
                    "all_shares": summarize_delta(b, a, indexes),
                    "by_share": {str(fraction): summarize_delta(b, a, indexes, fi)
                                 for fi, fraction in enumerate(FRACTIONS)}}
    for condition in CONDITIONS:
        results["per_share"][condition] = {}
        results["per_pair_type"][condition] = {}
        for fraction_index, fraction in enumerate(FRACTIONS):
            results["per_share"][condition][str(fraction)] = {}
            for version in VERSIONS:
                results["per_share"][condition][str(fraction)][version] = summarize(
                    cells[(condition, version)]["ALL"], indexes, fraction_index)
        for stratum in ("HH", "HA", "AA"):
            results["per_pair_type"][condition][stratum] = {}
            for fraction_index, fraction in enumerate(FRACTIONS):
                results["per_pair_type"][condition][stratum][str(fraction)] = {}
                for version in VERSIONS:
                    results["per_pair_type"][condition][stratum][str(fraction)][version] = summarize(
                        cells[(condition, version)][stratum], indexes, fraction_index)

    for condition in CONDITIONS:
        results["error_transitions"][condition] = {}
        model = frame[frame.condition == condition]
        for before, after in (("E", "E_BS"), ("E_args", "E_args_BS")):
            results["error_transitions"][condition][f"{after}_minus_{before}"] = transition_summary(
                model[model.version == before], model[model.version == after])
    for version in VERSIONS:
        results["error_transitions"][f"{version}:all_pairs_minus_sampled"] = transition_summary(
            frame[(frame.condition == CONDITIONS[0]) & (frame.version == version)],
            frame[(frame.condition == CONDITIONS[1]) & (frame.version == version)])

    results["fold_training_support"] = run["outer_folds"]
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("wrote", OUT)
    for version in VERSIONS:
        entry = results["comparisons"][version]["ALL"]["all_pairs_minus_sampled"]
        d, ci = entry["metrics"], entry["ci95"]
        print(version, "all pairs minus sampled ΔFMR/FSR/BA pp",
              *[round(100 * d[k], 2) for k in ("fmr", "fsr", "ba")],
              "BA CI pp", [round(100 * x, 2) for x in ci["ba"]])
    for condition in CONDITIONS:
        for after, before in (("E_BS", "E"), ("E_args_BS", "E_args")):
            entry = results["fingerprint_increment"][condition]["ALL"][
                f"{after}_minus_{before}"]["all_shares"]
            print(condition, after, "minus", before, "ΔFMR/FSR/BA pp",
                  *[round(100 * entry["metrics"][key], 2)
                    for key in ("fmr", "fsr", "ba")],
                  "BA CI pp", [round(100 * x, 2) for x in entry["ci95"]["ba"]])


if __name__ == "__main__":
    main()
