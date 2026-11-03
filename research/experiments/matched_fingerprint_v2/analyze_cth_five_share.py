#!/usr/bin/env python3
"""Normalize five-share CTH results from the frozen matched predictions."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "research/results/matched_fingerprint_v1/pair_predictions.csv.gz"
OUT = ROOT / "research/results/matched_fingerprint_v2/five_share"
LEARNERS = ("svm", "tree", "xgb")
VERSIONS = ("evidence", "BS")
SHARES = ("0.0", "0.25", "0.5", "0.75", "1.0")
PAIR_TYPES = ("ALL", "HH", "HA", "AA")
DRAWS, SEED = 10_000, 20261104


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_metric_matrix(frame: pd.DataFrame, tasks: list[str]) -> tuple[np.ndarray, dict]:
    values, counts = [], {}
    for task in tasks:
        x = frame[frame.query_task == task]
        pos, neg = x[x.same_task == 1], x[x.same_task == 0]
        if pos.empty or neg.empty:
            raise AssertionError(f"task/type cell lacks one label: {task}")
        fmr = float(neg.predicted_link.mean())
        fsr = float(1 - pos.predicted_link.mean())
        values.append((fmr, fsr, 1 - (fmr + fsr) / 2))
        counts[task] = {"positive_pair_decisions": len(pos),
                        "negative_pair_decisions": len(neg)}
    return np.asarray(values, dtype=float), counts


def summarize(matrix: np.ndarray, indexes: np.ndarray) -> dict:
    names = ("fmr", "fsr", "ba")
    point = matrix.mean(axis=0)
    ci = np.quantile(matrix[indexes].mean(axis=1), [0.025, 0.975], axis=0).T
    return {"metrics": dict(zip(names, map(float, point))),
            "ci95": {name: list(map(float, ci[i])) for i, name in enumerate(names)}}


def filtered(frame, share, pair_type):
    out = frame[frame.fraction == share]
    return out if pair_type == "ALL" else out[out.pair_type == pair_type]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT, compression="gzip")
    frame = frame[(frame.dataset == "cth") & frame.learner.isin(LEARNERS)
                  & frame.variant.isin(VERSIONS)].copy()
    frame["fraction"] = frame.fraction.astype(str)
    frame["same_task"] = frame.same_task.astype(int)
    frame["predicted_link"] = frame.predicted_link.astype(int)
    pair_key = ["fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
                "query_task", "candidate_task", "pair_type", "same_task", "candidate_position"]
    reference = frame[(frame.learner == "xgb") & (frame.variant == "evidence")][pair_key]
    if len(reference) != 54_400:
        raise AssertionError(f"unexpected test pair count: {len(reference)}")
    for learner in LEARNERS:
        for version in VERSIONS:
            actual = frame[(frame.learner == learner) & (frame.variant == version)][pair_key]
            pd.testing.assert_frame_equal(reference.reset_index(drop=True),
                                          actual.reset_index(drop=True), check_dtype=False)

    tasks = sorted(frame.query_task.unique())
    if len(tasks) != 17:
        raise AssertionError(f"expected 17 CTH task groups, got {len(tasks)}")
    rng = np.random.default_rng(SEED)
    indexes = rng.integers(0, len(tasks), size=(DRAWS, len(tasks)))
    rows, result = [], {"protocol": "matched CTH five-share evaluation",
                        "task_count": len(tasks), "task_ids": tasks,
                        "pair_decisions_per_learner_version": len(reference),
                        "shares_equal_weight": list(map(float, SHARES)),
                        "bootstrap": {"unit": "query task", "draws": DRAWS, "seed": SEED},
                        "pair_lists_identical_across_learners_and_versions": True,
                        "metrics_by_share_and_pair_type": {}, "five_share_equal_weight": {},
                        "pair_type_equal_weight_over_available_shares": {},
                        "error_transitions": {},
                        "input_sha256": sha(INPUT),
                        "analysis_script_sha256": sha(Path(__file__))}

    metric_matrices = {}
    for learner in LEARNERS:
        result["metrics_by_share_and_pair_type"][learner] = {}
        result["five_share_equal_weight"][learner] = {}
        metric_matrices[learner] = {}
        learner_frame = frame[frame.learner == learner]
        for pair_type in PAIR_TYPES:
            result["metrics_by_share_and_pair_type"][learner][pair_type] = {}
            per_version_share = {version: [] for version in VERSIONS}
            for share in SHARES:
                cells = {}
                for version in VERSIONS:
                    cell = filtered(learner_frame[learner_frame.variant == version],
                                    share, pair_type)
                    label_counts = cell.same_task.value_counts().to_dict()
                    if not label_counts.get(1) or not label_counts.get(0):
                        continue
                    valid_tasks = sorted(cell.query_task.unique())
                    matrix, counts = task_metric_matrix(cell, valid_tasks)
                    cell_indexes = rng.integers(0, len(valid_tasks),
                                                size=(DRAWS, len(valid_tasks)))
                    cells[version] = {"matrix": matrix, "tasks": valid_tasks,
                                      "counts": counts, "summary": summarize(matrix,
                                                                                 cell_indexes)}
                if set(cells) == set(VERSIONS):
                    if cells["evidence"]["tasks"] != cells["BS"]["tasks"]:
                        raise AssertionError("paired versions have different task support")
                    delta = cells["BS"]["matrix"] - cells["evidence"]["matrix"]
                    delta_indexes = rng.integers(0, len(cells["BS"]["tasks"]),
                                                 size=(DRAWS, len(cells["BS"]["tasks"])))
                    cells["delta_BS_minus_E"] = summarize(delta, delta_indexes)
                    for version in VERSIONS:
                        per_version_share[version].append(cells[version]["matrix"])
                    for version in VERSIONS:
                        src = cells[version]
                        rows.append({"learner": learner, "share": float(share),
                                     "pair_type": pair_type, "version": version,
                                     "valid_tasks": len(src["tasks"]),
                                     "positive_pair_decisions": sum(
                                         x["positive_pair_decisions"] for x in src["counts"].values()),
                                     "negative_pair_decisions": sum(
                                         x["negative_pair_decisions"] for x in src["counts"].values()),
                                     **{f"{k}_point": val for k, val in
                                        src["summary"]["metrics"].items()},
                                     **{f"{k}_ci_low": val[0] for k, val in
                                        src["summary"]["ci95"].items()},
                                     **{f"{k}_ci_high": val[1] for k, val in
                                        src["summary"]["ci95"].items()},
                                     **{f"delta_{k}_point": val for k, val in
                                        cells["delta_BS_minus_E"]["metrics"].items()},
                                     **{f"delta_{k}_ci_low": val[0] for k, val in
                                        cells["delta_BS_minus_E"]["ci95"].items()},
                                     **{f"delta_{k}_ci_high": val[1] for k, val in
                                        cells["delta_BS_minus_E"]["ci95"].items()}})
                result["metrics_by_share_and_pair_type"][learner][pair_type][share] = {
                    "counts": {version: {"positive_pair_decisions": int(
                        (filtered(learner_frame[learner_frame.variant == version],
                                  share, pair_type).same_task == 1).sum()),
                        "negative_pair_decisions": int(
                        (filtered(learner_frame[learner_frame.variant == version],
                                  share, pair_type).same_task == 0).sum())}
                        for version in VERSIONS}}
                if "delta_BS_minus_E" in cells:
                    result["metrics_by_share_and_pair_type"][learner][pair_type][share].update({
                        version: cells[version]["summary"] for version in VERSIONS})
                    result["metrics_by_share_and_pair_type"][learner][pair_type][share][
                        "delta_BS_minus_E"] = cells["delta_BS_minus_E"]

            # The overall five-share summary is the equal mean of share-level metrics per task.
            if pair_type == "ALL":
                for version in VERSIONS:
                    share_matrices = per_version_share[version]
                    if len(share_matrices) != len(SHARES):
                        raise AssertionError(f"missing share for {learner}/{version}")
                    five_matrix = np.stack(share_matrices, axis=1).mean(axis=1)
                    metric_matrices[learner][version] = five_matrix
                    result["five_share_equal_weight"][learner][version] = summarize(
                        five_matrix, indexes)
                delta = metric_matrices[learner]["BS"] - metric_matrices[learner]["evidence"]
                result["five_share_equal_weight"][learner]["delta_BS_minus_E"] = summarize(
                    delta, indexes)

        # Pair-type summary excludes concentrations where that type cannot exist.
        result["pair_type_equal_weight_over_available_shares"][learner] = {}
        for pair_type in ("HH", "HA", "AA"):
            share_cells = result["metrics_by_share_and_pair_type"][learner][pair_type]
            valid_shares = [share for share in SHARES
                            if "delta_BS_minus_E" in share_cells[share]]
            if not valid_shares:
                continue
            task_sets = []
            task_matrices = {version: [] for version in VERSIONS}
            pair_counts = {version: {"positive_pair_decisions": 0,
                                     "negative_pair_decisions": 0} for version in VERSIONS}
            for share in valid_shares:
                version_frames = {}
                for version in VERSIONS:
                    cell = filtered(learner_frame[learner_frame.variant == version],
                                    share, pair_type)
                    support = sorted(cell.query_task.unique())
                    version_frames[version] = (cell, support)
                common = sorted(set(version_frames["evidence"][1]) &
                                set(version_frames["BS"][1]))
                task_sets.append(common)
                for version in VERSIONS:
                    cell, support = version_frames[version]
                    if support != common:
                        raise AssertionError("pair-type task support differs between variants")
                    matrix, counts = task_metric_matrix(cell, common)
                    task_matrices[version].append(matrix)
                    pair_counts[version]["positive_pair_decisions"] += sum(
                        x["positive_pair_decisions"] for x in counts.values())
                    pair_counts[version]["negative_pair_decisions"] += sum(
                        x["negative_pair_decisions"] for x in counts.values())
            common_tasks = task_sets[0]
            if any(support != common_tasks for support in task_sets[1:]):
                raise AssertionError("pair-type share support differs by concentration")
            boot_indexes = rng.integers(0, len(common_tasks),
                                        size=(DRAWS, len(common_tasks)))
            averaged = {version: np.stack(task_matrices[version], axis=1).mean(axis=1)
                        for version in VERSIONS}
            delta = averaged["BS"] - averaged["evidence"]
            result["pair_type_equal_weight_over_available_shares"][learner][pair_type] = {
                "valid_shares": valid_shares, "valid_tasks": common_tasks,
                "counts": pair_counts,
                "evidence": summarize(averaged["evidence"], boot_indexes),
                "BS": summarize(averaged["BS"], boot_indexes),
                "delta_BS_minus_E": summarize(delta, boot_indexes),
                "share_weighting": "equal within task over concentrations where the pair type exists"}

        before = learner_frame[learner_frame.variant == "evidence"]
        after = learner_frame[learner_frame.variant == "BS"]
        left = before[pair_key + ["predicted_link"]]
        right = after[pair_key + ["predicted_link"]]
        joined = left.merge(right, on=pair_key, suffixes=("_e", "_bs"),
                            validate="one_to_one")
        changes = {}
        for share in SHARES:
            changes[share] = {}
            for pair_type in PAIR_TYPES:
                x = joined[joined.fraction == share]
                if pair_type != "ALL":
                    x = x[x.pair_type == pair_type]
                rel = x.same_task.astype(int).to_numpy()
                e = x.predicted_link_e.astype(int).to_numpy()
                bs = x.predicted_link_bs.astype(int).to_numpy()
                changes[share][pair_type] = {
                    "pairs": int(len(x)),
                    "positive_pairs": int((rel == 1).sum()),
                    "negative_pairs": int((rel == 0).sum()),
                    "errors_corrected": int(((e != rel) & (bs == rel)).sum()),
                    "errors_introduced": int(((e == rel) & (bs != rel)).sum()),
                    "positive_errors_corrected": int(((rel == 1) & (e == 0) & (bs == 1)).sum()),
                    "positive_errors_introduced": int(((rel == 1) & (e == 1) & (bs == 0)).sum()),
                    "negative_errors_corrected": int(((rel == 0) & (e == 1) & (bs == 0)).sum()),
                    "negative_errors_introduced": int(((rel == 0) & (e == 0) & (bs == 1)).sum()),
                }
        result["error_transitions"][learner] = changes

    out_json = OUT / "five_share_analysis.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf8")
    with (OUT / "five_share_rows.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out_json)
    for learner in LEARNERS:
        for version in VERSIONS:
            s = result["five_share_equal_weight"][learner][version]
            print(learner, version, "BA/FMR/FSR %",
                  *[round(s["metrics"][k] * 100, 2) for k in ("ba", "fmr", "fsr")],
                  "BA CI %", [round(x * 100, 2) for x in s["ci95"]["ba"]])
        d = result["five_share_equal_weight"][learner]["delta_BS_minus_E"]
        print(learner, "ΔBA/FMR/FSR pp",
              *[round(d["metrics"][k] * 100, 2) for k in ("ba", "fmr", "fsr")],
              "ΔBA CI pp", [round(x * 100, 2) for x in d["ci95"]["ba"]])


if __name__ == "__main__":
    main()
