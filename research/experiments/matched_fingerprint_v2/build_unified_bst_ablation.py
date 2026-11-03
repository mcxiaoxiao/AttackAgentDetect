#!/usr/bin/env python3
"""Recompute the matched CTH/AIT B/S/T ablation from frozen pair predictions."""
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
OUT = ROOT / "research/results/matched_fingerprint_v2/unified_ablation"
SEED, DRAWS = 20261104, 10_000
CTH_SHARES = ("0.0", "0.25", "0.5", "0.75", "1.0")
CONFIGS = {
    "cth": {"BS": (1, 1, 0), "S": (0, 1, 0), "B": (1, 0, 0),
            "evidence": (0, 0, 0)},
    "ait": {"BST": (1, 1, 1), "ST": (0, 1, 1), "BT": (1, 0, 1),
            "BS": (1, 1, 0), "evidence": (0, 0, 0)},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rates(frame: pd.DataFrame) -> tuple[float, float, float, int, int]:
    positive = frame[frame.same_task == 1]
    negative = frame[frame.same_task == 0]
    if positive.empty or negative.empty:
        raise AssertionError("each task/gallery must contain positive and negative pairs")
    fmr = float(negative.predicted_link.mean())
    fsr = float(1.0 - positive.predicted_link.mean())
    return fmr, fsr, 1.0 - (fmr + fsr) / 2.0, len(positive), len(negative)


def task_values(frame: pd.DataFrame, dataset: str) -> tuple[list[str], np.ndarray, dict]:
    tasks = sorted(frame.query_task.unique())
    values, counts = [], {}
    for task in tasks:
        one_task = frame[frame.query_task == task]
        share_rows = []
        positive_count = negative_count = 0
        shares = CTH_SHARES if dataset == "cth" else ("-1.0",)
        for share in shares:
            one_share = one_task[one_task.fraction.astype(str) == share]
            fmr, fsr, ba, n_pos, n_neg = rates(one_share)
            share_rows.append((fmr, fsr, ba))
            positive_count += n_pos
            negative_count += n_neg
        values.append(np.asarray(share_rows).mean(axis=0))
        counts[task] = {"positive_pair_decisions": positive_count,
                        "negative_pair_decisions": negative_count,
                        "shares_averaged": len(shares)}
    return tasks, np.asarray(values, dtype=float), counts


def metric_summary(matrix: np.ndarray, indexes: np.ndarray) -> dict:
    point = matrix.mean(axis=0)
    boot = matrix[indexes].mean(axis=1)
    ci = np.quantile(boot, [0.025, 0.975], axis=0).T
    names = ("fmr", "fsr", "ba")
    return {"metrics": dict(zip(names, map(float, point))),
            "ci95": {name: list(map(float, ci[i])) for i, name in enumerate(names)}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT, compression="gzip")
    frame = frame[(frame.learner == "xgb") & frame.dataset.isin(CONFIGS)].copy()
    frame["fraction"] = frame.fraction.astype(str)
    frame["same_task"] = frame.same_task.astype(int)
    frame["predicted_link"] = frame.predicted_link.astype(int)
    pair_keys = ["dataset", "fold", "pair_id", "replicate", "fraction", "query_id",
                 "candidate_id", "query_task", "candidate_task", "pair_type", "same_task",
                 "candidate_position"]
    ref = frame[(frame.dataset == "cth") & (frame.variant == "evidence")][pair_keys]
    for dataset, configs in CONFIGS.items():
        dataset_frame = frame[frame.dataset == dataset]
        dataset_ref = dataset_frame[dataset_frame.variant == "evidence"][pair_keys]
        for version in configs:
            found = dataset_frame[dataset_frame.variant == version][pair_keys]
            pd.testing.assert_frame_equal(dataset_ref.reset_index(drop=True),
                                          found.reset_index(drop=True), check_dtype=False)
        assert dataset_ref.dataset.nunique() == 1
    assert len(ref) == 54_400
    assert frame[(frame.dataset == "ait") & (frame.variant == "evidence")].shape[0] == 43_600

    rng = np.random.default_rng(SEED)
    results, matrix_cache, task_audits = {}, {}, {}
    row_records = []
    for dataset, configs in CONFIGS.items():
        results[dataset], matrix_cache[dataset], task_audits[dataset] = {}, {}, {}
        current = frame[frame.dataset == dataset]
        dataset_tasks = sorted(current.query_task.unique())
        indexes = rng.integers(0, len(dataset_tasks), size=(DRAWS, len(dataset_tasks)))
        for version in configs:
            subset = current[current.variant == version]
            tasks, matrix, counts = task_values(subset, dataset)
            if matrix_cache[dataset] and tasks != next(iter(matrix_cache[dataset].values()))[0]:
                raise AssertionError("configuration task lists differ")
            indexes = rng.integers(0, len(tasks), size=(DRAWS, len(tasks)))
            matrix_cache[dataset][version] = (tasks, matrix)
            task_audits[dataset][version] = counts
            summary = metric_summary(matrix, indexes)
            results[dataset][version] = summary
        full_version = "BS" if dataset == "cth" else "BST"
        full_tasks, full_matrix = matrix_cache[dataset][full_version]
        for version in configs:
            tasks, matrix = matrix_cache[dataset][version]
            assert tasks == full_tasks
            paired_delta = matrix - full_matrix
            delta_summary = metric_summary(paired_delta, indexes)
            results[dataset][version]["delta_vs_full"] = {
                "metrics": delta_summary["metrics"],
                "ci95": delta_summary["ci95"]}
            fallback = int(current[current.variant == version].fallback.astype(int).sum())
            results[dataset][version]["fallback_pair_decisions"] = fallback
            result = results[dataset][version]
            row_records.append({
                "dataset": dataset, "configuration": version,
                "B": configs[version][0], "S": configs[version][1], "T": configs[version][2],
                "tasks": len(tasks),
                "positive_pair_decisions": sum(v["positive_pair_decisions"]
                                                for v in task_audits[dataset][version].values()),
                "negative_pair_decisions": sum(v["negative_pair_decisions"]
                                                for v in task_audits[dataset][version].values()),
                "fallback_pair_decisions": fallback,
                "ba": result["metrics"]["ba"],
                "ba_ci_low": result["ci95"]["ba"][0],
                "ba_ci_high": result["ci95"]["ba"][1],
                "fmr": result["metrics"]["fmr"], "fsr": result["metrics"]["fsr"],
                "delta_ba_vs_full": result["delta_vs_full"]["metrics"]["ba"],
                "delta_ba_ci_low": result["delta_vs_full"]["ci95"]["ba"][0],
                "delta_ba_ci_high": result["delta_vs_full"]["ci95"]["ba"][1],
                "delta_fmr_vs_full": result["delta_vs_full"]["metrics"]["fmr"],
                "delta_fsr_vs_full": result["delta_vs_full"]["metrics"]["fsr"],
            })

    run = {
        "protocol": "post hoc, matched ablation recomputed from frozen v1 pair predictions; not a new model run",
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha256(INPUT),
        "source_metrics_sha256": sha256(ROOT / "research/results/matched_fingerprint_v1/metrics.json"),
        "analysis_script_sha256": sha256(Path(__file__)),
        "learner": "XGBoost; each fold uses the selected full-model parameter and fixed pair list",
        "cth_metric": "equal mean over the five exact Agent shares within each query task, then equal mean over 17 tasks",
        "ait_metric": "equal mean over 12 task groups",
        "bootstrap": {"unit": "task group", "draws": DRAWS, "seed": SEED,
                      "paired_versions": True},
        "counts_are_pair_decisions_not_independent_samples": True,
        "pair_lists_identical_within_dataset": True,
        "configurations": CONFIGS,
        "results": results,
        "task_pair_counts": task_audits,
    }
    (OUT / "unified_bst_ablation.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    with (OUT / "unified_bst_ablation.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_records[0]))
        writer.writeheader()
        writer.writerows(row_records)
    print("wrote", OUT / "unified_bst_ablation.json")
    for row in row_records:
        print(row["dataset"], row["configuration"],
              "BA/FMR/FSR %", *[round(100 * row[k], 2) for k in ("ba", "fmr", "fsr")],
              "BA CI %", [round(100 * row[k], 2) for k in ("ba_ci_low", "ba_ci_high")],
              "ΔBA pp", round(100 * row["delta_ba_vs_full"], 2),
              [round(100 * row[k], 2) for k in ("delta_ba_ci_low", "delta_ba_ci_high")])


if __name__ == "__main__":
    main()
