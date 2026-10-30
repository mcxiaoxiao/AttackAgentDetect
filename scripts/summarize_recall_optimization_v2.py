#!/usr/bin/env python3
"""Summarize query coverage from the saved outer-test pair scores."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_association_error_analysis as old
import run_association_operating_points as op

OUT = ROOT / "results" / "recall_optimization_v2"
BOOTSTRAPS = 5000
SEED = 20261030 + 970


def task_coverage(rows, acceptance_column, method):
    by_query = collections.defaultdict(list)
    for row in rows:
        by_query[(row["benchmark"], row["query_task"], row["query_trace_id"])].append(
            int(float(row[acceptance_column]))
        )
    by_task = collections.defaultdict(list)
    for (benchmark, task, _trace), values in by_query.items():
        by_task[(benchmark, task)].append(float(any(values)))
    return [{"benchmark": benchmark, "task": task, "method": method,
             "coverage": sum(values) / len(values)}
            for (benchmark, task), values in sorted(by_task.items())]


def run():
    with (OUT / "pair_scores.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    data = {"targets": {}, "paired_deltas_vs_behavior": {}, "joint_selected": None}
    records = []
    methods = ("behavior", "behavior_align", "behavior_multiscale", "behavior_align_profile",
               "behavior_align_taskweighted", "behavior_align_ha")
    for target in (10, 20, 30):
        key = str(target)
        data["targets"][key] = {}
        method_rows = {}
        for method in methods:
            for policy, col_suffix in (("global", f"{method}_{target}"),
                                       ("stratified", f"{method}_strat_{target}")):
                if policy == "stratified" and method not in (
                    "behavior", "behavior_align", "behavior_align_taskweighted", "behavior_align_ha"
                ):
                    continue
                acceptance_col = f"accepted_{col_suffix}"
                task_rows = task_coverage(rows, acceptance_col, f"{method}_{policy}")
                interval = old.task_cluster_interval(task_rows, "coverage", n_boot=BOOTSTRAPS,
                                                     seed=SEED + target + len(records))
                data["targets"][key][f"{method}_{policy}"] = interval
                method_rows[f"{method}_{policy}"] = task_rows
                records.extend({**row, "target": target, "policy": policy} for row in task_rows)
        deltas = {}
        for method in methods:
            for policy in ("global", "stratified"):
                name = f"{method}_{policy}"
                baseline_name = f"behavior_{policy}"
                if name not in method_rows or baseline_name not in method_rows or name == baseline_name:
                    continue
                left = [{**row, "method": name} for row in method_rows[name]]
                right = [{**row, "method": baseline_name} for row in method_rows[baseline_name]]
                deltas[name] = op.paired_delta([*left, *right], "coverage", name, baseline_name,
                                               BOOTSTRAPS, SEED + target * 31 + methods.index(method))
        data["paired_deltas_vs_behavior"][key] = deltas

    selected_rows = task_coverage(rows, "joint_accepted", "joint_selected")
    data["joint_selected"] = old.task_cluster_interval(selected_rows, "coverage", n_boot=BOOTSTRAPS,
                                                        seed=SEED + 71)
    records.extend({**row, "target": 20, "policy": "joint_selected"} for row in selected_rows)

    # Commit a transparent, reproducible coverage supplement to the main result.
    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["coverage_by_target"] = data["targets"]
    results["coverage_paired_deltas_vs_behavior"] = data["paired_deltas_vs_behavior"]
    results["coverage"] = {"joint_selected": data["joint_selected"]}
    results["coverage_summary_source_sha256"] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "coverage_task_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"targets": {k: {m: round(v["estimate"] * 100, 2)
                                     for m, v in value.items()}
                       for k, value in data["targets"].items()},
                      "joint_selected": data["joint_selected"]}, ensure_ascii=False))


if __name__ == "__main__":
    run()
