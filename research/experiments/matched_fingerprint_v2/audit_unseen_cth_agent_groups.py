#!/usr/bin/env python3
"""Freeze CTH Agent-only task groups not used by prior association evaluations."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/results/matched_fingerprint_v2/independent_cth_task_manifest.json"
CORPUS = ROOT / "research/results/behavior_corpus_v1"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def csv_tasks(path, fields, source_field=None, source_value=None):
    tasks = set()
    with path.open(newline="", encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            if source_field and row.get(source_field) != source_value:
                continue
            for field in fields:
                value = row.get(field, "")
                if value and value.startswith("cyber-task-horizons/"):
                    tasks.add(value)
    return tasks


def csv_task_column(path, task_field, source_field=None, source_value=None):
    return csv_tasks(path, (task_field,), source_field, source_value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    legacy = load_module(ROOT / "research/experiments/overnight_fingerprint_v1/run.py", "legacy")
    sys.path.insert(0, str(ROOT))
    from attackagent import fingerprint_v2 as fingerprints
    raw = [r for r in legacy.load_corpus()
           if r.get("source_dataset") == "cyber-task-horizons" and r.get("track") == "ctf"]
    by_task = defaultdict(list)
    for row in raw:
        by_task[row["task_group"]].append(row)

    sources = {}
    sources["mixture_40_group_cross_mode"] = csv_task_column(
        ROOT / "research/results/behavioral_fingerprint_v2/cth_cross_mode/query_predictions.csv",
        "query_task")
    sources["mixture_role_predictions"] = csv_tasks(
        ROOT / "research/results/behavioral_fingerprint_v2/cth_mixture/pair_predictions.csv",
        ("query_task", "candidate_task"))
    sources["overnight_agent_retrieval_48"] = csv_task_column(
        ROOT / "research/results/overnight_fingerprint_v1/retrieval/predictions.csv",
        "task", "source", "cyber-task-horizons")
    sources["short_trace_agent_retrieval_86"] = csv_task_column(
        ROOT / "research/results/short_trace_extension_v1/predictions.csv",
        "task", "source", "cyber-task-horizons")

    matched_v1 = ROOT / "research/results/matched_fingerprint_v1/pair_predictions.csv.gz"
    with __import__("gzip").open(matched_v1, "rt", newline="", encoding="utf8") as handle:
        sources["matched_fingerprint_v1"] = set()
        for row in csv.DictReader(handle):
            if row.get("dataset") == "cth":
                sources["matched_fingerprint_v1"].update(
                    value for value in (row.get("query_task"), row.get("candidate_task"))
                    if value)
    used = set().union(*sources.values())

    agent_rows_by_task = {task: [r for r in rows if r["mode"] == "agent"]
                          for task, rows in by_task.items()}
    human_tasks = {task for task, rows in by_task.items()
                   if any(r["mode"] == "human" for r in rows)}
    dev_tasks = {task for task, rows in by_task.items()
                 if sum(r["mode"] == "human" for r in rows) >= 2
                 and sum(r["mode"] == "agent" for r in rows) >= 2}
    dev_sequences = {tuple(row["commands"]) for task in dev_tasks
                     for row in by_task[task]}

    # CyBench task slugs also occur in the Antieval-only evaluation, so they
    # are considered previously seen even when source-qualified IDs differ.
    antieval = [r for r in legacy.load_corpus()
                if r.get("source_dataset") == "antieval/cybench-trajectories"
                and r.get("track") == "ctf"]
    antieval_slugs = {r["task_group"].rstrip("/").split("/")[-1]
                      .removesuffix(" (hard)").rsplit(":", 1)[-1] for r in antieval}

    examined = []
    exclusions = Counter()
    candidates = []
    for task, rows in sorted(agent_rows_by_task.items()):
        if task in human_tasks:
            exclusions["mixed_human_agent_task_already_development"] += 1
            continue
        if task in used:
            exclusions["task_appears_in_prior_association_predictions"] += 1
            continue
        slug = task.rsplit("/", 1)[-1]
        if "/cybench/" in task and slug in antieval_slugs:
            exclusions["same_cybench_slug_in_prior_antieval_retrieval"] += 1
            continue
        if len(rows) < 2:
            exclusions["fewer_than_two_agent_runs"] += 1
            continue
        usable = []
        duplicated_training_sequences = 0
        for row in rows:
            commands = row["commands"][:128]
            actions = row["actions"][:128]
            if len(commands) < 2:
                continue
            if tuple(row["commands"]) in dev_sequences:
                duplicated_training_sequences += 1
                continue
            blocks, audit = fingerprints.fingerprint_matrix(
                commands, actions=actions, timestamps=None, window=4,
                min_window_events=2, max_events=128)
            if blocks is None:
                continue
            b = fingerprints.summarize(blocks["B"])
            s = fingerprints.summarize(blocks["S"])
            usable.append({"trace_id": row["trace_id"], "command_count": len(commands),
                           "fingerprint_available": bool(np.isfinite(b).any() and np.isfinite(s).any())})
        usable = [r for r in usable if r["fingerprint_available"]]
        row = {"task": task, "source_qualified_task": task,
               "agent_run_count": len(rows), "eligible_trace_count": len(usable),
               "training_sequence_duplicates_removed": duplicated_training_sequences,
               "traces": sorted(usable, key=lambda x: x["trace_id"])}
        examined.append(row)
        if len(usable) >= 2:
            candidates.append(row)
        else:
            exclusions["fewer_than_two_fingerprint_eligible_agent_runs"] += 1

    external = {"status": "frozen_candidate_manifest_pending_pair_run",
                "source": "Cyber Task Horizons public CTF agent execution traces",
                "source_revision": "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae",
                "independence_unit": "source-qualified CTF task group; no task appears in prior CTH association predictions or in the prior cross-source CyBench gallery",
                "label_scope": "same source-qualified task is a retrieval proxy; it does not label actor, attack, campaign, or organization identity",
                "protocol_compatibility": "Agent-only (AA) confirmation; no HH or HA pairs are available in the candidate cohort",
                "feature_gate": {"window": 4, "minimum_events": 2,
                                 "max_events": 128, "blocks": ["B", "S"]},
                "raw_trace_policy": "only task IDs, trace IDs, and command counts are written; no command or argument values are copied",
                "input_sha256": {
                    "evidence": sha(CORPUS / "evidence.jsonl"),
                    "registry": sha(CORPUS / "trace_registry.csv")},
                "previous_evaluations": {name: {"task_group_count": len(tasks),
                                                 "sha256": sha(path)}
                                         for name, tasks, path in [
                    ("mixture_40_group_cross_mode", sources["mixture_40_group_cross_mode"], ROOT / "research/results/behavioral_fingerprint_v2/cth_cross_mode/query_predictions.csv"),
                    ("mixture_role_predictions", sources["mixture_role_predictions"], ROOT / "research/results/behavioral_fingerprint_v2/cth_mixture/pair_predictions.csv"),
                    ("overnight_agent_retrieval_48", sources["overnight_agent_retrieval_48"], ROOT / "research/results/overnight_fingerprint_v1/retrieval/predictions.csv"),
                    ("short_trace_agent_retrieval_86", sources["short_trace_agent_retrieval_86"], ROOT / "research/results/short_trace_extension_v1/predictions.csv"),
                    ("matched_fingerprint_v1", sources["matched_fingerprint_v1"], matched_v1)]},
                "audit_counts": {"cth_ctf_task_groups": len(by_task),
                                 "cth_ctf_mixed_role_task_groups": len(human_tasks),
                                 "development_tasks_with_two_runs_per_role": len(dev_tasks),
                                 "development_command_sequences": len(dev_sequences),
                                 "prior_evaluation_task_groups_union": len(used),
                                 "prior_antieval_cybench_challenge_slugs": len(antieval_slugs),
                                 "unseen_agent_only_groups_with_2plus_usable_runs": len(candidates),
                                 "exclusions": dict(exclusions)},
                "eligible_tasks": candidates}
    OUT.write_text(json.dumps(external, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(external["audit_counts"], ensure_ascii=False, indent=2))
    for row in candidates:
        print(row["task"], row["eligible_trace_count"],
              [x["command_count"] for x in row["traces"]])


if __name__ == "__main__":
    main()
