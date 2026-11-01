"""Check frozen inputs and study eligibility. Does not fit or evaluate a model."""
from __future__ import annotations

import collections
import csv
import hashlib
import importlib.util
import json
import pathlib
import platform
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
EXP = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "research/results/overnight_fingerprint_v1"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def group_rows(rows, minimum, mode=None):
    result = collections.defaultdict(list)
    for row in rows:
        if int(row["command_count"]) >= minimum and (mode is None or row["mode"] == mode):
            result[row["task_group"]].append(row)
    return result


def repeated_both(groups):
    return {key: rows for key, rows in groups.items()
            if all(sum(r["mode"] == mode for r in rows) >= 2 for mode in ("human", "agent"))}


def frozen_fold(group, seed):
    return hashlib.sha256(f"{seed}/{group}".encode()).hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = json.loads((EXP / "protocol.json").read_text())
    for rel, expected in config["inputs"].items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f"Frozen input changed: {rel}; do not silently re-freeze after seeing scores")
    archive = ROOT / config["ait_archive"]["path"]
    if sha(archive) != config["ait_archive"]["sha256"]:
        raise ValueError("AIT archive differs from its frozen source")
    rows = list(csv.DictReader((ROOT / "research/results/behavior_corpus_v1/trace_registry.csv").open()))
    evidence = {r["trace_id"]: r for r in map(json.loads, (ROOT / "research/results/behavior_corpus_v1/evidence.jsonl").read_text().splitlines())}
    if len(rows) != 1142 or len(evidence) != 1142 or {r["trace_id"] for r in rows} != set(evidence):
        raise ValueError("Corpus registry/evidence join failed")
    if any(len(evidence[r["trace_id"]]["commands"]) != int(r["command_count"]) for r in rows):
        raise ValueError("Corpus command counts do not match evidence")
    # Reuse only the audited parser/formulas. This never invokes the pilot main().
    pilot_path = ROOT / "research/experiments/log_fingerprint_pilot_v1/run.py"
    spec = importlib.util.spec_from_file_location("frozen_pilot", pilot_path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    pilot.checks()
    runs = pilot.load_runs()
    pilot_rows, _ = pilot.view_runs(runs, "login_exec")
    if len(pilot_rows) != 95:
        raise ValueError("Frozen pilot cohort no longer reproduces 95 records")
    minimum = config["matrix"]["minimum_commands"]
    maximum = config["matrix"]["maximum_prefix_commands"]
    ait, rejected = [], []
    for row in pilot_rows:
        commands = row["commands"][:maximum]
        _, quality = pilot.stats(commands)
        item = {"run_id": row["id"], "task": row["task"], "source_group": row["source"],
                "label": row["y"], "prefix_commands": len(commands), "eligible_action_pairs": quality["eligible_pairs"]}
        if len(commands) >= minimum and quality["eligible_pairs"] >= 5:
            ait.append(item)
        else:
            rejected.append(item)
    cth = [r for r in rows if r["track"] == "ctf" and r["source_dataset"] == "cyber-task-horizons"]
    coverage = []
    for n in (1, 4, 8, 16, 20, 32, 64):
        groups = group_rows(cth, n)
        both = {key: value for key, value in groups.items() if {r["mode"] for r in value} == {"human", "agent"}}
        coverage.append({"minimum_commands": n, "dual_class_tasks": len(both),
                         "two_human_two_agent_tasks": len(repeated_both(groups)),
                         "records_in_dual_class_tasks": sum(map(len, both.values()))})
    mixture_groups = repeated_both(group_rows(cth, 1))
    matrix_mixture = repeated_both(group_rows(cth, minimum))
    agent_tracks = []
    folds = []
    endpoints = {"cth_mixture": mixture_groups}
    for source in config["agent_retrieval"]["sources"]:
        subset = [r for r in rows if r["source_dataset"] == source and r["track"] == "ctf"]
        groups = {key: value for key, value in group_rows(subset, minimum, "agent").items() if len(value) >= 2}
        available = len(groups) >= config["agent_retrieval"]["minimum_tasks"]
        k = 5 if len(groups) >= 15 else 3 if available else None
        agent_tracks.append({"source": source, "tasks": len(groups), "records": sum(map(len, groups.values())),
                             "candidates_per_query": k, "available": available})
        if available:
            endpoints[f"agent_retrieval/{source}"] = groups
    for endpoint, groups in endpoints.items():
        for i, group in enumerate(sorted(groups, key=lambda g: frozen_fold(g, config["seeds"]["split"]))):
            folds.append({"endpoint": endpoint, "group": group, "fold": i % 3})
    for i, task in enumerate(sorted({r["task"] for r in ait})):
        folds.append({"endpoint": "ait_leave_one_task_out", "group": task, "fold": i})
    labels = collections.Counter(r["label"] for r in ait)
    gates = {
        "ait_matrix_recognition": {"available": min(labels[0], labels[1]) >= config["recognition"]["minimum_per_class"]
                                   and len({r["task"] for r in ait}) >= config["recognition"]["minimum_tasks"],
                                   "human": labels[0], "agent": labels[1], "tasks": len({r["task"] for r in ait})},
        "ctf_baseline_mixture": {"available": len(mixture_groups) >= config["ctf_mixture"]["minimum_tasks"],
                                 "tasks": len(mixture_groups)},
        "ctf_matrix_increment_in_mixture": {"available": len(matrix_mixture) >= config["ctf_mixture"]["matrix_increment_minimum_tasks"],
                                            "tasks": len(matrix_mixture), "minimum_tasks": config["ctf_mixture"]["matrix_increment_minimum_tasks"],
                                            "if_unavailable": "skip; do not claim fingerprint fixes Human/Agent mixture association"},
    }
    result = {"kind": "eligibility_audit_not_effectiveness_results", "protocol_sha256": sha(EXP / "protocol.json"),
              "runtime": {"python": platform.python_version(), "numpy": pilot.np.__version__,
                          "scipy": pilot.scipy.__version__, "sklearn": pilot.sklearn.__version__},
              "corpus_records": len(rows), "ctf_records": sum(r["track"] == "ctf" for r in rows),
              "cth_ctf_records": len(cth), "cth_command_coverage": coverage,
              "ait_pilot_records": len(pilot_rows), "ait_matrix_records": len(ait), "ait_excluded_for_matrix": len(rejected),
              "gates": gates, "agent_only_retrieval": agent_tracks,
              "experiment_execution_started": False, "score_based_selection_performed": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(OUT / "ait_matrix_eligibility.csv", ait, ["run_id", "task", "source_group", "label", "prefix_commands", "eligible_action_pairs"])
    write_csv(OUT / "ait_matrix_exclusions.csv", rejected, ["run_id", "task", "source_group", "label", "prefix_commands", "eligible_action_pairs"])
    write_csv(OUT / "task_folds.csv", folds, ["endpoint", "group", "fold"])
    print(json.dumps({"ait_matrix_records": len(ait), "gates": gates, "agent_only_retrieval": agent_tracks}, indent=2))
    return 0 if gates["ait_matrix_recognition"]["available"] and gates["ctf_baseline_mixture"]["available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
