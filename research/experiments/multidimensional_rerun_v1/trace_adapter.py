"""Pinned, label-light adapter for the predeclared cross-benchmark rerun."""
from __future__ import annotations

import collections
import gzip
import hashlib
import io
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "scripts"))

from attackagent.traces import ATOM_GROUP, Observation, command_heads, samples, terminal_lines  # noqa: E402
import run_mixed_gallery as gallery  # noqa: E402

DATA_ROOT = ROOT / "data/raw/cyber-task-horizons"
MANIFEST_PATH = pathlib.Path(__file__).resolve().parent / "source_manifest.json"
REVISION = "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae"
DEVELOPMENT = ("intercode-ctf", "cybench", "nyuctf")
EXTERNAL = ("cvebench", "cybashbench", "cybergym", "nl2bash")
MIN_EXTERNAL_TASK_GROUPS = 3
SONNET_EVALSETS = {
    "cybashbench": "eval-set-mrfumbtyv0pa6iab",
    "nl2bash": "eval-set-bcakpnavej4q4jd3",
    "intercode-ctf": "eval-set-83d0a1kx3i5t3u6b",
    "nyuctf": "eval-set-bjldgp2ll9qroagt",
    "cybench": "eval-set-jcdoiurrc2a17nmv",
    "cvebench": "eval-set-rnjcl1u90s49rfsx",
    "cybergym": "eval-set-zsam5w2kdrpva5pz",
}


def normalize_task(benchmark: str, raw: object) -> str:
    """Return the benchmark's path-free task key using only source identifiers."""
    value = str(raw or "").strip().rstrip("/").split("/")[-1]
    if benchmark == "cvebench":
        value = re.sub(r"-one_day$", "", value)
    return value


def _trace_id(path: str, member: str) -> str:
    return hashlib.sha256(f"{REVISION}/{path}/{member}".encode()).hexdigest()[:16]


def _human_commands(sample: dict) -> list[str]:
    logs = sample.get("store", {}).get("HumanAgentState:logs", {})
    lines = []
    for key, value in sorted(logs.items()):
        if key.endswith(".input") and isinstance(value, str):
            lines.extend(terminal_lines(value))
    return command_heads("\n".join(lines))


def _agent_commands(sample: dict) -> list[str]:
    names = []
    for message in sample.get("messages", []):
        for call in message.get("tool_calls", []) or []:
            arguments = call.get("arguments", {}) or {}
            if call.get("function") == "bash" and isinstance(arguments.get("cmd"), str):
                names.extend(command_heads(arguments["cmd"]))
    return names


def _matched(rows):
    labels_by_task = collections.defaultdict(set)
    for row in rows:
        if row.names:
            labels_by_task[(row.benchmark, row.task)].add(row.label)
    matched = {key for key, labels in labels_by_task.items() if labels == {"human", "agent"}}
    return [row for row in rows if row.names and (row.benchmark, row.task) in matched], matched


def _load_development():
    rows = gallery.load_rows()
    rows = [
        row for row in rows
        if row.benchmark in DEVELOPMENT and (
            (row.label == "human" and row.source == "expert")
            or (row.label == "agent" and row.source == "claude-sonnet-4.6")
        )
    ]
    raw_rows = list(rows)
    labels_by_task = collections.defaultdict(set)
    rows_by_task = collections.Counter()
    for row in raw_rows:
        if row.names:
            labels_by_task[(row.benchmark, row.task)].add(row.label)
            rows_by_task[(row.benchmark, row.task)] += 1
    unmatched_by_benchmark = {}
    for benchmark in DEVELOPMENT:
        unmatched_keys = [key for key, labels in labels_by_task.items()
                          if key[0] == benchmark and labels != {"human", "agent"}]
        unmatched_by_benchmark[benchmark] = {
            "unmatched_task_groups": len(unmatched_keys),
            "unmatched_records": sum(rows_by_task[key] for key in unmatched_keys),
        }
    rows, matched = _matched(raw_rows)
    by_benchmark = {}
    for benchmark in DEVELOPMENT:
        selected = [row for row in rows if row.benchmark == benchmark]
        by_benchmark[benchmark] = {
            "records": len(selected),
            "human_records": sum(row.label == "human" for row in selected),
            "agent_records": sum(row.label == "agent" for row in selected),
            "matched_task_groups": sum(key[0] == benchmark for key in matched),
        }
    return rows, by_benchmark, unmatched_by_benchmark


def _load_external(manifest: dict):
    rows = []
    file_audit = []
    human_samples = collections.Counter()
    agent_samples = collections.Counter()
    unkeyed_samples = collections.Counter()
    samples_without_commands = collections.Counter()
    agent_command_records = collections.Counter()

    for record in manifest["human_files"]:
        relative = record["path"]
        parts = pathlib.PurePosixPath(relative).parts
        if len(parts) < 5 or parts[3] not in EXTERNAL:
            continue
        benchmark = parts[3]
        path = DATA_ROOT / relative
        parsed_count = 0
        with gzip.open(path, "rb") as handle:
            for member, sample in samples(io.BytesIO(handle.read())):
                parsed_count += 1
                human_samples[benchmark] += 1
                task_id = normalize_task(benchmark, sample.get("id"))
                commands = _human_commands(sample)
                if not task_id:
                    unkeyed_samples[benchmark] += 1
                    continue
                if not commands:
                    samples_without_commands[benchmark] += 1
                    continue
                rows.append(Observation(
                    _trace_id(relative, member), f"{benchmark}/{task_id}", "human", "expert",
                    tuple(commands), tuple(ATOM_GROUP[name] for name in commands), 0, benchmark,
                ))
        file_audit.append({"benchmark": benchmark, "kind": "human", "sample_count": parsed_count})

    evalset_to_benchmark = {evalset: benchmark for benchmark, evalset in SONNET_EVALSETS.items()}
    latest_agents = {}
    for record in manifest["agent_files"]:
        relative = record["path"]
        parts = pathlib.PurePosixPath(relative).parts
        evalset = parts[2] if len(parts) > 2 else ""
        benchmark = evalset_to_benchmark.get(evalset)
        if benchmark not in EXTERNAL:
            continue
        path = DATA_ROOT / relative
        parsed_count = 0
        for member, sample in samples(path):
            parsed_count += 1
            agent_samples[benchmark] += 1
            task_id = normalize_task(benchmark, sample.get("id"))
            commands = _agent_commands(sample)
            if not task_id:
                unkeyed_samples[benchmark] += 1
                continue
            if not commands:
                samples_without_commands[benchmark] += 1
                continue
            order = (pathlib.PurePosixPath(relative).name, str(sample.get("completed_at", "")))
            key = (benchmark, task_id)
            if key not in latest_agents or order > latest_agents[key][0]:
                latest_agents[key] = (order, relative, member, commands)
        file_audit.append({"benchmark": benchmark, "kind": "agent", "sample_count": parsed_count})

    for (benchmark, task_id), (_order, relative, member, commands) in sorted(latest_agents.items()):
        rows.append(Observation(
            _trace_id(relative, member), f"{benchmark}/{task_id}", "agent", "claude-sonnet-4.6",
            tuple(commands), tuple(ATOM_GROUP[name] for name in commands), 0, benchmark,
        ))
        agent_command_records[benchmark] += 1

    labels_by_task = collections.defaultdict(set)
    rows_by_task = collections.Counter()
    for row in rows:
        if row.names:
            key = (row.benchmark, row.task)
            labels_by_task[key].add(row.label)
            rows_by_task[key] += 1
    unmatched_by_benchmark = {}
    for benchmark in EXTERNAL:
        unmatched_keys = [key for key, labels in labels_by_task.items()
                          if key[0] == benchmark and labels != {"human", "agent"}]
        unmatched_by_benchmark[benchmark] = {
            "unmatched_task_groups": len(unmatched_keys),
            "unmatched_records": sum(rows_by_task[key] for key in unmatched_keys),
        }

    rows, matched = _matched(rows)
    by_benchmark = {}
    for benchmark in EXTERNAL:
        selected = [row for row in rows if row.benchmark == benchmark]
        task_count = sum(key[0] == benchmark for key in matched)
        eligible = task_count >= MIN_EXTERNAL_TASK_GROUPS
        by_benchmark[benchmark] = {
            "records": len(selected),
            "human_records": sum(row.label == "human" for row in selected),
            "agent_records": sum(row.label == "agent" for row in selected),
            "matched_task_groups": task_count,
            "primary_external_eligible": eligible,
            "exclusion_reason": (None if eligible else
                                 "selected model set has no observed execution commands"
                                 if agent_command_records[benchmark] == 0 else
                                 f"fewer than {MIN_EXTERNAL_TASK_GROUPS} matched task groups"),
        }
    audit = {
        "human_sample_records_seen": dict(human_samples),
        "agent_sample_records_seen": dict(agent_samples),
        "agent_command_observation_records": dict(agent_command_records),
        "samples_without_task_key": dict(unkeyed_samples),
        "samples_without_observed_commands": dict(samples_without_commands),
        "unmatched_tasks_before_pairing": unmatched_by_benchmark,
        "source_files": file_audit,
        "by_benchmark": by_benchmark,
    }
    return rows, audit


def load_cohort():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Verified source manifest not ready: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("revision") != REVISION or manifest.get("model") != "claude-sonnet-4.6":
        raise ValueError("Pinned source manifest revision or model differs from the protocol")
    development, dev_by_benchmark, dev_unmatched = _load_development()
    external, external_audit = _load_external(manifest)
    audit = {
        "revision": REVISION,
        "model": "claude-sonnet-4.6",
        "task_key_rules": {
            "all": "remove path prefix after the final slash and pair within benchmark only",
            "cvebench": "also remove the public task suffix -one_day",
        },
        "development": {
            "by_benchmark": dev_by_benchmark,
            "unmatched_tasks_before_pairing": dev_unmatched,
        },
        "external": external_audit,
    }
    return development, external, audit
