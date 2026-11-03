#!/usr/bin/env python3
"""Freeze an unseen-task NYU CTF Agent–Agent confirmation manifest.

Raw transcript content is parsed in memory and never written to the project.
Only recognized command counts, public task IDs, hashed trace IDs, and file
hashes are retained.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/results/matched_fingerprint_v2/independent_nyu_task_manifest.json"
PROTOCOL = ROOT / "research/experiments/matched_fingerprint_v2/independent_nyu_aa_protocol.md"
SUBMISSIONS = ("dcipher_claude3.5_sonnet", "dcipher_claude3.7_sonnet",
              "dcipher_gpt4.1", "dcipher_gpt4o")


sys.path.insert(0, str(ROOT))
from attackagent import fingerprint_v2 as FP  # noqa: E402
from attackagent import traces as TRACE  # noqa: E402


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_task(value):
    value = (value or "").strip().lower().replace("\\", "/").rstrip("/")
    if value.startswith("antieval/"):
        value = value.rsplit("/", 1)[-1].removesuffix(" (hard)")
        if ":" in value:
            value = value.rsplit(":", 1)[-1]
    else:
        value = value.rsplit("/", 1)[-1]
    for marker in (" (hard)", " (medium)", " (easy)"):
        value = value.removesuffix(marker)
    return value


def add_csv_slugs(path, fields, used, source_filter=None):
    path = Path(path)
    with path.open(newline="", encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            if source_filter and row.get(source_filter[0]) != source_filter[1]:
                continue
            for field in fields:
                value = row.get(field, "")
                if value:
                    used.add(normalize_task(value))


def prior_used_slugs():
    used = set()
    checked = {}
    files = [
        ("research/results/behavioral_fingerprint_v2/cth_cross_mode/query_predictions.csv",
         ["query_task"], None),
        ("research/results/behavioral_fingerprint_v2/cth_mixture/pair_predictions.csv",
         ["query_task", "candidate_task"], None),
        ("research/results/overnight_fingerprint_v1/retrieval/predictions.csv",
         ["task"], ("source", "cyber-task-horizons")),
        ("research/results/short_trace_extension_v1/predictions.csv",
         ["task"], ("source", "cyber-task-horizons")),
    ]
    for rel, fields, source_filter in files:
        path = ROOT / rel
        add_csv_slugs(path, fields, used, source_filter)
        checked[rel] = sha_file(path)

    matched = ROOT / "research/results/matched_fingerprint_v1/pair_predictions.csv.gz"
    with gzip.open(matched, "rt", newline="", encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") == "cth":
                used.update(normalize_task(row.get(k, "")) for k in
                            ("query_task", "candidate_task") if row.get(k))
    checked[str(matched.relative_to(ROOT))] = sha_file(matched)

    cth_manifest = ROOT / "research/results/matched_fingerprint_v2/independent_cth_task_manifest.json"
    previous = json.loads(cth_manifest.read_text(encoding="utf8"))
    used.update(normalize_task(row["task"]) for row in previous["eligible_tasks"])
    checked[str(cth_manifest.relative_to(ROOT))] = sha_file(cth_manifest)
    return used, checked


def extract_commands(path):
    data = json.loads(path.read_text(encoding="utf8"))
    output = []
    run_calls = 0
    missing_command = 0
    executors = data.get("executors", [])
    if not isinstance(executors, list):
        return [], 0, 0
    for messages in executors:
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            tool = message.get("tool_call")
            if not isinstance(tool, dict) or tool.get("name") != "run_command":
                continue
            args = tool.get("parsed_args")
            command = args.get("command") if isinstance(args, dict) else None
            if not isinstance(command, str):
                missing_command += 1
                continue
            run_calls += 1
            output.extend(TRACE.command_heads(command))
    return output[:128], run_calls, missing_command


def eligible_profile(commands):
    if len(commands) < 2:
        return False
    actions = [TRACE.ATOM_GROUP[command] for command in commands]
    blocks, _ = FP.fingerprint_matrix(commands, actions=actions, timestamps=None,
                                      window=4, min_window_events=2, max_events=128)
    return bool(blocks is not None and
                np.isfinite(FP.summarize(blocks["B"])).any() and
                np.isfinite(FP.summarize(blocks["S"])).any())


def repository_revision(source):
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="local shallow checkout of NYU-LLM-CTF/leaderboard_submissions")
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    transcript_root = source / "transcripts"
    if not transcript_root.is_dir():
        raise FileNotFoundError(f"not a NYU submissions checkout: {source}")
    revision = repository_revision(source)
    used, checked_prior = prior_used_slugs()
    counters = Counter()
    files, source_rows, source_task_slugs = {}, defaultdict(list), set()
    recognized_by_submission = Counter()
    inventory_rows = []

    for submission in SUBMISSIONS:
        folder = transcript_root / submission
        summary_path = folder / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"required D-CIPHER submission missing: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf8"))
        metadata = summary.get("metadata", {})
        if metadata.get("agent") != "D-CIPHER":
            raise ValueError(f"unexpected submission identity in {submission}")
        counters[f"submission_{submission}_summary_challenges"] = len(summary.get("results", {}))

        for path in sorted(folder.glob("*.json")):
            if path.name == "summary.json":
                continue
            counters["json_transcript_files"] += 1
            raw_hash = sha_file(path)
            trace_id = sha_bytes(f"{submission}/{path.stem}/{raw_hash}".encode())[:20]
            commands, tool_calls, missing = extract_commands(path)
            counters["run_command_tool_calls"] += tool_calls
            counters["run_command_calls_missing_text"] += missing
            recognized_by_submission[submission] += len(commands)
            counters["recognized_command_heads_before_min_length"] += len(commands)
            inventory_rows.append((submission, path.stem, raw_hash))
            source_task_slugs.add(path.stem.lower())
            if len(commands) < 2:
                counters["transcripts_under_two_recognized_commands"] += 1
                continue
            if not eligible_profile(commands):
                counters["transcripts_without_complete_bs"] += 1
                continue
            source_rows[path.stem.lower()].append({
                "trace_id": trace_id, "submission": submission,
                "command_count": len(commands), "raw_file_sha256": raw_hash,
                "sequence_sha256": sha_bytes("\n".join(commands).encode())})

    inventory_text = "".join(f"{sub}/{task}:{digest}\n"
                             for sub, task, digest in sorted(inventory_rows))
    inventory_sha = sha_bytes(inventory_text.encode())
    overlaps = sorted(source_task_slugs & used)
    excluded_overlap = set(overlaps)
    candidates = []
    duplicates_removed = 0
    under_supported = 0
    for task, traces in sorted(source_rows.items()):
        if task in excluded_overlap:
            counters["eligible_traces_excluded_known_task"] += len(traces)
            continue
        seen_sequences = set()
        unique = []
        for trace in traces:
            sequence = trace["sequence_sha256"]
            if sequence in seen_sequences:
                duplicates_removed += 1
                continue
            seen_sequences.add(sequence)
            unique.append(trace)
        if len(unique) < 2:
            under_supported += 1
            continue
        partition = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) % 3
        candidates.append({"task": task, "candidate_partition": partition,
                           "eligible_trace_count": len(unique), "traces": unique})
    partition_counts = Counter(row["candidate_partition"] for row in candidates)
    if len(candidates) < 20 or any(partition_counts[k] < 4 for k in range(3)):
        raise RuntimeError(f"insufficient unseen tasks for frozen galleries: {dict(partition_counts)}")

    protocol_hash = sha_file(PROTOCOL)
    manifest = {
        "status": "frozen_before_confirmation_model_fit",
        "source": "NYU-LLM-CTF leaderboard submissions; D-CIPHER only",
        "source_url": "https://github.com/NYU-LLM-CTF/leaderboard_submissions",
        "source_revision": revision,
        "license_note": "Raw submitter transcripts are not redistributed; source terms and per-submission attribution remain applicable.",
        "label_scope": "same challenge filename across distinct D-CIPHER submissions is a task-retrieval proxy; not activity, actor, or campaign identity",
        "pair_scope": "Agent-only AA; four model snapshots within one Agent framework; no HH/HA estimates",
        "protocol_path": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": protocol_hash,
        "preexcluded_task_overlap_count": len(overlaps),
        "preexcluded_task_slugs": overlaps,
        "input_task_slug_count": len(source_task_slugs),
        "eligible_task_count": len(candidates),
        "eligible_trace_count": sum(row["eligible_trace_count"] for row in candidates),
        "exact_same_task_command_sequence_duplicates_removed": duplicates_removed,
        "otherwise_under_supported_task_count": under_supported,
        "candidate_partition_counts": {str(k): partition_counts[k] for k in range(3)},
        "pair_sampling": {"repetitions": 100, "positive_candidates": 1,
                          "negative_candidates": 3,
                          "negatives_from_distinct_tasks": True,
                          "negatives_restricted_to_same_sha256_task_partition": True,
                          "same_pair_manifest_for_all_learners_and_versions": True},
        "feature_contract": {"tool_calls_used": "run_command only",
                             "command_normalizer": "attackagent.traces.command_heads; shared CTH allowlist and wrapper handling",
                             "action_map": "attackagent.traces.ATOM_GROUP",
                             "fingerprints": "B/S, four-event windows, minimum two events, 128-event cap; no time field",
                             "excluded_fields": ["success outcome", "challenge text and task ID as model features",
                                                 "model/submission ID", "tool output", "file write contents"]},
        "audit": {"input_transcript_files": counters["json_transcript_files"],
                  "recognized_command_heads_by_submission": dict(recognized_by_submission),
                  "recognized_command_heads_before_min_length": counters["recognized_command_heads_before_min_length"],
                  "run_command_tool_calls": counters["run_command_tool_calls"],
                  "run_command_calls_missing_text": counters["run_command_calls_missing_text"],
                  "transcripts_under_two_recognized_commands": counters["transcripts_under_two_recognized_commands"],
                  "transcripts_without_complete_bs": counters["transcripts_without_complete_bs"],
                  "eligible_traces_excluded_known_task": counters["eligible_traces_excluded_known_task"],
                  "previously_used_association_slugs": len(used),
                  "same_task_exact_sequence_duplicates_removed": duplicates_removed,
                  "candidate_partition_counts": {str(k): partition_counts[k] for k in range(3)},
                  "input_inventory_sha256": inventory_sha,
                  "inputs_per_trace": [{"submission": sub, "challenge": task,
                                        "raw_file_sha256": digest}
                                       for sub, task, digest in sorted(inventory_rows)],
                  "previous_result_files_sha256": checked_prior},
        "eligible_tasks": candidates,
        "frozen_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf8")
    print(json.dumps({"source_revision": revision,
                      "preexcluded_task_overlaps": len(overlaps),
                      "eligible_tasks": len(candidates),
                      "eligible_traces": manifest["eligible_trace_count"],
                      "exact_duplicates_removed": duplicates_removed,
                      "candidate_partitions": dict(partition_counts),
                      "manifest": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
