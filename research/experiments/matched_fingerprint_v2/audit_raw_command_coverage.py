#!/usr/bin/env python3
"""Measure how much command evidence the frozen CTH adapter discards.

Only aggregate counts and hashes are written. Raw commands, arguments, and
task-specific strings are never copied to the result directory.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
import json
import pathlib
import shlex
import zipfile
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys_path = str(ROOT)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from attackagent.traces import ASSIGNMENT, WRAPPERS, command_heads, samples, terminal_lines

OUT = ROOT / "research/results/matched_fingerprint_v2"
RAW = ROOT / "research/data/raw/cyber-task-horizons"


def module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(obj)
    return obj


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_sample(locator: str):
    rel, member = locator.split("#", 1)
    path = RAW / rel
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            payload = f.read()
        archive = zipfile.ZipFile(io.BytesIO(payload))
    else:
        archive = zipfile.ZipFile(path)
    with archive, archive.open(member) as f:
        return json.load(f), path


def extract_chunks(sample: dict, mode: str) -> list[str]:
    if mode == "human":
        logs = sample.get("store", {}).get("HumanAgentState:logs", {})
        lines = []
        for key, value in sorted(logs.items()):
            if key.endswith(".input") and isinstance(value, str):
                lines.extend(terminal_lines(value))
        return lines
    chunks = []
    for msg in sample.get("messages", []):
        for call in msg.get("tool_calls", []) or []:
            args = call.get("arguments", {})
            if call.get("function") == "bash" and isinstance(args.get("cmd"), str):
                chunks.append(args["cmd"])
    return chunks


def recognized_invocations(chunk: str) -> list[tuple[str, int, int]]:
    """Return (head, token count, option count), never retaining arguments."""
    if "\n" in chunk:
        return [item for line in chunk.splitlines() for item in recognized_invocations(line)]
    try:
        lexer = shlex.shlex(chunk, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
    except ValueError:
        return []
    out = []
    known = __import__("attackagent.traces", fromlist=["ATOM_GROUP"]).ATOM_GROUP
    at_start, active = True, None
    token_count = option_count = 0
    for token in tokens + [";"]:
        if token and all(ch in ";&|" for ch in token):
            if active is not None:
                out.append((active, token_count, option_count))
            at_start, active = True, None
            token_count = option_count = 0
            continue
        if at_start:
            base = pathlib.PurePosixPath(token).name.lower()
            if ASSIGNMENT.match(token) or base in WRAPPERS:
                continue
            at_start = False
            if base in known:
                active = base
                token_count = 1
            continue
        if active is not None:
            token_count += 1
            option_count += int(token.startswith("-") and len(token) > 1)
    return out


def main():
    match = module(ROOT / "research/experiments/matched_fingerprint_v1/run.py", "matched_v1")
    rows, tasks, folds = match.build_cth()
    with (ROOT / "research/results/behavior_corpus_v1/trace_registry.csv").open() as f:
        registry = {r["trace_id"]: r for r in csv.DictReader(f)}

    totals = defaultdict(Counter)
    per_trace = []
    sources = {}
    mismatch = []
    all_head_counts = Counter()
    for row in rows:
        meta = registry[row["id"]]
        sample, path = load_sample(meta["record_locator"])
        sources[str(path.relative_to(ROOT))] = digest(path)
        chunks = extract_chunks(sample, row["mode"])
        invocations = [v for chunk in chunks for v in recognized_invocations(chunk)]
        heads = [head for chunk in chunks for head in command_heads(chunk)]
        if heads != row["commands"]:
            mismatch.append(row["id"])
        stat = totals[row["mode"]]
        stat["traces"] += 1
        stat["chunks"] += len(chunks)
        stat["recognized_invocations"] += len(invocations)
        stat["head_events"] += len(heads)
        stat["events_with_arguments"] += sum(tokens > 1 for _, tokens, _ in invocations)
        stat["events_with_options"] += sum(options > 0 for _, _, options in invocations)
        stat["argument_tokens"] += sum(tokens - 1 for _, tokens, _ in invocations)
        all_head_counts.update(heads)
        per_trace.append({"id": row["id"], "task": row["task"], "mode": row["mode"],
                          "head_events": len(heads), "raw_shell_chunks": len(chunks),
                          "recognized_invocations": len(invocations),
                          "invocations_with_arguments": sum(tokens > 1 for _, tokens, _ in invocations),
                          "invocations_with_options": sum(options > 0 for _, _, options in invocations)})

    summary = {
        "status": "aggregate_only_raw_audit",
        "protocol": "frozen 17-task CTH cohort used by matched_fingerprint_v1",
        "task_count": len(tasks), "trace_count": len(rows), "folds": folds,
        "raw_file_count": len(sources), "raw_file_sha256": sources,
        "adapter": "known-command-heads-lossy-v1",
        "trace_adapter_sha256": digest(ROOT / "attackagent/traces.py"),
        "mismatched_head_traces": mismatch,
        "head_vocabulary_size": len(all_head_counts),
        "head_vocabulary_top10_counts": all_head_counts.most_common(10),
        "by_mode": {},
        "per_trace": per_trace,
        "privacy_note": "raw command content and argument values are not saved",
    }
    for mode, counts in totals.items():
        n = counts["recognized_invocations"]
        summary["by_mode"][mode] = {
            **counts,
            "argument_event_fraction": counts["events_with_arguments"] / n if n else None,
            "option_event_fraction": counts["events_with_options"] / n if n else None,
            "mean_argument_tokens_per_invocation": counts["argument_tokens"] / n if n else None,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_command_coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_trace"}, indent=2))


if __name__ == "__main__":
    main()
