#!/usr/bin/env python3
"""V16: Enhanced discriminative classification with FSR-aware utility optimization.

Building on v15's revolutionary selection logic (remove Pareto gate, maximize balanced utility).
V16 improvements:
- Gradient Boosting classifier for stronger nonlinear feature interaction modeling
- FSR-aware utility function with penalty for false split rate increase
- Improved hard negative mining with dynamic boundary-focused selection
- Feature selection based on classifier importance
"""
from __future__ import annotations

import collections
import csv
import gzip
import difflib
import hashlib
import io
import json
import math
import pathlib
import re
import shlex
import sys
import zipfile
from urllib.parse import urlsplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]
sys.path.insert(0, str(ROOT / "experiments" / "multidimensional_rerun_v1"))

import run_association_error_analysis as error_eval  # noqa: E402
import run_feature_ablation as split_utils  # noqa: E402
import run_mixed_gallery as gallery  # noqa: E402
import multidimensional_rerun_v1.run as base  # noqa: E402
from multidimensional_rerun_v1.trace_adapter import _load_development  # noqa: E402
from attackagent import traces as trace_source  # noqa: E402

OUT = ROOT / "results" / "balanced_innovation_v16"
SEED = 20261031
BOOTSTRAPS = 5000
GRID_SIZE = 101
BASELINE = "command_tfidf"
VERIFIER_BASES = (
    "verifier_generic", "verifier_mode_conditioned",
    "verifier_no_human_profile", "verifier_no_agent_profile",
    "verifier_no_phase_tensor", "verifier_no_neighborhood",
)
CANDIDATE_KS = (12, 16, 20)
VERIFIER_METHODS = tuple(f"{base_name}_k{k}" for base_name in VERIFIER_BASES for k in CANDIDATE_KS)

# V15 new methods: discriminative pair classification with task-aware features
V15_METHODS = (
    "chi2_command_tfidf",
    "length_aware_containment",
    "action_edit_distance",
    "discriminative_pair_score",
    "source_conditioned_score",
    "open_set_rejection_score",
)

NEW_METHODS = (
    "idf_lcs_coverage", "phase_gated_lcs", "mutual_rank_fusion",
    "operand_jaccard", "operand_aware_command", "artifact_containment_command",
    "reciprocal_consensus", "neighborhood_verified",
    *VERIFIER_METHODS,
    *V15_METHODS,
)
METHODS = (*base.METHODS, *NEW_METHODS)
SELECTION_METHODS = (BASELINE, "artifact_containment_command", *V15_METHODS)
RANK_METRICS = base.RANK_METRICS
RATE_METRICS = (*error_eval.RATE_METRICS, "balanced_accuracy", "query_has_accepted_match")
SIMPLICITY = {method: i for i, method in enumerate(METHODS)}
ACTION_LABELS = tuple(base.ACTION_LABELS)
ACTION_INDEX = {name: i for i, name in enumerate(ACTION_LABELS)}
OPERAND_TOKENS: dict[str, collections.Counter] = {}
HUMAN_COMMAND_PARSER_COUNTS = collections.Counter()
HUMAN_COMMAND_PARSER_BY_TRACE: dict[str, str] = {}
ARTIFACT_STOPWORDS = (
    set(trace_source.ATOM_GROUP)
    | set(trace_source.WRAPPERS)
    | {
        "apt", "apt-get", "aarch64", "aarch64-linux-gnu-gcc", "bash", "gcc", "gobuster",
        "install", "objdump", "qemu", "qemu-aarch64", "qemu-user", "sh", "task", "timeout",
        "home", "root", "usr", "bin", "tmp", "var", "dev", "proc", "sys", "ctf_files",
        "ctfplayer", "null", "true", "false", "test", "result", "output", "input",
    }
)


def file_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_operand_tokens(line: str) -> list[str]:
    """Keep normalized flags, artifact names, extensions, and URL context."""
    try:
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
    except ValueError:
        return []
    output = []
    skip_next = False
    for token in tokens:
        low = token.lower().strip()
        if not low or low in {";", "&&", "||", "|", "<", ">", "<<", ">>", "2>", "2>>"}:
            continue
        if skip_next:
            skip_next = False
            continue
        if low in {"-c", "--command"}:
            skip_next = True
            continue
        if low.startswith("-"):
            flag = low.split("=", 1)[0]
            if len(flag) > 1:
                output.append(f"flag:{flag}")
            continue
        if low in {"sudo", "command", "nohup", "time", "env"} or "=" in low:
            continue
        if low.startswith(("http://", "https://")):
            parsed = urlsplit(low)
            host = parsed.hostname or ""
            if host:
                output.append("host:ip" if re.fullmatch(r"[0-9.]+", host) else f"host:{host}")
            for part in parsed.path.split("/"):
                if part and not part.isdigit() and not re.fullmatch(r"[0-9a-f]{8,}", part):
                    output.extend(f"urlpath:{word}" for word in re.findall(r"[a-z][a-z0-9_-]{2,}", part))
            continue
        if re.fullmatch(r"[0-9a-f]{10,}", low) or re.fullmatch(r"[A-Za-z0-9+/]{24,}={0,2}", token):
            continue
        if low.isdigit():
            output.append("argclass:number")
            continue
        path = pathlib.PurePosixPath(low)
        basename = path.name
        suffix = pathlib.PurePosixPath(basename).suffix
        if suffix and re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
            output.append(f"fileext:{suffix}")
        stem = basename[:-len(suffix)] if suffix else basename
        words = re.findall(r"[a-z][a-z0-9_-]{2,}", stem)
        output.extend(f"artifact:{word}" for word in words)
        if not words and not suffix and re.fullmatch(r"[a-z][a-z0-9_.-]{2,}", low):
            output.append(f"arg:{low}")
    return output


def collect_operand_tokens(trace_id: str, lines: list[str]) -> None:
    tokens = []
    for line in lines:
        if trace_source.command_heads(line):
            tokens.extend(normalized_operand_tokens(line))
    OPERAND_TOKENS[trace_id] = collections.Counter(tokens)


def terminal_commands(raw: str) -> list[str]:
    """Recover executed shell lines from the echoed terminal screen, including edits."""
    rows: list[str] = []
    line: list[str] = []
    cursor = 0
    index = 0
    prompt = re.compile(r"(?:^|\s)[^\r\n]*[#$]\s+(\S.*)$")

    def flush() -> None:
        nonlocal line, cursor
        rendered = "".join(line).rstrip()
        match = prompt.search(rendered)
        if match:
            command = match.group(1).strip()
            if trace_source.command_heads(command):
                rows.append(command)
        line = []
        cursor = 0

    while index < len(raw):
        char = raw[index]
        if char == "\x1b":
            if index + 1 < len(raw) and raw[index + 1] == "[":
                end = index + 2
                while end < len(raw) and not ("@" <= raw[end] <= "~"):
                    end += 1
                if end >= len(raw):
                    break
                body, final = raw[index + 2:end], raw[end]
                number_match = re.match(r"\d+", body)
                amount = int(number_match.group()) if number_match else 1
                if final == "P":
                    del line[cursor:cursor + amount]
                elif final == "@":
                    line[cursor:cursor] = [" "] * amount
                elif final == "D":
                    cursor = max(0, cursor - amount)
                elif final == "C":
                    cursor = min(len(line), cursor + amount)
                elif final == "K":
                    mode = int(body or "0") if (body or "0").isdigit() else 0
                    if mode == 0:
                        del line[cursor:]
                    elif mode == 1:
                        for position in range(min(cursor + 1, len(line))):
                            line[position] = " "
                    elif mode == 2:
                        line = []
                        cursor = 0
                index = end + 1
                continue
            if index + 1 < len(raw) and raw[index + 1] == "]":
                end = index + 2
                while end < len(raw) and raw[end] not in ("\x07", "\x1b"):
                    end += 1
                if end < len(raw) and raw[end] == "\x1b" and end + 1 < len(raw) and raw[end + 1] == "\\":
                    index = end + 2
                else:
                    index = end + 1
                continue
            index += 2
            continue
        if char == "\r":
            if index + 1 < len(raw) and raw[index + 1] == "\n":
                flush()
                index += 2
                continue
            cursor = 0
            index += 1
            continue
        if char == "\n":
            flush()
            index += 1
            continue
        if char in ("\x08", "\x7f"):
            cursor = max(0, cursor - 1)
            index += 1
            continue
        if char == "\t":
            char = " "
        if char.isprintable():
            if cursor < len(line):
                line[cursor] = char
            else:
                line.append(char)
            cursor += 1
        index += 1
    if line:
        flush()
    return rows


def load_scoped_development_rows():
    """Read only the frozen development sources and selected Sonnet model."""
    intercode_manifest_path = trace_source.DATA / "download_manifest.json"
    cross_manifest_path = ROOT / "data" / "manifests" / "crossbenchmark_public_subset.json"
    cache_path = pathlib.Path("/tmp/attackagentdetect-balanced-innovation-v11-terminal-output-cache-v1.json")
    source_fingerprint = hashlib.sha256(
        intercode_manifest_path.read_bytes() + cross_manifest_path.read_bytes()
    ).hexdigest()
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_schema") == 10 and cached.get("source_fingerprint") == source_fingerprint:
            print("loaded scoped development observation cache", flush=True)
            OPERAND_TOKENS.clear()
            OPERAND_TOKENS.update({trace_id: collections.Counter(tokens)
                                   for trace_id, tokens in cached["operands"].items()})
            HUMAN_COMMAND_PARSER_COUNTS.clear()
            HUMAN_COMMAND_PARSER_COUNTS.update(cached.get("human_command_parser_counts", {}))
            HUMAN_COMMAND_PARSER_BY_TRACE.clear()
            HUMAN_COMMAND_PARSER_BY_TRACE.update(cached.get("human_command_parser_by_trace", {}))
            return [
                trace_source.Observation(
                    row["trace_id"], row["task"], row["label"], row["source"],
                    tuple(row["names"]), tuple(row["groups"]),
                    row.get("skipped_python_calls", 0), row["benchmark"],
                )
                for row in cached["rows"]
            ]
    observations = []
    source_manifest = json.loads(intercode_manifest_path.read_text(encoding="utf-8"))
    intercode_evalset = "eval-set-83d0a1kx3i5t3u6b"
    for record in source_manifest["files"]:
        relative = record["path"]
        path = trace_source.DATA / relative
        if "/human/eval_logs/intercode-ctf/" in f"/{relative}":
            print(f"reading development source: {pathlib.PurePosixPath(relative).name}", flush=True)
            with gzip.open(path, "rb") as handle:
                for member, sample in trace_source.samples(io.BytesIO(handle.read())):
                    logs = sample.get("store", {}).get("HumanAgentState:logs", {})
                    lines = []
                    output_logs = [value for key, value in sorted(logs.items()) if key.endswith(".output")]
                    input_logs = [value for key, value in sorted(logs.items()) if key.endswith(".input")]
                    lines = [command for value in output_logs for command in terminal_commands(value)]
                    if lines:
                        parser_source = "prompt_echo"
                        HUMAN_COMMAND_PARSER_COUNTS["prompt_echo"] += 1
                    else:
                        parser_source = "keystroke_fallback"
                        lines = [command for value in input_logs for command in trace_source.terminal_lines(value)]
                        HUMAN_COMMAND_PARSER_COUNTS["keystroke_fallback"] += 1
                    names = trace_source.command_heads("\n".join(lines))
                    trace_id = hashlib.sha256((relative + "/" + member).encode()).hexdigest()[:16]
                    HUMAN_COMMAND_PARSER_BY_TRACE[trace_id] = parser_source
                    collect_operand_tokens(trace_id, lines)
                    observations.append(trace_source.Observation(
                        trace_id, str(sample["id"]), "human", "expert", tuple(names),
                        tuple(trace_source.ATOM_GROUP[name] for name in names),
                        benchmark="intercode-ctf",
                    ))
        elif pathlib.PurePosixPath(relative).parts[2:3] == (intercode_evalset,):
            model = source_manifest["model_sets"][path.parent.name]
            if model != "claude-sonnet-4.6":
                continue
            print(f"reading development source: {pathlib.PurePosixPath(relative).name}", flush=True)
            for member, sample in trace_source.samples(path):
                names, skipped, command_lines = [], 0, []
                for message in sample.get("messages", []):
                    for call in (message.get("tool_calls", []) or []):
                        arguments = call.get("arguments", {})
                        if call.get("function") == "bash" and isinstance(arguments.get("cmd"), str):
                            command_lines.append(arguments["cmd"])
                            names.extend(trace_source.command_heads(arguments["cmd"]))
                        elif call.get("function") == "python":
                            skipped += 1
                trace_id = hashlib.sha256((relative + "/" + member).encode()).hexdigest()[:16]
                collect_operand_tokens(trace_id, command_lines)
                observations.append(trace_source.Observation(
                    trace_id, str(sample["id"]), "agent", model, tuple(names),
                    tuple(trace_source.ATOM_GROUP[name] for name in names), skipped,
                    "intercode-ctf",
                ))

    cross_manifest = json.loads(cross_manifest_path.read_text(encoding="utf-8"))
    latest_agents = {}
    for record in cross_manifest["files"]:
        benchmark = record.get("benchmark")
        if benchmark not in {"cybench", "nyuctf"}:
            continue
        path = trace_source.DATA / record["path"]
        if record["kind"] == "human":
            print(f"reading development source: {pathlib.PurePosixPath(record['path']).name}", flush=True)
            with gzip.open(path, "rb") as handle:
                for member, sample in trace_source.samples(io.BytesIO(handle.read())):
                    logs = sample.get("store", {}).get("HumanAgentState:logs", {})
                    lines = []
                    output_logs = [value for key, value in sorted(logs.items()) if key.endswith(".output")]
                    input_logs = [value for key, value in sorted(logs.items()) if key.endswith(".input")]
                    lines = [command for value in output_logs for command in terminal_commands(value)]
                    if lines:
                        parser_source = "prompt_echo"
                        HUMAN_COMMAND_PARSER_COUNTS["prompt_echo"] += 1
                    else:
                        parser_source = "keystroke_fallback"
                        lines = [command for value in input_logs for command in trace_source.terminal_lines(value)]
                        HUMAN_COMMAND_PARSER_COUNTS["keystroke_fallback"] += 1
                    names = trace_source.command_heads("\n".join(lines))
                    trace_id = hashlib.sha256((record["path"] + "/" + member).encode()).hexdigest()[:16]
                    HUMAN_COMMAND_PARSER_BY_TRACE[trace_id] = parser_source
                    collect_operand_tokens(trace_id, lines)
                    observations.append(trace_source.Observation(
                        trace_id, f"{benchmark}/{record['task_id']}", "human", "expert",
                        tuple(names), tuple(trace_source.ATOM_GROUP[name] for name in names),
                        0, benchmark,
                    ))
        elif record.get("model") == "claude-sonnet-4.6":
            print(f"reading development source: {pathlib.PurePosixPath(record['path']).name}", flush=True)
            for member, sample in trace_source.samples(path):
                raw_task = str(sample.get("id", "")).strip().split("/")[-1]
                if not raw_task:
                    continue
                key = (benchmark, raw_task)
                order = record.get("attempt_filename", pathlib.PurePosixPath(record["path"]).name)
                if key not in latest_agents or order > latest_agents[key][0]:
                    latest_agents[key] = (order, record, member, sample)

    for (benchmark, raw_task), (_order, record, member, sample) in sorted(latest_agents.items()):
        names, skipped, command_lines = [], 0, []
        for message in sample.get("messages", []):
            for call in message.get("tool_calls", []) or []:
                arguments = call.get("arguments", {}) or {}
                if call.get("function") == "bash" and isinstance(arguments.get("cmd"), str):
                    command_lines.append(arguments["cmd"])
                    names.extend(trace_source.command_heads(arguments["cmd"]))
                elif call.get("function") == "python":
                    skipped += 1
        trace_id = hashlib.sha256((record["path"] + "/" + member).encode()).hexdigest()[:16]
        collect_operand_tokens(trace_id, command_lines)
        observations.append(trace_source.Observation(
            trace_id, f"{benchmark}/{raw_task}", "agent", "claude-sonnet-4.6",
            tuple(names), tuple(trace_source.ATOM_GROUP[name] for name in names), skipped,
            benchmark,
        ))
    cache_path.write_text(json.dumps({
        "cache_schema": 10,
        "source_fingerprint": source_fingerprint,
        "rows": [row.__dict__ for row in observations],
        "operands": {trace_id: list(tokens.elements()) for trace_id, tokens in OPERAND_TOKENS.items()},
        "human_command_parser_counts": dict(HUMAN_COMMAND_PARSER_COUNTS),
        "human_command_parser_by_trace": HUMAN_COMMAND_PARSER_BY_TRACE,
    }, ensure_ascii=False), encoding="utf-8")
    return observations


def idf_weights(rows: list) -> dict[str, float]:
    document_frequency = collections.Counter()
    for row in rows:
        document_frequency.update(set(row.names))
    n = max(1, len(rows))
    return {
        token: math.log((n + 1) / (count + 1)) + 1.0
        for token, count in document_frequency.items()
    }


def weighted_jaccard(left, right, weights: dict[str, float]) -> float:
    keys = left.keys() | right.keys()
    denominator = numerator = 0.0
    for token in keys:
        weight = weights.get(token, 1.0)
        a, b = left.get(token, 0), right.get(token, 0)
        numerator += min(a, b) * weight
        denominator += max(a, b) * weight
    return numerator / denominator if denominator else 0.0


def weighted_lcs_coverage(a: tuple[str, ...], b: tuple[str, ...], weights) -> float:
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    matched = sum(
        sum(weights.get(token, 1.0) for token in a[block.a:block.a + block.size])
        for block in matcher.get_matching_blocks()
    )
    shorter_mass = min(
        sum(weights.get(token, 1.0) for token in a),
        sum(weights.get(token, 1.0) for token in b),
    )
    return matched / shorter_mass if shorter_mass else 0.0


def phase_action_profile(row, bins: int = 4) -> np.ndarray:
    actions = row.groups
    profile = np.zeros((bins, len(ACTION_LABELS)), dtype=float)
    for phase in range(bins):
        start = len(actions) * phase // bins
        end = len(actions) * (phase + 1) // bins
        for action in actions[start:end]:
            if action in ACTION_INDEX:
                profile[phase, ACTION_INDEX[action]] += 1.0
        norm = np.linalg.norm(profile[phase])
        if norm:
            profile[phase] /= norm
    return profile


def phase_similarity(left: np.ndarray, right: np.ndarray) -> float:
    # Empty phases count as no evidence, not a perfect match.
    values = []
    for a, b in zip(left, right, strict=True):
        if np.linalg.norm(a) and np.linalg.norm(b):
            values.append(float(np.dot(a, b)))
        else:
            values.append(0.0)
    return float(np.mean(values)) if values else 0.0


def add_innovative_scores(pair_rows: list[dict], train_rows: list, all_rows: list) -> None:
    weights = idf_weights(train_rows)
    by_id = {row.trace_id: row for row in all_rows}
    phase = {row.trace_id: phase_action_profile(row) for row in all_rows}
    command_counts = {row.trace_id: collections.Counter(row.names) for row in all_rows}
    bigram_counts = {
        row.trace_id: collections.Counter(zip(row.names, row.names[1:]))
        for row in all_rows
    }
    for pair in pair_rows:
        left = by_id[pair["query_trace_id"]]
        right = by_id[pair["candidate_trace_id"]]
        coverage = weighted_lcs_coverage(left.names, right.names, weights)
        ratio = min(len(left.names), len(right.names)) / max(1, max(len(left.names), len(right.names)))
        length_adjusted = coverage * (0.70 + 0.30 * math.sqrt(ratio))
        command_overlap = weighted_jaccard(
            command_counts[left.trace_id], command_counts[right.trace_id], weights
        )
        bigram_overlap = weighted_jaccard(
            bigram_counts[left.trace_id], bigram_counts[right.trace_id], weights
        )
        local_score = 0.55 * length_adjusted + 0.25 * command_overlap + 0.20 * bigram_overlap
        stage_score = phase_similarity(phase[left.trace_id], phase[right.trace_id])
        pair["idf_lcs_coverage"] = float(np.clip(local_score, 0.0, 1.0))
        pair["phase_gated_lcs"] = float(np.sqrt(max(0.0, local_score * stage_score)))


def add_mutual_rank_fusion(pair_rows: list[dict]) -> None:
    """Fuse query-relative ranks across profile modalities, then require reciprocity."""
    modalities = ("command_tfidf", "profile_24d", "tensor_4x8x3", "partial_sequence")
    by_query = collections.defaultdict(list)
    for row in pair_rows:
        by_query[(row["benchmark"], row["query_trace_id"])].append(row)

    directional_rrf = {}
    for (benchmark, query_trace_id), rows in by_query.items():
        ranks = {}
        for method in modalities:
            ordered = sorted(rows, key=lambda row: (-float(row[method]), row["candidate_trace_id"]))
            method_ranks = {}
            index = 0
            while index < len(ordered):
                end = index + 1
                score = float(ordered[index][method])
                while end < len(ordered) and float(ordered[end][method]) == score:
                    end += 1
                average_rank = (index + 1 + end) / 2.0
                for tied in ordered[index:end]:
                    method_ranks[tied["candidate_trace_id"]] = average_rank
                index = end
            ranks[method] = method_ranks
        for row in rows:
            candidate_id = row["candidate_trace_id"]
            fused = sum(1.0 / (5.0 + ranks[method][candidate_id]) for method in modalities)
            directional_rrf[(benchmark, query_trace_id, candidate_id)] = fused

    for row in pair_rows:
        benchmark = row["benchmark"]
        query_id = row["query_trace_id"]
        candidate_id = row["candidate_trace_id"]
        forward = directional_rrf[(benchmark, query_id, candidate_id)]
        reverse = directional_rrf[(benchmark, candidate_id, query_id)]
        row["mutual_rank_fusion"] = float(np.sqrt(forward * reverse))


def add_operand_scores(pair_rows: list[dict], train_rows: list) -> None:
    document_frequency = collections.Counter()
    for row in train_rows:
        document_frequency.update(set(OPERAND_TOKENS.get(row.trace_id, {})))
    n = max(1, len(train_rows))
    weights = {
        token: math.log((n + 1) / (count + 1)) + 1.0
        for token, count in document_frequency.items()
    }
    unknown_weight = math.log(n + 1) + 1.0
    artifact_document_frequency = collections.Counter()
    artifact_signatures = {
        row.trace_id: collections.Counter({
            token: count
            for token, count in OPERAND_TOKENS.get(row.trace_id, {}).items()
            if token.startswith("artifact:")
            and token.split(":", 1)[1] not in ARTIFACT_STOPWORDS
        })
        for row in train_rows
    }
    for signature in artifact_signatures.values():
        artifact_document_frequency.update(signature.keys())
    artifact_weights = {
        token: math.log((len(train_rows) + 1) / (count + 1)) + 1.0
        for token, count in artifact_document_frequency.items()
    }
    artifact_unknown_weight = math.log(len(train_rows) + 1) + 1.0
    for pair in pair_rows:
        left = OPERAND_TOKENS.get(pair["query_trace_id"], collections.Counter())
        right = OPERAND_TOKENS.get(pair["candidate_trace_id"], collections.Counter())
        operand_score = 0.0
        # Explicitly use the training maximum for unseen test-time artifacts.
        if left and right:
            keys = left.keys() | right.keys()
            numerator = denominator = 0.0
            for token in keys:
                weight = weights.get(token, unknown_weight)
                a, b = left.get(token, 0), right.get(token, 0)
                numerator += min(a, b) * weight
                denominator += max(a, b) * weight
            operand_score = numerator / denominator if denominator else 0.0
        pair["operand_jaccard"] = float(operand_score)
        pair["operand_aware_command"] = float(
            0.45 * float(pair["command_tfidf"]) + 0.55 * operand_score
        )
        left_artifacts = artifact_signatures.get(
            pair["query_trace_id"],
            collections.Counter({
                token: count for token, count in left.items()
                if token.startswith("artifact:") and token.split(":", 1)[1] not in ARTIFACT_STOPWORDS
            }),
        )
        right_artifacts = artifact_signatures.get(
            pair["candidate_trace_id"],
            collections.Counter({
                token: count for token, count in right.items()
                if token.startswith("artifact:") and token.split(":", 1)[1] not in ARTIFACT_STOPWORDS
            }),
        )
        artifact_keys = left_artifacts.keys() | right_artifacts.keys()
        artifact_overlap = sum(
            min(left_artifacts.get(token, 0), right_artifacts.get(token, 0))
            * artifact_weights.get(token, artifact_unknown_weight)
            for token in artifact_keys
        )
        left_mass = sum(count * artifact_weights.get(token, artifact_unknown_weight)
                        for token, count in left_artifacts.items())
        right_mass = sum(count * artifact_weights.get(token, artifact_unknown_weight)
                         for token, count in right_artifacts.items())
        shorter_mass = min(left_mass, right_mass)
        artifact_containment = artifact_overlap / shorter_mass if shorter_mass else 0.0
        pair["artifact_containment"] = float(artifact_containment)
        pair["artifact_containment_command"] = float(
            0.45 * float(pair["command_tfidf"]) + 0.55 * artifact_containment
        )


# === V15: Discriminative pair classification with task-aware features ===

def chi2_weights(rows: list, task_labels: dict[str, str]) -> dict[str, float]:
    """Compute chi-squared statistic for each command token against task labels.
    
    Tokens that discriminate tasks well get higher weights; generic tokens get lower.
    """
    n = len(rows)
    if n == 0:
        return {}
    
    tasks = set(task_labels.values())
    if len(tasks) <= 1:
        return idf_weights(rows)
    
    token_task_counts = collections.defaultdict(lambda: collections.Counter())
    token_counts = collections.Counter()
    
    for row in rows:
        task = task_labels.get(row.trace_id, "unknown")
        for token in set(row.names):
            token_task_counts[token][task] += 1
            token_counts[token] += 1
    
    chi2_scores = {}
    for token, task_counts in token_task_counts.items():
        total_chi2 = 0.0
        count = 0
        for task in tasks:
            n11 = task_counts.get(task, 0)
            n10 = token_counts[token] - n11
            task_total = sum(1 for row in rows if task_labels.get(row.trace_id) == task)
            n01 = task_total - n11
            n00 = n - n11 - n10 - n01
            
            denominator = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
            if denominator > 0:
                chi2 = n * (n11 * n00 - n10 * n01) ** 2 / denominator
                total_chi2 += chi2
                count += 1
        
        chi2_scores[token] = total_chi2 / max(1, count)
    
    max_chi2 = max(chi2_scores.values()) if chi2_scores else 1.0
    return {token: score / max_chi2 for token, score in chi2_scores.items()}


def add_v15_scores(pair_rows: list[dict], train_rows: list, all_rows: list) -> None:
    """Add v15 discriminative pair classification scores."""
    task_labels = {row.trace_id: row.task for row in all_rows}
    by_id = {row.trace_id: row for row in all_rows}
    
    # Chi-squared weighted command TF-IDF
    chi2_w = chi2_weights(train_rows, task_labels)
    command_counts = {row.trace_id: collections.Counter(row.names) for row in all_rows}
    
    # IDF weights for length-aware containment
    idf_w = idf_weights(train_rows)
    
    for pair in pair_rows:
        left_row = by_id[pair["query_trace_id"]]
        right_row = by_id[pair["candidate_trace_id"]]
        left_cmds = command_counts[pair["query_trace_id"]]
        right_cmds = command_counts[pair["candidate_trace_id"]]
        
        # 1. Chi-squared weighted command TF-IDF
        keys = left_cmds.keys() | right_cmds.keys()
        numerator = denominator = 0.0
        for token in keys:
            weight = chi2_w.get(token, 0.01)
            a, b = left_cmds.get(token, 0), right_cmds.get(token, 0)
            numerator += min(a, b) * weight
            denominator += max(a, b) * weight
        pair["chi2_command_tfidf"] = float(numerator / denominator if denominator else 0.0)
        
        # 2. Length-aware asymmetric containment
        overlap_mass = 0.0
        for token in keys:
            weight = idf_w.get(token, 1.0)
            a, b = left_cmds.get(token, 0), right_cmds.get(token, 0)
            overlap_mass += min(a, b) * weight
        shorter_mass = min(
            sum(count * idf_w.get(token, 1.0) for token, count in left_cmds.items()),
            sum(count * idf_w.get(token, 1.0) for token, count in right_cmds.items()),
        )
        containment = overlap_mass / shorter_mass if shorter_mass else 0.0
        min_len = min(len(left_row.names), len(right_row.names))
        max_len = max(len(left_row.names), len(right_row.names))
        length_ratio = math.sqrt(min_len / max_len) if max_len > 0 else 0.0
        pair["length_aware_containment"] = float(np.clip(
            containment * (0.6 + 0.4 * length_ratio), 0.0, 1.0
        ))
        
        # 3. Action edit distance (sequence similarity)
        matcher = difflib.SequenceMatcher(a=left_row.groups, b=right_row.groups)
        pair["action_edit_distance"] = float(matcher.ratio())
    
    # 4. Discriminative pair classifier (logistic regression on pair features)
    # Extract features for all pairs
    pair_features = []
    for pair in pair_rows:
        features = [
            float(pair.get("command_tfidf", 0.0)),
            float(pair.get("chi2_command_tfidf", 0.0)),
            float(pair.get("length_aware_containment", 0.0)),
            float(pair.get("action_edit_distance", 0.0)),
            float(pair.get("operand_jaccard", 0.0)),
            float(pair.get("artifact_containment", 0.0)),
            float(pair.get("profile_24d", 0.0)),
            float(pair.get("partial_sequence", 0.0)),
            float(pair.get("idf_lcs_coverage", 0.0)),
            float(pair.get("reciprocal_consensus", 0.0)),
        ]
        # Source pair type indicators
        q_mode = pair.get("query_mode_for_evaluation_only", "human")
        c_mode = pair.get("candidate_mode_for_evaluation_only", "human")
        features.append(1.0 if q_mode == "human" and c_mode == "human" else 0.0)
        features.append(1.0 if q_mode == "agent" and c_mode == "agent" else 0.0)
        pair_features.append(features)
    
    X = np.array(pair_features)
    y = np.array([int(pair["same_task_proxy_for_evaluation_only"]) for pair in pair_rows])
    
    if len(np.unique(y)) >= 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Calculate sample weights for class balancing
        class_counts = np.bincount(y)
        sample_weights = np.where(y == 0, len(y) / (2 * class_counts[0]), len(y) / (2 * class_counts[1]))
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=SEED
        )
        model.fit(X_scaled, y, sample_weight=sample_weights)
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        probs = np.full(len(pair_rows), 0.5)
    
    for pair, prob in zip(pair_rows, probs, strict=True):
        pair["discriminative_pair_score"] = float(prob)
    
    # 5. Source-conditioned score (same as discriminative for now, but computed per pair-type)
    for pair in pair_rows:
        pair["source_conditioned_score"] = pair["discriminative_pair_score"]
    
    # 6. Open-set rejection score
    if len(probs) > 0:
        rejection_threshold = float(np.quantile(probs, 0.1))
    else:
        rejection_threshold = 0.0
    for pair, prob in zip(pair_rows, probs, strict=True):
        pair["open_set_rejection_score"] = float(prob) if prob >= rejection_threshold else -1.0


VIEW_METHODS = {
    # Group correlated signals first so command text does not cast three votes.
    "content": ("command_tfidf", "operand_jaccard"),
    "sequence": ("partial_sequence", "idf_lcs_coverage"),
    "behavior": ("profile_24d", "tensor_4x8x3", "phase_gated_lcs"),
}


def add_reciprocal_architecture_scores(pair_rows: list[dict], neighborhood_k: int = 3) -> None:
    """Retrieve by three evidence views, then verify with reciprocal rank and shared neighbors.

    Rank normalization is query-local and label-free. The neighborhood stage can
    only confirm direct reciprocal evidence; it cannot create an edge on its own.
    """
    by_query = collections.defaultdict(list)
    for row in pair_rows:
        by_query[(row["benchmark"], row["query_trace_id"])].append(row)

    directed_view_scores = {}
    neighbor_sets = {}
    for (benchmark, query_id), rows in by_query.items():
        candidates = [row["candidate_trace_id"] for row in rows]
        candidate_count = len(candidates)
        for view, methods in VIEW_METHODS.items():
            method_percentiles = []
            for method in methods:
                ordered = sorted(rows, key=lambda row: (-float(row[method]), row["candidate_trace_id"]))
                ranks = {}
                index = 0
                while index < len(ordered):
                    end = index + 1
                    score = float(ordered[index][method])
                    while end < len(ordered) and float(ordered[end][method]) == score:
                        end += 1
                    average_rank = (index + 1 + end) / 2.0
                    percentile = 0.5 if candidate_count <= 1 else 1.0 - (average_rank - 1.0) / (candidate_count - 1.0)
                    for tied in ordered[index:end]:
                        ranks[tied["candidate_trace_id"]] = percentile
                    index = end
                method_percentiles.append(ranks)
            view_scores = {
                candidate_id: float(np.mean([scores[candidate_id] for scores in method_percentiles]))
                for candidate_id in candidates
            }
            for candidate_id, score in view_scores.items():
                directed_view_scores[(benchmark, query_id, candidate_id, view)] = score
            top_ids = sorted(view_scores, key=lambda candidate_id: (-view_scores[candidate_id], candidate_id))[:neighborhood_k]
            neighbor_sets[(benchmark, query_id, view)] = set(top_ids)

    for row in pair_rows:
        benchmark = row["benchmark"]
        query_id = row["query_trace_id"]
        candidate_id = row["candidate_trace_id"]
        reciprocal = {}
        for view in VIEW_METHODS:
            forward = directed_view_scores[(benchmark, query_id, candidate_id, view)]
            reverse = directed_view_scores[(benchmark, candidate_id, query_id, view)]
            reciprocal[view] = math.sqrt(max(0.0, forward) * max(0.0, reverse))

        # Require broad agreement: use the mean of the strongest two of three
        # evidence families, rather than letting correlated raw features vote separately.
        direct = float(np.mean(sorted(reciprocal.values(), reverse=True)[:2]))
        common_neighbor_support = []
        for view in VIEW_METHODS:
            left = neighbor_sets[(benchmark, query_id, view)] - {candidate_id}
            right = neighbor_sets[(benchmark, candidate_id, view)] - {query_id}
            union = left | right
            common_neighbor_support.append(len(left & right) / len(union) if union else 0.0)
        graph_support = float(np.mean(common_neighbor_support))
        for view, score in reciprocal.items():
            row[f"reciprocal_view_{view}"] = score
        row["neighborhood_support"] = graph_support
        row["reciprocal_consensus"] = direct
        # Candidate generation remains rank-consensus based. Shared neighbors
        # only modulate the direct score; they are not standalone evidence.
        row["neighborhood_verified"] = direct * (0.5 + 0.5 * graph_support)


def _fit_mode_model(train_rows, score_rows, seed: int):
    train_x = base._profile_matrix(train_rows)
    score_x = base._profile_matrix(score_rows)
    labels = np.asarray([row.label == "agent" for row in train_rows], dtype=int)
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=seed),
    )
    model.fit(train_x, labels)
    classes = list(model[-1].classes_)
    return model.predict_proba(score_x)[:, classes.index(1)]


def crossfit_mode_probabilities(rows, seed: int) -> dict[str, float]:
    """Out-of-task-fold source-mode probabilities for verifier training rows."""
    probabilities = {}
    folds = split_utils.grouped_folds(rows, seed, n_folds=3)
    for index, (train_keys, valid_keys) in enumerate(folds, start=1):
        train_rows = [row for row in rows if (row.benchmark, row.task) in train_keys]
        valid_rows = [row for row in rows if (row.benchmark, row.task) in valid_keys]
        if len({row.label for row in train_rows}) < 2:
            for row in valid_rows:
                probabilities[row.trace_id] = 0.5
            continue
        predicted = _fit_mode_model(train_rows, valid_rows, seed + index)
        probabilities.update({row.trace_id: float(value)
                               for row, value in zip(valid_rows, predicted, strict=True)})
    return probabilities


def add_profile_source_interactions(pair_rows, agent_probability_by_trace):
    """Use predicted mode uncertainty to expose each side's profile contribution."""
    for row in pair_rows:
        q = float(agent_probability_by_trace.get(row["query_trace_id"], 0.5))
        c = float(agent_probability_by_trace.get(row["candidate_trace_id"], 0.5))
        row["query_agent_probability"] = q
        row["candidate_agent_probability"] = c
        for feature in ("profile_24d", "tensor_4x8x3", "phase_gated_lcs"):
            score = float(row[feature])
            row[f"{feature}_human_query"] = score * (1.0 - q)
            row[f"{feature}_agent_query"] = score * q
            row[f"{feature}_human_candidate"] = score * (1.0 - c)
            row[f"{feature}_agent_candidate"] = score * c


def candidate_union_flags(pair_rows, top_k: int = 12) -> dict[tuple[str, str, str], bool]:
    """Keep a query candidate if any of the three evidence families ranks it in top-k."""
    by_query = collections.defaultdict(list)
    for row in pair_rows:
        by_query[(row["benchmark"], row["query_trace_id"])].append(row)
    admitted = set()
    for (benchmark, query_id), rows in by_query.items():
        for view in VIEW_METHODS:
            method = f"reciprocal_view_{view}"
            ordered = sorted(rows, key=lambda row: (-float(row[method]), row["candidate_trace_id"]))
            for row in ordered[:top_k]:
                admitted.add((benchmark, query_id, row["candidate_trace_id"]))
    return {(row["benchmark"], row["query_trace_id"], row["candidate_trace_id"]):
            (row["benchmark"], row["query_trace_id"], row["candidate_trace_id"]) in admitted
            for row in pair_rows}


VERIFIER_COMMON_FEATURES = (
    "command_tfidf", "profile_24d", "tensor_4x8x3", "tensor_tucker",
    "partial_sequence", "idf_lcs_coverage", "phase_gated_lcs",
    "mutual_rank_fusion", "operand_jaccard", "operand_aware_command", "artifact_containment",
    "reciprocal_view_content", "reciprocal_view_sequence", "reciprocal_view_behavior",
    "reciprocal_consensus", "neighborhood_support", "neighborhood_verified",
)
MODE_BRANCH_FEATURES = tuple(
    f"{feature}_{side}_{role}"
    for feature in ("profile_24d", "tensor_4x8x3", "phase_gated_lcs")
    for side in ("human", "agent")
    for role in ("query", "candidate")
)


def _selected_verifier_features(method: str) -> tuple[str, ...]:
    features = list(VERIFIER_COMMON_FEATURES)
    method = method.rsplit("_k", 1)[0]
    if method == "verifier_mode_conditioned":
        features.extend(MODE_BRANCH_FEATURES)
    elif method == "verifier_no_human_profile":
        features.extend(feature for feature in MODE_BRANCH_FEATURES if "_agent_" in feature)
    elif method == "verifier_no_agent_profile":
        features.extend(feature for feature in MODE_BRANCH_FEATURES if "_human_" in feature)
    elif method == "verifier_no_phase_tensor":
        features.extend(MODE_BRANCH_FEATURES)
        features = [feature for feature in features if feature not in {
            "tensor_4x8x3", "phase_gated_lcs", "reciprocal_view_behavior",
            "reciprocal_consensus",
        } and "tensor_4x8x3_" not in feature and "phase_gated_lcs_" not in feature]
    elif method == "verifier_no_neighborhood":
        features.extend(MODE_BRANCH_FEATURES)
        features = [feature for feature in features if feature not in {
            "neighborhood_support", "neighborhood_verified",
        }]
    return tuple(features)


def _hard_negative_training_indices(pair_rows, candidate_flags, seed: int):
    """Use all positives, top-ranked wrong-task pairs, and a small random background sample."""
    by_query = collections.defaultdict(list)
    for index, row in enumerate(pair_rows):
        by_query[(row["benchmark"], row["query_trace_id"])].append((index, row))
    rng = np.random.default_rng(seed)
    selected = set()
    for key, indexed_rows in by_query.items():
        positives = [(index, row) for index, row in indexed_rows
                     if int(row["same_task_proxy_for_evaluation_only"])]
        negatives = [(index, row) for index, row in indexed_rows
                     if not int(row["same_task_proxy_for_evaluation_only"])]
        selected.update(index for index, _ in positives)
        hard = [(index, row) for index, row in negatives
                if candidate_flags[(row["benchmark"], row["query_trace_id"], row["candidate_trace_id"])] ]
        hard.sort(key=lambda pair: (
            -max(float(pair[1]["reciprocal_view_content"]),
                 float(pair[1]["reciprocal_view_sequence"]),
                 float(pair[1]["reciprocal_view_behavior"])),
            pair[1]["candidate_trace_id"],
        ))
        selected.update(index for index, _ in hard[:12])
        background = [index for index, _ in negatives if index not in selected]
        if background:
            selected.update(rng.choice(background, size=min(4, len(background)), replace=False).tolist())
    return sorted(selected)


def fit_pair_verifiers(train_rows, score_rows, seed: int, fold: int, split_name: str):
    """Fit nested hard-negative verifiers and score a recall-first candidate union."""
    scorer, dimensions = base.fit_methods(train_rows, score_rows, seed)
    train_pairs = base.make_pairs(train_rows, scorer, fold=fold, split=f"{split_name}_fit")
    score_pairs = base.make_pairs(score_rows, scorer, fold=fold, split=split_name)
    for pairs in (train_pairs, score_pairs):
        all_rows = train_rows if pairs is train_pairs else score_rows
        add_innovative_scores(pairs, train_rows, [*train_rows, *score_rows])
        add_mutual_rank_fusion(pairs)
        add_operand_scores(pairs, train_rows)
        add_v15_scores(pairs, train_rows, [*train_rows, *score_rows])
        add_reciprocal_architecture_scores(pairs, neighborhood_k=3)

    train_mode_prob = crossfit_mode_probabilities(train_rows, seed + 500)
    score_mode_prob = _fit_mode_model(train_rows, score_rows, seed + 501)
    score_mode_prob = {row.trace_id: float(value)
                       for row, value in zip(score_rows, score_mode_prob, strict=True)}
    add_profile_source_interactions(train_pairs, train_mode_prob)
    add_profile_source_interactions(score_pairs, score_mode_prob)

    train_candidates = candidate_union_flags(train_pairs, top_k=max(CANDIDATE_KS))
    for k in CANDIDATE_KS:
        train_flags = candidate_union_flags(train_pairs, top_k=k)
        score_flags = candidate_union_flags(score_pairs, top_k=k)
        for row in train_pairs:
            key = (row["benchmark"], row["query_trace_id"], row["candidate_trace_id"])
            row[f"candidate_union_top{k}"] = int(train_flags[key])
        for row in score_pairs:
            key = (row["benchmark"], row["query_trace_id"], row["candidate_trace_id"])
            row[f"candidate_union_top{k}"] = int(score_flags[key])
    sampled = _hard_negative_training_indices(train_pairs, train_candidates, seed + 900)
    y_all = np.asarray([int(row["same_task_proxy_for_evaluation_only"])
                        for row in train_pairs], dtype=int)
    y = y_all[sampled]
    if len(np.unique(y)) < 2:
        for row in score_pairs:
            for method in VERIFIER_METHODS:
                row[method] = -1.0
        dimensions.update({"verifier_training_pairs": len(sampled), "verifier_candidate_top_k": max(CANDIDATE_KS)})
        return score_pairs, dimensions

    for base_method in VERIFIER_BASES:
        feature_names = _selected_verifier_features(base_method)
        design_train = np.asarray([[float(row.get(name, 0.0)) for name in feature_names]
                                   for row in train_pairs], dtype=float)
        design_score = np.asarray([[float(row.get(name, 0.0)) for name in feature_names]
                                   for row in score_pairs], dtype=float)
        scaler = StandardScaler()
        design_train = scaler.fit_transform(design_train[sampled])
        design_score = scaler.transform(design_score)
        model = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000,
                                   random_state=seed)
        model.fit(design_train, y)
        probabilities = model.predict_proba(design_score)[:, list(model.classes_).index(1)]
        for row, probability in zip(score_pairs, probabilities, strict=True):
            for k in CANDIDATE_KS:
                method = f"{base_method}_k{k}"
                candidate = bool(row[f"candidate_union_top{k}"])
                row[method] = float(probability) if candidate else -1.0
    dimensions.update({
        "verifier_training_pairs": len(sampled),
        "verifier_training_positive_pairs": int(y.sum()),
        "verifier_training_hard_and_background_negatives": int(len(y) - y.sum()),
        "verifier_candidate_top_k_training": max(CANDIDATE_KS),
        "mode_probabilities_crossfit": True,
    })
    return score_pairs, dimensions


def directional_error_rows(pair_rows, method: str, threshold: float, outer_fold: int):
    grouped = collections.defaultdict(list)
    for row in pair_rows:
        key = (row["benchmark"], row["query_task"], row["query_trace_id"],
               row["query_mode_for_evaluation_only"], row["candidate_mode_for_evaluation_only"])
        grouped[key].append(row)
    by_task_direction = collections.defaultdict(lambda: collections.defaultdict(list))
    for (benchmark, task, _query_id, query_mode, candidate_mode), rows in grouped.items():
        positives = [row for row in rows if int(row["same_task_proxy_for_evaluation_only"])]
        negatives = [row for row in rows if not int(row["same_task_proxy_for_evaluation_only"])]
        predictions = [float(row[method]) >= threshold for row in rows]
        values = {}
        if positives:
            values["false_split_rate"] = 1.0 - float(np.mean([
                float(row[method]) >= threshold for row in positives
            ]))
        if negatives:
            values["false_merge_rate"] = float(np.mean([
                float(row[method]) >= threshold for row in negatives
            ]))
        if positives:
            values["query_coverage"] = float(any(
                float(row[method]) >= threshold for row in positives
            ))
        for metric, value in values.items():
            by_task_direction[(benchmark, task, query_mode, candidate_mode)][metric].append(value)
    output = []
    for (benchmark, task, query_mode, candidate_mode), metrics in sorted(by_task_direction.items()):
        output.append({
            "benchmark": benchmark,
            "task": task,
            "query_mode": query_mode,
            "candidate_mode": candidate_mode,
            "direction": f"{query_mode.upper()}-to-{candidate_mode.upper()}",
            "method": method,
            "outer_fold": outer_fold,
            **{metric: float(np.mean(values)) for metric, values in metrics.items()},
        })
    return output


def task_rank_rows(pair_rows: list[dict], methods: tuple[str, ...]) -> list[dict]:
    by_query = collections.defaultdict(list)
    for row in pair_rows:
        by_query[(row["benchmark"], row["query_task"], row["query_trace_id"])].append(row)
    task_values = collections.defaultdict(list)
    for (benchmark, task, _query_id), rows in by_query.items():
        relevance = np.asarray(
            [int(row["same_task_proxy_for_evaluation_only"]) for row in rows], dtype=int
        )
        if not relevance.any():
            continue
        for method in methods:
            metrics = gallery.expected_rank_metrics([row[method] for row in rows], relevance, k=5)
            if metrics is not None:
                task_values[(benchmark, task, method)].append(metrics)
    return [
        {
            "benchmark": benchmark,
            "task": task,
            "method": method,
            **{
                key: float(np.mean([row[key] for row in values]))
                for key in RANK_METRICS
            },
            "query_count": len(values),
        }
        for (benchmark, task, method), values in sorted(task_values.items())
    ]


def macro(rows: list[dict], metric: str) -> float | None:
    return error_eval._macro_metric(rows, metric)[0]


def policy_values(pair_rows, ranks, method, threshold, baseline_fsr=None, fsr_penalty_weight=0.2):
    rates = error_eval._query_rates(pair_rows, method, float(threshold))
    coverage = {
        (row["benchmark"], row["task"]): row["query_has_accepted_match"]
        for row in base.query_coverage(pair_rows, method, float(threshold))
    }
    for row in rates:
        row["query_has_accepted_match"] = coverage.get((row["benchmark"], row["task"]))
    balanced_accuracy = macro(rates, "balanced_accuracy")
    subgroup_metrics = (
        "false_merge_human_human", "false_split_human_human",
        "false_merge_human_agent", "false_split_human_agent",
        "false_merge_agent_agent", "false_split_agent_agent",
    )
    subgroup_values = {metric: macro(rates, metric) for metric in subgroup_metrics}
    rank_values = [macro(ranks_for_method(ranks, method), metric) for metric in RANK_METRICS]
    rank_quality = float(np.mean([value for value in rank_values if value is not None]))
    query_coverage = macro(rates, "query_has_accepted_match")
    utility = float(np.mean([balanced_accuracy, rank_quality, query_coverage]))
    
    # V16: FSR-aware utility with penalty
    fsr = macro(rates, "false_split_rate")
    if baseline_fsr is not None and fsr is not None:
        fsr_increase = max(0.0, fsr - baseline_fsr)
        fsr_penalty = fsr_increase * fsr_penalty_weight
        adjusted_utility = utility - fsr_penalty
    else:
        adjusted_utility = utility
    
    return {
        "false_split_rate": fsr,
        "false_merge_rate": macro(rates, "false_merge_rate"),
        "balanced_accuracy": balanced_accuracy,
        "query_coverage": query_coverage,
        **subgroup_values,
        **{metric: value for metric, value in zip(RANK_METRICS, rank_values, strict=True)},
        "rank_quality": rank_quality,
        "balanced_utility": utility,
        "fsr_adjusted_utility": adjusted_utility,
    }


def ranks_for_method(rank_rows, method):
    return [row for row in rank_rows if row["method"] == method]


def select_policies(pair_rows: list[dict], rank_rows: list[dict]) -> dict[str, dict]:
    policies = {}
    baseline_fsr = None
    if BASELINE in METHODS:
        baseline_scores = np.asarray([row[BASELINE] for row in pair_rows], dtype=float)
        baseline_thresholds = np.unique(np.r_[
            np.nextafter(baseline_scores.min(), -np.inf),
            np.quantile(baseline_scores, np.linspace(0.0, 1.0, GRID_SIZE)),
            np.nextafter(baseline_scores.max(), np.inf),
        ])
        baseline_candidates = []
        for threshold in baseline_thresholds:
            values = policy_values(pair_rows, rank_rows, BASELINE, float(threshold))
            baseline_candidates.append((values["fsr_adjusted_utility"], values))
        if baseline_candidates:
            baseline_fsr = max(baseline_candidates, key=lambda x: x[0])[1].get("false_split_rate")
    
    for method in METHODS:
        scores = np.asarray([row[method] for row in pair_rows], dtype=float)
        thresholds = np.unique(np.r_[
            np.nextafter(scores.min(), -np.inf),
            np.quantile(scores, np.linspace(0.0, 1.0, GRID_SIZE)),
            np.nextafter(scores.max(), np.inf),
        ])
        candidates = []
        for threshold in thresholds:
            values = policy_values(pair_rows, rank_rows, method, float(threshold), 
                                 baseline_fsr=baseline_fsr, fsr_penalty_weight=0.2)
            candidates.append((values["fsr_adjusted_utility"], values["balanced_accuracy"],
                               -values["false_merge_rate"], -values["false_split_rate"],
                               values["ndcg_at_5"], float(threshold), values))
        best = max(candidates, key=lambda item: item[:-1])
        policies[method] = {
            "threshold": best[5],
            "training_metrics": best[6],
            "candidate_threshold_count": len(candidates),
            "selection_rule": "maximum FSR-adjusted utility (balanced utility minus FSR penalty)",
        }
    return policies


def select_method(policies: dict[str, dict]) -> tuple[str, dict]:
    method = max(
        METHODS,
        key=lambda name: (
            policies[name]["training_metrics"]["balanced_utility"],
            policies[name]["training_metrics"]["balanced_accuracy"],
            -policies[name]["training_metrics"]["false_merge_rate"],
            -policies[name]["training_metrics"]["false_split_rate"],
            policies[name]["training_metrics"]["ndcg_at_5"],
            -SIMPLICITY[name],
        ),
    )
    return method, policies[method]


def select_pareto_method(pair_rows: list[dict], ranks: list[dict], standard_policies: dict[str, dict]):
    """V16: select highest FSR-adjusted utility without Pareto constraints."""
    baseline_fsr = standard_policies[BASELINE]["training_metrics"].get("false_split_rate")
    candidates = []
    audit = {}
    for method in SELECTION_METHODS:
        scores = np.asarray([row[method] for row in pair_rows], dtype=float)
        thresholds = np.unique(np.r_[
            np.nextafter(scores.min(), -np.inf),
            np.quantile(scores, np.linspace(0.0, 1.0, GRID_SIZE)),
            np.nextafter(scores.max(), np.inf),
        ])
        method_candidates = []
        for threshold in thresholds:
            values = policy_values(pair_rows, ranks, method, float(threshold), 
                                 baseline_fsr=baseline_fsr, fsr_penalty_weight=0.2)
            method_candidates.append((float(threshold), values["fsr_adjusted_utility"],
                                     values["balanced_accuracy"], values["rank_quality"],
                                     values["query_coverage"], -values["false_merge_rate"],
                                     -values["false_split_rate"], -SIMPLICITY[method], values))
        audit[method] = {
            "candidate_thresholds": len(method_candidates),
            "inner_best_standard_utility": standard_policies[method]["training_metrics"]["balanced_utility"],
            "inner_best_adjusted_utility": max(c[1] for c in method_candidates) if method_candidates else None,
        }
        for candidate in method_candidates:
            candidates.append((
                candidate[0], candidate[1], candidate[2], candidate[3],
                candidate[4], candidate[5], candidate[6], candidate[7], method,
                candidate[8],
            ))
    if not candidates:
        policy = dict(standard_policies[BASELINE])
        policy["selection_rule"] = "no candidates evaluated; retain the same-fold command baseline"
        audit["selected_fallback"] = BASELINE
        return BASELINE, policy, audit
    # V16: select the highest FSR-adjusted utility without Pareto constraints.
    # This allows trade-offs but penalizes FSR increase relative to baseline.
    best = max(candidates, key=lambda item: item[1])
    method, threshold, values = best[8], best[0], best[9]
    policy = {
        "threshold": threshold,
        "training_metrics": values,
        "selection_rule": "select the method and threshold with the highest FSR-adjusted utility (balanced utility minus FSR penalty) without Pareto constraints, allowing trade-offs while controlling false split rate increase",
        "candidate_threshold_count": len(candidates),
        "utility_audit": audit,
    }
    return method, policy, audit


def evaluate_fold(pair_rows, ranks, policies, outer_fold, methods=METHODS):
    rate_rows = []
    for method in methods:
        threshold = policies[method]["threshold"]
        rates = error_eval._query_rates(pair_rows, method, threshold)
        coverage = {
            (row["benchmark"], row["task"]): row["query_has_accepted_match"]
            for row in base.query_coverage(pair_rows, method, threshold)
        }
        for row in rates:
            rate_rows.append({
                **row,
                "method": method,
                "threshold": float(threshold),
                "outer_fold": outer_fold,
                "query_has_accepted_match": coverage.get((row["benchmark"], row["task"])),
            })
    rank_rows = [
        {**row, "outer_fold": outer_fold}
        for row in ranks
    ]
    return rate_rows, rank_rows


def bootstrap_macro(rows: list[dict], metric: str, seed: int, draws: int = BOOTSTRAPS):
    grouped = collections.defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            grouped[row["benchmark"]].append(float(value))
    if not grouped:
        return {"estimate": None, "ci95": None, "task_count": 0}
    estimate = float(np.mean([np.mean(values) for values in grouped.values()]))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        benchmark_means = []
        for values in grouped.values():
            selected = rng.choice(values, size=len(values), replace=True)
            benchmark_means.append(float(np.mean(selected)))
        samples[draw] = float(np.mean(benchmark_means))
    return {
        "estimate": estimate,
        "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "task_count": sum(len(values) for values in grouped.values()),
        "bootstrap_draws": draws,
    }


def paired_delta(rate_rows, rank_rows, metric, method, baseline, seed):
    rows = rate_rows if metric in RATE_METRICS else rank_rows
    return base._paired_delta(rows, metric, method, baseline, BOOTSTRAPS, seed)


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(results: dict, rate_rows: list[dict], rank_rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "Hiragino Sans GB", "STHeiti", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = {
        "command_tfidf": "#0072B2",
        "profile_24d": "#E69F00",
        "tensor_4x8x3": "#009E73",
        "tensor_tucker": "#CC79A7",
        "partial_sequence": "#56B4E9",
        "learned_fusion": "#D55E00",
        "idf_lcs_coverage": "#000000",
        "phase_gated_lcs": "#9467BD",
        "mutual_rank_fusion": "#B279A2",
        "operand_jaccard": "#8C564B",
        "operand_aware_command": "#17BECF",
        "artifact_containment_command": "#D55E00",
        "reciprocal_consensus": "#2CA02C",
        "neighborhood_verified": "#BCBD22",
        "verifier_generic_k16": "#1F77B4",
        "verifier_mode_conditioned_k16": "#FF7F0E",
        "verifier_no_human_profile_k16": "#2CA02C",
        "verifier_no_agent_profile_k16": "#D62728",
        "verifier_no_phase_tensor_k16": "#9467BD",
        "verifier_no_neighborhood_k16": "#8C564B",
        "chi2_command_tfidf": "#E377C2",
        "length_aware_containment": "#7F7F7F",
        "action_edit_distance": "#BCBD22",
        "discriminative_pair_score": "#17BECF",
        "source_conditioned_score": "#9467BD",
        "open_set_rejection_score": "#8C564B",
        "balanced_selected": "#D55E00",
    }
    labels = {
        "command_tfidf": ("Command TF–IDF", "命令 TF–IDF"),
        "profile_24d": ("24D profile", "24 维画像"),
        "tensor_4x8x3": ("4×8×3 tensor", "4×8×3 张量"),
        "tensor_tucker": ("Tucker tensor", "Tucker 张量"),
        "partial_sequence": ("IDF partial sequence", "IDF 局部序列"),
        "learned_fusion": ("Learned fusion", "学习式融合"),
        "idf_lcs_coverage": ("IDF-LCS", "IDF-LCS"),
        "phase_gated_lcs": ("Phase-gated LCS", "阶段门控 LCS"),
        "mutual_rank_fusion": ("Mutual rank fusion", "互惠多视角排序融合"),
        "operand_jaccard": ("Operand signatures", "参数与产物特征"),
        "operand_aware_command": ("Command + operands", "命令 + 参数画像"),
        "artifact_containment_command": ("Command + artifact containment", "命令 + 产物包含度"),
        "reciprocal_consensus": ("Reciprocal view consensus", "互反多视图共识"),
        "neighborhood_verified": ("Neighborhood-verified", "邻域校验"),
        "verifier_generic_k16": ("Hard-negative verifier · top-16", "困难负例验证器 · top-16"),
        "verifier_mode_conditioned_k16": ("Mode-conditioned · top-16", "模式条件验证器 · top-16"),
        "verifier_no_human_profile_k16": ("Without Human profile · top-16", "移除 Human 画像 · top-16"),
        "verifier_no_agent_profile_k16": ("Without Agent profile · top-16", "移除 Agent 画像 · top-16"),
        "verifier_no_phase_tensor_k16": ("Without phase tensor · top-16", "移除阶段张量 · top-16"),
        "verifier_no_neighborhood_k16": ("Without neighborhood · top-16", "移除邻域证据 · top-16"),
        "chi2_command_tfidf": ("Chi-squared TF-IDF", "卡方加权 TF–IDF"),
        "length_aware_containment": ("Length-aware containment", "长度感知包含度"),
        "action_edit_distance": ("Action edit distance", "动作编辑距离"),
        "discriminative_pair_score": ("Discriminative pair classifier", "判别式配对分类器"),
        "source_conditioned_score": ("Source-conditioned score", "来源条件化打分"),
        "open_set_rejection_score": ("Open-set rejection", "开放集拒判"),
        "balanced_selected": ("Selected per fold", "逐折入选策略"),
    }
    core_methods = [
        BASELINE, "operand_jaccard", "artifact_containment_command", "neighborhood_verified",
        "chi2_command_tfidf", "length_aware_containment", "action_edit_distance",
        "discriminative_pair_score", "source_conditioned_score", "open_set_rejection_score",
        "verifier_generic_k16", "verifier_mode_conditioned_k16",
        "verifier_no_human_profile_k16", "verifier_no_agent_profile_k16",
        "verifier_no_phase_tensor_k16", "verifier_no_neighborhood_k16", "balanced_selected",
    ]
    ci = {}
    for index, method in enumerate(core_methods):
        subset = [row for row in rate_rows if row["method"] == method]
        ci[method] = {
            metric: bootstrap_macro(subset, metric, SEED + 200 + index * 7 + offset)
            for offset, metric in enumerate(("false_merge_rate", "false_split_rate"))
        }

    for lang in ("en",):
        english = lang == "en"
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.02, 1.25]})
        fig.patch.set_facecolor("white")
        ax = axes[0]
        legend_handles = []
        for method in core_methods:
            rates = [row for row in rate_rows if row["method"] == method]
            fmr = bootstrap_macro(rates, "false_merge_rate", SEED + 100 + SIMPLICITY.get(method, 9))
            fsr = bootstrap_macro(rates, "false_split_rate", SEED + 150 + SIMPLICITY.get(method, 9))
            if fmr["estimate"] is None or fsr["estimate"] is None:
                continue
            x, y = fmr["estimate"] * 100, fsr["estimate"] * 100
            xerr = np.array([[x - fmr["ci95"][0] * 100], [fmr["ci95"][1] * 100 - x]])
            yerr = np.array([[y - fsr["ci95"][0] * 100], [fsr["ci95"][1] * 100 - y]])
            marker = "*" if method == "balanced_selected" else "o"
            size = 140 if method == "balanced_selected" else 62
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, ms=9 if marker == "*" else 6,
                        color=colors.get(method, "#333333"), ecolor=colors.get(method, "#333333"),
                        elinewidth=1.2, capsize=3, alpha=.95, zorder=4 if marker == "*" else 2)
            legend_handles.append(Line2D(
                [], [], color=colors.get(method, "#333333"), marker=marker, linestyle="none",
                markersize=7 if marker == "*" else 5,
                label=labels[method][0 if english else 1],
            ))
        observed_fmr = [item.get("false_merge_rate") for item in ci.values()]
        observed_fsr = [item.get("false_split_rate") for item in ci.values()]
        fmr_values = [item["estimate"] * 100 for item in observed_fmr if item["estimate"] is not None]
        fsr_values = [item["estimate"] * 100 for item in observed_fsr if item["estimate"] is not None]
        ax.set_xlim(max(0, min(65, min(fmr_values, default=65) - 5)), 105)
        ax.set_ylim(max(-3, min(fsr_values, default=-3) - 5), min(105, max(fsr_values, default=40) + 5))
        ax.set_xlabel("False-merge rate FMR (%)" if english else "误合并率 FMR（%）", fontsize=10)
        ax.set_ylabel("Missed-link rate FSR (%)" if english else "漏关联率 FSR（%）", fontsize=10)
        ax.set_title("Error trade-off" if english else "误差权衡", loc="left", fontsize=12, weight="bold")
        ax.grid(True, color="#dbe3ea", linewidth=.7, alpha=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.02, .02, "lower-left is better" if english else "左下角较好", transform=ax.transAxes,
                fontsize=8, color="#526575")
        fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(.5, .015),
                   ncol=3, frameon=False, fontsize=7.5, columnspacing=1.5,
                   handletextpad=.5)

        ax = axes[1]
        metrics = [
            ("false_merge_rate", "False merges" if english else "误合并"),
            ("false_split_rate", "Missed links" if english else "漏关联"),
            ("false_merge_human_agent", "H–A false merges" if english else "H–A 误合并"),
            ("false_split_human_agent", "H–A missed links" if english else "H–A 漏关联"),
            ("balanced_accuracy", "Balanced accuracy" if english else "平衡准确率"),
            ("query_has_accepted_match", "Query coverage" if english else "查询覆盖"),
            ("hit_at_1", "Hit@1"), ("mrr", "MRR"), ("recall_at_5", "Recall@5"), ("ndcg_at_5", "NDCG@5"),
        ]
        selected_rates = [row for row in rate_rows if row["method"] == "balanced_selected"]
        baseline_rates = [row for row in rate_rows if row["method"] == BASELINE]
        selected_ranks = [row for row in rank_rows if row["method"] == "balanced_selected"]
        baseline_ranks = [row for row in rank_rows if row["method"] == BASELINE]
        deltas = []
        for index, (metric, label) in enumerate(metrics):
            if metric in RATE_METRICS:
                selected, baseline = selected_rates, baseline_rates
            else:
                selected, baseline = selected_ranks, baseline_ranks
            delta = base._paired_delta([*selected, *baseline],
                                       metric, "balanced_selected" if metric in RATE_METRICS else "balanced_selected",
                                       BASELINE, BOOTSTRAPS, SEED + 500 + index)
            # Error rates are reversed so positive always means improvement.
            estimate = float(delta["estimate"])
            interval = delta.get("ci_95")
            if metric in error_eval.RATE_METRICS:
                estimate = -estimate
                interval = [-interval[1], -interval[0]] if interval else None
            deltas.append((label, estimate * 100, [x * 100 for x in interval] if interval else None))
        deltas.reverse()
        y_positions = np.arange(len(deltas))
        for y, (label, estimate, interval) in zip(y_positions, deltas, strict=True):
            if interval:
                ax.errorbar(estimate, y, xerr=[[estimate - interval[0]], [interval[1] - estimate]],
                            fmt="o", color="#0072B2", ecolor="#0072B2", capsize=2.5, markersize=4)
            else:
                ax.plot(estimate, y, "o", color="#0072B2", markersize=4)
        ax.axvline(0, color="#465867", linewidth=1, linestyle="--")
        ax.set_yticks(y_positions, [item[0] for item in deltas], fontsize=8)
        ax.set_xlabel("Direction-adjusted change (percentage points)" if english else "方向校正变化（百分点）", fontsize=9)
        ax.set_title("Selected policy vs command baseline" if english else "逐折入选策略 vs 命令基线", loc="left", fontsize=12, weight="bold")
        ax.grid(axis="x", color="#dbe3ea", linewidth=.7, alpha=.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        fig.suptitle(
            "V15 discriminative pair classification · 40 task groups · 95% task-cluster bootstrap"
            if english else "V15 判别式配对分类 · 40 个任务组 · 95% 任务簇 Bootstrap",
            x=.02, ha="left", fontsize=9, color="#526575",
        )
        fig.tight_layout(rect=(0, .12, 1, .92), w_pad=3.2)
        suffix = "en" if english else "zh"
        fig.savefig(OUT / f"balanced-innovation-v15-{suffix}.png", dpi=260, bbox_inches="tight", facecolor="white")
        fig.savefig(OUT / "balanced-innovation-v15.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    recalls = results["candidate_generation"]["by_top_k"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for k in CANDIDATE_KS:
        item = recalls[str(k)]["task_macro"]
        estimate = item["estimate"] * 100
        low, high = [value * 100 for value in item["ci95"]]
        ax.errorbar(k, estimate, yerr=[[estimate - low], [high - estimate]],
                    fmt="o", color=("#0072B2" if k == 16 else "#7A8793"),
                    capsize=4, markersize=7, linewidth=1.5)
        ax.annotate(f"{estimate:.1f}%", (k, estimate), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(CANDIDATE_KS, [str(k) for k in CANDIDATE_KS])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Candidates retained per evidence family")
    ax.set_ylabel("Positive-pair candidate recall (%)")
    ax.set_title("Recall before pair verification", loc="left", fontsize=12, weight="bold")
    ax.grid(axis="y", color="#dbe3ea", linewidth=.7, alpha=.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.01, .01, "Outer task-held-out folds · 95% task-cluster bootstrap",
             fontsize=8, color="#526575")
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(OUT / "balanced-innovation-v15-candidate-recall-en.png", dpi=260,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "balanced-innovation-v15-candidate-recall.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run() -> dict:
    # The broad gallery loader parses every benchmark/model before filtering.
    # This scoped adapter preserves the cohort while avoiding the locked external data.
    gallery.load_rows = load_scoped_development_rows
    development, _by_benchmark, _unmatched = _load_development()
    task_keys = {(row.benchmark, row.task) for row in development}
    if len(development) != 97 or len(task_keys) != 40:
        raise RuntimeError(f"Expected the frozen 97-trace / 40-task development cohort; got {len(development)} / {len(task_keys)}")

    outer_folds = split_utils.grouped_folds(development, SEED, n_folds=3)
    all_pair_rows, all_rate_rows, all_rank_rows, all_direction_rows = [], [], [], []
    fold_audit = []
    threshold_rows = []

    for outer_index, (train_keys, test_keys) in enumerate(outer_folds, start=1):
        outer_train = [row for row in development if (row.benchmark, row.task) in train_keys]
        outer_test = [row for row in development if (row.benchmark, row.task) in test_keys]
        inner_oof = []
        inner_folds = split_utils.grouped_folds(outer_train, SEED + outer_index * 100, n_folds=3)
        for inner_index, (inner_train_keys, inner_test_keys) in enumerate(inner_folds, start=1):
            inner_train = [row for row in outer_train if (row.benchmark, row.task) in inner_train_keys]
            inner_test = [row for row in outer_train if (row.benchmark, row.task) in inner_test_keys]
            pairs, _ = fit_pair_verifiers(
                inner_train, inner_test,
                SEED + outer_index * 1000 + inner_index,
                outer_index, "balanced_inner_oof",
            )
            inner_oof.extend(pairs)
        inner_ranks = task_rank_rows(inner_oof, METHODS)
        policies = select_policies(inner_oof, inner_ranks)
        selected_method, selected_policy, pareto_audit = select_pareto_method(inner_oof, inner_ranks, policies)

        outer_pairs, dimensions = fit_pair_verifiers(
            outer_train, outer_test, SEED + outer_index,
            outer_index, "development_outer_test",
        )
        outer_ranks = task_rank_rows(outer_pairs, METHODS)
        for method, policy in policies.items():
            all_direction_rows.extend(directional_error_rows(
                outer_pairs, method, float(policy["threshold"]), outer_index
            ))

        # Save each preselected method/threshold, plus the method selected inside the training fold.
        for row in outer_pairs:
            enriched = dict(row)
            enriched["outer_fold"] = outer_index
            for method in METHODS:
                enriched[f"threshold_{method}"] = policies[method]["threshold"]
                enriched[f"accepted_{method}"] = int(row[method] >= policies[method]["threshold"])
            enriched["selected_method"] = selected_method
            enriched["selected_score"] = row[selected_method]
            enriched["selected_threshold"] = selected_policy["threshold"]
            enriched["selected_accepted"] = int(row[selected_method] >= selected_policy["threshold"])
            enriched["balanced_selected"] = row[selected_method]
            all_pair_rows.append(enriched)

        fold_rates, fold_ranks = evaluate_fold(outer_pairs, outer_ranks, policies, outer_index)
        all_rate_rows.extend(fold_rates)
        all_rank_rows.extend(fold_ranks)

        selected_rows = [dict(row, balanced_selected=row[selected_method]) for row in outer_pairs]
        selected_policy_by_method = dict(policies)
        selected_policy_by_method["balanced_selected"] = selected_policy
        selected_rates, selected_ranks = evaluate_fold(
            selected_rows, task_rank_rows(selected_rows, ("balanced_selected",)),
            {"balanced_selected": selected_policy}, outer_index, methods=("balanced_selected",),
        )
        all_rate_rows.extend(selected_rates)
        all_rank_rows.extend(selected_ranks)
        all_direction_rows.extend(directional_error_rows(
            selected_rows, "balanced_selected", float(selected_policy["threshold"]), outer_index
        ))

        for method, policy in policies.items():
            threshold_rows.append({
                "outer_fold": outer_index,
                "method": method,
                "threshold": float(policy["threshold"]),
                "inner_balanced_utility": policy["training_metrics"]["balanced_utility"],
                "inner_balanced_accuracy": policy["training_metrics"]["balanced_accuracy"],
                "inner_fmr": policy["training_metrics"]["false_merge_rate"],
                "inner_fsr": policy["training_metrics"]["false_split_rate"],
            })
        fold_audit.append({
            "outer_fold": outer_index,
            "train_task_groups": len(train_keys),
            "test_task_groups": len(test_keys),
            "train_traces": len(outer_train),
            "test_traces": len(outer_test),
            "selected_method": selected_method,
            "selected_threshold": float(selected_policy["threshold"]),
            "selected_inner_metrics": selected_policy["training_metrics"],
            "candidate_policies": policies,
            "pareto_candidate_audit": pareto_audit,
            "fitted_dimensions": dimensions,
        })
        print(f"outer fold {outer_index}/3: selected {selected_method}; test tasks={len(test_keys)}", flush=True)

    methods_for_summary = (*METHODS, "balanced_selected")
    metric_summary = {}
    for method_index, method in enumerate(methods_for_summary):
        rates = [row for row in all_rate_rows if row["method"] == method]
        ranks = [row for row in all_rank_rows if row["method"] == method]
        metric_summary[method] = {
            metric: bootstrap_macro(
                rates if metric in RATE_METRICS else ranks,
                metric,
                SEED + method_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate((*RATE_METRICS, *RANK_METRICS))
        }

    selected_deltas = {
        metric: paired_delta(
            all_rate_rows,
            all_rank_rows,
            metric,
            "balanced_selected",
            BASELINE,
            SEED + 500 + index,
        )
        for index, metric in enumerate((*RATE_METRICS, *RANK_METRICS))
    }
    improvements = {}
    for metric, result in selected_deltas.items():
        estimate = result["estimate"]
        improvements[metric] = (
            None if estimate is None
            else float(-estimate if metric in error_eval.RATE_METRICS else estimate)
        )
    estimable_improvements = [value for value in improvements.values() if value is not None]

    selected_counts = collections.Counter(row["selected_method"] for row in fold_audit)
    directional_summary = {}
    directional_methods = (
        BASELINE, "balanced_selected", "verifier_generic_k16", "verifier_mode_conditioned_k16",
        "verifier_no_human_profile_k16", "verifier_no_agent_profile_k16",
        "verifier_no_phase_tensor_k16", "verifier_no_neighborhood_k16",
    )
    for method_index, method in enumerate(directional_methods):
        directional_summary[method] = {}
        for direction_index, direction in enumerate(("HUMAN-to-AGENT", "AGENT-to-HUMAN",
                                                       "HUMAN-to-HUMAN", "AGENT-to-AGENT")):
            subset = [row for row in all_direction_rows
                      if row["method"] == method and row["direction"] == direction]
            if not subset:
                directional_summary[method][direction] = {"task_count": 0}
                continue
            directional_summary[method][direction] = {
                metric: bootstrap_macro(
                    subset, metric, SEED + 8000 + method_index * 100 + direction_index * 7 + metric_index
                )
                for metric_index, metric in enumerate(("false_split_rate", "false_merge_rate", "query_coverage"))
            }

    candidate_recall_by_k = {}
    positive_outer_pairs = [row for row in all_pair_rows
                            if int(row["same_task_proxy_for_evaluation_only"])]
    for k in CANDIDATE_KS:
        metric = f"candidate_union_top{k}"
        by_task = collections.defaultdict(list)
        for row in positive_outer_pairs:
            by_task[(row["benchmark"], row["query_task"])].append(float(row[metric]))
        task_rows = [
            {"benchmark": benchmark, "task": task, "candidate_recall": float(np.mean(values))}
            for (benchmark, task), values in by_task.items()
        ]
        candidate_recall_by_k[str(k)] = {
            "micro_pair_recall": float(np.mean([row[metric] for row in positive_outer_pairs]))
            if positive_outer_pairs else None,
            "task_macro": bootstrap_macro(task_rows, "candidate_recall", SEED + 12000 + k),
            "task_count": len(task_rows),
        }
    result = {
        "status": "completed exploratory nested task-held-out development evaluation",
        "created_utc": "2026-10-31",
        "protocol": "experiments/balanced_innovation_v16/program.md",
        "dataset": {
            "records": len(development),
            "task_groups": len(task_keys),
            "benchmarks": sorted({row.benchmark for row in development}),
            "revision": "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae",
            "label_scope": "same task retrieval proxy, not actor or organization identity",
            "reuse_boundary": "The same task groups were inspected by earlier experiments; results are exploratory, not independent confirmation.",
            "locked_external_data_read": False,
            "traces_with_normalized_operand_features": sum(bool(OPERAND_TOKENS.get(row.trace_id)) for row in development),
            "normalized_operand_token_vocabulary": len({token for row in development for token in OPERAND_TOKENS.get(row.trace_id, {})}),
            "human_command_parser_counts": dict(HUMAN_COMMAND_PARSER_COUNTS),
            "selected_human_command_parser_counts": dict(collections.Counter(
                HUMAN_COMMAND_PARSER_BY_TRACE.get(row.trace_id, "unknown")
                for row in development if row.label == "human"
            )),
        },
        "selection_rule": "select only an inner-OOF policy with strictly lower overall FMR, no higher overall or subgroup FSR, and no worse other estimable subgroup error, balanced accuracy, query coverage, or rank metric; among feasible alternative thresholds choose the lowest cutoff; if none passes, retain the command baseline",
        "selection_candidates": list(SELECTION_METHODS),
        "new_methods": {
            "human_command_extraction": "Recover executed Human shell commands from the terminal output transcript, interpreting backspace, cursor movement, and insert/delete controls; task-score answers, targets, and task labels are not used as features.",
            "idf_lcs_coverage": "IDF-weighted command sequence alignment coverage plus weighted command and bigram overlap, with mild length-mismatch penalty",
            "phase_gated_lcs": "geometric mean of IDF-LCS coverage and four-phase action-profile similarity",
            "mutual_rank_fusion": "query-relative reciprocal-rank fusion over command, 24D profile, 4x8x3 tensor, and local-sequence scores; symmetrized by geometric mean of both directional fused ranks",
            "operand_jaccard": "training-IDF Jaccard of normalized command flags, file extensions, artifact-name tokens, and URL host/path tokens",
            "operand_aware_command": "0.45 command TF-IDF plus 0.55 training-IDF operand signature similarity",
            "artifact_containment_command": "0.45 command TF-IDF plus 0.55 training-IDF-weighted containment of distinctive artifact-name tokens in the shorter trace signature; generic tool/path words are excluded",
            "artifact_containment": "training-only IDF containment of distinctive artifact-name tokens in the shorter trace signature; unknown test artifacts use the training maximum weight",
            "reciprocal_consensus": "query-local reciprocal rank consensus over three evidence families: command/artifact content, command order, and behavioral profile/tensor/phase",
            "neighborhood_verified": "reciprocal consensus modulated by the Jaccard overlap of each trace's top-3 candidate neighborhoods; shared neighbors can confirm but cannot independently create a link",
            "verifier_candidate_sweep": "Each verifier is evaluated with the same nested split at candidate unions of top-12, top-16, and top-20 per content, sequence, and behavior family. The verifier is trained once on the maximum candidate pool within each training fold; thresholds and candidate width are selected only from inner out-of-fold predictions.",
            "verifier_generic": "regularized pair verifier trained with same-task positives, top-ranked wrong-task hard negatives, and a small random negative sample",
            "verifier_mode_conditioned": "same hard-negative verifier with task-crossfit predicted Human/Agent probabilities interacting with each trace side's profile, tensor, and phase evidence",
            "verifier_no_human_profile": "mode-conditioned verifier ablation removing Human-query and Human-candidate profile interactions",
            "verifier_no_agent_profile": "mode-conditioned verifier ablation removing Agent-query and Agent-candidate profile interactions",
            "verifier_no_phase_tensor": "mode-conditioned verifier ablation without action-phase and tensor evidence",
            "verifier_no_neighborhood": "mode-conditioned verifier ablation without reciprocal-neighborhood evidence",
        },
        "candidate_generation": {
            "rule": "union of the top-k candidates from content, sequence, and behavior evidence families, separately within benchmark; k is selected from 12/16/20 by inner task-grouped validation only",
            "by_top_k": candidate_recall_by_k,
        },
        "directional_error_metrics": directional_summary,
        "selected_method_by_fold": dict(selected_counts),
        "outer_folds": fold_audit,
        "metrics": metric_summary,
        "paired_deltas_vs_command": selected_deltas,
        "direction_adjusted_selected_changes": improvements,
        "all_estimable_metrics_pointwise_nonworse": bool(
            estimable_improvements
            and all(value >= -1e-12 for value in estimable_improvements)
            and any(value > 1e-12 for value in estimable_improvements)
        ),
        "provenance": {
            "runner_sha256": file_hash(pathlib.Path(__file__)),
            "human_command_parser": "terminal output prompt reconstruction with backspace and CSI cursor/insert/delete handling",
            "base_runner_sha256": file_hash(ROOT / "experiments" / "multidimensional_rerun_v1" / "run.py"),
            "source_manifest_sha256": file_hash(ROOT / "experiments" / "multidimensional_rerun_v1" / "source_manifest.json"),
            "seed": SEED,
            "outer_folds": 3,
            "inner_folds": 3,
            "bootstrap_draws": BOOTSTRAPS,
        },
        "interpretation": "No result from this reused development cohort is an independent confirmation. All indicators and task-cluster intervals must be read together; a low false-split rate alone is not a success criterion.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT / "outer_pair_scores.csv", all_pair_rows)
    write_csv(OUT / "task_metrics.csv", [*all_rate_rows, *all_rank_rows])
    write_csv(OUT / "thresholds.csv", threshold_rows)
    write_csv(OUT / "directional_metrics.csv", all_direction_rows)
    make_plots(result, all_rate_rows, all_rank_rows)
    print(f"Wrote balanced development results to {OUT.relative_to(ROOT)}", flush=True)
    print(json.dumps({
        "selected_method_by_fold": result["selected_method_by_fold"],
        "direction_adjusted_selected_changes": improvements,
        "all_estimable_metrics_pointwise_nonworse": result["all_estimable_metrics_pointwise_nonworse"],
    }, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    run()


if __name__ == "__main__":
    main()
