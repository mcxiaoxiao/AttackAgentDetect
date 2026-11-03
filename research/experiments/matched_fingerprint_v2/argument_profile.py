#!/usr/bin/env python3
"""Create value-redacted command-syntax features from public CTH raw logs.

Argument values never leave this process. The profile retains known command
heads' option names, coarse argument types, and coarse argument-count buckets.
"""
from __future__ import annotations

import gzip
import io
import ipaddress
import json
import pathlib
import re
import shlex
import zipfile

from attackagent.traces import (ASSIGNMENT, ATOM_GROUP, WRAPPERS,
                                command_heads, terminal_lines)

RAW = pathlib.Path(__file__).resolve().parents[2] / "data/raw/cyber-task-horizons"

Heredoc = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)")
NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
HEX = re.compile(r"^(?:0x)?[0-9a-fA-F]{8,}$")
ATTACHED_SHORT = re.compile(r"^(-[A-Za-z]+)(.+)$")


def _value_type(token: str) -> str:
    value = token.strip()
    if not value:
        return "empty"
    if value.startswith(("http://", "https://", "ftp://", "sftp://")):
        return "url"
    if value.startswith(("$", "${", "%")):
        return "variable"
    try:
        ipaddress.ip_address(value.strip("[]"))
        return "ip"
    except ValueError:
        pass
    if HEX.fullmatch(value):
        return "hex"
    if NUMERIC.fullmatch(value):
        return "number"
    if "/" in value or "\\" in value or value.startswith((".", "~")):
        return "path"
    if re.match(r"^[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8}$", value):
        return "filename"
    if "=" in value and not value.startswith("-"):
        return "assignment"
    if value in {"|", ">", ">>", "<", "2>", "2>&1"}:
        return "redirection"
    return "literal"


def _flag_feature(head: str, token: str) -> str | None:
    if token == "--":
        return f"flag__{head}__end_options"
    if token.startswith("--"):
        return f"flag__{head}__{token.split('=', 1)[0].lower()}"
    if token.startswith("-") and len(token) > 1 and token[1].isalpha():
        if re.fullmatch(r"-[A-Za-z]+", token):
            return f"flag__{head}__{token.lower()}"
        attached = ATTACHED_SHORT.match(token)
        if attached:
            return f"flag__{head}__attached_short"
        return f"flag__{head}__short_option"
    return None


def _count_bucket(value: int, kind: str) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2_3"
    return "4_plus"


def _invocations_for_line(line: str) -> list[tuple[str, list[str]]]:
    try:
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
    except ValueError:
        return []
    out, at_start, head, args = [], True, None, []
    for token in tokens + [";"]:
        if token and all(ch in ";&|" for ch in token):
            if head is not None:
                out.append((head, args))
            at_start, head, args = True, None, []
            continue
        if at_start:
            base = pathlib.PurePosixPath(token).name.lower()
            if ASSIGNMENT.match(token) or base in WRAPPERS:
                continue
            at_start = False
            if base in ATOM_GROUP:
                head = base
            continue
        if head is not None:
            args.append(token)
    return out


def profile_for_chunks(chunks: list[str]) -> tuple[list[str], int]:
    """Return sanitized syntax tokens and the recognized command-head count."""
    features: list[str] = []
    recognized_heads: list[str] = []
    for chunk in chunks:
        heredoc_end = None
        for raw_line in chunk.splitlines() or [chunk]:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if heredoc_end is not None:
                if line == heredoc_end:
                    heredoc_end = None
                continue
            marker = Heredoc.search(line)
            if marker:
                heredoc_end = marker.group(1)
            for head, args in _invocations_for_line(line):
                recognized_heads.append(head)
                flag_count = 0
                for pos, token in enumerate(args):
                    flag = _flag_feature(head, token)
                    if flag is not None:
                        flag_count += 1
                        features.append(flag)
                        if "=" in token:
                            value_kind = _value_type(token.split("=", 1)[1])
                            features.append(f"argtype__{head}__option_value_{value_kind}")
                        else:
                            attached = ATTACHED_SHORT.match(token)
                            if attached:
                                features.append(f"argtype__{head}__option_value_{_value_type(attached.group(2))}")
                        continue
                    attached = ATTACHED_SHORT.match(token)
                    if attached:
                        flag_count += 1
                        features.append(f"flag__{head}__attached_short")
                        features.append(f"argtype__{head}__option_value_{_value_type(attached.group(2))}")
                        continue
                    value_kind = _value_type(token)
                    index = "first" if pos == 0 else "second" if pos == 1 else "third_plus"
                    features.append(f"argpos__{head}__{index}__{value_kind}")
                features.append(f"arity__{head}__{_count_bucket(len(args), 'args')}")
                features.append(f"option_count__{head}__{_count_bucket(flag_count, 'flags')}")
    # The existing parser is the contract for which shell commands are usable.
    counted = sum(len(command_heads(chunk)) for chunk in chunks)
    if len(recognized_heads) != counted:
        raise ValueError(f"abstract profile/parser disagreement: {len(recognized_heads)} != {counted}")
    return features, counted


def _sample(path: pathlib.Path, member: str, cache: dict[pathlib.Path, zipfile.ZipFile]):
    if path not in cache:
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as f:
                payload = f.read()
            cache[path] = zipfile.ZipFile(io.BytesIO(payload))
        else:
            cache[path] = zipfile.ZipFile(path)
    with cache[path].open(member) as f:
        return json.load(f)


def cth_profiles(rows) -> dict[str, dict]:
    """Return one sanitized syntax profile per trace ID; no raw text is retained."""
    output = {}
    cache: dict[pathlib.Path, zipfile.ZipFile] = {}
    try:
        for row in rows:
            rel, member = row["locator"].split("#", 1)
            path = RAW / rel
            sample = _sample(path, member, cache)
            if row["mode"] == "human":
                logs = sample.get("store", {}).get("HumanAgentState:logs", {})
                chunks = []
                for key, value in sorted(logs.items()):
                    if key.endswith(".input") and isinstance(value, str):
                        chunks.extend(terminal_lines(value))
                chunks = ["\n".join(chunks)] if chunks else []
            else:
                chunks = []
                for msg in sample.get("messages", []):
                    for call in msg.get("tool_calls", []) or []:
                        args = call.get("arguments", {})
                        if call.get("function") == "bash" and isinstance(args.get("cmd"), str):
                            chunks.append(args["cmd"])
            features, heads = profile_for_chunks(chunks)
            output[row["id"]] = {"features": features, "recognized_heads": heads,
                                  "feature_count": len(features)}
    finally:
        for archive in cache.values():
            archive.close()
    return output
