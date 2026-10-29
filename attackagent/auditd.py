"""Conservative Linux auditd EXECVE/SYSCALL normalizer.

The parser never executes command text. It keeps event references and coarse
quality counters, and deliberately leaves execution-mode labels out of output.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import pathlib
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, TextIO

from .evidence import TraceEvidence

_AUDIT_ID = re.compile(r"audit\((?P<time>[^:)]*):(?P<serial>\d+)\)")
_HEX = re.compile(r"^[0-9a-fA-F]+$")
_ARG_KEY = re.compile(r"^a(?P<index>\d+)$")
_UNSET = {"4294967295", "-1", "unset", "4294967295L"}
_X86_64_SYSCALLS = {
    "0": "read", "1": "write", "2": "open", "3": "close", "41": "socket",
    "42": "connect", "43": "accept", "44": "sendto", "45": "recvfrom",
    "59": "execve", "82": "rename", "87": "unlink", "257": "openat",
    "263": "unlinkat", "257": "openat", "322": "execveat",
}
_COMMAND_ACTIONS = {
    **{name: "discovery" for name in ("ls", "pwd", "find", "which", "file", "stat", "id", "whoami", "ps", "uname")},
    **{name: "read" for name in ("cat", "head", "tail", "less", "strings", "xxd", "jq")},
    **{name: "search" for name in ("grep", "rg", "ag", "awk")},
    **{name: "network" for name in ("curl", "wget", "nc", "netcat", "ssh", "scp", "nmap")},
    **{name: "modify" for name in ("cp", "mv", "mkdir", "rm", "touch", "chmod", "sed", "tee")},
    **{name: "execute" for name in ("sh", "bash", "python", "python3", "perl", "ruby", "node", "gcc", "make", "gdb", "strace")},
}


@dataclass
class _AuditEvent:
    host: str
    timestamp_text: str
    timestamp: float | None
    serial: str
    records: dict[str, list[dict[str, str]]] = field(default_factory=lambda: defaultdict(list))
    record_hashes: set[str] = field(default_factory=set)
    duplicate_lines: int = 0
    source_lines: int = 0

    @property
    def ref(self) -> str:
        return f"{self.host}/audit({self.timestamp_text}:{self.serial})"


def _parse_fields(line: str) -> tuple[str | None, dict[str, str]]:
    try:
        tokens = shlex.split(line, posix=True)
    except ValueError:
        # An unmatched quote should not make the remainder of a log disappear.
        tokens = line.split()
    record_type = None
    fields: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key == "type":
            record_type = value
        else:
            fields[key] = value
    return record_type, fields


def _event_id(fields: dict[str, str]) -> tuple[str, str, float | None, str] | None:
    match = _AUDIT_ID.search(fields.get("msg", ""))
    if not match:
        return None
    timestamp_text = match.group("time")
    try:
        timestamp = float(timestamp_text)
    except ValueError:
        timestamp = None
    return timestamp_text, match.group("serial"), timestamp, fields.get("node", "")


def _decode_hex(value: str) -> str:
    """Decode printable audit hex fields, preserving unrecognized values."""
    if len(value) < 2 or len(value) % 2 or not _HEX.fullmatch(value):
        return value
    try:
        decoded = bytes.fromhex(value).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return value
    printable = sum(ch.isprintable() or ch in "\x00\t" for ch in decoded)
    if decoded and printable / len(decoded) >= 0.85:
        return decoded.replace("\x00", " ").strip()
    return value


def _basename(value: str) -> str:
    value = value.strip().strip("\"'")
    value = value.rsplit("/", 1)[-1]
    return value.lower()


def _event_evidence(event: _AuditEvent) -> tuple[str | None, str, int, int]:
    syscall_records = event.records.get("SYSCALL", [])
    exec_records = event.records.get("EXECVE", [])
    syscall = syscall_records[0] if syscall_records else {}
    argv_by_index: dict[int, str] = {}
    for args in exec_records:
        for key, value in args.items():
            match = _ARG_KEY.match(key)
            if match:
                argv_by_index[int(match.group("index"))] = _decode_hex(value)
    argv = [argv_by_index[index] for index in sorted(argv_by_index)]

    executable = None
    if argv:
        executable = _basename(argv[0])
    if not executable:
        executable = _basename(syscall.get("exe", "")) or _basename(syscall.get("comm", ""))
    if executable in {"", "?", "(null)"}:
        executable = None

    raw_syscall = syscall.get("syscall", "")
    arch = syscall.get("arch", "").lower()
    syscall_name = raw_syscall.lower()
    if syscall_name.isdigit() and arch in {"c000003e", "x86_64"}:
        syscall_name = _X86_64_SYSCALLS.get(syscall_name, f"syscall_{syscall_name}")
    if not syscall_name:
        syscall_name = "unknown_syscall"

    command_action = _COMMAND_ACTIONS.get(executable or "", "")
    syscall_action = (
        "process" if syscall_name in {"execve", "execveat", "fork", "clone", "clone3", "vfork"}
        else "network" if syscall_name in {"socket", "connect", "accept", "accept4", "sendto", "recvfrom"}
        else "file_write" if syscall_name in {"write", "pwrite64", "rename", "renameat", "unlink", "unlinkat", "mkdir", "mkdirat"}
        else "file_access" if syscall_name in {"read", "open", "openat", "close", "stat", "newfstatat"}
        else "system_call"
    )
    action = command_action or syscall_action
    return executable, action, int(bool(exec_records)), int(not bool(syscall_records))


def _valid_session(fields: dict[str, str]) -> tuple[str, str] | None:
    session = fields.get("ses", "").strip().lower()
    if not session or session in _UNSET:
        return None
    auid = fields.get("auid", "unknown").strip().lower()
    return auid, session


def _open_source(source: str | pathlib.Path | TextIO | Iterable[str]):
    if hasattr(source, "read"):
        return source
    if isinstance(source, pathlib.Path):
        if source.suffix == ".gz":
            return gzip.open(source, "rt", encoding="utf-8", errors="replace")
        return source.open("r", encoding="utf-8", errors="replace")
    if isinstance(source, str):
        return io.StringIO(source)
    return source


def load_auditd(
    source: str | pathlib.Path | TextIO | Iterable[str],
    *,
    host: str = "host-unknown",
    inactivity_seconds: int = 300,
    return_stats: bool = False,
) -> list[TraceEvidence] | tuple[list[TraceEvidence], dict[str, int]]:
    """Normalize auditd records into session traces.

    SYSCALL/EXECVE and other records with the same ``audit(timestamp:serial)``
    are joined independent of line order. Valid sessions are sorted and split
    after an inactivity gap. Events without a valid session remain individual
    traces. Exact repeated lines are removed and counted. ``source`` strings
    are treated as raw log text; pass a :class:`~pathlib.Path` to read a file.
    """
    if inactivity_seconds < 0:
        raise ValueError("inactivity_seconds must be non-negative")
    handle = _open_source(source)
    should_close = isinstance(handle, io.TextIOBase) and not hasattr(source, "read") and isinstance(source, pathlib.Path)
    events: dict[tuple[str, str, str], _AuditEvent] = {}
    stats = Counter(lines=0, malformed_lines=0, duplicate_lines=0, parsed_records=0, joined_events=0)
    seen_raw: set[tuple[str, str]] = set()
    for line_number, raw_line in enumerate(handle, start=1):
        stats["lines"] += 1
        raw_line = raw_line.rstrip("\r\n")
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        record_type, fields = _parse_fields(raw_line)
        if not record_type:
            stats["malformed_lines"] += 1
            continue
        parsed_id = _event_id(fields)
        if parsed_id is None:
            stats["malformed_lines"] += 1
            continue
        timestamp_text, serial, timestamp, node = parsed_id
        actual_host = node or host
        event_key = (actual_host, timestamp_text, serial)
        if event_key not in events:
            events[event_key] = _AuditEvent(actual_host, timestamp_text, timestamp, serial)
        event = events[event_key]
        raw_digest = hashlib.sha256(raw_line.encode("utf-8", errors="replace")).hexdigest()
        repeat_key = ("/".join(event_key), raw_digest)
        event.source_lines += 1
        if repeat_key in seen_raw:
            event.duplicate_lines += 1
            stats["duplicate_lines"] += 1
            continue
        seen_raw.add(repeat_key)
        event.records[record_type].append(fields)
        stats["parsed_records"] += 1

    if should_close:
        handle.close()

    session_events: dict[tuple[str, str, str], list[_AuditEvent]] = defaultdict(list)
    isolated_events: list[_AuditEvent] = []
    for event in events.values():
        syscall_fields = event.records.get("SYSCALL", [])
        session = _valid_session(syscall_fields[0]) if syscall_fields else None
        if session is None:
            isolated_events.append(event)
            continue
        session_events[(event.host, session[0], session[1])].append(event)

    groups: list[list[_AuditEvent]] = [[event] for event in isolated_events]
    for key, grouped in sorted(session_events.items()):
        ordered = sorted(grouped, key=lambda event: (
            event.timestamp if event.timestamp is not None else float("inf"),
            int(event.serial) if event.serial.isdigit() else event.serial,
        ))
        current: list[_AuditEvent] = []
        previous_time: float | None = None
        for event in ordered:
            if (current and event.timestamp is not None and previous_time is not None
                    and event.timestamp - previous_time > inactivity_seconds):
                groups.append(current)
                current = []
            current.append(event)
            if event.timestamp is not None:
                previous_time = event.timestamp
        if current:
            groups.append(current)

    traces: list[TraceEvidence] = []
    for group in groups:
        group = sorted(group, key=lambda event: (
            event.timestamp if event.timestamp is not None else float("inf"),
            int(event.serial) if event.serial.isdigit() else event.serial,
        ))
        commands: list[str] = []
        actions: list[str] = []
        refs: list[str] = []
        trace_quality = Counter(events=0, exec_events=0, missing_syscall=0, duplicate_lines=0, incomplete_events=0)
        for event in group:
            command, action, exec_count, missing_syscall = _event_evidence(event)
            trace_quality["events"] += 1
            trace_quality["exec_events"] += exec_count
            trace_quality["missing_syscall"] += missing_syscall
            trace_quality["duplicate_lines"] += event.duplicate_lines
            if command:
                commands.append(command)
            else:
                trace_quality["incomplete_events"] += 1
            if action:
                actions.append(action)
            refs.append(event.ref)
        digest = hashlib.sha256("|".join(refs).encode()).hexdigest()[:16]
        traces.append(TraceEvidence(
            trace_id=f"auditd-{digest}",
            source_format="linux-auditd-v1",
            benchmark="auditd",
            commands=tuple(commands),
            actions=tuple(actions),
            event_refs=tuple(refs),
            quality=dict(trace_quality),
        ))
    traces.sort(key=lambda trace: trace.trace_id)
    stats["joined_events"] = len(events)
    stats["normalized_traces"] = len(traces)
    if return_stats:
        return traces, {key: int(value) for key, value in stats.items()}
    return traces
