"""Extract a conservative comparable command vocabulary from public trajectories.

Terminal keystrokes and agent tool calls are not equivalent to system audit logs.
This parser is a deliberately lossy research adapter, not a forensic extractor.
"""

import gzip
import hashlib
import io
import json
import pathlib
import re
import shlex
import zipfile
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cyber-task-horizons"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)")
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

ATOM_GROUP = {
    **{x: "discover" for x in ("ls", "pwd", "find", "which", "whereis", "locate", "file", "stat")},
    **{x: "read" for x in ("cat", "head", "tail", "less", "more", "strings", "xxd", "hexdump", "od", "jq")},
    **{x: "search" for x in ("grep", "rg", "ag")},
    **{x: "execute" for x in ("python", "python3", "bash", "sh", "node", "perl", "ruby", "java", "gdb", "strace", "ltrace")},
    **{x: "build" for x in ("gcc", "g++", "clang", "make", "cmake", "javac")},
    **{x: "network" for x in ("curl", "wget", "nc", "netcat", "ssh", "scp", "nmap")},
    **{x: "modify" for x in ("cp", "mv", "mkdir", "rm", "tee", "touch", "chmod", "sed", "awk", "echo", "printf", "nano", "vi", "vim")},
    **{x: "environment" for x in ("cd", "env", "printenv", "whoami", "id", "uname", "ps", "export")},
}
WRAPPERS = {"sudo", "command", "nohup", "time"}


@dataclass(frozen=True)
class Observation:
    trace_id: str
    task: str
    label: str
    source: str
    names: tuple
    groups: tuple
    skipped_python_calls: int = 0


def terminal_lines(raw):
    """Approximate submitted shell lines from script(1) keystrokes."""
    raw = ANSI.sub("", raw)
    if raw.startswith("Script started") and "\n" in raw:
        raw = raw.split("\n", 1)[1]
    out, line = [], []
    for char in raw:
        if char in "\r\n":
            item = "".join(line).strip()
            if item and not item.startswith("Script done"):
                out.append(item)
            line = []
        elif char in ("\x7f", "\b"):
            if line:
                line.pop()
        elif char == "\x03":
            line = []
        elif char == "\t":
            # Completion can change the arguments; do not invent its output.
            line.append(" ")
        elif char.isprintable():
            line.append(char)
    return out


def command_heads(text):
    """Extract known command names; discard unrecognized and harness commands."""
    names = []
    heredoc_end = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if heredoc_end is not None:
            if line == heredoc_end:
                heredoc_end = None
            continue
        marker = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)", line)
        if marker:
            heredoc_end = marker.group(1)
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
        except ValueError:
            continue
        at_start = True
        for token in tokens:
            if token and all(ch in ";&|" for ch in token):
                at_start = True
                continue
            if not at_start:
                continue
            base = pathlib.PurePosixPath(token).name.lower()
            if ASSIGNMENT.match(token) or base in WRAPPERS:
                continue
            if base in ATOM_GROUP:
                names.append(base)
            at_start = False
    return names


def samples(data):
    with zipfile.ZipFile(data) as archive:
        names = [n for n in archive.namelist() if n.startswith("samples/") and n.endswith(".json")]
        return [(n, json.loads(archive.read(n))) for n in names]


def load_observations():
    """Load selected public data; only shell command names become features."""
    manifest = json.loads((DATA / "download_manifest.json").read_text(encoding="utf-8"))
    output = []
    for row in manifest["files"]:
        path = DATA / row["path"]
        if "/human/" in row["path"]:
            with gzip.open(path, "rb") as f:
                parsed = samples(io.BytesIO(f.read()))
            for name, sample in parsed:
                logs = sample.get("store", {}).get("HumanAgentState:logs", {})
                lines = []
                for key, value in sorted(logs.items()):
                    if key.endswith(".input"):
                        lines.extend(terminal_lines(value))
                heads = command_heads("\n".join(lines))
                trace_id = hashlib.sha256((row["path"] + "/" + name).encode()).hexdigest()[:16]
                output.append(Observation(trace_id, str(sample["id"]), "human", "expert", tuple(heads), tuple(ATOM_GROUP[x] for x in heads)))
        else:
            for name, sample in samples(path):
                heads, skipped = [], 0
                for msg in sample.get("messages", []):
                    for call in (msg.get("tool_calls", []) or []):
                        args = call.get("arguments", {})
                        if call.get("function") == "bash" and isinstance(args.get("cmd"), str):
                            heads.extend(command_heads(args["cmd"]))
                        elif call.get("function") == "python":
                            skipped += 1
                trace_id = hashlib.sha256((row["path"] + "/" + name).encode()).hexdigest()[:16]
                model = manifest["model_sets"][path.parent.name]
                output.append(Observation(trace_id, str(sample["id"]), "agent", model, tuple(heads), tuple(ATOM_GROUP[x] for x in heads), skipped))
    return output
