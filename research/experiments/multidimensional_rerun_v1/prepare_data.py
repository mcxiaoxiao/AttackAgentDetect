#!/usr/bin/env python3
"""Fetch a bounded, pinned CTH subset for the preregistered 7-benchmark rerun."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import pathlib
import re
import tempfile
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/cyber-task-horizons"
TREE_PATH = ROOT / "data/raw/github-tree.json"
OUT = pathlib.Path(__file__).resolve().parent / "source_manifest.json"
REVISION = "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae"
BENCHMARKS = ("cybashbench", "nl2bash", "intercode-ctf", "nyuctf",
              "cybench", "cvebench", "cybergym")
SONNET_EVALSETS = {
    "cybashbench": ("eval-set-mrfumbtyv0pa6iab",),
    "nl2bash": ("eval-set-bcakpnavej4q4jd3",),
    "intercode-ctf": ("eval-set-83d0a1kx3i5t3u6b",),
    "nyuctf": ("eval-set-bjldgp2ll9qroagt",),
    "cybench": ("eval-set-jcdoiurrc2a17nmv",),
    "cvebench": ("eval-set-rnjcl1u90s49rfsx",),
    "cybergym": ("eval-set-zsam5w2kdrpva5pz",),
}
MAX_TOTAL_BYTES = 2_000_000_000
RANGE_THRESHOLD_BYTES = 32_000_000
RANGE_CHUNK_BYTES = 12_000_000
MEDIA_ROOT = f"https://media.githubusercontent.com/media/lyptus-research/cyber-task-horizons-data/{REVISION}/"
RAW_ROOT = f"https://raw.githubusercontent.com/lyptus-research/cyber-task-horizons-data/{REVISION}/"


def fetch(url: str, *, timeout: int = 120, headers: dict | None = None):
    request_headers = {"User-Agent": "AttackAgentDetect-research/0.2", **(headers or {})}
    req = urllib.request.Request(url, headers=request_headers)
    return urllib.request.urlopen(req, timeout=timeout)


def remote_pointer(path: str) -> dict:
    url = RAW_ROOT + urllib.parse.quote(path, safe="/")
    with fetch(url, timeout=30) as response:
        pointer = response.read(4096).decode("utf-8", "replace")
    sha = re.search(r"^oid sha256:([0-9a-f]{64})$", pointer, re.M)
    size = re.search(r"^size (\d+)$", pointer, re.M)
    if not sha or not size:
        raise ValueError(f"Missing Git LFS pointer metadata for {path}")
    return {"path": path, "sha256": sha.group(1), "size": int(size.group(1))}


def _fetch_range(url: str, start: int, end: int, total: int, output: pathlib.Path) -> pathlib.Path:
    expected = end - start + 1
    for attempt in range(3):
        try:
            with fetch(url, timeout=180, headers={"Range": f"bytes={start}-{end}"}) as response:
                expected_range = f"bytes {start}-{end}/{total}"
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise ValueError(f"Unexpected range response: {response.status} {response.headers.get('Content-Range')}")
                written = 0
                with output.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        written += len(chunk)
            if written != expected:
                raise ValueError(f"Range length mismatch: expected {expected}, got {written}")
            return output
        except Exception:
            output.unlink(missing_ok=True)
            if attempt == 2:
                raise
    raise AssertionError("unreachable range retry loop")


def _acquire_ranged(row: dict, target: pathlib.Path, url: str) -> dict:
    ranges = [(start, min(start + RANGE_CHUNK_BYTES, row["size"]) - 1)
              for start in range(0, row["size"], RANGE_CHUNK_BYTES)]
    hasher = hashlib.sha256()
    size = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=target.name + ".parts.") as directory:
        part_paths = [pathlib.Path(directory) / f"part-{index:04d}.bin" for index in range(len(ranges))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(ranges))) as pool:
            futures = [pool.submit(_fetch_range, url, start, end, row["size"], part_paths[index])
                       for index, (start, end) in enumerate(ranges)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False
        ) as output:
            temp_path = pathlib.Path(output.name)
            for part_path in part_paths:
                with part_path.open("rb") as part:
                    while chunk := part.read(1024 * 1024):
                        output.write(chunk)
                        hasher.update(chunk)
                        size += len(chunk)
    if size != row["size"] or hasher.hexdigest() != row["sha256"]:
        temp_path.unlink(missing_ok=True)
        raise ValueError(f"LFS size/hash verification failed after ranged download: {row['path']}")
    temp_path.replace(target)
    return row | {"status": "downloaded_verified"}


def target_paths(tree: dict) -> tuple[list[str], list[str]]:
    entries = [row["path"] for row in tree["tree"]]
    human = [path for path in entries
             if path.startswith("data/human/eval_logs/") and path.endswith(".eval.gz")
             and path.split("/")[3] in BENCHMARKS]
    agent = [path for path in entries
             if path.startswith("data/eval_logs/") and path.endswith(".eval")
             and any(f"/{evalset}/" in path for evalsets in SONNET_EVALSETS.values()
                     for evalset in evalsets)]
    missing_sets = [evalset for evalsets in SONNET_EVALSETS.values() for evalset in evalsets
                    if not any(f"/{evalset}/" in path for path in agent)]
    if missing_sets:
        raise ValueError(f"Pinned tree does not include selected model sets: {missing_sets}")
    if not human or not agent:
        raise ValueError("Pinned tree is missing Human or Agent records")
    return sorted(human), sorted(agent)


def acquire(row: dict) -> dict:
    target = RAW / row["path"]
    if target.is_file() and target.stat().st_size == row["size"]:
        hasher = hashlib.sha256()
        with target.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest == row["sha256"]:
            return row | {"status": "verified_existing"}
    target.parent.mkdir(parents=True, exist_ok=True)
    url = MEDIA_ROOT + urllib.parse.quote(row["path"], safe="/")
    if row["size"] >= RANGE_THRESHOLD_BYTES:
        return _acquire_ranged(row, target, url)
    digest = hashlib.sha256()
    size = 0
    with fetch(url) as response, tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False
    ) as temp:
        temp_path = pathlib.Path(temp.name)
        while chunk := response.read(1024 * 1024):
            temp.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != row["size"] or digest.hexdigest() != row["sha256"]:
        temp_path.unlink(missing_ok=True)
        raise ValueError(f"LFS size/hash verification failed: {row['path']}")
    temp_path.replace(target)
    return row | {"status": "downloaded_verified"}


def main():
    tree = json.loads(TREE_PATH.read_text(encoding="utf-8"))
    if tree.get("sha") != REVISION:
        raise ValueError(f"Pinned Git tree mismatch: {tree.get('sha')}")
    human_paths, agent_paths = target_paths(tree)
    rows = [remote_pointer(path) for path in (*human_paths, *agent_paths)]
    total = sum(row["size"] for row in rows)
    if total > MAX_TOTAL_BYTES or any(row["size"] > 500_000_000 for row in rows):
        raise ValueError(f"Bounded acquisition budget exceeded: {total} bytes")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        acquired = list(pool.map(acquire, rows))
    by_path = {row["path"]: row for row in acquired}
    manifest = {
        "source": "https://github.com/lyptus-research/cyber-task-horizons-data",
        "revision": REVISION,
        "license": "CC BY 4.0",
        "model": "claude-sonnet-4.6",
        "benchmarks": list(BENCHMARKS),
        "human_files": [by_path[path] for path in human_paths],
        "agent_evalsets": SONNET_EVALSETS,
        "agent_files": [by_path[path] for path in agent_paths],
        "requested_file_count": len(rows),
        "requested_bytes": total,
        "verified_status_counts": {
            status: sum(row["status"] == status for row in acquired)
            for status in sorted({row["status"] for row in acquired})
        },
        "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"human_files", "agent_files"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
