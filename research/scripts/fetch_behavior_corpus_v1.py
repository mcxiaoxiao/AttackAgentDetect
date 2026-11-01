"""Retrieve only the pinned catalog inputs, validating bytes and SHA-256.

Without --download this checks the local cache and reports missing files.
Raw logs stay outside the published catalog. No source code is downloaded or run.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import tempfile
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
REV = "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae"
MANIFESTS = ROOT / "research/data/manifests"
HF = ROOT / "research/data/external/dataset-review-20261101/hf"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def source_files():
    sources = [MANIFESTS / "intercode_public_subset.json",
               MANIFESTS / "crossbenchmark_public_subset.json",
               ROOT / "research/experiments/multidimensional_rerun_v1/source_manifest.json"]
    files = {}
    for manifest in sources:
        data = json.loads(manifest.read_text())
        if data["revision"] != REV:
            raise ValueError(f"Unexpected CTH revision: {manifest}")
        for field in ("files", "human_files", "agent_files"):
            entries = data.get(field, [])
            if not isinstance(entries, list):
                continue
            for row in entries:
                rel = row["path"]
                item = {"source": "cth", "path": ROOT / "research/data/raw/cyber-task-horizons" / rel,
                        "url": f"https://media.githubusercontent.com/media/lyptus-research/cyber-task-horizons-data/{REV}/"
                               + urllib.parse.quote(rel, safe="/"),
                        "bytes": row["size"], "sha256": row["sha256"]}
                if rel in files and item != files[rel]:
                    raise ValueError(f"Conflicting source manifests: {rel}")
                files[rel] = item
    for label, folder in [("cybench", "antieval-cybench-files"), ("agenttrace", "agenttrace-files")]:
        data = json.loads((MANIFESTS / f"behavior_corpus_hf_{label}_v1.json").read_text())
        for row in data["files"]:
            if f"/resolve/{data['revision']}/" not in row["url"]:
                raise ValueError("HF source must point to the pinned revision")
            files[f"{label}/{row['path']}"] = {"source": "hf", "path": HF / folder / row.get("local", row["path"]),
                                                  "url": row["url"], "bytes": row["bytes"], "sha256": row["sha256"]}
    return list(files.values())


def retrieve(item, download):
    path = item["path"]
    if path.is_file():
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Cache validation failed; preserved existing file: {path}")
        return "verified_existing"
    if not download:
        return "missing"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(item["url"], headers={"User-Agent": "AttackAgentDetect-research/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = pathlib.Path(stream.name)
            size = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                size += len(chunk)
                if size > item["bytes"]:
                    raise ValueError("Source response exceeds its pinned size")
                stream.write(chunk)
        if size != item["bytes"] or digest(temporary) != item["sha256"]:
            raise ValueError(f"Downloaded bytes/hash mismatch: {path}")
        temporary.replace(path)
        return "downloaded_verified"
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing pinned files; valid cached files are reused")
    parser.add_argument("--source", choices=["all", "cth", "hf"], default="all")
    args = parser.parse_args()
    files = [f for f in source_files() if args.source == "all" or f["source"] == args.source]
    states = collections.Counter(retrieve(f, args.download) for f in files)
    print(json.dumps({"files": len(files), "total_bytes": sum(f["bytes"] for f in files),
                      "status_counts": dict(states)}, indent=2))
    return 1 if states["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
