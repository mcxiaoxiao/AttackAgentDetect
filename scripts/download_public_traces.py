#!/usr/bin/env python3
"""Download a bounded, pinned public corpus subset; verify every Git LFS object."""

import concurrent.futures
import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cyber-task-horizons"
REV = "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae"
MEDIA_URL = f"https://media.githubusercontent.com/media/lyptus-research/cyber-task-horizons-data/{REV}/"
MAX_FILE_BYTES = 35_000_000
MAX_TOTAL_BYTES = 300_000_000


def fetch(url, attempts=3):
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AttackAgentDetect research"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(1 + attempt)


def download(row):
    target = RAW / row["path"]
    if target.exists() and target.stat().st_size == row["size"]:
        if hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]:
            return row | {"status": "verified_existing"}
    target.parent.mkdir(parents=True, exist_ok=True)
    data = fetch(MEDIA_URL + urllib.parse.quote(row["path"], safe="/"))
    if len(data) != row["size"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError(f"Size/hash mismatch: {row['path']}")
    target.write_bytes(data)
    return row | {"status": "downloaded_verified"}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data" / "manifests" / "intercode_public_subset.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest["revision"] != REV:
        raise ValueError("Unexpected dataset revision")
    pointers = manifest["files"]
    size = sum(x["size"] for x in pointers)
    if size != manifest["total_bytes"] or any(x["size"] > MAX_FILE_BYTES for x in pointers) or size > MAX_TOTAL_BYTES:
        raise ValueError(f"Configured download budget exceeded: {size} bytes")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, pointers))
    manifest["files"] = results
    (RAW / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
