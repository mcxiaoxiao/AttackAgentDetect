#!/usr/bin/env python3
"""Download the pinned CyBench/NYUCTF human and agent trace subset."""

import concurrent.futures
import hashlib
import json
import pathlib
import tempfile
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cyber-task-horizons"
MANIFEST = ROOT / "data" / "manifests" / "crossbenchmark_public_subset.json"
REV = "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae"
MEDIA_URL = f"https://media.githubusercontent.com/media/lyptus-research/cyber-task-horizons-data/{REV}/"


def download(row):
    target = RAW / row["path"]
    if target.is_file() and target.stat().st_size == row["size"]:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest == row["sha256"]:
            return {"path": row["path"], "status": "verified_existing", "size": row["size"]}

    target.parent.mkdir(parents=True, exist_ok=True)
    url = MEDIA_URL + urllib.parse.quote(row["path"], safe="/")
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AttackAgentDetect-research/0.1"})
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=90) as response, tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False
            ) as temp:
                temp_path = pathlib.Path(temp.name)
                while chunk := response.read(1024 * 1024):
                    temp.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != row["size"] or digest.hexdigest() != row["sha256"]:
                temp_path.unlink(missing_ok=True)
                raise ValueError(f"Size/hash mismatch: {row['path']}")
            temp_path.replace(target)
            return {"path": row["path"], "status": "downloaded_verified", "size": size}
        except Exception as exc:  # retry transient network failures only
            last_error = exc
            if attempt < 2:
                time.sleep(1 + attempt)
    raise RuntimeError(f"Failed to download {row['path']}: {last_error}")


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["revision"] != REV:
        raise ValueError("Unexpected dataset revision")
    expected = sum(row["size"] for row in manifest["files"])
    if expected != manifest["total_bytes"]:
        raise ValueError("Manifest total byte count is inconsistent")
    RAW.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, manifest["files"]))
    download_manifest = {
        "source": manifest["source"],
        "revision": REV,
        "benchmark": "cybench, nyuctf",
        "files": results,
        "total_files": len(results),
        "total_bytes": sum(row["size"] for row in results),
    }
    (RAW / "crossbenchmark_download_manifest.json").write_text(
        json.dumps(download_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "downloaded_or_verified": len(results),
        "total_files": len(manifest["files"]),
        "total_bytes": expected,
        "status_counts": {status: sum(row["status"] == status for row in results)
                          for status in sorted({row["status"] for row in results})},
    }, indent=2))


if __name__ == "__main__":
    main()
