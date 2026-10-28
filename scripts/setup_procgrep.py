#!/usr/bin/env python3
"""Fetch a pinned ProcGrep source snapshot and install into this project's venv."""

import hashlib
import pathlib
import subprocess
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
REV = "a98b30e7391a1f42d93e05db7999b0ad21b881f5"
SHA256 = "c2809048b04a72e68aa2fa1b9222780d2394c2dbfc6941aff101abb460433fd0"
URL = f"https://codeload.github.com/hamidahoderinwale/procgrep/zip/{REV}"


def main():
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    archive = EXTERNAL / "procgrep-pinned.zip"
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=90) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("ProcGrep archive SHA-256 does not match")
    base = EXTERNAL.resolve()
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            if not (base / item.filename).resolve().is_relative_to(base):
                raise ValueError("Unsafe path in archive")
        zf.extractall(base)
    source = EXTERNAL / f"procgrep-{REV}"
    python = ROOT / ".venv" / "bin" / "python"
    subprocess.run(["uv", "pip", "install", "--python", str(python), str(source)], check=True)
    print(f"Installed ProcGrep from pinned source {REV}")


if __name__ == "__main__":
    main()
