#!/usr/bin/env python3
"""Run the complete, bounded analysis after the pinned public traces are downloaded."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STEPS = (
    "audit_public_traces.py",
    "run_exploration.py",
    "run_sensitivity.py",
    "run_trace_baseline.py",
    "run_split_diagnostics.py",
    "summarize_results.py",
    "verify_results.py",
)


def main():
    for name in STEPS:
        path = ROOT / "scripts" / name
        print(f"\n==> {path.relative_to(ROOT)}", flush=True)
        subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
