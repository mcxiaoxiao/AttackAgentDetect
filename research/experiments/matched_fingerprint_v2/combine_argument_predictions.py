#!/usr/bin/env python3
"""Combine CTH fold-local argument-profile predictions."""
import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/results/matched_fingerprint_v2/argument_profile"
paths = sorted((OUT / "cth").glob("fold-*-*-*.csv.gz"))
assert len(paths) == 33, len(paths)
target = OUT / "pair_predictions.csv.gz"
with gzip.open(target, "wt", encoding="utf8") as dst:
    header = None
    for path in paths:
        with gzip.open(path, "rt", encoding="utf8") as src:
            current = src.readline()
            if header is None:
                header = current
                dst.write(current)
            assert current == header, path
            for line in src:
                dst.write(line)
print(f"combined {len(paths)} files into {target}")
