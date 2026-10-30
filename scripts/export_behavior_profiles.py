#!/usr/bin/env python3
"""Save label-free numeric profiles with optional public baseline predictions."""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attackagent import fit_detector, predict_modes
from attackagent.jsonl_cli import read_evidence_jsonl, read_labeled_jsonl
from attackagent.profiles import build_behavior_profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--train", type=pathlib.Path, help="optional format-matched labels for the existing public TF-IDF baseline")
    args = parser.parse_args(argv)
    if args.output.resolve() in {args.input.resolve(), args.train.resolve() if args.train else None}:
        parser.error("output must differ from input and training files")
    rows = read_evidence_jsonl(args.input)
    scores = predict_modes(fit_detector(read_labeled_jsonl(args.train)), rows) if args.train else [None] * len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row, score in zip(rows, scores):
            handle.write(json.dumps(build_behavior_profile(row, score), ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "profiles": len(rows), "numeric_features": 24,
                      "tensor_shape": [4, 8, 3],
                      "scorer": "existing_public_tfidf_baseline" if args.train else None}))


if __name__ == "__main__":
    main()
