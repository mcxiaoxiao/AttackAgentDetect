#!/usr/bin/env python3
"""Rebuild the unweighted CTH inner/outer pair manifest for argument study."""
import csv
import gzip
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "matched_v1", ROOT / "research/experiments/matched_fingerprint_v1/run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)
OUT = ROOT / "research/results/matched_fingerprint_v2/argument_profile/cth"


def pair_type(a, b):
    if a == b == "human":
        return "HH"
    if a == b == "agent":
        return "AA"
    return "HA"


def append_train(dest, rows, pairs, fold, stage, inner=-1):
    for index, (a, b, y) in enumerate(pairs):
        left, right = rows[a], rows[b]
        dest.append({"dataset": "cth", "outer_fold": fold, "stage": stage,
                     "inner_fold": inner, "index": index, "query_id": left["id"],
                     "candidate_id": right["id"], "query_task": left["task"],
                     "candidate_task": right["task"], "query_mode": left["mode"],
                     "candidate_mode": right["mode"],
                     "pair_type": pair_type(left["mode"], right["mode"]),
                     "same_task": int(y), "training_weight": "1"})


def append_validation(dest, rows, pairs, fold, inner):
    for index, p in enumerate(pairs):
        left, right = rows[p["query_idx"]], rows[p["candidate_idx"]]
        dest.append({"dataset": "cth", "outer_fold": fold, "stage": "inner_validation",
                     "inner_fold": inner, "index": index, "query_id": left["id"],
                     "candidate_id": right["id"], "query_task": left["task"],
                     "candidate_task": right["task"], "query_mode": left["mode"],
                     "candidate_mode": right["mode"], "pair_type": p["pair_type"],
                     "same_task": p["same_task"], "training_weight": ""})


def main():
    rows, tasks, folds = RUN.build_cth()
    records = []
    for fold in range(3):
        train_tasks = {t for t in tasks if folds[t] != fold}
        audit = json.loads((OUT / f"fold-{fold}-audit.json").read_text())
        for inner in (0, 1):
            mapping = {t: i % 2 for i, t in enumerate(sorted(train_tasks))}
            val_tasks = {t for t in train_tasks if mapping[t] == inner}
            fit_tasks = train_tasks - val_tasks
            train = RUN.train_pairs(rows, fit_tasks, RUN.SEED + 10000 * fold + 257 * inner)
            val = RUN.gallery(rows, val_tasks, inner, 1, RUN.SEED + 50000 * fold + 101 * inner)
            expected = audit["inner_folds"][inner]
            assert len(train) == expected["train_pairs"] and len(val) == expected["validation_pairs"]
            append_train(records, rows, train, fold, "inner_train", inner)
            append_validation(records, rows, val, fold, inner)
        outer = RUN.train_pairs(rows, train_tasks, RUN.SEED + 900000 + fold)
        assert len(outer) == audit["train_pairs"]
        append_train(records, rows, outer, fold, "outer_train")
    path = OUT / "training_validation_pairs.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} pair records to {path}")


if __name__ == "__main__":
    main()
