#!/usr/bin/env python3
"""Rebuild every inner/outer training and validation pair with role weights."""
import csv
import gzip
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.experiments.matched_fingerprint_v2 import run_role_balanced as runner

runner.ROLE_BALANCED = True
OUT = ROOT / "research/results/matched_fingerprint_v2/role_balanced/cth"


def add_train_rows(dest, rows, pairs, dataset, fold, stage, inner=-1):
    a, b, y = runner.train_arrays(pairs)
    weights = runner.pair_type_weights(rows, a, b, y, enabled=True)
    for i, (x, z, label, weight) in enumerate(zip(a, b, y, weights)):
        dest.append({"dataset": dataset, "outer_fold": fold, "stage": stage,
                     "inner_fold": inner, "index": i,
                     "query_id": rows[x]["id"], "candidate_id": rows[z]["id"],
                     "query_task": rows[x]["task"], "candidate_task": rows[z]["task"],
                     "query_mode": rows[x]["mode"], "candidate_mode": rows[z]["mode"],
                     "pair_type": runner.pair_type(rows[x]["mode"], rows[z]["mode"]),
                     "same_task": int(label), "training_weight": f"{weight:.12g}"})


def add_validation_rows(dest, rows, pairs, dataset, fold, inner):
    for i, pair in enumerate(pairs):
        a, b = rows[pair["query_idx"]], rows[pair["candidate_idx"]]
        dest.append({"dataset": dataset, "outer_fold": fold, "stage": "inner_validation",
                     "inner_fold": inner, "index": i, "query_id": a["id"],
                     "candidate_id": b["id"], "query_task": a["task"],
                     "candidate_task": b["task"], "query_mode": a["mode"],
                     "candidate_mode": b["mode"], "pair_type": pair["pair_type"],
                     "same_task": pair["same_task"], "training_weight": ""})


def main():
    dataset = "cth"
    rows, tasks, folds = runner.build_cth()
    records = []
    for fold in range(3):
        train_tasks = {t for t in tasks if folds[t] != fold}
        fold_audit = json.loads((OUT / f"fold-{fold}-audit.json").read_text())
        for inner in (0, 1):
            mapping = {task: i % 2 for i, task in enumerate(sorted(train_tasks))}
            val_tasks = {t for t in train_tasks if mapping[t] == inner}
            fit_tasks = train_tasks - val_tasks
            train = runner.train_pairs(rows, fit_tasks, runner.SEED + 10000 * fold + 257 * inner)
            val = runner.gallery(rows, val_tasks, inner, 1, runner.SEED + 50000 * fold + 101 * inner)
            expected = fold_audit["inner_folds"][inner]
            assert len(train) == expected["train_pairs"] and len(val) == expected["validation_pairs"]
            add_train_rows(records, rows, train, dataset, fold, "inner_train", inner)
            add_validation_rows(records, rows, val, dataset, fold, inner)
        train = runner.train_pairs(rows, train_tasks, runner.SEED + 900000 + fold)
        assert len(train) == fold_audit["train_pairs"]
        add_train_rows(records, rows, train, dataset, fold, "outer_train")
    path = OUT / "training_validation_pairs.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} pair records to {path}")


if __name__ == "__main__":
    main()
