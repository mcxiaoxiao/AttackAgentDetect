#!/usr/bin/env python3
"""Recreate deterministic training and inner-validation pair manifests."""
import collections
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("matched", ROOT / "research/experiments/matched_fingerprint_v1/run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

for dataset in ("cth", "ait"):
    rows, tasks, folds = m.build_cth() if dataset == "cth" else m.build_ait()
    records = []
    for fold in range(3):
        train_tasks = {t for t in tasks if folds[t] != fold}
        fold_audit = __import__("json").loads((m.OUT / dataset / f"fold-{fold}-audit.json").read_text())
        for inner in (0, 1):
            inner_map = {task: i % 2 for i, task in enumerate(sorted(train_tasks))}
            valid_tasks = {t for t in train_tasks if inner_map[t] == inner}
            fit_tasks = train_tasks - valid_tasks
            train = m.train_pairs(rows, fit_tasks, m.SEED + 10000 * fold + 257 * inner)
            val = m.gallery(rows, valid_tasks, inner, 1, m.SEED + 50000 * fold + 101 * inner)
            assert len(train) == fold_audit["inner_folds"][inner]["train_pairs"]
            assert len(val) == fold_audit["inner_folds"][inner]["validation_pairs"]
            for j, (a, b, y) in enumerate(train):
                records.append({"dataset": dataset, "outer_fold": fold, "stage": "inner_train",
                                "inner_fold": inner, "index": j, "query_id": rows[a]["id"],
                                "candidate_id": rows[b]["id"], "query_task": rows[a]["task"],
                                "candidate_task": rows[b]["task"], "same_task": y})
            for j, p in enumerate(val):
                records.append({"dataset": dataset, "outer_fold": fold, "stage": "inner_validation",
                                "inner_fold": inner, "index": j, "query_id": p["query_id"],
                                "candidate_id": p["candidate_id"], "query_task": p["query_task"],
                                "candidate_task": p["candidate_task"], "same_task": p["same_task"]})
        outer = m.train_pairs(rows, train_tasks, m.SEED + 900000 + fold)
        assert len(outer) == fold_audit["train_pairs"]
        for j, (a, b, y) in enumerate(outer):
            records.append({"dataset": dataset, "outer_fold": fold, "stage": "outer_train",
                            "inner_fold": -1, "index": j, "query_id": rows[a]["id"],
                            "candidate_id": rows[b]["id"], "query_task": rows[a]["task"],
                            "candidate_task": rows[b]["task"], "same_task": y})
    m.write_gz_csv(m.OUT / dataset / "training_validation_pairs.csv.gz", records)
    print(dataset, "training/validation pair records", len(records), flush=True)
