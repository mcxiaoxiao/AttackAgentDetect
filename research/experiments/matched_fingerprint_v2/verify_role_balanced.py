#!/usr/bin/env python3
"""Verify exact pairing, task isolation, weights, predictions, and metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT / "research/results/matched_fingerprint_v1"
OUT = ROOT / "research/results/matched_fingerprint_v2/role_balanced"
checks = {}


def frame(path):
    return pd.read_csv(path, compression="gzip", keep_default_na=False)


old_pairs = frame(V1 / "cth/pairs.csv.gz")
new_pairs = frame(OUT / "cth/pairs.csv.gz")
pair_cols = ["pair_id", "fold", "fraction", "query_id", "candidate_id", "query_task",
             "candidate_task", "query_mode", "candidate_mode", "pair_type", "same_task"]
assert len(old_pairs) == len(new_pairs) == 54_400
assert old_pairs[pair_cols].equals(new_pairs[pair_cols])
checks["frozen_candidate_pairs"] = {"rows": len(new_pairs), "equal_to_v1": True}

roles = frame(OUT / "cth/role_allocations.csv.gz")
groups = roles.groupby(["replicate", "agent_slots"], observed=True)
assert len(groups) == 850 and (groups.size() == 32).all()
assert all(int((g["mode"] == "agent").sum()) == key[1] for key, g in groups)
checks["exact_role_allocations"] = 850

training = frame(OUT / "cth/training_validation_pairs.csv.gz")
assert len(training) == 1768
for fold in range(3):
    audit = json.loads((OUT / "cth" / f"fold-{fold}-audit.json").read_text())
    train_tasks, test_tasks = set(audit["train_tasks"]), set(audit["test_tasks"])
    assert not train_tasks & test_tasks
    part = training[training.outer_fold.astype(int) == fold]
    assert set(part[part.stage == "outer_train"].query_task).issubset(train_tasks)
    assert set(part[part.stage == "outer_train"].candidate_task).issubset(train_tasks)
    assert not set(part[part.stage == "outer_train"].query_task) & test_tasks
    for inner in (0, 1):
        fit = set(audit["inner_folds"][inner]["fit_tasks"])
        val = set(audit["inner_folds"][inner]["validation_tasks"])
        assert not fit & val and fit | val == train_tasks
        for stage, tasks in (("inner_train", fit), ("inner_validation", val)):
            x = part[(part.stage == stage) & (part.inner_fold.astype(int) == inner)]
            assert set(x.query_task).issubset(tasks) and set(x.candidate_task).issubset(tasks)
            assert np.all(x.query_id != x.candidate_id)
    trains = part[part.stage.isin(["inner_train", "outer_train"])]
    for key, x in trains.groupby(["stage", "inner_fold", "same_task"], observed=True):
        sums = x.groupby("pair_type", observed=True).training_weight.apply(
            lambda z: pd.to_numeric(z).sum())
        assert sums.max() - sums.min() < 1e-8, (fold, key, sums.to_dict())
checks["training_pair_rows"] = len(training)
checks["task_split_and_role_weights"] = "passed"

pred = frame(OUT / "pair_predictions.csv.gz")
assert len(pred) == 435_200
base_cols = pair_cols
checked = 0
for learner, variants in (("xgb", ("evidence", "BS", "B", "S")),
                          ("svm", ("evidence", "BS")),
                          ("tree", ("evidence", "BS"))):
    for fold in range(3):
        baseline = frame(OUT / "cth" / f"fold-{fold}-{learner}-evidence.csv.gz")
        for variant in variants:
            p = frame(OUT / "cth" / f"fold-{fold}-{learner}-{variant}.csv.gz")
            assert p[base_cols].equals(baseline[base_cols]), (fold, learner, variant)
            score, threshold = pd.to_numeric(p.score).to_numpy(), pd.to_numeric(p.threshold).to_numpy()
            assert np.array_equal((score >= threshold).astype(int), p.predicted_link.astype(int).to_numpy())
            mask = p.fallback.astype(int).to_numpy() == 1
            if mask.any():
                assert np.array_equal(p.predicted_link.to_numpy()[mask], baseline.predicted_link.to_numpy()[mask])
                assert np.allclose(score[mask], pd.to_numeric(baseline.score).to_numpy()[mask])
            checked += len(p)
checks["prediction_rows_verified"] = checked

summary = pd.read_csv(OUT / "summary.csv")
for _, row in summary[summary.pair_type == "ALL"].iterrows():
    assert abs(row.ba - (1 - (row.fmr + row.fsr) / 2)) < 1e-10
checks["task_macro_ba_identity"] = True
checks["unique_pair_versions"] = int(pred.groupby(["learner", "variant"]).size().min())
checks["prediction_sha256"] = hashlib.sha256((OUT / "pair_predictions.csv.gz").read_bytes()).hexdigest()
(OUT / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
