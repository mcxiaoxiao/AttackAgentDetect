#!/usr/bin/env python3
"""Verify matched CTH command-argument evidence and B/S predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT / "research/results/matched_fingerprint_v1"
OUT = ROOT / "research/results/matched_fingerprint_v2/argument_profile"
checks = {}


def frame(path):
    return pd.read_csv(path, compression="gzip", keep_default_na=False)


pair_columns = ["pair_id", "fold", "fraction", "query_id", "candidate_id", "query_task",
                "candidate_task", "query_mode", "candidate_mode", "pair_type", "same_task"]
old_pairs = frame(V1 / "cth/pairs.csv.gz")
new_pairs = frame(OUT / "cth/pairs.csv.gz")
assert len(old_pairs) == len(new_pairs) == 54_400
assert old_pairs[pair_columns].equals(new_pairs[pair_columns])
checks["frozen_candidate_pairs"] = {"rows": len(new_pairs), "equal_to_v1": True}

roles = frame(OUT / "cth/role_allocations.csv.gz")
groups = roles.groupby(["replicate", "agent_slots"], observed=True)
assert len(groups) == 850 and (groups.size() == 32).all()
assert all(int((g["mode"] == "agent").sum()) == key[1] for key, g in groups)
checks["exact_role_allocations"] = len(groups)

training = frame(OUT / "cth/training_validation_pairs.csv.gz")
assert len(training) == 1768
assert set(training[training.stage != "inner_validation"].training_weight) == {"1"}
for fold in range(3):
    audit = json.loads((OUT / "cth" / f"fold-{fold}-audit.json").read_text())
    train_tasks, test_tasks = set(audit["train_tasks"]), set(audit["test_tasks"])
    assert not train_tasks & test_tasks
    rows = training[training.outer_fold.astype(int) == fold]
    outer = rows[rows.stage == "outer_train"]
    assert set(outer.query_task).issubset(train_tasks) and set(outer.candidate_task).issubset(train_tasks)
    assert not (set(outer.query_task) | set(outer.candidate_task)) & test_tasks
    for inner in (0, 1):
        fit = set(audit["inner_folds"][inner]["fit_tasks"])
        val = set(audit["inner_folds"][inner]["validation_tasks"])
        assert not fit & val and fit | val == train_tasks
        for stage, allowed in (("inner_train", fit), ("inner_validation", val)):
            part = rows[(rows.stage == stage) & (rows.inner_fold.astype(int) == inner)]
            assert set(part.query_task).issubset(allowed) and set(part.candidate_task).issubset(allowed)
            assert np.all(part.query_id != part.candidate_id)
checks["training_validation_pair_rows"] = len(training)
checks["task_isolation"] = "passed"

pred = frame(OUT / "pair_predictions.csv.gz")
assert len(pred) == 598_400
assert not {"commands", "arguments", "argument_profile"} & set(pred.columns)
checked = 0
for learner, variants in (("xgb", ("head_only", "evidence", "BS", "B", "S")),
                          ("svm", ("head_only", "evidence", "BS")),
                          ("tree", ("head_only", "evidence", "BS"))):
    for fold in range(3):
        baseline = frame(OUT / "cth" / f"fold-{fold}-{learner}-evidence.csv.gz")
        for variant in variants:
            p = frame(OUT / "cth" / f"fold-{fold}-{learner}-{variant}.csv.gz")
            assert p[pair_columns].equals(baseline[pair_columns]), (fold, learner, variant)
            assert p.representation.nunique() == 1 and p.parameter.nunique() == 1
            assert p.representation.iloc[0] == baseline.representation.iloc[0]
            assert float(p.parameter.iloc[0]) == float(baseline.parameter.iloc[0])
            score, threshold = pd.to_numeric(p.score).to_numpy(), pd.to_numeric(p.threshold).to_numpy()
            assert np.array_equal((score >= threshold).astype(int), p.predicted_link.astype(int).to_numpy())
            if variant not in ("head_only", "evidence"):
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
checks["argument_profile_traces"] = json.loads((OUT / "cth/input_audit.json").read_text())["argument_profile_traces"]
checks["prediction_sha256"] = hashlib.sha256((OUT / "pair_predictions.csv.gz").read_bytes()).hexdigest()
checks["source_profile_sha256"] = json.loads((OUT / "environment.json").read_text())["argument_profile_sha256"]
(OUT / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
