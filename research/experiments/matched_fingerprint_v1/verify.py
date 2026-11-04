#!/usr/bin/env python3
"""Check frozen pairing, fallback decisions, table preservation, and PDF bounds."""
import csv
import gzip
import hashlib
import json
import pathlib
import re
import shutil

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "research/results/matched_fingerprint_v1"
PAPER = ROOT / "research/paper"
checks = {}


def load(path):
    return pd.read_csv(path, keep_default_na=False)


for dataset, full, methods in (
        ("cth", "BS", {"xgb": ["evidence", "BS", "B", "S"],
                       "svm": ["evidence", "BS"], "tree": ["evidence", "BS"]}),
        ("ait", "BST", {"xgb": ["evidence", "BST", "ST", "BT", "BS"]})):
    audit = json.loads((OUT / dataset / "input_audit.json").read_text())
    folds = audit["folds"]
    assert audit["run_ids_unique"]
    if dataset == "ait":
        assert audit["raw_hashes_unique"]
    keys = ["pair_id", "replicate", "fraction", "query_id", "candidate_id", "query_task",
            "candidate_task", "query_mode", "candidate_mode", "pair_type", "same_task",
            "candidate_position"]
    checked = 0
    for fold in range(3):
        fold_audit = json.loads((OUT / dataset / f"fold-{fold}-audit.json").read_text())
        assert not set(fold_audit["train_tasks"]) & set(fold_audit["test_tasks"])
        assert len(fold_audit["selection"]["xgb"]["candidates"]) == 36
        manifest = load(OUT / dataset / "training_validation_pairs.csv.gz")
        manifest = manifest[manifest.outer_fold.astype(int) == fold]
        for inner in fold_audit["inner_folds"]:
            fit, val = set(inner["fit_tasks"]), set(inner["validation_tasks"])
            assert not fit & val
            assert fit | val == set(fold_audit["train_tasks"])
            assert not (fit | val) & set(fold_audit["test_tasks"])
            for stage, allowed in (("inner_train", fit), ("inner_validation", val)):
                m = manifest[(manifest.stage == stage) &
                             (manifest.inner_fold.astype(int) == inner["inner"])]
                assert set(m.query_task).issubset(allowed)
                assert set(m.candidate_task).issubset(allowed)
                assert np.all(m.query_id != m.candidate_id)
                assert np.all(m.same_task.astype(int).to_numpy() ==
                              (m.query_task == m.candidate_task).to_numpy().astype(int))
        outer = manifest[manifest.stage == "outer_train"]
        assert set(outer.query_task).issubset(set(fold_audit["train_tasks"]))
        assert set(outer.candidate_task).issubset(set(fold_audit["train_tasks"]))
        expected = load(OUT / dataset / f"fold-{fold}-xgb-evidence.csv.gz")
        assert all(folds[x] == fold for x in expected.query_task)
        assert all(folds[x] == fold for x in expected.candidate_task)
        assert np.all(expected.same_task.astype(int).to_numpy()
                      == (expected.query_task == expected.candidate_task).to_numpy().astype(int))
        assert np.all(expected.query_id != expected.candidate_id)
        for model, variants in methods.items():
            baseline = load(OUT / dataset / f"fold-{fold}-{model}-evidence.csv.gz")
            assert baseline[keys].equals(expected[keys])
            for v in variants:
                p = load(OUT / dataset / f"fold-{fold}-{model}-{v}.csv.gz")
                assert p[keys].equals(expected[keys]), (dataset, fold, model, v)
                assert len(p) == len(baseline)
                score = pd.to_numeric(p.score).to_numpy()
                threshold = pd.to_numeric(p.threshold).to_numpy()
                assert np.all((score >= threshold).astype(int) == p.predicted_link.astype(int))
                mask = p.fallback.astype(int).to_numpy() == 1
                if mask.any():
                    assert np.all(p.predicted_link.to_numpy()[mask] == baseline.predicted_link.to_numpy()[mask])
                    assert np.allclose(score[mask], pd.to_numeric(baseline.score).to_numpy()[mask])
                    assert np.allclose(threshold[mask], pd.to_numeric(baseline.threshold).to_numpy()[mask])
                assert p.representation.nunique() == 1 and p.parameter.nunique() == 1
                assert p.representation.iloc[0] == baseline.representation.iloc[0]
                assert float(p.parameter.iloc[0]) == float(baseline.parameter.iloc[0])
                checked += len(p)
    checks[dataset] = {"prediction_records_verified": checked, "tasks": audit["task_count"],
                       "traces": audit["trace_count"], "frozen_pairs": audit["pair_count"]}

roles = load(OUT / "cth/role_allocations.csv.gz")
groups = roles.groupby(["replicate", "agent_slots"], observed=True)
assert len(groups) == 850
assert (groups.size() == 32).all()
assert all((g["mode"] == "agent").sum() == key[1] for key, g in groups)
checks["exact_role_allocations"] = len(groups)
repro = json.loads((OUT / "baseline_reproduction.json").read_text())
assert repro["max_error"] < 1e-12
checks["legacy_baseline_max_abs_error"] = repro["max_error"]

original = (PAPER / "human-agent-fingerprint-overnight.tex").read_text()
snapshot = json.loads((OUT / "original_table_snapshot.json").read_text())
pattern = (r"\\begin\{table\}(?:\[[^]]+\])?.*?\\end\{table\}|"
           r"\\begin\{longtable\}(?:\[[^]]+\])?\{.*?\\end\{longtable\}")
source_tables = [m.group() for m in re.finditer(pattern, original, re.S)]
assert len(source_tables) == len(snapshot["tables"])
for block, expected in zip(source_tables, snapshot["tables"]):
    assert hashlib.sha256(block.encode()).hexdigest() == expected["sha256"]
checks["original_table_snapshot"] = {
    "count": len(source_tables), "source": snapshot["source"],
    "sha256": [x["sha256"] for x in snapshot["tables"]]}
checks["original_tex_sha256"] = hashlib.sha256(original.encode()).hexdigest()

data = json.loads((OUT / "metrics.json").read_text())
for d in ("cth", "ait"):
    for x in data[d]["methods"]:
        v = x["metrics"]
        if all(v[k]["point"] is not None for k in ("fmr", "fsr", "ba")):
            # Strata may use different task support for positive and negative outcomes.
            if x["pair_type"] == "ALL":
                assert abs(v["ba"]["point"] - (1 - (v["fmr"]["point"] + v["fsr"]["point"])/2)) < 1e-12
checks["task_macro_ba_identity"] = True

# Save a single compressed record stream in addition to fold-local checkpoints.
paths = sorted((OUT / "cth").glob("fold-*-*-*.csv.gz")) + sorted((OUT / "ait").glob("fold-*-*-*.csv.gz"))
with gzip.open(OUT / "pair_predictions.csv.gz", "wt", encoding="utf8") as dest:
    header = None
    for path in paths:
        with gzip.open(path, "rt", encoding="utf8") as src:
            h = src.readline()
            if header is None:
                header = h
                dest.write(h)
            assert h == header
            shutil.copyfileobj(src, dest)
checks["combined_prediction_files"] = len(paths)
(OUT / "verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(checks, ensure_ascii=False, indent=2))
