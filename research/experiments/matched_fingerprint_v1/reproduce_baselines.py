#!/usr/bin/env python3
"""Recheck that the new frozen pair list reproduces the prior two baseline curves."""
import collections
import importlib.util
import json
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("match", ROOT / "research/experiments/matched_fingerprint_v1/run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
legacy = m.module(ROOT / "research/experiments/overnight_fingerprint_v1/run.py", "legacy")
old = json.loads((ROOT / "research/results/association_fingerprint_v3/exact_mixture.json").read_text())
rows, tasks, folds = m.build_cth()
pairs, roles = m.cth_pairs(rows, tasks, folds)
out = {"exact_old_code_reproduced": True, "per_pair_reconstruction": []}
for fold in range(3):
    train = [i for i, r in enumerate(rows) if folds[r["task"]] != fold]
    test = [i for i, r in enumerate(rows) if folds[r["task"]] == fold]
    for method in ("command_cosine", "action_bigram_cosine"):
        mat, mapping = legacy.cos_matrix(rows, train, test, method)
        threshold = old["thresholds"][str((fold, method))]
        selected = [p for p in pairs if p["fold"] == fold]
        a = np.asarray([mapping[p["query_idx"]] for p in selected])
        b = np.asarray([mapping[p["candidate_idx"]] for p in selected])
        scores = np.asarray(mat[a].multiply(mat[b]).sum(axis=1)).ravel()
        for p, pred in zip(selected, scores >= threshold):
            p[method] = bool(pred)
for count in (0, 8, 16, 24, 32):
    for method in ("command_cosine", "action_bigram_cosine"):
        delta = []
        for task in tasks:
            selected = [p for p in pairs if p["query_task"] == task and p["fraction"] == count / 32]
            pos = [1 - p[method] for p in selected if p["same_task"]]
            neg = [p[method] for p in selected if not p["same_task"]]
            metrics = np.asarray([np.mean(neg), np.mean(pos), 1 - (np.mean(neg)+np.mean(pos))/2])
            expected = np.asarray(old["task_means"][f"{count}:{method}"][task])
            delta.append(float(np.max(np.abs(metrics-expected))))
        err = max(delta)
        out["per_pair_reconstruction"].append({"agent_slots":count, "method":method, "max_error":err})
        assert err < 1e-12, (count, method, err)
out["max_error"] = max(r["max_error"] for r in out["per_pair_reconstruction"])
m.write_json(m.OUT / "baseline_reproduction.json", out)
print(json.dumps(out, indent=2))
