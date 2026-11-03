#!/usr/bin/env python3
"""Verify the frozen NYU confirmation panel, pair manifest, and predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "research/results/matched_fingerprint_v2"
OUT = RESULTS / "independent_nyu_agent"
TASKS = RESULTS / "independent_nyu_task_manifest.json"
PROTOCOL = ROOT / "research/experiments/matched_fingerprint_v2/independent_nyu_aa_protocol.md"
RUN = OUT / "run_manifest.json"
PREDICTIONS = OUT / "predictions.csv.gz"
PAIRS = OUT / "pair_manifest.csv.gz"
TRAINING = OUT / "training_pairs.csv.gz"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    task_manifest = json.loads(TASKS.read_text(encoding="utf8"))
    run = json.loads(RUN.read_text(encoding="utf8"))
    analysis = json.loads((OUT / "analysis.json").read_text(encoding="utf8"))
    assert task_manifest["status"] == "frozen_before_confirmation_model_fit"
    assert run["status"] == "executed_once_frozen_cross_source_confirmation"
    assert run["source_revision"] == task_manifest["source_revision"]
    assert run["protocol_sha256"] == task_manifest["protocol_sha256"] == sha(PROTOCOL)
    assert run["task_manifest_sha256"] == sha(TASKS)
    assert run["inputs_sha256"]["runner"] == sha(
        ROOT / "research/experiments/matched_fingerprint_v2/run_independent_nyu_agent.py")
    assert run["pair_manifest_sha256"] == sha(PAIRS)
    assert analysis["provenance_sha256"]["predictions"] == sha(PREDICTIONS)
    assert analysis["provenance_sha256"]["pair_manifest"] == sha(PAIRS)
    assert analysis["provenance_sha256"]["run_manifest"] == sha(RUN)

    task_rows = task_manifest["eligible_tasks"]
    task_partition = {row["task"]: int(row["candidate_partition"]) for row in task_rows}
    eligible = set(task_partition)
    assert eligible.isdisjoint(set(task_manifest["preexcluded_task_slugs"]))
    assert len(eligible) == task_manifest["eligible_task_count"] == 160
    assert sum(len(row["traces"]) for row in task_rows) == task_manifest[
        "eligible_trace_count"] == 585
    pair = pd.read_csv(PAIRS, compression="gzip")
    assert len(pair) == run["pair_manifest_rows"] == 234_000
    assert set(pair.pair_type) == {"AA"}
    assert set(pair.query_task) == eligible and set(pair.candidate_task) <= eligible
    assert np.all(pair.same_task.astype(int).to_numpy() ==
                  (pair.query_task == pair.candidate_task).to_numpy().astype(int))
    assert np.all(pair.query_id != pair.candidate_id)
    assert all(task_partition[row.query_task] == int(row.fold) and
               task_partition[row.candidate_task] == int(row.fold)
               for row in pair[["query_task", "candidate_task", "fold"]].itertuples(index=False))
    gallery = pair.groupby(["replicate", "query_id", "query_task"], observed=True)
    assert gallery.size().eq(4).all() and len(gallery) == 58_500
    for (_, query_id, query_task), rows in gallery:
        assert rows.query_id.nunique() == 1 and rows.query_id.iloc[0] == query_id
        assert rows.query_task.nunique() == 1 and rows.query_task.iloc[0] == query_task
        assert rows.same_task.astype(int).sum() == 1
        assert len(rows[rows.same_task == 0].candidate_task.unique()) == 3
        assert rows.candidate_position.astype(int).tolist() == [0, 1, 2, 3]

    training = pd.read_csv(TRAINING, compression="gzip")
    calibration_folds = list(run["threshold_calibration"]["folds"].values())
    development_tasks = set().union(*[
        set(fold["training_tasks"]) | set(fold["validation_tasks"])
        for fold in calibration_folds])
    assert len(development_tasks) == 17
    assert len(training) == run["training_pair_count"] == 3_570
    assert set(training.query_task) <= development_tasks
    assert set(training.candidate_task) <= development_tasks
    assert eligible.isdisjoint(set(training.query_task) | set(training.candidate_task))
    assert np.all(training.query_id != training.candidate_id)
    assert np.all(training.same_task.astype(int).to_numpy() ==
                  (training.query_task == training.candidate_task).to_numpy().astype(int))
    for label in (0, 1):
        mass = training.loc[training.same_task == label, "sample_weight"].sum()
        assert abs(float(mass) - 136.0) < 1e-8
        cell_mass = training[training.same_task == label].groupby("pair_type").sample_weight.sum()
        assert cell_mass.max() - cell_mass.min() < 1e-8

    for fold in calibration_folds:
        fit, validation = set(fold["training_tasks"]), set(fold["validation_tasks"])
        assert fit.isdisjoint(validation) and fit | validation == development_tasks
        assert not (fit | validation) & eligible
    for learner, versions in run["threshold_calibration"]["threshold_metrics"].items():
        for version in ("E", "E_BS"):
            assert versions[version]["oof_pairs"] > 0
            assert versions[version]["oof_macro_fmr"] <= 0.10 + 1e-12

    prediction = pd.read_csv(PREDICTIONS, compression="gzip")
    assert len(prediction) == run["prediction_rows"] == 1_404_000
    assert set(prediction.learner) == {"svm", "tree", "xgb"}
    assert set(prediction.variant) == {"E", "E_BS"}
    assert prediction.groupby(["learner", "variant"], observed=True).size().eq(234_000).all()
    assert prediction.fallback.astype(int).sum() == 0
    assert np.isfinite(pd.to_numeric(prediction.score)).all()
    assert prediction.groupby(["learner", "variant"], observed=True).threshold.nunique().eq(1).all()
    assert np.all((pd.to_numeric(prediction.score) >=
                   pd.to_numeric(prediction.threshold)).astype(int).to_numpy() ==
                  prediction.predicted_link.astype(int).to_numpy())

    checks = {
        "frozen_protocol_hashes_match": True,
        "source_revision": run["source_revision"],
        "confirmation_tasks": len(eligible), "confirmation_traces": 585,
        "candidate_pairs": len(pair), "query_galleries": len(gallery),
        "predictions": len(prediction), "training_pairs": len(training),
        "all_pair_lists_identical": analysis["pair_keys_identical_across_learners_and_versions"],
        "external_fallback_pairs": int(prediction.fallback.astype(int).sum()),
        "training_task_isolation": True, "candidate_partition_isolation": True,
        "threshold_oof_fmr_limit": True, "prediction_threshold_arithmetic": True,
        "pair_manifest_sha256": sha(PAIRS), "prediction_sha256": sha(PREDICTIONS),
    }
    (OUT / "verification.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
