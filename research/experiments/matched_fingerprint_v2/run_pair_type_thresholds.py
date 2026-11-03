#!/usr/bin/env python3
"""Isolate type-specific threshold calibration on the frozen argument model."""
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "research/experiments/matched_fingerprint_v2/run_arguments.py"
SPEC = importlib.util.spec_from_file_location("argument_run", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
OUT = ROOT / "research/results/matched_fingerprint_v2/type_threshold"
INPUT = ROOT / "research/results/matched_fingerprint_v2/argument_profile"


def threshold_by_type(y, scores, pair_types, fallback=None, base_prediction=None):
    pair_types = np.asarray(pair_types, dtype=object)
    result = {}
    for name in sorted(set(pair_types)):
        mask = pair_types == name
        result[name] = M.threshold(
            np.asarray(y)[mask], np.asarray(scores)[mask],
            None if fallback is None else np.asarray(fallback)[mask],
            None if base_prediction is None else np.asarray(base_prediction)[mask])
    return result


def decision_by_type(scores, thresholds, pair_types, fallback=None, base_prediction=None):
    pair_types = np.asarray(pair_types, dtype=object)
    limits = np.asarray([thresholds[name] for name in pair_types], dtype=float)
    predictions = np.asarray(scores, dtype=float) >= limits
    if fallback is not None:
        mask = np.asarray(fallback, dtype=bool)
        predictions[mask] = np.asarray(base_prediction, dtype=bool)[mask]
    return predictions


def finite_threshold(value):
    value = float(value)
    return value if math.isfinite(value) else None


def validation_rates(y, pred, pair_types):
    out = {}
    y, pred = np.asarray(y), np.asarray(pred)
    for name in sorted(set(pair_types)):
        mask = np.asarray(pair_types) == name
        pos, neg = mask & (y == 1), mask & (y == 0)
        out[name] = {"positive_pairs": int(pos.sum()), "negative_pairs": int(neg.sum()),
                     "fmr": float(pred[neg].mean()), "fsr": float(1 - pred[pos].mean())}
    return out


def write_predictions(path, rows, learner, variant, representation, parameter,
                      scores, thresholds, fallback=None, base_predictions=None,
                      base_scores=None, base_thresholds=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("dataset", "fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
               "query_task", "candidate_task", "query_mode", "candidate_mode", "pair_type",
               "same_task", "candidate_position", "learner", "variant", "representation",
               "parameter", "score", "threshold", "predicted_link", "fallback")
    with gzip.open(path, "wt", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i, pair in enumerate(rows):
            is_fallback = bool(fallback is not None and fallback[i])
            if is_fallback:
                score = float(base_scores[i])
                limit = float(base_thresholds[pair["pair_type"]])
                predicted = bool(base_predictions[i])
            else:
                score = float(scores[i])
                limit = float(thresholds[pair["pair_type"]])
                predicted = score >= limit
            record = {k: pair[k] for k in columns if k in pair}
            record.update(dataset="cth", fold=pair["fold"], learner=learner, variant=variant,
                          representation="/".join(map(str, representation)), parameter=parameter,
                          score=f"{score:.12g}",
                          threshold="inf" if math.isinf(limit) else f"{limit:.12g}",
                          predicted_link=int(predicted), fallback=int(is_fallback))
            writer.writerow(record)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, tasks, folds = M.build_cth()
    profiles = M.profile_cache(rows, minimum=2)
    pairs, roles = M.cth_pairs(rows, tasks, folds)
    M.write_gz_csv(OUT / "cth-pairs.csv.gz", pairs)
    M.write_gz_csv(OUT / "cth-role-allocations.csv.gz", roles)
    all_thresholds = {"protocol": "per-pair-type inner-OOF FMR <= 10%; fixed global-run model configurations",
                      "source_pair_prediction_sha256": hashlib.sha256(
                          (INPUT / "pair_predictions.csv.gz").read_bytes()).hexdigest(),
                      "folds": {}}
    for fold in sorted(set(folds.values())):
        test_tasks = {t for t in tasks if folds[t] == fold}
        train_tasks = set(tasks) - test_tasks
        audit = json.loads((INPUT / "cth" / f"fold-{fold}-audit.json").read_text())
        representation = tuple(audit["representation"])
        train_idx = [i for i, r in enumerate(rows) if r["task"] in train_tasks]
        test_pairs = [p for p in pairs if p["fold"] == fold]
        ctx = M.inner_contexts(rows, train_tasks, profiles, fold, "cth")
        y_oof = np.concatenate([c["val_y"] for c in ctx])
        task_oof = np.concatenate([c["val_tasks"] for c in ctx])
        # Reconstruct each exact validation gallery from its frozen seed. The
        # runner stores the pair count rather than the pair rows in its context.
        type_chunks = []
        for c in ctx:
            val_pairs = M.gallery(
                rows, set(c["validation_tasks"]), c["inner"], 1,
                M.SEED + 50000 * fold + 101 * c["inner"])
            va, vb, vy, vtask = M.as_arrays(val_pairs)
            assert np.array_equal(va, c["val_a"])
            assert np.array_equal(vb, c["val_b"])
            assert np.array_equal(vy, c["val_y"])
            assert np.array_equal(vtask, c["val_tasks"])
            type_chunks.append(np.asarray([p["pair_type"] for p in val_pairs], dtype=object))
        type_oof = np.concatenate(type_chunks)
        assert len(type_oof) == len(y_oof)
        outer_train = M.train_pairs(rows, train_tasks, M.SEED + 900000 + fold)
        ta, tb, ty = M.train_arrays(outer_train)
        va, vb, _, _ = M.as_arrays(test_pairs)
        features = M.Features(rows, train_idx, profiles, representation[0], representation[1])
        e_train, e_test = features.evidence(ta, tb), features.evidence(va, vb)
        test_types = np.asarray([p["pair_type"] for p in test_pairs], dtype=object)
        fold_thresholds = {"representation": representation, "models": {}}
        for learner in ("xgb", "svm", "tree"):
            selection = audit["selection"][learner]
            parameter = selection["parameter"]
            base_oof = M.oof_base(ctx, learner, parameter)
            full_oof, missing_oof = M.oof_variant(
                ctx, learner, parameter, representation, ("B", "S"))
            base_limits = threshold_by_type(y_oof, base_oof, type_oof)
            base_oof_pred = decision_by_type(base_oof, base_limits, type_oof)
            full_limits = threshold_by_type(y_oof, full_oof, type_oof,
                                            missing_oof, base_oof_pred)
            full_oof_pred = decision_by_type(full_oof, full_limits, type_oof,
                                             missing_oof, base_oof_pred)

            base_test_scores = M.fit_scores(learner, parameter, e_train, ty, e_test)
            x_train, _ = features.full(e_train, ta, tb, ("B", "S"), representation[2])
            x_test, missing_test = features.full(e_test, va, vb, ("B", "S"), representation[2])
            full_test_scores = M.fit_scores(learner, parameter, x_train, ty, x_test)
            base_test_pred = decision_by_type(base_test_scores, base_limits, test_types)
            full_test_pred = decision_by_type(full_test_scores, full_limits, test_types,
                                              missing_test, base_test_pred)

            write_predictions(OUT / f"fold-{fold}-{learner}-evidence.csv.gz", test_pairs,
                              learner, "evidence", representation, parameter,
                              base_test_scores, base_limits)
            write_predictions(OUT / f"fold-{fold}-{learner}-BS.csv.gz", test_pairs,
                              learner, "BS", representation, parameter,
                              full_test_scores, full_limits, missing_test,
                              base_test_pred, base_test_scores, base_limits)
            fold_thresholds["models"][learner] = {
                "parameter": parameter,
                "thresholds": {"evidence": {k: finite_threshold(v) for k, v in base_limits.items()},
                               "BS": {k: finite_threshold(v) for k, v in full_limits.items()}},
                "inner_validation_metrics": {
                    "evidence": validation_rates(y_oof, base_oof_pred, type_oof),
                    "BS": validation_rates(y_oof, full_oof_pred, type_oof)},
                "fallback_pairs": int(missing_oof.sum()),
                "outer_fallback_pairs": int(missing_test.sum())}
        all_thresholds["folds"][str(fold)] = fold_thresholds
        print("fold", fold, "type-specific thresholds selected from inner OOF only", flush=True)
    (OUT / "threshold_audit.json").write_text(json.dumps(all_thresholds, indent=2) + "\n")
    print("completed fixed-configuration threshold diagnostic")


if __name__ == "__main__":
    main()
