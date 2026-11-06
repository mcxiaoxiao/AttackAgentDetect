#!/usr/bin/env python3
"""Nested single-measure ablation on CTH development folds.

One B/S window measurement is selected for removal using only inner task folds.
The selected compact model and full model are then compared on the outer fold.
All results are exploratory because this CTH cohort has been used in development.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V1_RUN = ROOT / "research/experiments/matched_fingerprint_v1/run.py"
V1_PRED = ROOT / "research/results/matched_fingerprint_v1/pair_predictions.csv.gz"
V1_AUDIT = ROOT / "research/results/matched_fingerprint_v1/cth"
OUT = ROOT / "research/results/matched_fingerprint_v2/dimension_selection_20261104"
N_BOOT = 10_000
BOOT_SEED = 20261104
DIMENSIONS = tuple([f"B{i}" for i in range(1, 9)] + [f"S{i}" for i in range(1, 9)])


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("matched_v1_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_features(features, evidence, left, right, pooling, fusion,
                  dropped: str | None = None):
    parts = [evidence]
    for block in ("B", "S"):
        values = np.abs(features.z[block][left] - features.z[block][right])
        if fusion == "conditional":
            values = values * evidence[:, :1]
        if dropped is not None and dropped.startswith(block):
            dimension = int(dropped[1:]) - 1
            width = 3 if pooling == "mean_std_trend" else 2
            mask = np.ones(values.shape[1], dtype=bool)
            mask[dimension * width:(dimension + 1) * width] = False
            values = values[:, mask]
        parts.append(values)
    missing = ~(features.ok["B"][left] & features.ok["B"][right] &
                features.ok["S"][left] & features.ok["S"][right])
    return np.column_stack(parts), missing


def inner_predictions(runner, contexts, kind, param, rep):
    window, pooling, fusion = rep
    task_parts, base_parts, command_parts = [], [], []
    candidates = {"full": [], **{name: [] for name in DIMENSIONS}}
    missing = {name: [] for name in candidates}
    for context in contexts:
        features, train_e, val_e = context["features"][(window, pooling)]
        base_parts.append(runner.fit_scores(kind, param, train_e, context["train_y"], val_e))
        command_parts.append(runner.fit_scores(kind, param, train_e[:, :1],
                                                context["train_y"], val_e[:, :1]))
        task_parts.append((context["val_y"], context["val_tasks"]))
        for dropped in candidates:
            x_train, _ = pair_features(features, train_e, context["train_a"],
                                       context["train_b"], pooling, fusion,
                                       None if dropped == "full" else dropped)
            x_val, no = pair_features(features, val_e, context["val_a"],
                                      context["val_b"], pooling, fusion,
                                      None if dropped == "full" else dropped)
            candidates[dropped].append(runner.fit_scores(
                kind, param, x_train, context["train_y"], x_val))
            missing[dropped].append(no)
    y = np.concatenate([part[0] for part in task_parts])
    tasks = np.concatenate([part[1] for part in task_parts])
    base = np.concatenate(base_parts)
    command = np.concatenate(command_parts)
    base_threshold = runner.threshold(y, base)
    base_pred = runner.decision(base, base_threshold)
    command_threshold = runner.threshold(y, command)
    command_pred = runner.decision(command, command_threshold)
    summaries = {
        "evidence": {"threshold": base_threshold,
                     "metrics": runner.macro(y, base_pred, tasks)},
        "command_only": {"threshold": command_threshold,
                         "metrics": runner.macro(y, command_pred, tasks)},
    }
    scores = {"evidence": base, "command_only": command}
    thresholds = {"evidence": base_threshold, "command_only": command_threshold}
    fallback_masks = {"command_only": np.zeros(len(y), dtype=bool)}
    for name, chunks in candidates.items():
        score = np.concatenate(chunks)
        no = np.concatenate(missing[name])
        threshold = runner.threshold(y, score, no, base_pred)
        pred = runner.decision(score, threshold, no, base_pred)
        summaries[name] = {"threshold": threshold,
                           "metrics": runner.macro(y, pred, tasks),
                           "fallback_pairs": int(no.sum())}
        scores[name], thresholds[name], fallback_masks[name] = score, threshold, no
    compact_candidates = ["full", *DIMENSIONS]
    selected = max(compact_candidates, key=lambda name: (
        round(summaries[name]["metrics"][2], 12),
        int(name != "full"),
        name,
    ))
    return {"y": y, "tasks": tasks, "base_scores": base,
            "base_threshold": base_threshold, "base_pred": base_pred,
            "scores": scores, "thresholds": thresholds,
            "fallback_masks": fallback_masks, "summaries": summaries,
            "selected": selected}


def outer_scores(runner, rows, training_tasks, fold, rep, param, test_frame,
                 profiles, chosen_dimension: str | None):
    train_indices = [i for i, row in enumerate(rows) if row["task"] in training_tasks]
    features = runner.Features(rows, train_indices, profiles, rep[0], rep[1])
    training_pairs = runner.train_pairs(rows, training_tasks,
                                        runner.SEED + 900000 + fold)
    train_a, train_b, train_y = runner.train_arrays(training_pairs)
    trace_index = {row["id"]: i for i, row in enumerate(rows)}
    test_a = np.asarray([trace_index[x] for x in test_frame.query_id], dtype=np.int32)
    test_b = np.asarray([trace_index[x] for x in test_frame.candidate_id], dtype=np.int32)
    test_y = test_frame.same_task.to_numpy(dtype=np.int8)
    e_train = features.evidence(train_a, train_b)
    e_test = features.evidence(test_a, test_b)

    base_model = runner.classifier("xgb", param)
    base_model.fit(e_train, train_y)
    base_test = base_model.predict_proba(e_test)[:, 1]
    command_model = runner.classifier("xgb", param)
    command_model.fit(e_train[:, :1], train_y)
    command_test = command_model.predict_proba(e_test[:, :1])[:, 1]

    full_x_train, _ = pair_features(features, e_train, train_a, train_b,
                                    rep[1], rep[2])
    full_x_test, fallback = pair_features(features, e_test, test_a, test_b,
                                          rep[1], rep[2])
    full_model = runner.classifier("xgb", param)
    full_model.fit(full_x_train, train_y)
    full_test = full_model.predict_proba(full_x_test)[:, 1]

    compact_x_train, _ = pair_features(features, e_train, train_a, train_b,
                                       rep[1], rep[2], chosen_dimension)
    compact_x_test, compact_fallback = pair_features(
        features, e_test, test_a, test_b, rep[1], rep[2], chosen_dimension)
    compact_model = runner.classifier("xgb", param)
    compact_model.fit(compact_x_train, train_y)
    compact_test = compact_model.predict_proba(compact_x_test)[:, 1]
    if not np.array_equal(fallback, compact_fallback):
        raise AssertionError("dropping one measurement changed block eligibility")

    return {"evidence": (base_test, np.zeros(len(test_y), dtype=bool)),
            "command_only": (command_test, np.zeros(len(test_y), dtype=bool)),
            "full": (full_test, fallback),
            "selected_compact": (compact_test, compact_fallback)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runner = load_module(V1_RUN)
    rows, tasks, folds = runner.build_cth()
    predictions = pd.read_csv(V1_PRED, compression="gzip")
    predictions = predictions[(predictions.dataset == "cth") &
                              (predictions.learner == "xgb") &
                              (predictions.variant == "evidence")].copy()
    predictions["fraction"] = predictions.fraction.astype(str)
    predictions["same_task"] = predictions.same_task.astype(int)
    predictions["score"] = predictions.score.astype(float)
    predictions["threshold"] = predictions.threshold.astype(float)
    predictions["predicted_link"] = predictions.predicted_link.astype(int)
    profiles = runner.profile_cache(rows, minimum=2)

    fold_audits = {}
    output_rows = []
    source_index = predictions.groupby("fold", sort=True)
    for fold in (0, 1, 2):
        fold_predictions = source_index.get_group(fold).copy()
        audit_path = V1_AUDIT / f"fold-{fold}-audit.json"
        prior_audit = json.loads(audit_path.read_text())
        rep = tuple(prior_audit["representation"])
        param = int(prior_audit["selection"]["xgb"]["parameter"])
        test_tasks = {task for task in tasks if folds[task] == fold}
        training_tasks = set(tasks) - test_tasks
        contexts = runner.inner_contexts(rows, training_tasks, profiles, fold, "cth")
        inner = inner_predictions(runner, contexts, "xgb", param, rep)
        chosen = inner["selected"]
        dimension = None if chosen == "full" else chosen
        outer = outer_scores(runner, rows, training_tasks, fold, rep, param,
                             fold_predictions, profiles, dimension)
        fold_audits[str(fold)] = {
            "training_tasks": sorted(training_tasks), "test_tasks": sorted(test_tasks),
            "representation": list(rep), "xgb_parameter": param,
            "inner_selected": chosen,
            "inner_oof_metrics": {
                name: {"threshold": value["threshold"],
                      "fmr": float(value["metrics"][0]),
                      "fsr": float(value["metrics"][1]),
                      "ba": float(value["metrics"][2]),
                      "fallback_pairs": value.get("fallback_pairs", 0)}
                for name, value in inner["summaries"].items()},
        }
        if dimension is not None:
            fold_audits[str(fold)]["selected_dimension_drop"] = dimension
        for variant, (score, fallback) in outer.items():
            subset = fold_predictions.reset_index(drop=True)
            if variant == "evidence":
                threshold = inner["thresholds"]["evidence"]
            elif variant == "full":
                selected = "full"
                threshold = inner["thresholds"][selected]
            elif variant == "selected_compact":
                selected = chosen
                if selected == "full":
                    threshold = inner["thresholds"]["full"]
                else:
                    threshold = inner["thresholds"][selected]
            else:
                threshold = inner["thresholds"][variant]
            base_test, _ = outer["evidence"]
            effective_score = np.where(fallback, base_test, score)
            effective_threshold = np.where(fallback, inner["base_threshold"], threshold)
            pred = effective_score >= effective_threshold
            name = "selected_compact" if variant == "selected_compact" else variant
            for i, (_, pair) in enumerate(subset.iterrows()):
                output_rows.append({
                    "dataset": "cth", "fold": fold, "pair_id": pair.pair_id,
                    "replicate": pair.replicate, "fraction": pair.fraction,
                    "query_id": pair.query_id, "candidate_id": pair.candidate_id,
                    "query_task": pair.query_task, "candidate_task": pair.candidate_task,
                    "query_mode": pair.query_mode, "candidate_mode": pair.candidate_mode,
                    "pair_type": pair.pair_type, "same_task": int(pair.same_task),
                    "candidate_position": pair.candidate_position,
                    "learner": "xgb", "variant": name,
                    "representation": "/".join(map(str, rep)), "parameter": param,
                    "score": float(effective_score[i]),
                    "threshold": float(effective_threshold[i]), "predicted_link": int(pred[i]),
                    "fallback": int(fallback[i]),
                    "selected_dimension_drop": dimension or "",
                })

    pred_out = OUT / "outer_fold_predictions.csv.gz"
    pd.DataFrame(output_rows).to_csv(pred_out, index=False, compression="gzip")
    report = {
        "protocol": "nested single-window-measure selection on CTH task-held-out development folds",
        "selection": "maximize inner task-macro BA at each arm's threshold calibrated for pooled validation FMR <= 10%; compare full B/S and 16 one-measure removals; exact ties prefer a removal",
        "model": "XGBoost; each outer fold reuses the representation and parameter selected inside that fold's training tasks by the original matched study",
        "comparison_arms": ["command_only", "evidence", "full", "selected_compact"],
        "fallback": "the saved score and threshold both use E on pairs missing a required block",
        "independent_unit": "CTH challenge task; each task's 5 role-share cells are averaged equally",
        "development_status": "exploratory because this 17-task CTH cohort has been examined in prior iterations",
        "bootstrap": {"unit": "query task", "draws": N_BOOT, "seed": BOOT_SEED},
        "input_sha256": {str(V1_PRED.relative_to(ROOT)): sha256(V1_PRED),
                         str(V1_RUN.relative_to(ROOT)): sha256(V1_RUN)},
        "outer_folds": fold_audits,
        "prediction_file": str(pred_out.relative_to(ROOT)),
        "prediction_sha256": sha256(pred_out),
    }
    (OUT / "dimension_selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({fold: {"drop": value.get("selected_dimension_drop", "none"),
                            "inner_ba": value["inner_oof_metrics"][value["inner_selected"]]["ba"]}
                      for fold, value in fold_audits.items()}, indent=2))
    print(pred_out)


if __name__ == "__main__":
    main()
