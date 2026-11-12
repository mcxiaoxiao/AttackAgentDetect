#!/usr/bin/env python3
"""Nested task-held-out screening of compact S variants and gallery rejection.

The outer task folds and candidate construction match ``run.py``. Inner
out-of-fold galleries choose one feature arm and its cutoff. Non-E cutoffs are
constrained to no higher task-macro pair FMR and no higher no-match acceptance
than E on those same inner galleries. The script is exploratory: these public
cohorts have already been examined during development.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import pathlib
import sys
from typing import Iterable

import numpy as np
import scipy
import sklearn
from scipy.spatial.distance import jensenshannon
import xgboost

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.experiments.same_mode_bst import run as core  # noqa: E402
from research.experiments.behavioral_fingerprint_v2.command_categories import (  # noqa: E402
    categorize_command,
    CATEGORY_LIST,
)

SEED = 20261111
INNER_REPETITIONS = 20
THRESHOLD_FMR_CEILING = 0.10
DIRICHLET_ALPHA = 0.5

# Each profile is appended to E+B (and observed T on AIT).
ARM_SPEC = {
    "E": {"s": "none", "hard_negative_multiplier": 1.0},
    "base": {"s": "none", "hard_negative_multiplier": 1.0},
    "fullS": {"s": "full", "hard_negative_multiplier": 1.0},
    "S_scalar": {"s": "scalar", "hard_negative_multiplier": 1.0},
    "S_revisit_reuse": {"s": "revisit_reuse", "hard_negative_multiplier": 1.0},
    "S_smoothed_transition": {"s": "smoothed_transition", "hard_negative_multiplier": 1.0},
    "S_ordered_ngrams": {"s": "ordered_ngrams", "hard_negative_multiplier": 1.0},
    "fullS_hardneg2": {"s": "full", "hard_negative_multiplier": 2.0},
    "S_smoothed_hardneg2": {"s": "smoothed_transition", "hard_negative_multiplier": 2.0},
}
ARM_COMPLEXITY = {
    "E": 0, "base": 1, "S_revisit_reuse": 2, "S_scalar": 3,
    "S_smoothed_transition": 4, "S_ordered_ngrams": 5,
    "fullS": 6, "fullS_hardneg2": 6, "S_smoothed_hardneg2": 4,
}


def categorical_sequences(rows: list[dict]) -> tuple[list[list[str | None]], dict[str, np.ndarray]]:
    """Build fixed category sequences and symmetric pair-distance matrices."""
    sequences = [
        [None if (cat := categorize_command(str(command))) == "unknown" else cat
         for command in row["commands"][:128]]
        for row in rows
    ]
    n_rows = len(rows)
    transition_counts = np.zeros((n_rows, len(CATEGORY_LIST) ** 2), dtype=float)
    bigrams: list[set[tuple[str, str]]] = []
    trigrams: list[set[tuple[str, str, str]]] = []
    for idx, seq in enumerate(sequences):
        for i in range(len(seq) - 1):
            left, right = seq[i], seq[i + 1]
            if left is not None and right is not None:
                transition_counts[idx,
                    CATEGORY_LIST.index(left) * len(CATEGORY_LIST) + CATEGORY_LIST.index(right)] += 1
        bigrams.append({(seq[i], seq[i + 1]) for i in range(len(seq) - 1)
                        if seq[i] is not None and seq[i + 1] is not None})
        trigrams.append({(seq[i], seq[i + 1], seq[i + 2]) for i in range(len(seq) - 2)
                         if seq[i] is not None and seq[i + 1] is not None and seq[i + 2] is not None})

    js_distance = np.full((n_rows, n_rows), 0.5, dtype=float)
    js_missing = np.ones((n_rows, n_rows), dtype=float)
    for i in range(n_rows):
        if transition_counts[i].sum() == 0:
            continue
        p = (transition_counts[i] + DIRICHLET_ALPHA) / (
            transition_counts[i].sum() + DIRICHLET_ALPHA * transition_counts.shape[1]
        )
        for j in range(i):
            if transition_counts[j].sum() == 0:
                continue
            q = (transition_counts[j] + DIRICHLET_ALPHA) / (
                transition_counts[j].sum() + DIRICHLET_ALPHA * transition_counts.shape[1]
            )
            js_distance[i, j] = js_distance[j, i] = float(jensenshannon(p, q, base=2.0))
            js_missing[i, j] = js_missing[j, i] = 0.0

    bigram_distance = np.full((n_rows, n_rows), 0.5, dtype=float)
    bigram_missing = np.ones((n_rows, n_rows), dtype=float)
    trigram_distance = np.full((n_rows, n_rows), 0.5, dtype=float)
    trigram_missing = np.ones((n_rows, n_rows), dtype=float)
    for i in range(n_rows):
        for j in range(i):
            if bigrams[i] and bigrams[j]:
                union = bigrams[i] | bigrams[j]
                bigram_distance[i, j] = bigram_distance[j, i] = 1.0 - len(bigrams[i] & bigrams[j]) / len(union)
                bigram_missing[i, j] = bigram_missing[j, i] = 0.0
            if trigrams[i] and trigrams[j]:
                union = trigrams[i] | trigrams[j]
                trigram_distance[i, j] = trigram_distance[j, i] = 1.0 - len(trigrams[i] & trigrams[j]) / len(union)
                trigram_missing[i, j] = trigram_missing[j, i] = 0.0

    return sequences, {
        "js_distance": js_distance,
        "js_missing": js_missing,
        "bigram_distance": bigram_distance,
        "bigram_missing": bigram_missing,
        "trigram_distance": trigram_distance,
        "trigram_missing": trigram_missing,
    }


def pair_custom(pairlist: list[tuple], custom: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode not in {"smoothed_transition", "ordered_ngrams"}:
        return np.empty((len(pairlist), 0), dtype=float)
    a = np.asarray([p[0] for p in pairlist], dtype=int)
    b = np.asarray([p[1] for p in pairlist], dtype=int)
    if mode == "smoothed_transition":
        return np.column_stack([custom["js_distance"][a, b], custom["js_missing"][a, b]])
    return np.column_stack([
        custom["bigram_distance"][a, b], custom["bigram_missing"][a, b],
        custom["trigram_distance"][a, b], custom["trigram_missing"][a, b],
    ])


def s_features(evidence: np.ndarray, blocks: dict[str, np.ndarray],
               pairlist: list[tuple], custom: dict[str, np.ndarray],
               variant: str, dataset: str) -> np.ndarray:
    if variant == "E":
        return evidence
    parts = [evidence, blocks["B"]]
    if dataset == "ait":
        parts.append(blocks["T"])
    name = ARM_SPEC[variant]["s"]
    if name == "none":
        return np.column_stack(parts)
    s = blocks["S"]
    if name == "full":
        parts.append(s)
    elif name == "scalar":
        parts.append(s[:, :24])
    elif name == "revisit_reuse":
        # S2--S7: revisit span, returns, and reused fragments; drop novelty and drift.
        parts.append(s[:, 3:21])
    elif name == "smoothed_transition":
        parts.extend([s[:, :24], pair_custom(pairlist, custom, name)])
    elif name == "ordered_ngrams":
        parts.extend([s[:, :24], pair_custom(pairlist, custom, name)])
    else:
        raise ValueError(f"unknown S profile: {name}")
    return np.column_stack(parts)


def missing_for(missing: dict[str, np.ndarray], variant: str, dataset: str) -> np.ndarray:
    if variant == "E":
        return np.zeros(len(missing["B"]), dtype=bool)
    keys = ["B"] + (["T"] if dataset == "ait" else [])
    if ARM_SPEC[variant]["s"] != "none":
        keys.append("S")
    return np.logical_or.reduce([missing[key] for key in keys])


def fit_weights(rows: list[dict], pairlist: list[tuple], multiplier: float) -> np.ndarray:
    weights = core.weights_for(rows, pairlist)
    if multiplier <= 1.0:
        return weights
    cells: dict[tuple[int, str], list[int]] = collections.defaultdict(list)
    hard = np.zeros(len(pairlist), dtype=bool)
    for idx, (a, b, label) in enumerate(pairlist):
        pair_kind = core.base.pair_type(rows[a]["mode"], rows[b]["mode"])
        cells[(int(label), pair_kind)].append(idx)
        hard[idx] = (
            label == 0
            and rows[a]["task"] != rows[b]["task"]
            and rows[a]["mode"] == rows[b]["mode"]
        )
    # Increase this one negative subtype while preserving the original total
    # sample mass in each label x Human/Agent pair-type cell.
    for indices in cells.values():
        ix = np.asarray(indices, dtype=int)
        if not hard[ix].any():
            continue
        target_mass = float(weights[ix].sum())
        weights[ix[hard[ix]]] *= multiplier
        weights[ix] *= target_mass / float(weights[ix].sum())
    return weights


def gallery_metrics(rows: list[dict], galleries: list[tuple], scores: np.ndarray,
                    missing: np.ndarray, threshold: float,
                    baseline_scores: np.ndarray, baseline_threshold: float) -> dict:
    matched: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    unmatched: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    task_counts: dict[str, np.ndarray] = collections.defaultdict(lambda: np.zeros(4, dtype=float))
    no_match_by_task: dict[str, list[float]] = collections.defaultdict(list)
    hit_by_task: dict[str, list[float]] = collections.defaultdict(list)

    final_score = np.where(missing, baseline_scores, scores)
    cutoffs = np.where(missing, baseline_threshold, threshold)
    decisions = final_score >= cutoffs
    for idx, (query, _candidate, replicate, position) in enumerate(galleries):
        key = (query, replicate)
        if position < 4:
            matched[key].append(idx)
        else:
            unmatched[key].append(idx)

    for (query, _replicate), indices in matched.items():
        task = rows[query]["task"]
        # Ranking uses one common score scale per gallery; if any required
        # block is missing, the entire gallery falls back to E.
        rank_scores = baseline_scores[indices] if missing[indices].any() else scores[indices]
        maximum = np.max(rank_scores)
        tied = np.flatnonzero(rank_scores == maximum)
        correct = []
        for tied_index in tied:
            candidate = galleries[indices[tied_index]][1]
            correct.append(float(rows[query]["task"] == rows[candidate]["task"]
                                 and rows[query]["mode"] == rows[candidate]["mode"]))
        hit_by_task[task].append(float(np.mean(correct)))
        for idx in indices:
            candidate = galleries[idx][1]
            label = int(rows[query]["task"] == rows[candidate]["task"]
                        and rows[query]["mode"] == rows[candidate]["mode"])
            if label:
                task_counts[task][3] += 1
                task_counts[task][2] += int(not decisions[idx])
            else:
                task_counts[task][1] += 1
                task_counts[task][0] += int(decisions[idx])

    for (query, _replicate), indices in unmatched.items():
        task = rows[query]["task"]
        no_match_by_task[task].append(float(decisions[indices].any()))

    tasks = sorted(set(task_counts) | set(no_match_by_task))
    per_task = {}
    for task in tasks:
        fp, neg, fn, pos = task_counts[task]
        per_task[task] = {
            "fmr": float(fp / neg) if neg else float("nan"),
            "fsr": float(fn / pos) if pos else float("nan"),
            "hit1": float(np.mean(hit_by_task[task])) if hit_by_task[task] else float("nan"),
            "no_match_accept": float(np.mean(no_match_by_task[task])) if no_match_by_task[task] else float("nan"),
        }
        per_task[task]["ba"] = 1.0 - (per_task[task]["fmr"] + per_task[task]["fsr"]) / 2.0
    metrics = {
        name: float(np.nanmean([per_task[t][name] for t in tasks]))
        for name in ("hit1", "ba", "fmr", "fsr", "no_match_accept")
    }
    return {"metrics": metrics, "per_task": per_task}


def guarded_cutoff(rows: list[dict], galleries: list[tuple], scores: np.ndarray,
                   missing: np.ndarray, baseline_scores: np.ndarray,
                   baseline_threshold: float) -> dict:
    baseline = gallery_metrics(rows, galleries, baseline_scores,
                               np.zeros(len(galleries), dtype=bool),
                               baseline_threshold, baseline_scores, baseline_threshold)
    base_fmr = baseline["metrics"]["fmr"]
    base_nm = baseline["metrics"]["no_match_accept"]
    # As the cutoff rises, pair FMR and any-accept no-match rate can only fall,
    # while recall can only fall. Search the first feasible score instead of
    # rescanning every candidate score; this preserves the exact constraints.
    candidates = np.r_[np.unique(scores[~missing]), np.inf]
    cache: dict[int, tuple[dict, bool]] = {}

    def evaluate(index: int) -> tuple[dict, bool]:
        if index not in cache:
            result = gallery_metrics(rows, galleries, scores, missing,
                                     float(candidates[index]), baseline_scores,
                                     baseline_threshold)
            metric = result["metrics"]
            feasible = (
                metric["fmr"] <= min(THRESHOLD_FMR_CEILING, base_fmr) + 1e-12
                and metric["no_match_accept"] <= base_nm + 1e-12
            )
            cache[index] = (result, feasible)
        return cache[index]

    last = len(candidates) - 1
    if not evaluate(last)[1]:
        return {"threshold": float("inf"), "feasible": False,
                "metrics": {"hit1": 0.0, "ba": 0.5, "fmr": 0.0,
                            "fsr": 1.0, "no_match_accept": 0.0}}
    low, high = -1, last
    while high - low > 1:
        middle = (low + high) // 2
        if evaluate(middle)[1]:
            high = middle
        else:
            low = middle
    first_feasible = high
    best_recall = 1.0 - evaluate(first_feasible)[0]["metrics"]["fsr"]
    # If later cutoffs keep the same recall, prefer the higher cutoff as stated
    # in the protocol. This is a second monotone search over the recall plateau.
    low, high = first_feasible, last + 1
    while high - low > 1:
        middle = (low + high) // 2
        recall = 1.0 - evaluate(middle)[0]["metrics"]["fsr"]
        if recall >= best_recall - 1e-12:
            low = middle
        else:
            high = middle
    result, _ = evaluate(low)
    return {"threshold": float(candidates[low]), "feasible": True,
            "metrics": result["metrics"]}


def paired_task_deltas(task_rows: list[dict], arm: str, baseline_arm: str = "E") -> dict:
    by_arm_task = {(r["arm"], r["task"]): r for r in task_rows}
    result = {}
    for metric in ("hit1", "ba", "fmr", "fsr", "no_match_accept"):
        deltas = [by_arm_task[(arm, task)][metric] - by_arm_task[(baseline_arm, task)][metric]
                  for a, task in by_arm_task if a == arm and (baseline_arm, task) in by_arm_task]
        result[metric] = float(np.mean(deltas)) if deltas else None
    return result


def run(dataset: str, output: pathlib.Path, repetitions: int) -> None:
    rows, tasks, folds = core.base.build_cth() if dataset == "cth" else core.base.build_ait()
    minimum = 2 if dataset == "cth" else 4
    _, custom = categorical_sequences(rows)
    output.mkdir(parents=True, exist_ok=True)
    oof_selections = []
    all_predictions = []
    all_task_metrics = []
    outer_task_metrics: dict[str, list[dict]] = collections.defaultdict(list)
    variant_counts = collections.Counter()

    for outer_fold in range(3):
        train = [i for i, row in enumerate(rows) if folds[row["task"]] != outer_fold]
        test = [i for i, row in enumerate(rows) if folds[row["task"]] == outer_fold]
        outer_gallery = core.galleries(rows, test, outer_fold, repetitions)
        outer_pairs = [(a, b, 0) for a, b, _, _ in outer_gallery]
        inner_galleries: list[tuple] = []
        inner_pair_labels: list[np.ndarray] = []
        inner_pair_e_scores: list[np.ndarray] = []
        inner_gallery_scores: dict[str, list[np.ndarray]] = {arm: [] for arm in ARM_SPEC}
        inner_gallery_missing: dict[str, list[np.ndarray]] = {arm: [] for arm in ARM_SPEC}
        inner_pair_scores: dict[str, list[np.ndarray]] = {arm: [] for arm in ARM_SPEC}
        inner_pair_missing: dict[str, list[np.ndarray]] = {arm: [] for arm in ARM_SPEC}

        for inner_fold in sorted({folds[rows[i]["task"]] for i in train}):
            fit = [i for i in train if folds[rows[i]["task"]] != inner_fold]
            val = [i for i in train if folds[rows[i]["task"]] == inner_fold]
            train_pairs = core.pairs(rows, fit, "mode")
            val_pairs = core.pairs(rows, val, "mode")
            gallery = core.galleries(rows, val, inner_fold, INNER_REPETITIONS)
            gallery_pairs = [(a, b, 0) for a, b, _, _ in gallery]
            feature_map = core.make_features(rows, fit, minimum)
            train_e, train_blocks, _, _ = feature_map(train_pairs)
            val_e, val_blocks, val_missing, _ = feature_map(val_pairs)
            gallery_e, gallery_blocks, gallery_missing, _ = feature_map(gallery_pairs)
            y_train = np.asarray([p[2] for p in train_pairs], dtype=int)
            y_val = np.asarray([p[2] for p in val_pairs], dtype=int)
            default_weights = core.weights_for(rows, train_pairs)
            e_val = core.model_scores("xgb", train_e, y_train, val_e, default_weights)
            e_gallery = core.model_scores("xgb", train_e, y_train, gallery_e, default_weights)
            inner_pair_labels.append(y_val)
            inner_pair_e_scores.append(e_val)
            inner_galleries.extend(gallery)

            for arm, spec in ARM_SPEC.items():
                if arm == "E":
                    pair_score = e_val
                    gallery_score = e_gallery
                    pair_missing = np.zeros(len(val_pairs), dtype=bool)
                    gal_missing = np.zeros(len(gallery), dtype=bool)
                else:
                    train_x = s_features(train_e, train_blocks, train_pairs, custom, arm, dataset)
                    val_x = s_features(val_e, val_blocks, val_pairs, custom, arm, dataset)
                    gallery_x = s_features(gallery_e, gallery_blocks, gallery_pairs, custom, arm, dataset)
                    weights = fit_weights(rows, train_pairs, spec["hard_negative_multiplier"])
                    pair_score = core.model_scores("xgb", train_x, y_train, val_x, weights)
                    gallery_score = core.model_scores("xgb", train_x, y_train, gallery_x, weights)
                    pair_missing = missing_for(val_missing, arm, dataset)
                    gal_missing = missing_for(gallery_missing, arm, dataset)
                inner_pair_scores[arm].append(pair_score)
                inner_pair_missing[arm].append(pair_missing)
                inner_gallery_scores[arm].append(gallery_score)
                inner_gallery_missing[arm].append(gal_missing)
            print(dataset, "outer", outer_fold, "inner", inner_fold,
                  "pairs", len(train_pairs), "galleries", len(gallery) // 4, flush=True)

        labels_oof = np.concatenate(inner_pair_labels)
        e_oof = np.concatenate(inner_pair_e_scores)
        e_threshold = core.base.threshold(labels_oof, e_oof)
        legacy_thresholds = {"E": e_threshold}
        e_gal_scores = np.concatenate(inner_gallery_scores["E"])
        all_inner_missing = np.zeros(len(e_gal_scores), dtype=bool)
        e_inner = gallery_metrics(rows, inner_galleries, e_gal_scores, all_inner_missing,
                                  e_threshold, e_gal_scores, e_threshold)

        calibrated = {}
        for arm in ARM_SPEC:
            pair_scores = np.concatenate(inner_pair_scores[arm])
            pair_miss = np.concatenate(inner_pair_missing[arm])
            gal_scores = np.concatenate(inner_gallery_scores[arm])
            gal_miss = np.concatenate(inner_gallery_missing[arm])
            if arm == "E":
                legacy_thresholds[arm] = e_threshold
                calibrated[arm] = {"threshold": e_threshold, "feasible": True,
                                   "metrics": e_inner["metrics"]}
            else:
                legacy_thresholds[arm] = core.base.threshold(
                    labels_oof, pair_scores, pair_miss, e_oof >= e_threshold
                )
                calibrated[arm] = guarded_cutoff(
                    rows, inner_galleries, gal_scores, gal_miss, e_gal_scores, e_threshold
                )
            ranked = gallery_metrics(rows, inner_galleries, gal_scores, gal_miss,
                                     calibrated[arm]["threshold"], e_gal_scores, e_threshold)
            calibrated[arm]["inner_metrics"] = ranked["metrics"]
            calibrated[arm]["inner_feasible"] = bool(calibrated[arm]["feasible"])

        # Choose by inner Hit@1 only among arms whose calibrated cutoff respects
        # E's inner task-macro FMR and no-match acceptance. Ties prefer fewer S
        # dimensions; outer folds are never consulted for this choice.
        eligible_arms = [arm for arm in ARM_SPEC if calibrated[arm]["feasible"]]
        if not eligible_arms:
            eligible_arms = ["E"]
        selected = max(
            eligible_arms,
            key=lambda arm: (
                calibrated[arm]["inner_metrics"]["hit1"],
                -ARM_COMPLEXITY[arm],
                calibrated[arm]["inner_metrics"]["ba"],
            ),
        )
        variant_counts[selected] += 1
        oof_selections.append({
            "outer_fold": outer_fold,
            "selected_arm": selected,
            "baseline_e_threshold": float(e_threshold),
            "baseline_inner_metrics": e_inner["metrics"],
            "calibration": {
                arm: {"threshold": val["threshold"] if np.isfinite(val["threshold"]) else None,
                      "pairwise_threshold": legacy_thresholds[arm]
                      if np.isfinite(legacy_thresholds[arm]) else None,
                      "feasible": val["feasible"], "inner_metrics": val["inner_metrics"]}
                for arm, val in calibrated.items()
            },
        })

        outer_train_pairs = core.pairs(rows, train, "mode")
        y_train = np.asarray([p[2] for p in outer_train_pairs], dtype=int)
        final_map = core.make_features(rows, train, minimum)
        outer_train_e, outer_train_blocks, _, _ = final_map(outer_train_pairs)
        outer_e, outer_blocks, outer_missing_dict, _ = final_map(outer_pairs)
        default_weights = core.weights_for(rows, outer_train_pairs)
        e_outer = core.model_scores("xgb", outer_train_e, y_train, outer_e, default_weights)
        selected_threshold = calibrated[selected]["threshold"]

        for arm, spec in ARM_SPEC.items():
            if arm == "E":
                score = e_outer
                miss = np.zeros(len(outer_pairs), dtype=bool)
            else:
                train_x = s_features(outer_train_e, outer_train_blocks, outer_train_pairs,
                                     custom, arm, dataset)
                test_x = s_features(outer_e, outer_blocks, outer_pairs, custom, arm, dataset)
                weights = fit_weights(rows, outer_train_pairs, spec["hard_negative_multiplier"])
                score = core.model_scores("xgb", train_x, y_train, test_x, weights)
                miss = missing_for(outer_missing_dict, arm, dataset)
            threshold = calibrated[arm]["threshold"]
            result = gallery_metrics(rows, outer_gallery, score, miss, threshold,
                                     e_outer, e_threshold)
            arm_task_rows = []
            for task, metrics in result["per_task"].items():
                row_metric = {"dataset": dataset, "outer_fold": outer_fold,
                              "arm": arm, "task": task, **metrics}
                arm_task_rows.append(row_metric)
                all_task_metrics.append(row_metric)
                outer_task_metrics[arm].append(row_metric)
            all_predictions.extend(prediction_rows(
                rows, outer_gallery, dataset, arm, outer_fold, score, miss, threshold,
                e_outer, e_threshold, result["metrics"],
            ))
            if arm == selected:
                all_predictions.extend(prediction_rows(
                    rows, outer_gallery, dataset, "selected_nested", outer_fold, score, miss,
                    threshold, e_outer, e_threshold, result["metrics"],
                ))
                for task, metrics in result["per_task"].items():
                    selected_row = {"dataset": dataset, "outer_fold": outer_fold,
                                    "arm": "selected_nested", "task": task, **metrics}
                    all_task_metrics.append(selected_row)
                    outer_task_metrics["selected_nested"].append(selected_row)
            if arm in {"base", "fullS"}:
                pairwise_threshold = legacy_thresholds[arm]
                pairwise_result = gallery_metrics(
                    rows, outer_gallery, score, miss, pairwise_threshold,
                    e_outer, e_threshold,
                )
                pairwise_arm = f"{arm}_pairwise"
                for task, metrics in pairwise_result["per_task"].items():
                    pairwise_row = {"dataset": dataset, "outer_fold": outer_fold,
                                    "arm": pairwise_arm, "task": task, **metrics}
                    all_task_metrics.append(pairwise_row)
                    outer_task_metrics[pairwise_arm].append(pairwise_row)
                all_predictions.extend(prediction_rows(
                    rows, outer_gallery, dataset, pairwise_arm, outer_fold,
                    score, miss, pairwise_threshold, e_outer, e_threshold,
                    pairwise_result["metrics"],
                ))
            print(dataset, "outer", outer_fold, "arm", arm,
                  "threshold", threshold, "Hit@1", result["metrics"]["hit1"], flush=True)

    aggregated = {}
    for arm, task_rows in outer_task_metrics.items():
        metrics = {}
        for name in ("hit1", "ba", "fmr", "fsr", "no_match_accept"):
            values = [row[name] for row in task_rows]
            metrics[name] = float(np.mean(values))
        metrics["delta_vs_E"] = paired_task_deltas(all_task_metrics, arm)
        metrics["tasks"] = len(task_rows)
        aggregated[arm] = metrics

    prediction_path = output / f"{dataset}_predictions.csv.gz"
    with gzip.open(prediction_path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_predictions[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_predictions)
    task_path = output / f"{dataset}_task_metrics.csv"
    with task_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_task_metrics[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_task_metrics)

    source_paths = {
        "task_folds": ROOT / "research/results/overnight_fingerprint_v1/task_folds.csv",
        "command_categories": ROOT / "research/experiments/behavioral_fingerprint_v2/command_categories.py",
        "command_profile": ROOT / "research/experiments/behavioral_fingerprint_v2/command_profile.py",
        "feature_implementation": ROOT / "attackagent/fingerprint_v2.py",
        "same_mode_core": ROOT / "research/experiments/same_mode_bst/run.py",
    }
    if dataset == "cth":
        source_paths["trace_registry"] = ROOT / "research/results/behavior_corpus_v1/trace_registry.csv"
        source_paths["trace_evidence"] = ROOT / "research/results/behavior_corpus_v1/evidence.jsonl"
    else:
        source_paths["ait_archive"] = ROOT / "research/data/external/ait-shell-pilot-v1/data.zip"
    dependency_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }
    manifest = {
        "dataset": dataset,
        "seed": SEED,
        "outer_folds": folds,
        "outer_gallery_repetitions": repetitions,
        "inner_gallery_repetitions": INNER_REPETITIONS,
        "runs": len(rows), "tasks": len(tasks),
        "candidate_arms": ARM_SPEC,
        "selected_nested_counts": dict(variant_counts),
        "mode_counts": dict(collections.Counter(row["mode"] for row in rows)),
        "threshold_policy": "guarded inner gallery threshold: task-macro pair FMR <= min(10%, E FMR), no-match acceptance <= E, then minimize FSR; pairwise-threshold comparator uses the original inner pair FMR<=10% rule",
        "selection_policy": "maximize inner task-macro Hit@1 after guarded calibration; ties prefer fewer S feature dimensions",
        "smoothing": {"transition_dirichlet_alpha": DIRICHLET_ALPHA,
                      "unknown_category": "breaks category n-grams; excluded from transition counts"},
        "results": aggregated,
        "outer_fold_selections": oof_selections,
        "dependency_sha256": dependency_hashes,
        "software_versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                              "scikit_learn": sklearn.__version__,
                              "xgboost": xgboost.__version__},
        "runner_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "feature_sha256": hashlib.sha256((ROOT / "attackagent/fingerprint_v2.py").read_bytes()).hexdigest(),
        "status": "retrospective exploratory; cohorts inspected during development",
    }
    (output / f"{dataset}_metrics.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": dataset, "output": str(output),
                      "selected_nested_counts": dict(variant_counts),
                      "results": aggregated}, indent=2), flush=True)


def prediction_rows(rows: list[dict], galleries: list[tuple], dataset: str,
                    arm: str, fold: int,
                    scores: np.ndarray, missing: np.ndarray, threshold: float,
                    baseline_scores: np.ndarray, baseline_threshold: float,
                    arm_metrics: dict) -> list[dict]:
    result = []
    for idx, (query, candidate, replicate, position) in enumerate(galleries):
        query_row, candidate_row = rows[query], rows[candidate]
        fallback = bool(missing[idx])
        score = float(baseline_scores[idx] if fallback else scores[idx])
        cutoff = float(baseline_threshold if fallback else threshold)
        result.append({
            "dataset": dataset,
            "fold": fold, "target": "mode", "learner": "xgb", "arm": arm,
            "query_id": query_row["id"], "candidate_id": candidate_row["id"],
            "query_task": query_row["task"], "candidate_task": candidate_row["task"],
            "query_mode": query_row["mode"], "candidate_mode": candidate_row["mode"],
            "replicate": replicate, "position": position,
            "same_task": int(query_row["task"] == candidate_row["task"]),
            "label": int(query_row["task"] == candidate_row["task"]
                        and query_row["mode"] == candidate_row["mode"]
                        and position < 4),
            "score": score, "baseline_score": float(baseline_scores[idx]),
            "threshold": cutoff, "predicted_link": int(score >= cutoff),
            "fallback": int(fallback),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cth", "ait"), required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    run(args.dataset, args.out, args.repetitions)


if __name__ == "__main__":
    main()
