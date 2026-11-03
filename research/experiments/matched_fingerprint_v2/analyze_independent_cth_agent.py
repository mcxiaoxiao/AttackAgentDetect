#!/usr/bin/env python3
"""Paired task-cluster analysis for the frozen independent AA confirmation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
IN = ROOT / "research/results/matched_fingerprint_v2/independent_cth_agent/predictions.csv.gz"
RUN = ROOT / "research/results/matched_fingerprint_v2/independent_cth_agent/run_manifest.json"
OUT = ROOT / "research/results/matched_fingerprint_v2/independent_cth_agent/analysis.json"
NBOOT, SEED = 10_000, 20261104
LEARNERS = ("svm", "tree", "xgb")
VERSIONS = ("E", "E_args", "E_BS", "E_args_BS")


def metric_frame(df, tasks):
    values = []
    for task in tasks:
        x = df[df.query_task == task]
        pos, neg = x[x.same_task == 1], x[x.same_task == 0]
        assert len(pos) and len(neg)
        fmr = float(neg.predicted_link.mean())
        fsr = float(1 - pos.predicted_link.mean())
        values.append((fmr, fsr, 1 - (fmr + fsr) / 2))
    return np.asarray(values, dtype=float)


def boot_summary(values, indexes):
    point = values.mean(axis=0)
    boot = values[indexes].mean(axis=1)
    bounds = np.quantile(boot, [0.025, 0.975], axis=0).T
    return {"metrics": dict(zip(("fmr", "fsr", "ba"), map(float, point))),
            "ci95": {name: list(map(float, bounds[i]))
                     for i, name in enumerate(("fmr", "fsr", "ba"))}}


def delta_summary(before, after, indexes):
    return boot_summary(after - before, indexes)


def ranking_rows(df, tasks, excluded_full):
    rows = []
    group_keys = ["fold", "replicate", "query_id", "query_task"]
    for keys, group in df.groupby(group_keys, sort=False, observed=True):
        if len(group) != 4 or int(group.same_task.sum()) != 1:
            raise AssertionError(f"invalid gallery group {keys}: {len(group)} rows")
        if excluded_full and group.fallback.astype(int).any():
            continue
        scores = group.score.astype(float).to_numpy()
        labels = group.same_task.astype(int).to_numpy()
        positive = scores[labels == 1][0]
        negatives = scores[labels == 0]
        greater = int(np.sum(negatives > positive))
        ties = 1 + int(np.sum(negatives == positive))
        rank = 1 + greater
        hit1 = (1.0 / ties) if greater == 0 else 0.0
        mrr = float(np.mean([1 / (rank + j) for j in range(ties)]))
        auc = float(np.mean([(positive > n) + 0.5 * (positive == n) for n in negatives]))
        rows.append({"fold": keys[0], "replicate": keys[1], "query_id": keys[2],
                     "query_task": keys[3], "hit1": hit1, "mrr": mrr, "auc": auc})
    return pd.DataFrame(rows)


def ranking_task_matrix(rows, tasks, metric):
    by_task = rows.groupby("query_task", observed=True)[metric].mean()
    return np.asarray([float(by_task.get(task, np.nan)) for task in tasks], dtype=float)


def ranking_effect(before, after, tasks, indexes):
    names = ("hit1", "mrr", "auc")
    result = {}
    coverage = {}
    for name in names:
        b = ranking_task_matrix(before, tasks, name)
        a = ranking_task_matrix(after, tasks, name)
        common = np.isfinite(b) & np.isfinite(a)
        assert common.all(), "paired ranking coverage differs by task"
        delta = a - b
        boot = delta[indexes].mean(axis=1)
        result[name] = {"delta": float(delta.mean()),
                        "ci95": list(map(float, np.quantile(boot, [0.025, 0.975])))}
    coverage = {"query_galleries_before": int(len(before)),
                "query_galleries_after": int(len(after)),
                "tasks": len(tasks)}
    return result, coverage


def transitions(before, after):
    keys = ["fold", "pair_id", "replicate", "query_task", "pair_type", "same_task"]
    x = before[keys + ["predicted_link"]].merge(
        after[keys + ["predicted_link"]], on=keys,
        suffixes=("_before", "_after"), validate="one_to_one")
    output = {}
    for label, group in x.groupby("same_task", observed=True):
        y = group.same_task.astype(int).to_numpy()
        b = group.predicted_link_before.astype(int).to_numpy()
        a = group.predicted_link_after.astype(int).to_numpy()
        output["same_task" if label else "different_task"] = {
            "pairs": int(len(group)),
            "errors_corrected": int(((b != y) & (a == y)).sum()),
            "errors_introduced": int(((b == y) & (a != y)).sum()),
            "tasks_with_net_error_reduction": int(sum(
                ((g.predicted_link_before.to_numpy() != g.same_task.to_numpy()) &
                 (g.predicted_link_after.to_numpy() == g.same_task.to_numpy())).sum() >
                ((g.predicted_link_before.to_numpy() == g.same_task.to_numpy()) &
                 (g.predicted_link_after.to_numpy() != g.same_task.to_numpy())).sum()
                for _, g in group.groupby("query_task", observed=True))),
        }
    return output


def main():
    df = pd.read_csv(IN, compression="gzip")
    run = json.loads(RUN.read_text())
    tasks = sorted(df.query_task.unique())
    assert len(tasks) == 26
    assert set(df.dataset) == {"cth_independent_aa"}
    assert set(df.pair_type) == {"AA"}
    assert len(df) == run["pair_manifest_rows"] * len(LEARNERS) * len(VERSIONS)
    assert df.groupby(["learner", "variant"]).size().nunique() == 1
    assert df.groupby(["learner", "variant"]).size().iloc[0] == run["pair_manifest_rows"]
    base = df[(df.learner == "xgb") & (df.variant == "E")]
    expected = set(zip(base.fold, base.pair_id))
    for learner in LEARNERS:
        for variant in VERSIONS:
            current = df[(df.learner == learner) & (df.variant == variant)]
            assert set(zip(current.fold, current.pair_id)) == expected
    indexes = np.random.default_rng(SEED).integers(0, len(tasks), size=(NBOOT, len(tasks)))
    result = {"protocol": "frozen AA-only CTH task-group confirmation; 100 repeated fixed-gallery draws; task is bootstrap cluster",
              "scope": "same-task means the same CTF challenge proxy, not activity/actor/campaign/organization identity",
              "test_tasks": len(tasks), "test_traces": run["test_trace_count"],
              "query_galleries_per_repetition": run["test_query_count_per_repetition"],
              "repetitions": run["repetitions"], "candidate_count": run["candidate_count"],
              "pair_count_per_version": run["pair_manifest_rows"],
              "bootstrap_draws": NBOOT, "bootstrap_seed": SEED,
              "bootstrap_note": "100 gallery draws are averaged within each of 26 task clusters; they are not independent samples",
              "pair_keys_identical_across_all_models_and_versions": True,
              "models": {}}
    for learner in LEARNERS:
        model_rows = df[df.learner == learner]
        per_version = {}
        matrices = {}
        for variant in VERSIONS:
            current = model_rows[model_rows.variant == variant].copy()
            metrics = metric_frame(current, tasks)
            matrices[variant] = metrics
            threshold_values = []
            for value in current.threshold.unique().tolist():
                numeric = float(value)
                threshold_values.append("inf" if np.isinf(numeric) else numeric)
            per_version[variant] = {"threshold_metrics": boot_summary(metrics, indexes),
                                    "fallback_pairs": int(current.fallback.sum()),
                                    "fallback_rate": float(current.fallback.mean()),
                                    "thresholds": sorted(threshold_values, key=str)}
        contrasts = {}
        for after, before in (("E_BS", "E"), ("E_args", "E"),
                              ("E_args_BS", "E_args")):
            delta = delta_summary(matrices[before], matrices[after], indexes)
            transitions_out = transitions(
                model_rows[model_rows.variant == before],
                model_rows[model_rows.variant == after])
            b_rank = ranking_rows(model_rows[model_rows.variant == before], tasks,
                                  excluded_full=after in ("E_BS", "E_args_BS"))
            a_rank = ranking_rows(model_rows[model_rows.variant == after], tasks,
                                  excluded_full=after in ("E_BS", "E_args_BS"))
            # Recompute the baseline on precisely the same fingerprint-supported query galleries.
            common_keys = ["fold", "replicate", "query_id", "query_task"]
            common = set(map(tuple, a_rank[common_keys].to_numpy()))
            b_rank = b_rank[b_rank[common_keys].apply(tuple, axis=1).isin(common)]
            ranking_delta, ranking_coverage = ranking_effect(b_rank, a_rank, tasks, indexes)
            contrasts[f"{after}_minus_{before}"] = {
                "paired_threshold_metric_change": delta,
                "ranking_change_on_fully_supported_galleries": ranking_delta,
                "ranking_coverage": ranking_coverage,
                "error_transitions": transitions_out}
        per_type = {}
        for category in ("cybench", "intercode-ctf_crypto", "intercode-ctf_general_skills",
                         "intercode-ctf_pwn", "intercode-ctf_reverse", "nyuctf"):
            if category == "cybench":
                selected = [t for t in tasks if "/cybench/" in t]
            elif category == "nyuctf":
                selected = [t for t in tasks if "/nyuctf/" in t]
            else:
                selected = [t for t in tasks if category in t]
            if not selected:
                continue
            per_type[category] = {variant: {
                "tasks": len(selected),
                "metrics": dict(zip(("fmr", "fsr", "ba"),
                                     map(float, matrices[variant][[tasks.index(t) for t in selected]].mean(axis=0))))}
                for variant in VERSIONS}
        result["models"][learner] = {"versions": per_version, "contrasts": contrasts,
                                     "benchmark_task_families_descriptive": per_type}

    result["frozen_configuration"] = {
        "representation": run["representation"], "parameters": run["parameters"],
        "thresholds": run["thresholds"], "training_pairs": run["training_pair_count"],
        "unseen_task_manifest_sha256": run["inputs_sha256"]["task_manifest"]}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("wrote", OUT)
    for learner in LEARNERS:
        print("\n", learner)
        for contrast, value in result["models"][learner]["contrasts"].items():
            d = value["paired_threshold_metric_change"]["metrics"]
            ci = value["paired_threshold_metric_change"]["ci95"]
            print(contrast,
                  "ΔFMR/FSR/BA pp", *[round(100*d[k], 2) for k in ("fmr", "fsr", "ba")],
                  "ΔBA CI pp", [round(100*x, 2) for x in ci["ba"]],
                  "ranking", {k: [round(100*v["delta"], 2),
                                 [round(100*x, 2) for x in v["ci95"]]]
                              for k, v in value["ranking_change_on_fully_supported_galleries"].items()})


if __name__ == "__main__":
    main()
