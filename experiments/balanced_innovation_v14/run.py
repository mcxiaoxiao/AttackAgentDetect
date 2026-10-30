#!/usr/bin/env python3
"""Nested test of a query-local top-k rule over the v11 artifact score."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "experiments" / "balanced_innovation_v11" / "run.py"
OUT = ROOT / "results" / "balanced_innovation_v14"
METHOD = "rank_capped_artifact"
RANK_SOURCE = "artifact_containment_command"
MAX_K = 60

sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("balanced_innovation_v11_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load frozen base runner at {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.OUT = OUT
base.METHODS = (*base.METHODS, METHOD)
base.NEW_METHODS = (*base.NEW_METHODS, METHOD)
base.SIMPLICITY = {**base.SIMPLICITY, METHOD: len(base.SIMPLICITY)}
base.SELECTION_METHODS = (base.BASELINE, METHOD)


def add_rank_cap_scores(rows: list[dict]) -> None:
    """Map the artifact score to a within-query reciprocal average rank."""
    by_query: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["benchmark"], row["query_task"], row["query_trace_id"])
        by_query.setdefault(key, []).append(row)
    for query_rows in by_query.values():
        ordered = sorted(query_rows, key=lambda row: -float(row[RANK_SOURCE]))
        start = 0
        while start < len(ordered):
            end = start + 1
            value = float(ordered[start][RANK_SOURCE])
            while end < len(ordered) and np.isclose(
                float(ordered[end][RANK_SOURCE]), value, rtol=0.0, atol=1e-12
            ):
                end += 1
            average_rank = (start + 1 + end) / 2.0
            reciprocal_rank = 1.0 / average_rank
            for row in ordered[start:end]:
                row["rank_cap_average_rank"] = average_rank
                row["rank_cap_candidate_count"] = len(ordered)
                row[METHOD] = reciprocal_rank
            start = end


_fit_pair_verifiers = base.fit_pair_verifiers


def fit_pair_verifiers_with_rank_cap(train_rows, score_rows, seed: int, fold: int, split_name: str):
    rows, dimensions = _fit_pair_verifiers(train_rows, score_rows, seed, fold, split_name)
    add_rank_cap_scores(rows)
    return rows, dimensions


base.fit_pair_verifiers = fit_pair_verifiers_with_rank_cap


def failed_hard_gates(values: dict, baseline: dict) -> list[str]:
    metrics = (
        "false_merge_rate", "false_split_rate",
        "false_merge_human_human", "false_split_human_human",
        "false_merge_human_agent", "false_split_human_agent",
        "false_merge_agent_agent", "false_split_agent_agent",
        "balanced_accuracy", "query_coverage", *base.RANK_METRICS,
    )
    failed = []
    for metric in metrics:
        reference = baseline.get(metric)
        value = values.get(metric)
        if reference is None:
            continue
        if value is None:
            failed.append(f"{metric}:unavailable")
            continue
        if metric == "false_merge_rate":
            if value >= reference - 1e-12:
                failed.append(f"{metric}:must_decrease")
        elif metric in base.error_eval.RATE_METRICS:
            if value > reference + 1e-12:
                failed.append(f"{metric}:worsened")
        elif value < reference - 1e-12:
            failed.append(f"{metric}:worsened")
    return failed


def select_rank_cap_policy(pair_rows: list[dict], ranks: list[dict], standard_policies: dict[str, dict]):
    baseline = standard_policies[base.BASELINE]["training_metrics"]
    feasible = []
    candidate_audit = []
    for k in range(1, MAX_K + 1):
        threshold = 1.0 / k
        values = base.policy_values(pair_rows, ranks, METHOD, threshold)
        failures = failed_hard_gates(values, baseline)
        candidate_audit.append({
            "k": k,
            "false_merge_rate": values["false_merge_rate"],
            "false_split_rate": values["false_split_rate"],
            "false_merge_human_human": values["false_merge_human_human"],
            "false_split_human_human": values["false_split_human_human"],
            "false_merge_human_agent": values["false_merge_human_agent"],
            "false_split_human_agent": values["false_split_human_agent"],
            "false_merge_agent_agent": values["false_merge_agent_agent"],
            "false_split_agent_agent": values["false_split_agent_agent"],
            "balanced_accuracy": values["balanced_accuracy"],
            "query_coverage": values["query_coverage"],
            "hit_at_1": values["hit_at_1"],
            "mrr": values["mrr"],
            "recall_at_5": values["recall_at_5"],
            "ndcg_at_5": values["ndcg_at_5"],
            "rank_quality": values["rank_quality"],
            "failed_gates": failures,
        })
        if not failures:
            feasible.append((threshold, k, values))

    audit = {
        METHOD: {
            "candidate_k_count": MAX_K,
            "feasible_k_values": [item[1] for item in feasible],
            "inner_selected_k": min((item[1] for item in feasible), default=None),
            "candidate_audit": candidate_audit,
            "rule": "smallest k satisfying every hard gate; ties at the average-rank cutoff are kept together",
        }
    }
    if not feasible:
        policy = dict(standard_policies[base.BASELINE])
        policy["selection_rule"] = "no query-local rank cap passed every inner hard gate; retain the same-fold command baseline"
        audit["selected_fallback"] = base.BASELINE
        return base.BASELINE, policy, audit

    # Highest reciprocal-rank threshold is equivalent to the smallest feasible k.
    threshold, k, values = max(feasible, key=lambda item: (item[0], -item[1]))
    policy = {
        "threshold": threshold,
        "top_k": k,
        "training_metrics": values,
        "candidate_threshold_count": MAX_K,
        "selection_rule": "choose the smallest query-local top-k that strictly lowers FMR and preserves all inner subgroup, FSR, balanced-accuracy, coverage, and ranking gates",
        "hard_gate_baseline": base.BASELINE,
    }
    audit["selected_k"] = k
    return METHOD, policy, audit


base.select_pareto_method = select_rank_cap_policy


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize_result(result: dict) -> dict:
    result["protocol"] = "experiments/balanced_innovation_v14/program.md"
    result["selection_rule"] = (
        "select the smallest inner-OOF query-local top-k whose artifact-score rank strictly lowers FMR "
        "without worsening any estimable subgroup error, FSR, balanced accuracy, query coverage, or ranking; "
        "if none passes, retain command TF-IDF"
    )
    result["new_methods"][METHOD] = (
        "reciprocal average rank of the v11 artifact-containment score, computed within each query; "
        "a hard top-k boundary is selected by nested task-grouped validation"
    )
    result["provenance"]["runner_sha256"] = file_sha256(pathlib.Path(__file__))
    result["provenance"]["selection_grid"] = f"k=1..{MAX_K}; threshold=1/k"
    result["interpretation"] = (
        "This is exploratory reuse of the development cohort. A lower false-merge point is not operationally useful "
        "if its absolute FMR remains high; outer-fold FSR increases are failures."
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    renames = (
        ("balanced-innovation-v11-en.png", "balanced-innovation-v14-en.png"),
        ("balanced-innovation-v11.pdf", "balanced-innovation-v14.pdf"),
        ("balanced-innovation-v11-candidate-recall-en.png", "balanced-innovation-v14-candidate-recall-en.png"),
        ("balanced-innovation-v11-candidate-recall.pdf", "balanced-innovation-v14-candidate-recall.pdf"),
    )
    for old_name, new_name in renames:
        old_path, new_path = OUT / old_name, OUT / new_name
        if old_path.exists() and not new_path.exists():
            old_path.replace(new_path)
    print(json.dumps({
        "selected_method_by_fold": result["selected_method_by_fold"],
        "direction_adjusted_selected_changes": result["direction_adjusted_selected_changes"],
        "all_estimable_metrics_pointwise_nonworse": result["all_estimable_metrics_pointwise_nonworse"],
        "outer_folds": [
            {
                "fold": row["outer_fold"],
                "method": row["selected_method"],
                "top_k": (int(round(1.0 / row["selected_threshold"]))
                          if row["selected_method"] == METHOD else None),
                "inner_fsr": row["selected_inner_metrics"].get("false_split_rate"),
            }
            for row in result["outer_folds"]
        ],
    }, ensure_ascii=False, indent=2))
    return result


def run() -> dict:
    return finalize_result(base.run())


if __name__ == "__main__":
    if sys.argv[1:] == ["--finalize-existing"]:
        result_path = OUT / "results.json"
        finalize_result(json.loads(result_path.read_text(encoding="utf-8")))
    else:
        run()
