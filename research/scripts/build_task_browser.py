#!/usr/bin/env python3
"""Build the static, browser-readable pair explorer from frozen result artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "multidimensional_rerun_v1"
PAIR_FILE = RESULTS / "external_pair_scores.csv"
SUMMARY_FILE = RESULTS / "results.json"
PROFILE_FILE = RESULTS / "behavior_profiles.jsonl"
OUTPUT = ROOT.parent / "site" / "data" / "task-association-v1.json"

METHODS = [
    ("command_tfidf", "命令 TF–IDF", "Command TF–IDF"),
    ("profile_24d", "24 维画像 · 折中对照", "24D profile · balanced comparison"),
    ("tensor_4x8x3", "4×8×3 阶段张量", "4×8×3 phase tensor"),
    ("tensor_tucker", "Tucker 张量分解", "Tucker tensor"),
    ("partial_sequence", "IDF 局部序列", "IDF partial sequence"),
    ("learned_fusion", "学习式融合", "Learned fusion"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_profiles() -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    with PROFILE_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            profiles[record["trace_id"]] = record
    return profiles


def metric(summary: dict, key: str) -> dict | None:
    value = summary.get(key)
    if not isinstance(value, dict) or "estimate" not in value:
        return None
    return {
        "estimate": value["estimate"],
        "ci95": value.get("ci_95"),
        "task_count": value.get("task_count"),
    }


def outcome(positive: bool, accepted: bool) -> str:
    if positive:
        return "TP" if accepted else "FN"
    return "FP" if accepted else "TN"


def main() -> None:
    raw = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    source_profiles = read_profiles()
    with PAIR_FILE.open(encoding="utf-8", newline="") as handle:
        source_pairs = list(csv.DictReader(handle))

    thresholds = raw["selection"]["thresholds_by_method"]
    method_meta = []
    for key, name_zh, name_en in METHODS:
        summary = raw["metrics"]["primary_error_and_ranking_summaries"][key]
        method_meta.append(
            {
                "key": key,
                "name_zh": name_zh,
                "name_en": name_en,
                "threshold": thresholds[key]["threshold"],
                "threshold_source": "development task-held-out selection",
                "metrics": {
                    name: metric(summary, name)
                    for name in (
                        "false_split_rate",
                        "false_merge_rate",
                        "balanced_accuracy",
                        "query_has_accepted_match",
                        "hit_at_1",
                        "mrr",
                        "recall_at_5",
                        "ndcg_at_5",
                    )
                    if metric(summary, name) is not None
                },
            }
        )

    # Rank each candidate within its query, using scores only.
    query_groups: dict[str, list[dict]] = defaultdict(list)
    for row in source_pairs:
        query_groups[row["query_trace_id"]].append(row)
    ranks: dict[tuple[str, str], dict[str, int]] = {}
    for query_id, rows in query_groups.items():
        for key, _, _ in METHODS:
            ordered = sorted(rows, key=lambda row: (-float(row[key]), row["candidate_trace_id"]))
            for position, row in enumerate(ordered, start=1):
                ranks[(query_id, row["candidate_trace_id"])] = ranks.setdefault(
                    (query_id, row["candidate_trace_id"]), {}
                )
                ranks[(query_id, row["candidate_trace_id"])][key] = position

    trace_labels: dict[str, dict] = {}
    pairs = []
    confusion: dict[str, Counter] = {key: Counter() for key, _, _ in METHODS}
    ordered_rows = sorted(
        source_pairs,
        key=lambda row: (
            row["benchmark"],
            row["query_task"],
            row["query_trace_id"],
            row["candidate_trace_id"],
        ),
    )
    for index, row in enumerate(ordered_rows, start=1):
        positive = row["same_task_proxy_for_evaluation_only"] == "1"
        for side, trace_id_key, task_key, mode_key in (
            (
                "query",
                "query_trace_id",
                "query_task",
                "query_mode_for_evaluation_only",
            ),
            (
                "candidate",
                "candidate_trace_id",
                "candidate_task",
                "candidate_mode_for_evaluation_only",
            ),
        ):
            trace_id = row[trace_id_key]
            trace_labels.setdefault(
                trace_id,
                {
                    "trace_id": trace_id,
                    "benchmark": row["benchmark"],
                    "task_for_evaluation_only": row[task_key],
                    "mode_for_evaluation_only": row[mode_key],
                },
            )
        predictions = {}
        scores = {}
        for key, _, _ in METHODS:
            score = float(row[key])
            accepted = score >= thresholds[key]["threshold"]
            result = outcome(positive, accepted)
            scores[key] = score
            predictions[key] = {
                "accepted": accepted,
                "outcome": result,
                "rank": ranks[(row["query_trace_id"], row["candidate_trace_id"])][key],
            }
            confusion[key][result] += 1
        pairs.append(
            {
                "pair_id": f"P{index:04d}",
                "benchmark": row["benchmark"],
                "query_task_for_evaluation_only": row["query_task"],
                "candidate_task_for_evaluation_only": row["candidate_task"],
                "query_trace_id": row["query_trace_id"],
                "candidate_trace_id": row["candidate_trace_id"],
                "pair_type_for_evaluation_only": row["pair_type_for_evaluation_only"],
                "same_task_proxy_for_evaluation_only": positive,
                "scores": scores,
                "predictions": predictions,
            }
        )

    traces = {}
    for trace_id, labels in trace_labels.items():
        profile = source_profiles[trace_id]
        traces[trace_id] = {
            **labels,
            "feature_dimensions": profile["feature_dimensions"],
            "features": profile["features"],
            "missing_features": profile["missing_features"],
            "tensor_shape": profile["tensor_shape"],
            "tensor_action_labels": profile["tensor_action_labels"],
            "tensor_channels": profile["tensor_channels"],
            "tensor": profile["tensor"],
        }

    for method in method_meta:
        key = method["key"]
        method["pair_counts"] = {
            state: confusion[key].get(state, 0) for state in ("TP", "FP", "FN", "TN")
        }

    output = {
        "schema_version": "task-association-browser-v1",
        "title": "Locked external task-pair inspection",
        "dataset": {
            "name": raw["cohort"]["external"].get("name", "CVEBench + CyberGym"),
            "benchmark_count": len({row["benchmark"] for row in ordered_rows}),
            "task_count": len({row["query_task"] for row in ordered_rows}),
            "query_count": len(query_groups),
            "pair_count": len(pairs),
            "same_task_pair_count": sum(
                row["same_task_proxy_for_evaluation_only"] for row in pairs
            ),
            "different_task_pair_count": sum(
                not row["same_task_proxy_for_evaluation_only"] for row in pairs
            ),
            "label_scope": "same benchmark-qualified task retrieval proxy; not actor or organization identity",
            "threshold_boundary": "Every method threshold was selected using development tasks before this locked external evaluation.",
        },
        "selected_method": raw["selection"]["method"],
        "inspection_default_method": "profile_24d",
        "inspection_default_note": "Display-only comparison point: lower external FMR than the 0%-FSR partial-sequence point, with nonzero FSR. Not a retuned or selected method.",
        "methods": method_meta,
        "traces": traces,
        "pairs": pairs,
        "sources": {
            "pair_scores": "results/multidimensional_rerun_v1/external_pair_scores.csv",
            "summary": "results/multidimensional_rerun_v1/results.json",
            "profiles": "results/multidimensional_rerun_v1/behavior_profiles.jsonl",
            "pair_scores_sha256": sha256(PAIR_FILE),
            "summary_sha256": sha256(SUMMARY_FILE),
            "profiles_sha256": sha256(PROFILE_FILE),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {OUTPUT.relative_to(ROOT)}: {len(pairs)} pairs, "
        f"{len(traces)} traces, {OUTPUT.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
