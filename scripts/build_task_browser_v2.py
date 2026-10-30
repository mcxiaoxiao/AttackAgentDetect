#!/usr/bin/env python3
"""Build the static pair inspector for the nested development experiment."""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attackagent.profiles import (  # noqa: E402
    ACTION_LABELS,
    TENSOR_CHANNELS,
    TENSOR_PHASES,
    extract_behavior_features,
    extract_behavior_tensor,
)
import experiments.balanced_innovation_v11.run as experiment  # noqa: E402

RESULTS = ROOT / "results" / "balanced_innovation_v11"
PAIR_FILE = RESULTS / "outer_pair_scores.csv"
SUMMARY_FILE = RESULTS / "results.json"
OUTPUT = ROOT / "site" / "data" / "task-association-v2.json"
METHODS = (
    ("command_tfidf", "命令 TF–IDF 基线", "Command TF–IDF baseline", "threshold_command_tfidf", "accepted_command_tfidf"),
    ("balanced_selected", "逐折内层入选", "Inner-selected per fold", "selected_threshold", "selected_accepted"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(summary: dict, name: str) -> dict | None:
    value = summary.get(name)
    if not isinstance(value, dict) or "estimate" not in value:
        return None
    return {
        "estimate": value.get("estimate"),
        "ci95": value.get("ci95", value.get("ci_95")),
        "task_count": value.get("task_count", value.get("paired_task_count")),
    }


def outcome(positive: bool, accepted: bool) -> str:
    if positive:
        return "TP" if accepted else "FN"
    return "FP" if accepted else "TN"


def main() -> None:
    experiment.gallery.load_rows = experiment.load_scoped_development_rows
    traces, _by_benchmark, _unmatched = experiment._load_development()
    raw = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    with PAIR_FILE.open(encoding="utf-8", newline="") as handle:
        source_pairs = list(csv.DictReader(handle))

    trace_records = {}
    for trace in traces:
        features = extract_behavior_features(trace.names, trace.groups)
        tensor = extract_behavior_tensor(trace.names, trace.groups)
        artifact_tokens = sorted(
            token.removeprefix("artifact:")
            for token in experiment.OPERAND_TOKENS.get(trace.trace_id, {})
            if token.startswith("artifact:")
            and token.split(":", 1)[1] not in experiment.ARTIFACT_STOPWORDS
        )
        trace_records[trace.trace_id] = {
            "trace_id": trace.trace_id,
            "benchmark": trace.benchmark,
            "task_for_evaluation_only": trace.task,
            "mode_for_evaluation_only": trace.label,
            "feature_dimensions": len(features),
            "features": features,
            "missing_features": [key for key, value in features.items() if value is None],
            "tensor_shape": [TENSOR_PHASES, len(ACTION_LABELS), len(TENSOR_CHANNELS)],
            "tensor_action_labels": list(ACTION_LABELS),
            "tensor_channels": list(TENSOR_CHANNELS),
            "tensor": tensor,
            "commands": list(trace.names),
            "artifact_tokens": artifact_tokens,
        }

    query_groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in source_pairs:
        query_groups[row["query_trace_id"]].append(row)
    ranks: dict[tuple[str, str], dict[str, int]] = {}
    for query_id, rows in query_groups.items():
        for key, *_ in METHODS:
            ordered = sorted(rows, key=lambda row: (-float(row[key]), row["candidate_trace_id"]))
            for position, row in enumerate(ordered, start=1):
                ranks.setdefault((query_id, row["candidate_trace_id"]), {})[key] = position

    pairs = []
    counts = {key: collections.Counter() for key, *_ in METHODS}
    ordered_rows = sorted(
        source_pairs,
        key=lambda row: (row["benchmark"], row["query_task"], row["query_trace_id"], row["candidate_trace_id"]),
    )
    for index, row in enumerate(ordered_rows, start=1):
        positive = row["same_task_proxy_for_evaluation_only"] == "1"
        scores, thresholds, predictions = {}, {}, {}
        for key, _zh, _en, threshold_key, accepted_key in METHODS:
            score = float(row[key])
            threshold = float(row[threshold_key])
            accepted = bool(int(row[accepted_key]))
            result = outcome(positive, accepted)
            scores[key] = score
            thresholds[key] = threshold
            predictions[key] = {
                "accepted": accepted,
                "outcome": result,
                "rank": ranks[(row["query_trace_id"], row["candidate_trace_id"])][key],
                "outer_fold": int(row["outer_fold"]),
            }
            counts[key][result] += 1
        pairs.append({
            "pair_id": f"D{index:05d}",
            "benchmark": row["benchmark"],
            "query_task_for_evaluation_only": row["query_task"],
            "candidate_task_for_evaluation_only": row["candidate_task"],
            "query_trace_id": row["query_trace_id"],
            "candidate_trace_id": row["candidate_trace_id"],
            "pair_type_for_evaluation_only": row["pair_type_for_evaluation_only"],
            "same_task_proxy_for_evaluation_only": positive,
            "outer_fold": int(row["outer_fold"]),
            "scores": scores,
            "thresholds": thresholds,
            "predictions": predictions,
        })

    positive_tasks = set()
    missed_tasks = set()
    for pair in pairs:
        if pair["same_task_proxy_for_evaluation_only"]:
            key = (pair["benchmark"], pair["query_task_for_evaluation_only"])
            positive_tasks.add(key)
            if pair["predictions"]["balanced_selected"]["outcome"] == "FN":
                missed_tasks.add(key)
    task_cluster_count = len(positive_tasks)
    zero_event_upper = 1.0 - 0.05 ** (1.0 / task_cluster_count) if task_cluster_count else None

    method_records = []
    for key, name_zh, name_en, *_ in METHODS:
        estimates = raw["metrics"][key]
        method_records.append({
            "key": key,
            "name_zh": name_zh,
            "name_en": name_en,
            "threshold": None,
            "threshold_source": "per-fold inner grouped out-of-fold selection",
            "metrics": {
                name: metric(estimates, name)
                for name in (
                    "false_split_rate", "false_merge_rate", "balanced_accuracy",
                    "query_has_accepted_match", "hit_at_1", "mrr", "recall_at_5", "ndcg_at_5",
                )
            },
            "pair_counts": {state: counts[key].get(state, 0) for state in ("TP", "FP", "FN", "TN")},
        })

    output = {
        "schema_version": "task-association-browser-v2",
        "title": "Nested development task-pair inspection",
        "dataset": {
            "name": "97 traces · 40 development task groups",
            "benchmark_count": len({row["benchmark"] for row in ordered_rows}),
            "task_count": len({row["query_task"] for row in ordered_rows}),
            "query_count": len(query_groups),
            "pair_count": len(pairs),
            "same_task_pair_count": sum(pair["same_task_proxy_for_evaluation_only"] for pair in pairs),
            "different_task_pair_count": sum(not pair["same_task_proxy_for_evaluation_only"] for pair in pairs),
            "label_scope": "same benchmark-qualified task retrieval proxy; not actor, campaign, organization, or identity",
            "threshold_boundary": "Each outer-fold cutoff was selected only from inner task-grouped out-of-fold predictions.",
            "description_zh": "97 条轨迹、40 个任务组的嵌套留出开发结果。可逐条检查误合并、漏关联、分数、阈值、候选排名及经修正命令提取器计算的画像。此队列反复用于方法开发，不是独立确认；同题不是身份标签。",
            "description_en": "Nested task-held-out development results for 97 traces across 40 task groups. Inspect false merges, missed links, scores, fold-specific thresholds, ranks, and profiles from the corrected command extractor. This reused cohort is not independent confirmation; same task is not identity.",
            "zero_miss_sensitivity_zh": (
                f"逐折入选方案在 {task_cluster_count} 个有正配对的任务组中观测到 {len(missed_tasks)} 个含漏关联的任务组。"
                f"零错下普通任务簇 Bootstrap 区间退化；若把任务组视为独立簇，0/{task_cluster_count} 的单侧 95% 精确上限是 {zero_event_upper * 100:.1f}%，"
                "这是“出现任一漏关联的任务组比例”的敏感性界限，不是任务宏 FSR 的置信区间。"
            ),
            "zero_miss_sensitivity_en": (
                f"The selected policy had a missed link in {len(missed_tasks)} of {task_cluster_count} positive task groups. "
                "With zero observed events, the ordinary cluster-bootstrap interval is degenerate; under an independent-task-cluster sensitivity assumption, "
                f"0/{task_cluster_count} gives a one-sided exact 95% upper bound of {zero_event_upper * 100:.1f}% for the share of task groups with any miss. "
                "This is not a confidence interval for task-macro FSR."
            ),
            "zero_miss_sensitivity": {
                "positive_task_groups": task_cluster_count,
                "task_groups_with_any_miss": len(missed_tasks),
                "one_sided_exact_95_upper_task_group_event_rate": zero_event_upper,
                "interpretation": "sensitivity bound for the proportion of independent positive task clusters with any missed link; not an interval for task-macro FSR",
            },
        },
        "selected_method": "balanced_selected",
        "inspection_default_method": "balanced_selected",
        "methods": method_records,
        "traces": trace_records,
        "pairs": pairs,
        "sources": {
            "pair_scores": "results/balanced_innovation_v11/outer_pair_scores.csv",
            "summary": "results/balanced_innovation_v11/results.json",
            "protocol": "experiments/balanced_innovation_v11/program.md",
            "github_json": "https://github.com/mcxiaoxiao/AttackAgentDetect/blob/main/site/data/task-association-v2.json",
            "github_pair_scores": "https://github.com/mcxiaoxiao/AttackAgentDetect/blob/main/results/balanced_innovation_v11/outer_pair_scores.csv",
            "github_summary": "https://github.com/mcxiaoxiao/AttackAgentDetect/blob/main/results/balanced_innovation_v11/results.json",
            "pair_scores_sha256": sha256(PAIR_FILE),
            "summary_sha256": sha256(SUMMARY_FILE),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(pairs)} pairs, {len(trace_records)} traces, {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
