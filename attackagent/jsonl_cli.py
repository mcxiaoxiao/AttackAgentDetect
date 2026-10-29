"""Small JSONL command-line adapter for the existing trace research prototype."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .prototype import ModeAwareAssociationPrototype, TraceRecord

SCHEMA_VERSION = 1


def read_jsonl(path: pathlib.Path, *, training: bool) -> list[TraceRecord]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            if item.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_number}: schema_version must be {SCHEMA_VERSION}")
            required = ("trace_id", "benchmark", "commands", "actions")
            missing = [key for key in required if key not in item]
            if training:
                missing += [key for key in ("activity_id", "mode") if key not in item]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields: {', '.join(missing)}")
            for key in ("trace_id", "benchmark"):
                if not isinstance(item[key], str) or not item[key].strip():
                    raise ValueError(f"{path}:{line_number}: {key} must be a non-empty string")
            if training and (not isinstance(item["activity_id"], str) or not item["activity_id"].strip()):
                raise ValueError(f"{path}:{line_number}: activity_id must be a non-empty string")
            for key in ("commands", "actions"):
                if not isinstance(item[key], list) or any(not isinstance(x, str) for x in item[key]):
                    raise ValueError(f"{path}:{line_number}: {key} must be an array of strings")
            mode = item.get("mode", "unknown")
            if training and mode not in {"human", "agent"}:
                raise ValueError(f"{path}:{line_number}: training mode must be human or agent")
            if not training and mode not in {"human", "agent", "unknown", "script"}:
                raise ValueError(f"{path}:{line_number}: unsupported mode label {mode!r}")
            records.append(TraceRecord(
                trace_id=str(item["trace_id"]),
                benchmark=str(item["benchmark"]),
                activity_id=str(item.get("activity_id", "unlabeled")),
                mode=str(mode),
                commands=tuple(item["commands"]),
                actions=tuple(item["actions"]),
                source=str(item.get("source", "unknown")),
            ))
    return records


def _mode_row(record, prediction):
    return {
        "trace_id": record.trace_id,
        "prediction": prediction["supported_prediction"],
        "agent_like_score": prediction["agent_like_score"],
        "score_calibrated": prediction["score_calibrated"],
        "evidence": {
            "command_count": prediction["command_count"],
            "recognized_command_features": prediction["recognized_feature_count"],
        },
        "abstention_reason": prediction["abstention_reason"],
        "supported_classes": ["human", "agent"],
        "unsupported_class": "script",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=pathlib.Path, help="labeled training JSONL")
    parser.add_argument("--queries", required=True, type=pathlib.Path, help="query trace JSONL")
    parser.add_argument("--candidates", required=True, type=pathlib.Path, help="candidate trace JSONL")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    try:
        training = read_jsonl(args.train, training=True)
        queries = read_jsonl(args.queries, training=False)
        candidates = read_jsonl(args.candidates, training=False)
        if not training or not queries or not candidates:
            raise ValueError("training, queries, and candidates must each contain at least one record")
        model = ModeAwareAssociationPrototype().fit(training)
        all_records = queries + candidates
        mode_predictions = model.predict_mode(all_records)
        output = []
        for query_index, query in enumerate(queries):
            query_prediction = mode_predictions[query_index]
            query_result = {
                "schema_version": SCHEMA_VERSION,
                "query": _mode_row(query, query_prediction),
                "scope": "same-benchmark trace ranking under the training distribution; not actor attribution",
                "rank_score_semantics": "uncalibrated pair-model score; ranking only",
            }
            if query_prediction["abstention_reason"]:
                query_result["candidate_ranking"] = []
                output.append(query_result)
                continue
            candidate_indices = [
                len(queries) + index for index, row in enumerate(candidates)
                if row.benchmark == query.benchmark
                and mode_predictions[len(queries) + index]["abstention_reason"] is None
            ]
            if not candidate_indices:
                query_result["candidate_ranking"] = []
                query_result["abstention_reason"] = "no_supported_same_benchmark_candidates"
                output.append(query_result)
                continue
            pairs = [(query_index, index) for index in candidate_indices]
            scores = model.predict_pair_scores(all_records, pairs)
            baseline_order = sorted(range(len(pairs)), key=lambda i: (-scores["similarity_only"][i], candidates[candidate_indices[i] - len(queries)].trace_id))
            mode_order = sorted(range(len(pairs)), key=lambda i: (-scores["mode_aware"][i], candidates[candidate_indices[i] - len(queries)].trace_id))
            baseline_rank = {pair_index: rank for rank, pair_index in enumerate(baseline_order, start=1)}
            mode_rank = {pair_index: rank for rank, pair_index in enumerate(mode_order, start=1)}
            rows = []
            for pair_index, candidate_index in enumerate(candidate_indices):
                candidate = candidates[candidate_index - len(queries)]
                rows.append({
                    "candidate_trace_id": candidate.trace_id,
                    "similarity_rank": baseline_rank[pair_index],
                    "mode_aware_rank": mode_rank[pair_index],
                    "similarity_rank_score": float(scores["similarity_only"][pair_index]),
                    "mode_aware_rank_score": float(scores["mode_aware"][pair_index]),
                })
            query_result["candidate_ranking"] = sorted(rows, key=lambda row: row["mode_aware_rank"])[:args.top_k]
            output.append(query_result)
        json.dump({"schema_version": SCHEMA_VERSION, "results": output}, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
