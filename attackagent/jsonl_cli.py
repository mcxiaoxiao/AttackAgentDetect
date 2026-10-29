"""JSONL CLI for the source-first AttackAgentDetect research library."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .api import fit_detector, list_methods, predict_modes, rank_candidates
from .auditd import load_auditd
from .evidence import LabeledEvidence, TraceEvidence
from .prototype import ModeAwareAssociationPrototype, TraceRecord

SCHEMA_VERSION = 1
COMMANDS = {"normalize", "rank", "detect", "methods"}


def _rows(path: pathlib.Path) -> list[dict]:
    output = []
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
            if item.get("schema_version") not in {1, 2}:
                raise ValueError(f"{path}:{line_number}: schema_version must be 1 or 2")
            output.append(item)
    return output


def read_evidence_jsonl(path: pathlib.Path) -> list[TraceEvidence]:
    """Read inference evidence and intentionally ignore all annotation fields."""
    records = []
    for line_number, item in enumerate(_rows(path), start=1):
        required = ("trace_id", "benchmark", "commands", "actions")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields: {', '.join(missing)}")
        if any(not isinstance(item[key], str) or not item[key].strip() for key in ("trace_id", "benchmark")):
            raise ValueError(f"{path}:{line_number}: trace_id and benchmark must be non-empty strings")
        for key in ("commands", "actions"):
            if not isinstance(item[key], list) or any(not isinstance(value, str) for value in item[key]):
                raise ValueError(f"{path}:{line_number}: {key} must be an array of strings")
        source_format = item.get("source_format", "jsonl-command-v1")
        if not isinstance(source_format, str) or not source_format:
            raise ValueError(f"{path}:{line_number}: source_format must be a non-empty string")
        records.append(TraceEvidence(
            trace_id=item["trace_id"],
            source_format=source_format,
            benchmark=item["benchmark"],
            commands=tuple(item["commands"]),
            actions=tuple(item["actions"]),
            event_refs=tuple(item.get("event_refs", ())),
            quality=item.get("quality", {}),
        ))
    return records


def read_labeled_jsonl(path: pathlib.Path) -> list[LabeledEvidence]:
    records = []
    evidence = read_evidence_jsonl(path)
    for index, (item, row) in enumerate(zip(_rows(path), evidence), start=1):
        if item.get("mode") not in {"human", "agent"}:
            raise ValueError(f"{path}:{index}: training mode must be human or agent")
        records.append(LabeledEvidence(
            evidence=row,
            mode=item["mode"],
            activity_id=item.get("activity_id"),
            source=item.get("source"),
        ))
    return records


def read_jsonl(path: pathlib.Path, *, training: bool) -> list[TraceRecord]:
    """Compatibility reader for the original in-repo research prototype.

    The legacy dataclass has label fields for training/evaluation. At inference,
    labels in the JSONL are never used and are replaced with ``unknown``.
    """
    records = []
    for index, item in enumerate(_rows(path), start=1):
        missing = [key for key in ("trace_id", "benchmark", "commands", "actions") if key not in item]
        if training:
            missing.extend(key for key in ("activity_id", "mode") if key not in item)
        if missing:
            raise ValueError(f"{path}:{index}: missing fields: {', '.join(missing)}")
        for key in ("commands", "actions"):
            if not isinstance(item[key], list) or any(not isinstance(value, str) for value in item[key]):
                raise ValueError(f"{path}:{index}: {key} must be an array of strings")
        mode = item.get("mode") if training else "unknown"
        if training and mode not in {"human", "agent"}:
            raise ValueError(f"{path}:{index}: training mode must be human or agent")
        records.append(TraceRecord(
            trace_id=str(item["trace_id"]),
            benchmark=str(item["benchmark"]),
            activity_id=str(item.get("activity_id", "unlabeled")),
            mode=str(mode),
            commands=tuple(item["commands"]),
            actions=tuple(item["actions"]),
            source=str(item.get("source", "unknown")) if training else "unknown",
        ))
    return records


def _json_evidence(row: TraceEvidence) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": row.trace_id,
        "source_format": row.source_format,
        "benchmark": row.benchmark,
        "commands": list(row.commands),
        "actions": list(row.actions),
        "event_refs": list(row.event_refs),
        "quality": dict(row.quality),
    }


def _mode_json(score) -> dict:
    return {
        "trace_id": score.trace_id,
        "prediction": score.prediction,
        "agent_like_score": score.agent_like_score,
        "score_calibrated": score.calibrated,
        "available": score.available,
        "abstention_reason": score.reason,
        "evidence_count": score.evidence_count,
        "source_format": score.source_format,
        "supported_classes": ["human", "agent"],
        "unsupported_class": "script",
    }


def _legacy_rank(args) -> dict:
    training = read_labeled_jsonl(args.train)
    detector = fit_detector(training)
    queries = read_evidence_jsonl(args.queries)
    all_candidates = read_evidence_jsonl(args.candidates)
    if not queries:
        raise ValueError("queries must contain at least one record")
    results = []
    for query in queries:
        candidates = [row for row in all_candidates if row.benchmark == query.benchmark]
        scores = predict_modes(detector, [query])[0]
        ranked = rank_candidates(query, candidates, method="mode_adjusted", detector=detector, mode_weight=0.25)
        candidate_rows = [{
            "candidate_trace_id": row.trace_id,
            "rank": row.rank,
            "rank_score": row.score,
            "score_semantics": row.score_semantics,
        } for row in ranked[:args.top_k]]
        results.append({
            "schema_version": SCHEMA_VERSION,
            "query": _mode_json(scores),
            "scope": "same-benchmark, label-free trace ranking; task labels are not consumed at inference",
            "candidate_ids_preserved": [row.trace_id for row in candidates if row.trace_id != query.trace_id],
            "candidate_ranking": candidate_rows,
        })
    return {"schema_version": SCHEMA_VERSION, "results": results}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    normalize = subparsers.add_parser("normalize", help="normalize Linux auditd records into trace JSONL")
    normalize.add_argument("--input", required=True, type=pathlib.Path)
    normalize.add_argument("--output", type=pathlib.Path)
    normalize.add_argument("--host", default="host-unknown")
    normalize.add_argument("--inactivity-seconds", type=int, default=300)

    rank = subparsers.add_parser("rank", help="rank a fixed, label-free candidate gallery")
    rank.add_argument("--query", required=True, type=pathlib.Path)
    rank.add_argument("--candidates", required=True, type=pathlib.Path)
    rank.add_argument("--method", default="command_cosine", choices=("command_cosine", "action_bigram", "procgrep_jsd", "mode_adjusted"))
    rank.add_argument("--train", type=pathlib.Path, help="optional format-matched human/agent labels for mode_adjusted")
    rank.add_argument("--base-method", default="command_cosine", choices=("command_cosine", "action_bigram", "procgrep_jsd"))
    rank.add_argument("--top-k", type=int, default=10)

    detect = subparsers.add_parser("detect", help="fit a local mode baseline and score evidence")
    detect.add_argument("--train", required=True, type=pathlib.Path)
    detect.add_argument("--input", required=True, type=pathlib.Path)
    detect.add_argument("--seed", type=int, default=20261030)

    subparsers.add_parser("methods", help="list implemented baselines and cited external work")
    return parser


def _run_subcommand(args) -> dict | list[dict]:
    if args.command == "methods":
        return list_methods()
    if args.command == "normalize":
        traces, stats = load_auditd(args.input, host=args.host, inactivity_seconds=args.inactivity_seconds, return_stats=True)
        rows = [_json_evidence(row) for row in traces]
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"schema_version": SCHEMA_VERSION, "stats": stats, "traces": rows}
    if args.command == "detect":
        detector = fit_detector(read_labeled_jsonl(args.train), seed=args.seed)
        predictions = predict_modes(detector, read_evidence_jsonl(args.input))
        return {
            "schema_version": SCHEMA_VERSION,
            "detector": {"source_format": detector.source_format, "training_rows": detector.training_rows, "score_calibrated": detector.calibrated},
            "predictions": [_mode_json(row) for row in predictions],
        }
    if args.command == "rank":
        if args.top_k < 1:
            raise ValueError("--top-k must be at least 1")
        queries = read_evidence_jsonl(args.query)
        candidates = read_evidence_jsonl(args.candidates)
        if len(queries) != 1:
            raise ValueError("--query must contain exactly one trace")
        query = queries[0]
        gallery = [row for row in candidates if row.benchmark == query.benchmark]
        detector = fit_detector(read_labeled_jsonl(args.train)) if args.train else None
        if args.method == "mode_adjusted" and detector is None:
            raise ValueError("mode_adjusted requires --train; use command_cosine for label-free ranking")
        ranked = rank_candidates(query, gallery, method=args.method, detector=detector, base_method=args.base_method)
        return {
            "schema_version": SCHEMA_VERSION,
            "query_trace_id": query.trace_id,
            "method": args.method,
            "candidate_count": len([row for row in gallery if row.trace_id != query.trace_id]),
            "candidate_ids_preserved": [row.trace_id for row in gallery if row.trace_id != query.trace_id],
            "results": [row.__dict__ for row in ranked[:args.top_k]],
            "scope": "same-benchmark trace ranking; not actor or APT-group attribution",
        }
    raise ValueError("a subcommand is required")


def _legacy_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Compatibility form: --train --queries --candidates")
    parser.add_argument("--train", required=True, type=pathlib.Path)
    parser.add_argument("--queries", required=True, type=pathlib.Path)
    parser.add_argument("--candidates", required=True, type=pathlib.Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    try:
        result = _legacy_rank(args)
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0].startswith("-"):
        return _legacy_main(arguments)
    parser = _build_parser()
    try:
        args = parser.parse_args(arguments)
        result = _run_subcommand(args)
        if args.command == "normalize" and args.output:
            print(json.dumps({"schema_version": SCHEMA_VERSION, "stats": result["stats"], "output": str(args.output)}, ensure_ascii=False, indent=2))
        else:
            json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
