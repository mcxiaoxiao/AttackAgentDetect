#!/usr/bin/env python3
"""Build the unified, source-qualified command-view dataset and web bundle."""

from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
RESEARCH = ROOT / "research"
DATA = SITE / "data"
RESULTS = RESEARCH / "results"
OUT = RESULTS / "behavior_trace_dataset_v2"

FIELDS = [
    "trace_id", "source_dataset", "source_revision", "benchmark", "track",
    "task_id", "task_group", "mode", "participant_id", "model", "framework",
    "observation_format", "command_count", "commands", "actions", "action_counts",
    "sequence_scope", "has_human_agent_peer", "evaluation_role", "relation_scope",
    "source_task_id", "task_condition", "cross_source_task_equivalence", "license",
    "source_url", "record_locator", "normalization",
]


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["trace_id"]] = row
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def make_record(meta: dict, commands: list[str], actions: list[str], action_counts: dict,
                sequence_scope: str, participant_id: str = "") -> dict:
    record = {field: "" for field in FIELDS}
    for key in FIELDS:
        if key in meta:
            record[key] = meta[key]
    record["commands"] = commands
    record["actions"] = actions
    record["action_counts"] = action_counts
    record["sequence_scope"] = sequence_scope
    record["participant_id"] = participant_id or meta.get("participant_id", "")
    record["has_human_agent_peer"] = to_bool(meta.get("has_human_agent_peer", False))
    return record


def source_rows() -> list[dict]:
    catalog = json.loads((DATA / "behavior-corpus-v1.json").read_text(encoding="utf-8"))
    public_by_id = {row["trace_id"]: row for row in catalog["traces"]}
    core_meta = {row["trace_id"]: row for row in read_csv(RESULTS / "behavior_corpus_v1" / "trace_registry.csv")}
    core_evidence = read_jsonl(RESULTS / "behavior_corpus_v1" / "evidence.jsonl")
    records = []

    for trace_id, meta in core_meta.items():
        public = public_by_id[trace_id]
        evidence = core_evidence[trace_id]
        if meta["source_dataset"] == "antieval/cybench-trajectories":
            # Antieval does not state a source license. Retain only the already
            # published capped command view, aggregates and upstream locator.
            commands = list(public.get("commands_preview", []))[:24]
            actions = []
            action_counts = public.get("action_counts", {})
            sequence_scope = "public-prefix-up-to-24; source-license-not-stated"
        else:
            commands = list(evidence.get("commands", []))
            actions = list(evidence.get("actions", []))
            action_counts = public.get("action_counts", {})
            sequence_scope = "complete-normalized-command-view"
        records.append(make_record(meta, commands, actions, action_counts, sequence_scope))

    human_registry = read_csv(RESULTS / "human_task_supplement_v1" / "trace_registry.csv")
    human_evidence = read_jsonl(RESULTS / "human_task_supplement_v1" / "evidence.jsonl")
    for row in human_registry:
        if int(row["common_command_count"]) < 1:
            continue
        evidence = human_evidence[row["trace_id"]]
        variant = row["variant"]
        level = row["task_group"].rsplit("/", 1)[-1]
        meta = {
            "trace_id": row["trace_id"],
            "source_dataset": "kypo-locust-human-training",
            "source_revision": f"zenodo:10298187/{variant}",
            "benchmark": "kypo-locust-training",
            "track": "human_task_supplement",
            "task_id": row["task_group"],
            "task_group": row["task_group"],
            "mode": "human",
            "participant_id": row["participant_id"],
            "model": "",
            "framework": "not-reported-by-source",
            "observation_format": "process-mining-bash-command-name",
            "command_count": int(row["common_command_count"]),
            "has_human_agent_peer": False,
            "evaluation_role": "human-only-task-level-supplement",
            "relation_scope": row["relation_scope"],
            "source_task_id": level,
            "task_condition": variant,
            "cross_source_task_equivalence": "not-asserted",
            "license": row["license"],
            "source_url": "https://zenodo.org/records/10298187",
            "record_locator": row["source_record"],
            "normalization": "known-command-heads-lossy-v1",
        }
        records.append(make_record(
            meta, list(evidence.get("commands", [])), [], evidence.get("action_counts", {}),
            "complete-normalized-command-view", row["participant_id"],
        ))

    ids = [row["trace_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate trace IDs in unified dataset")
    nonempty_human_records = sum(int(row["common_command_count"]) > 0 for row in human_registry)
    if len(records) != len(catalog["traces"]) + nonempty_human_records:
        raise ValueError("Unexpected unified dataset record count")
    return records


def compact_cell(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def csv_text(records: list[dict]) -> str:
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in records:
        writer.writerow({key: compact_cell(value) for key, value in row.items()})
    return buffer.getvalue()


def build_analysis() -> dict:
    mixture = json.loads((RESULTS / "overnight_fingerprint_v1" / "mixture" / "composite-balance.json").read_text())
    short_trace = json.loads((RESULTS / "short_trace_extension_v1" / "metrics.json").read_text())
    matched = json.loads((RESULTS / "matched_human_agent_probe_v1" / "metrics.json").read_text())
    pair_summary = json.loads((DATA / "ctf-behavior-trace-summary.json").read_text())
    return {
        "schema_version": 2,
        "dataset_release": "behavior-trace-dataset-v2",
        "fingerprint_v2": {
            key: json.loads((RESULTS / "behavioral_fingerprint_v2" / directory / filename).read_text())
            for key, directory, filename in [
                ("cth_cross_mode", "cth_cross_mode", "metrics.json"),
                ("cth_role_mixture", "cth_mixture", "metrics.json"),
                ("ait_recognition", "ait_recognition", "metrics.json"),
                ("ait_ablation", "ait_ablation", "metrics.json"),
                ("honey", "honey", "metrics.json"),
                ("cybiasbench", "cybiasbench", "metrics.json"),
                ("kypo_human", "kypo_human", "metrics.json"),
                ("nested_refinement", "refinement", "metrics.json"),
                ("mode_assisted", "refinement", "mode_assisted_metrics.json"),
                ("error_strata", "refinement", "error_strata.json"),
            ]
        },
        "execution_mode_recognition": {
            "source": "AIT / AECID Linux audit logs; separate from the 1,407 trace catalog records",
            "eligible_records": 89,
            "human": 45,
            "agent": 44,
            "task_groups": 12,
            "task_held_out_balanced_accuracy": {
                "command_tfidf": 0.968,
                "window_matrix_8d": 0.563,
                "whole_trace_8d": 0.552,
                "matrix_vs_whole_trace_delta": 0.011,
                "matrix_delta_95ci": [-0.143, 0.154],
            },
            "source_held_out_matrix_balanced_accuracy": 0.399,
            "interpretation": "Labels are entangled with task and capture process; this is a dataset-label baseline, not validated person or agent identification.",
        },
        "ctf_mixture_association": mixture,
        "short_trace_agent_only_retrieval": short_trace,
        "cross_source_same_challenge_human_query": matched,
        "ctf_pair_association": pair_summary,
    }


def main() -> None:
    records = source_rows()
    analysis = build_analysis()
    source_counts = Counter(row["source_dataset"] for row in records)
    mode_counts = Counter(row["mode"] for row in records)
    track_counts = Counter(row["track"] for row in records)
    summary = {
        "record_count": len(records),
        "human_records": mode_counts["human"],
        "agent_records": mode_counts["agent"],
        "source_count": len(source_counts),
        "sources": dict(sorted(source_counts.items())),
        "tracks": dict(sorted(track_counts.items())),
        "minimum_command_coverage": {
            str(gate): sum(int(row["command_count"]) >= gate for row in records)
            for gate in (1, 4, 8, 16, 32)
        },
        "license_policy": "Licenses and source terms are recorded per trace; the package has no single blanket license.",
    }
    dataset = {
        "schema_version": 2,
        "title": "Execution Behavior Trace Dataset",
        "normalization": "Known command heads and coarse action labels; raw arguments, prompts, outputs and timestamps are not included.",
        "summary": summary,
        "fields": FIELDS,
        "records": records,
    }
    json_output = json.dumps(dataset, ensure_ascii=False, indent=2) + "\n"
    csv_output = csv_text(records)
    analysis_output = json.dumps(analysis, ensure_ascii=False, indent=2) + "\n"

    readme = "\n".join([
        "# Execution Behavior Trace Dataset v2", "",
        "A source-qualified command-view dataset with current analysis outputs.", "",
        "## Contents", "",
        "- `traces/trace-catalog-v2.json`: normalized records and ordered command/action sequences where source terms allow.",
        "- `traces/trace-catalog-v2.csv`: the same records; sequence and count fields are JSON-encoded cells.",
        "- `relations/human-agent-cybench-crosswalk.csv`: candidate shared challenge slugs, not verified fixture equivalence.",
        "- `analysis/analysis-v2.json`: recognition, CTF mixture, short-trace, matched challenge, and pair-evaluation summaries.",
        "- `analysis/ctf-behavior-trace-pairs.csv` and `ctf-behavior-trace-summary.json`: the 97-record CTF pair-evaluation panel.",
        "- `protocols/`: experiment reports documenting methods and limits.", "",
        "## Dataset composition", "",
        "The catalog has 1,407 source-qualified records: 1,142 from Cyber Task Horizons, Antieval CyBench, and AgentTrace, plus 265 Human-only KYPO task segments. It contains 344 Human and 1,063 Agent records across four source families.",
        "Sequences contain normalized command names, not raw terminal logs. CTH, AgentTrace, and KYPO include the complete normalized sequence. Antieval CyBench does not state a source license; those records retain only the already-published command prefix (up to 24 entries), aggregates, and upstream locator. Review the pinned source terms before reuse or redistribution.", "",
        "## How to use", "",
        "Use `source_dataset` plus `task_group` as the grouping key; do not merge same-named tasks across sources by default. Keep every KYPO record from a participant in one split. The five-row CyBench crosswalk is exploratory: challenge version, fixture, and capture parity have not been verified.",
        "Suitable analyses include within-source same-task retrieval, Human-only task-level variation, command-view sensitivity, and source-held-out pilots. The records do not label real campaigns, operators, or person identities. The AIT recognition experiment is a separate analysis panel and is not part of the 1,407 trace catalog.", "",
        "## License and attribution", "",
        "No single license applies to the whole bundle. Use each row's `license`, `source_url`, `source_revision`, and `record_locator`. `not-stated-by-source` is not permission to reuse or redistribute. KYPO and CTH are CC BY 4.0; AgentTrace includes Apache 2.0 and upstream task terms. Preserve upstream attribution.", "",
        "## Citation", "",
        "AttackAgent Detect. *Execution Behavior Trace Dataset v2: Source-qualified command-view records and analysis*. 2026. Cite the upstream source for each subset used.",
    ]) + "\n"

    package_files = {
        "traces/trace-catalog-v2.json": json_output,
        "traces/trace-catalog-v2.csv": csv_output,
        "relations/human-agent-cybench-crosswalk.csv": (DATA / "human-agent-cybench-crosswalk-v1.csv").read_text(encoding="utf-8"),
        "analysis/analysis-v2.json": analysis_output,
        "analysis/ctf-behavior-trace-pairs.csv": (DATA / "ctf-behavior-trace-pairs.csv").read_text(encoding="utf-8"),
        "analysis/ctf-behavior-trace-summary.json": (DATA / "ctf-behavior-trace-summary.json").read_text(encoding="utf-8"),
        "protocols/overnight-fingerprint-v1.md": (RESEARCH / "reports" / "overnight-fingerprint-v1.md").read_text(encoding="utf-8"),
        "protocols/short-trace-extension-v1.md": (RESEARCH / "reports" / "short-trace-extension-v1.md").read_text(encoding="utf-8"),
        "protocols/human-task-data-extension-v1.md": (RESEARCH / "reports" / "human-task-data-extension-v1.md").read_text(encoding="utf-8"),
        "protocols/pairwise-trace-association.md": (RESEARCH / "reports" / "pairwise-trace-association.md").read_text(encoding="utf-8"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    for relative, content in package_files.items():
        path = OUT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (DATA / "behavior-trace-dataset-v2.json").write_text(json_output, encoding="utf-8")
    (DATA / "behavior-trace-dataset-v2.csv").write_text(csv_output, encoding="utf-8")
    (DATA / "behavior-trace-analysis-v2.json").write_text(analysis_output, encoding="utf-8")
    (SITE / "assets" / "behavior-trace-dataset-v2-README.md").write_text(readme, encoding="utf-8")
    zip_path = DATA / "behavior-trace-dataset-v2.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(OUT / "README.md", "behavior-trace-dataset-v2/README.md")
        archive.write(OUT / "summary.json", "behavior-trace-dataset-v2/summary.json")
        for relative in package_files:
            archive.write(OUT / relative, f"behavior-trace-dataset-v2/{relative}")
    print(json.dumps({
        "summary": summary,
        "csv_bytes": len(csv_output.encode("utf-8")),
        "json_bytes": len(json_output.encode("utf-8")),
        "zip_bytes": zip_path.stat().st_size,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
