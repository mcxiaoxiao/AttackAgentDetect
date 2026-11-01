"""Audit existing exported profiles; no model fitting or performance access."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/results/delegation_behavior_v1"
SOURCES = (
    ("multidimensional_profile_association_v1", "behavior_profiles_v3.jsonl"),
    ("multidimensional_rerun_v1", "behavior_profiles.jsonl"),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_specs():
    source = ROOT / "attackagent/profiles.py"
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "FEATURE_SPECS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("FEATURE_SPECS not found")


def audit(folder, filename, specs):
    base = ROOT / "research/results" / folder
    profile_path, label_path = base / filename, base / "evaluation_labels.csv"
    profiles = [json.loads(line) for line in profile_path.read_text().splitlines() if line.strip()]
    with label_path.open(newline="") as source:
        labels = list(csv.DictReader(source))
    ids = [row["trace_id"] for row in profiles]
    label_ids = [row["trace_id"] for row in labels]
    if len(ids) != len(set(ids)) or len(label_ids) != len(set(label_ids)):
        raise ValueError(f"Duplicate profile/label trace ID in {folder}")
    joined = {row["trace_id"]: row for row in labels}
    by_role, by_benchmark, formats, tensor_shapes = Counter(), Counter(), Counter(), Counter()
    missing, invalid = Counter(), Counter()
    for row in profiles:
        label = joined.get(row["trace_id"], {})
        role = label.get("record_label_for_evaluation_only", label.get("actor_label_for_evaluation_only", "unknown"))
        by_role[role] += 1
        by_benchmark[label.get("benchmark", row.get("benchmark", "unknown"))] += 1
        formats[row.get("source_format", "not_exported")] += 1
        tensor = row.get("tensor_profile", {})
        tensor_shapes[str(tensor.get("shape", row.get("tensor_shape", "not_exported")))] += 1
        values = row.get("feature_values", row.get("features", {}))
        if not isinstance(values, dict):
            raise ValueError(f"Unexpected feature representation in {folder}")
        for name, *_ in specs:
            value = values.get(name)
            if value is None:
                missing[name] += 1
            elif not isinstance(value, (int, float)) or not math.isfinite(value):
                invalid[name] += 1
    required = ("activity_id", "episode_id", "controller_id", "delegation_condition", "repeat_id", "harness_version")
    observed_keys = set().union(*(set(row) for row in [*profiles, *labels]))
    return {
        "source": folder,
        "profile_records": len(profiles),
        "label_records": len(labels),
        "profiles_without_labels": len(set(ids) - set(label_ids)),
        "labels_without_profiles": len(set(label_ids) - set(ids)),
        "by_record_label": dict(sorted(by_role.items())),
        "by_benchmark": dict(sorted(by_benchmark.items())),
        "exported_source_formats": dict(formats),
        "tensor_shapes": dict(tensor_shapes),
        "feature_missing_counts": {name: missing[name] for name, *_ in specs},
        "nonfinite_or_nonnumeric_counts": dict(invalid),
        "required_metadata_keys_present": {key: key in observed_keys for key in required},
        "eligibility": {
            "descriptive_profile_audit": True,
            "controlled_delegation_effect": False,
            "independent_activity_association_validation": False,
            "reason": "Exports lack verified activity, delegation, controller, repetition and harness registry; task IDs and record labels cannot supply them.",
        },
        "notes": [
            "Record labels describe released trace sources, not independently verified delegation conditions.",
            "Matching source_format strings do not establish matching underlying capture pipelines.",
            "Datasets may overlap; record counts must not be added as independent samples.",
        ],
        "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (profile_path, label_path)},
    }


def main():
    specs = feature_specs()
    reports = [audit(folder, filename, specs) for folder, filename in SOURCES]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "feature_catalog.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("feature_id", "name", "label_zh", "rationale", "origin_annotation"))
        for index, spec in enumerate(specs, 1):
            writer.writerow((f"P{index:02d}", *spec))
    report = {
        "status": "input_audit_only_no_effectiveness_experiment",
        "feature_count": len(specs),
        "feature_source_sha256": digest(ROOT / "attackagent/profiles.py"),
        "audit_script_sha256": digest(Path(__file__)),
        "datasets": reports,
    }
    (OUT / "input_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for row in reports:
        print(row["source"], row["profile_records"], dict(row["by_record_label"]))
    print("Saved input audit and 24-dimension catalog; no effect or performance metrics computed.")


if __name__ == "__main__":
    main()
