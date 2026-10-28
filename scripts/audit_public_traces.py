#!/usr/bin/env python3
"""Audit matched task IDs and logging modalities without exporting raw commands."""

import collections
import gzip
import io
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cyber-task-horizons"
OUT = ROOT / "results" / "data-audit.json"


def samples_from_zip(data):
    with zipfile.ZipFile(data) as archive:
        names = [x for x in archive.namelist() if x.startswith("samples/") and x.endswith(".json")]
        return [json.loads(archive.read(name)) for name in names]


def main():
    manifest = json.loads((DATA / "download_manifest.json").read_text(encoding="utf-8"))
    human, agents, no_sample = [], [], []
    for row in manifest["files"]:
        path = DATA / row["path"]
        if "/human/" in row["path"]:
            with gzip.open(path, "rb") as f:
                samples = samples_from_zip(io.BytesIO(f.read()))
            if not samples:
                no_sample.append(path.name)
            for sample in samples:
                logs = sample.get("store", {}).get("HumanAgentState:logs", {})
                human.append({
                    "task": sample.get("id"),
                    "input_logs": sum(str(k).endswith(".input") for k in logs),
                })
        else:
            samples = samples_from_zip(path)
            for sample in samples:
                calls = [
                    call for message in sample.get("messages", [])
                    for call in (message.get("tool_calls", []) or [])
                ]
                counts = collections.Counter(call.get("function") for call in calls)
                agents.append({"task": sample.get("id"), "model": path.parent.name, "calls": counts})
    h_tasks = {x["task"] for x in human if x["task"] is not None}
    a_tasks = {x["task"] for x in agents if x["task"] is not None}
    matched = h_tasks & a_tasks
    matching_h = [x for x in human if x["task"] in matched]
    matching_a = [x for x in agents if x["task"] in matched]
    result = {
        "source": manifest["source"],
        "revision": manifest["revision"],
        "benchmark": manifest["benchmark"],
        "human_files": manifest["human_file_count"],
        "human_samples": len(human),
        "human_distinct_tasks": len(h_tasks),
        "human_without_input": sum(x["input_logs"] == 0 for x in human),
        "human_files_without_sample": no_sample,
        "agent_files": manifest["model_file_count"],
        "agent_samples": len(agents),
        "agent_distinct_tasks": len(a_tasks),
        "agent_samples_by_model": dict(sorted(collections.Counter(x["model"] for x in agents).items())),
        "agent_tool_calls": dict(sorted(sum((x["calls"] for x in agents), collections.Counter()).items())),
        "matched_task_count": len(matched),
        "matched_human_samples": len(matching_h),
        "matched_agent_samples": len(matching_a),
        "matched_agent_without_bash": sum(not x["calls"].get("bash") for x in matching_a),
        "caveat": "Task IDs are not attack-actor or campaign labels; instrumentation differs by class.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
