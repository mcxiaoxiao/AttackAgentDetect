#!/usr/bin/env python3
"""Export the selected public traces as compact, ordered command-name steps."""

import csv
import json
import pathlib
import sys

RESEARCH_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = RESEARCH_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import attackagent.traces as trace_loader  # noqa: E402

trace_loader.ROOT = RESEARCH_ROOT
trace_loader.DATA = RESEARCH_ROOT / "data" / "raw" / "cyber-task-horizons"

PHASE_LABELS = {
    "discover": "发现与定位",
    "read": "读取与解析",
    "search": "搜索",
    "execute": "执行与分析",
    "build": "构建",
    "network": "网络交互",
    "modify": "文件操作",
    "environment": "环境检查",
}


def main():
    observations = trace_loader.load_observations() + trace_loader.load_crossbenchmark_observations()
    by_id = {row.trace_id: row for row in observations}
    pair_file = REPO_ROOT / "site" / "data" / "ctf-behavior-trace-pairs.csv"
    with pair_file.open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    trace_ids = sorted({row[key] for row in pairs for key in ("query_trace_id", "candidate_trace_id")})
    missing = sorted(set(trace_ids) - by_id.keys())
    if missing:
        raise RuntimeError(f"Missing source traces for {len(missing)} IDs: {missing[:5]}")

    details = {}
    for trace_id in trace_ids:
        row = by_id[trace_id]
        details[trace_id] = {
            "task": row.task,
            "mode": row.label,
            "steps": [
                {"command": command, "phase": phase, "phase_label": PHASE_LABELS.get(phase, phase)}
                for command, phase in zip(row.names, row.groups)
            ],
        }
    target = REPO_ROOT / "site" / "data" / "trace-details.json"
    target.write_text(json.dumps(details, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {len(details)} traces and {sum(len(row['steps']) for row in details.values())} ordered steps to {target}")


if __name__ == "__main__":
    main()
