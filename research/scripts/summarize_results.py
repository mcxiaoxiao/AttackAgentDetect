#!/usr/bin/env python3
"""Emit a compact, aggregate-only summary from the checked-in experiment outputs."""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    audit = json.loads((ROOT / "results/data-audit.json").read_text(encoding="utf-8"))
    primary = json.loads((ROOT / "results/preliminary.json").read_text(encoding="utf-8"))
    test = primary["classification"]["test"]
    retrieval = primary["association"]["command_cosine"]["test"]["agent-human"]
    result = {
        "dataset": "InterCode-CTF",
        "revision": audit["revision"],
        "sessions": {
            "human": audit["human_samples"],
            "model_agent": audit["agent_samples"],
            "script": 0,
        },
        "test": {
            "held_out_tasks": primary["splits"]["test"]["task_count"],
            "mode_classifier": {
                "labels": "human vs selected model-agent sessions",
                "balanced_accuracy": round(test["balanced_accuracy"], 3),
            },
            "same_task_retrieval": {
                "pair": "human-agent",
                "auc": round(retrieval["pair_auc"], 3),
                "label_limit": "same task is not same actor or campaign",
            },
        },
        "scope": "exploratory; capture pipeline is confounded with class",
    }
    out = ROOT / "results/quick-summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
