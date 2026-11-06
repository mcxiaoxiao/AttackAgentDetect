#!/usr/bin/env python3
"""Count observed B/S/T window measurements in the matched-study cohorts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "research/experiments/matched_fingerprint_v1/run.py"
OUT = ROOT / "research/results/matched_fingerprint_v2/dimension_selection_20261104"
WINDOW = 8
FEATURE_NAMES = {
    "B": ("command_diversity", "command_entropy", "dominant_command_share",
          "action_switch_rate", "next_action_conditional_entropy",
          "longest_action_run_share", "action_pair_reuse", "adjacent_command_repeat"),
    "S": ("first_seen_command_share", "revisit_share", "normalized_revisit_span",
          "aba_action_return_share", "lag3_action_return_share",
          "reused_command_bigram_share", "reused_command_trigram_share",
          "action_distribution_shift"),
    "T": ("median_log_gap", "log_gap_iqr", "gap_burstiness", "fast_gap_share",
          "slow_gap_share", "lag1_log_gap_correlation", "median_log_duration",
          "median_log_idle_interval"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("matched_v1_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def count_features(rows: list[dict], profiles: list[dict]) -> dict:
    result = {"trace_count": len(rows), "complete_block_traces": {},
              "features": {}, "window_size": WINDOW}
    for block, names in FEATURE_NAMES.items():
        values = np.vstack([row[block] for row in profiles])
        # The profile readout stores mean, SD, and trend consecutively per item.
        feature_means = values[:, ::3]
        result["complete_block_traces"][block] = sum(
            bool(row[f"{block}_ok"]) for row in profiles)
        result["features"][block] = {
            name: {"traces_with_any_finite_window_value": int(
                np.isfinite(feature_means[:, i]).sum())}
            for i, name in enumerate(names)
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runner = load_module(RUNNER)
    datasets = {}
    for name in ("cth", "ait"):
        rows, _, _ = runner.build_cth() if name == "cth" else runner.build_ait()
        profiles = runner.profile_cache(rows, minimum=2 if name == "cth" else 4)[WINDOW]
        datasets[name] = count_features(rows, profiles)
    nyu_manifest = json.loads((ROOT / "research/results/matched_fingerprint_v2/independent_nyu_agent/run_manifest.json").read_text())
    datasets["nyu"] = {
        "trace_count": nyu_manifest["confirmation_traces"],
        "complete_block_traces": {"B": nyu_manifest["confirmation_traces"],
                                   "S": nyu_manifest["confirmation_traces"], "T": 0},
        "features": {"B": "all eligible runs have complete B/S under the frozen protocol",
                     "S": "all eligible runs have complete B/S under the frozen protocol",
                     "T": "timestamps and durations are unavailable"},
        "source": "frozen NYU run manifest; no raw timing fields in the model contract",
    }
    output = {
        "protocol": "finite feature availability by trace at 8-event windows; no raw values emitted",
        "feature_definition_source": "attackagent/fingerprint_v2.py",
        "input_sha256": {str(RUNNER.relative_to(ROOT)): sha256(RUNNER)},
        "datasets": datasets,
    }
    target = OUT / "window_feature_coverage.json"
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(target)
    for dataset, info in datasets.items():
        print(dataset, info["trace_count"], info["complete_block_traces"])
        for block, values in info["features"].items():
            if isinstance(values, dict):
                print(block, " ".join(f"{key}={value['traces_with_any_finite_window_value']}"
                                      for key, value in values.items()))


if __name__ == "__main__":
    main()
