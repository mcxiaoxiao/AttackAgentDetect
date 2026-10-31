#!/usr/bin/env python3
"""Build privacy-preserving cohort-mean data and static tensor SVG fallbacks."""
from __future__ import annotations

import argparse
import html
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/multidimensional_profile_association_v1/behavior_profiles_v3.jsonl"
DATA_OUTPUT = ROOT.parent / "site/assets/profile-tensor-means-v1.json"

CHANNELS = {
    "zh": ["命令占比", "词汇多样度", "相邻重复率"],
    "en": ["Command share", "Vocabulary diversity", "Adjacent repeat rate"],
}
ACTION_ZH = {
    "build": "构建", "discover": "发现", "environment": "环境", "execute": "执行",
    "modify": "修改", "network": "网络", "read": "读取", "search": "搜索",
}
PALETTE = ("#f2f6fa", "#d8e8f2", "#a8ccdf", "#6ba6c8", "#277eae")


def colour(value: float) -> str:
    index = max(0, min(4, round(max(0.0, min(1.0, value)) * 4)))
    return PALETTE[index]


def svg_for(data: dict, language: str) -> str:
    actions = data["action_labels"]
    values = data["mean_tensor"]
    channel_titles = CHANNELS[language]
    width, height, panel_w = 930, 324, 310
    grid_x, grid_y, cell_w, cell_h, gap_x, gap_y = 101, 67, 42, 23, 4, 4
    phase_names = ["01", "02", "03", "04"] if language == "en" else ["一", "二", "三", "四"]
    axis = ("Ordered phase", "Action family") if language == "en" else ("有序阶段", "操作类别")
    mean_label = "Mean across all traces" if language == "en" else "全部轨迹的平均值"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(mean_label)} · 4 × 8 × 3</title>',
        f'<desc id="desc">{data["trace_count"]} traces, four ordered phases, eight action families, three behavioral channels.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for channel in range(3):
        panel_x = channel * panel_w
        lines.extend([
            f'<text x="{panel_x + 7}" y="21" fill="#172331" font-family="Arial,sans-serif" font-size="13" font-weight="600">{html.escape(channel_titles[channel])}</text>',
            f'<text x="{panel_x + grid_x}" y="44" fill="#69798b" font-family="Arial,sans-serif" font-size="10">{html.escape(axis[0])}</text>',
            f'<text x="{panel_x + 7}" y="57" fill="#69798b" font-family="Arial,sans-serif" font-size="10">{html.escape(axis[1])}</text>',
            f'<line x1="{panel_x + 7}" y1="29" x2="{panel_x + panel_w - 12}" y2="29" stroke="#d9e0e8"/>',
        ])
        for phase in range(4):
            x = panel_x + grid_x + phase * (cell_w + gap_x) + cell_w / 2
            lines.append(f'<text x="{x}" y="61" fill="#8290a0" font-family="monospace" font-size="9" text-anchor="middle">{phase_names[phase]}</text>')
        for action_index, action in enumerate(actions):
            y = grid_y + action_index * (cell_h + gap_y)
            label = action if language == "en" else ACTION_ZH.get(action, action)
            lines.append(f'<text x="{panel_x + 7}" y="{y + 15}" fill="#69798b" font-family="Arial,sans-serif" font-size="10">{html.escape(label)}</text>')
            for phase in range(4):
                value = float(values[phase][action_index][channel])
                x = panel_x + grid_x + phase * (cell_w + gap_x)
                lines.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" rx="4" fill="{colour(value)}" stroke="#fff" stroke-width="1.5"/>')
                lines.append(f'<text x="{x + cell_w / 2}" y="{y + 15}" fill="#172331" font-family="monospace" font-size="9" text-anchor="middle">{round(value * 100)}%</text>')
        if channel < 2:
            x = panel_x + panel_w - 3
            lines.append(f'<line x1="{x}" y1="10" x2="{x}" y2="308" stroke="#d9e0e8"/>')
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    args = parser.parse_args(argv)
    profiles = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not profiles or any(not row["tensor_profile"]["available"] for row in profiles):
        raise SystemExit("Saved profile set is empty or contains unavailable tensors")
    action_labels = profiles[0]["tensor_profile"]["axes"]["action_family"]
    tensor_values = np.asarray([row["tensor_profile"]["values"] for row in profiles], dtype=float)
    if tensor_values.shape[1:] != (4, 8, 3):
        raise SystemExit(f"Unexpected tensor shape: {tensor_values.shape[1:]}")
    result = {
        "schema_version": "phase-action-channel-cohort-mean-v1",
        "trace_count": len(profiles),
        "shape": [4, 8, 3],
        "phase_labels": ["early", "early_middle", "late_middle", "late"],
        "action_labels": action_labels,
        "channels": profiles[0]["tensor_profile"]["axes"]["channel"],
        "mean_tensor": np.round(tensor_values.mean(axis=0), 6).tolist(),
        "scope": "descriptive mean over the cohort; record labels are not stratified",
    }
    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DATA_OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATA_OUTPUT.parent / "profile-tensor-means-v1.svg").write_text(svg_for(result, "zh"), encoding="utf-8")
    (DATA_OUTPUT.parent / "profile-tensor-means-v1-en.svg").write_text(svg_for(result, "en"), encoding="utf-8")
    print(json.dumps({"profiles": len(profiles), "shape": result["shape"],
                      "outputs": [str(DATA_OUTPUT), str(DATA_OUTPUT.parent / "profile-tensor-means-v1.svg"),
                                  str(DATA_OUTPUT.parent / "profile-tensor-means-v1-en.svg")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
