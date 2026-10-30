#!/usr/bin/env python3
"""Publication-style, direction-adjusted metric comparison heatmap."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/multidimensional_rerun_v1/results.json"
OUT = ROOT / "results/figures/multidimensional-rerun-v1"

METRICS = (
    ("false_merge_rate", "FMR"),
    ("false_split_rate", "FSR"),
    ("false_merge_human_agent", "H–A FMR"),
    ("false_split_human_agent", "H–A FSR"),
    ("false_merge_human_human", "H–H FMR"),
    ("false_split_human_human", "H–H FSR"),
    ("false_merge_agent_agent", "A–A FMR"),
    ("false_split_agent_agent", "A–A FSR"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("query_has_accepted_match", "Query coverage"),
    ("hit_at_1", "Hit@1"),
    ("mrr", "MRR"),
    ("recall_at_5", "Recall@5"),
    ("ndcg_at_5", "NDCG@5"),
)
LOWER_IS_BETTER = {metric for metric, _ in METRICS[:8]}
METHOD_LABELS = {
    "profile_24d": "24D profile",
    "tensor_4x8x3": "4×8×3 tensor",
    "tensor_tucker": "Tucker profile",
    "partial_sequence": "IDF partial sequence",
    "learned_fusion": "Learned fusion",
}
METHOD_LABELS_ZH = {
    "profile_24d": "24 维画像",
    "tensor_4x8x3": "4×8×3 张量",
    "tensor_tucker": "Tucker 张量画像",
    "partial_sequence": "IDF 局部序列",
    "learned_fusion": "训练集融合",
}


def _draw(values, supported, selected, *, chinese=False):
    methods = list(METHOD_LABELS)
    method_labels = METHOD_LABELS_ZH if chinese else METHOD_LABELS
    metric_labels = [
        "误合并率" if chinese else "FMR",
        "漏关联率" if chinese else "FSR",
        "H–A 误合并" if chinese else "H–A FMR",
        "H–A 漏关联" if chinese else "H–A FSR",
        "H–H 误合并" if chinese else "H–H FMR",
        "H–H 漏关联" if chinese else "H–H FSR",
        "A–A 误合并" if chinese else "A–A FMR",
        "A–A 漏关联" if chinese else "A–A FSR",
        "平衡准确率" if chinese else "Balanced accuracy",
        "查询覆盖率" if chinese else "Query coverage",
        "Hit@1", "MRR", "Recall@5", "NDCG@5",
    ] if chinese else [label for _metric, label in METRICS]
    # Supplied arrays are already in percentage-point units and direction-adjusted.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Heiti SC", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(11.2, 4.6), constrained_layout=True)
    finite = np.abs(values[np.isfinite(values)])
    limit = max(10.0, float(np.ceil(np.max(finite) / 5) * 5)) if finite.size else 10.0
    image = ax.imshow(values, cmap="PuOr", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(methods)), [method_labels[name] for name in methods])
    for tick, method in zip(ax.get_yticklabels(), methods):
        if method == selected:
            tick.set_fontweight("bold")
            tick.set_color("#1758a8")
    ax.set_xticks(range(len(METRICS)), metric_labels, rotation=38, ha="right")
    ax.set_title("外部留出基准：相对命令 TF-IDF 的变化" if chinese else
                 "External task-held-out effects relative to command TF–IDF",
                 loc="left", fontweight="semibold")
    ax.set_ylabel("开发集选定方法" if chinese else "Methods chosen on development tasks")
    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(len(methods)):
        for col in range(len(METRICS)):
            if not np.isfinite(values[row, col]):
                label = "—"
                color = "#39424e"
            else:
                label = f"{values[row, col]:+.1f}" + ("*" if supported[row, col] else "")
                color = "#17212d" if abs(values[row, col]) < limit * 0.58 else "white"
            ax.text(col, row, label, ha="center", va="center", fontsize=7, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_ticks(np.linspace(-limit, limit, 5))
    colorbar.set_label("改善（百分点，正值更好）" if chinese else "Improvement (percentage points; positive is better)")
    fig.text(0.01, -0.035,
             "错误率按反方向显示。* 配对 95% 任务簇区间支持改善；区间在符合条件的外部基准内按任务重采样。" if chinese else
             "Error rates are direction-reversed. * Paired 95% task-cluster interval supports improvement. "
             "Intervals resample tasks within eligible external benchmarks.",
             ha="left", va="top", fontsize=7, color="#4c5968")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if chinese:
        fig.savefig(OUT.with_name(OUT.stem + "-zh.png"), dpi=320, bbox_inches="tight", facecolor="white")
    else:
        for extension in ("pdf", "png"):
            fig.savefig(OUT.with_suffix(f".{extension}"), dpi=320 if extension == "png" else None,
                        bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    deltas = result["metrics"]["paired_deltas_vs_command_baseline"]
    selected = result["selection"]["method"]
    methods = list(METHOD_LABELS)
    values = np.full((len(methods), len(METRICS)), np.nan, dtype=float)
    supported = np.zeros_like(values, dtype=bool)
    for row, method in enumerate(methods):
        for col, (metric, _label) in enumerate(METRICS):
            item = deltas.get(method, {}).get(metric, {})
            estimate, interval = item.get("estimate"), item.get("ci_95")
            if estimate is None:
                continue
            if metric in LOWER_IS_BETTER:
                estimate = -estimate
                interval = [-interval[1], -interval[0]] if interval else None
            values[row, col] = estimate * 100
            supported[row, col] = bool(interval and interval[0] > 0)
    _draw(values, supported, selected)
    _draw(values, supported, selected, chinese=True)
    print(f"Figures written to {OUT}.pdf, {OUT}.png and {OUT.stem}-zh.png")


if __name__ == "__main__":
    main()
