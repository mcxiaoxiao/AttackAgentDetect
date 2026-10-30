#!/usr/bin/env python3
"""Create a compact, honest comparison figure for the nested experiments."""
from __future__ import annotations

import csv
import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_association_operating_points as op

OUT = ROOT / "results"
R1 = json.loads((ROOT / "results/recall_optimization_v1/results.json").read_text())
R2 = json.loads((ROOT / "results/recall_optimization_v2/results.json").read_text())
R3 = json.loads((ROOT / "results/recall_joint_pareto_v3/results.json").read_text())

COLORS = {
    "baseline": "#34495e", "alignment": "#0072B2", "multiscale": "#009E73",
    "profile": "#CC79A7", "taskweighted": "#E69F00", "ha_specialist": "#D55E00",
    "stratified": "#56B4E9", "v2": "#6A5ACD", "v3": "#222222",
}


def rate_points():
    p = []
    r20 = R2["false_merge_targets"]["20"]
    for name, key, label, color, offset in [
        ("baseline", "behavior", "baseline", COLORS["baseline"], (-52, 40)),
        ("alignment", "behavior_align", "sequence alignment", COLORS["alignment"], (22, -20)),
        ("multiscale", "behavior_multiscale", "multi-scale", COLORS["multiscale"], (-82, -40)),
        ("profile", "behavior_align_profile", "24D profile", COLORS["profile"], (-70, -12)),
        ("taskweighted", "behavior_align_taskweighted", "task-weighted", COLORS["taskweighted"], (-105, -15)),
        ("ha_specialist", "behavior_align_ha", "H–A specialist", COLORS["ha_specialist"], (24, 24)),
        ("stratified", "behavior_stratified", "mode thresholds", COLORS["stratified"], (-98, -54)),
        ("v2", "joint_selected", "v2 selector", COLORS["v2"], (30, -44)),
    ]:
        source = r20[key]
        fmr, fsr = source["false_merge_rate"], source["false_split_rate"]
        p.append((label, color, fmr, fsr, offset, name == "baseline"))
    r3 = R3["false_split_merge"]["joint_selected"]
    p.append(("v3 Pareto selector", COLORS["v3"], r3["false_merge_rate"],
              r3["false_split_rate"], (32, 48), False))
    return p


def pair_delta(rows, metric, method, baseline="behavior"):
    return op.paired_delta(rows, metric, method, baseline, 5000, 20263010 + len(metric) + len(method))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ranking_delta(path, method):
    rows = read_csv(path)
    return {metric: pair_delta(rows, metric, method)
            for metric in ("hit_at_1", "mrr", "recall_at_5", "ndcg_at_5")}


def rate_delta(source, target="20"):
    return source


def build_effects():
    r1d = R1["paired_deltas_vs_behavior"]["20"]["behavior_align"]
    r2d = R2["paired_deltas_vs_behavior"]["20"]
    v2_coverage = R2["coverage_paired_deltas_vs_behavior"]["20"]
    v2_coverage_rows = read_csv(ROOT / "results/recall_optimization_v2/coverage_task_metrics.csv")
    v1_ranking = ranking_delta(ROOT / "results/recall_optimization_v1/task_metrics.csv", "behavior_align")
    v2_weighted_rank = ranking_delta(ROOT / "results/recall_optimization_v2/task_metrics.csv", "behavior_align_taskweighted")
    v2_ha_rank = ranking_delta(ROOT / "results/recall_optimization_v2/task_metrics.csv", "behavior_align_ha")
    v3 = R3

    def coverage_pair(method_name):
        if method_name == "behavior_stratified":
            left = [r for r in v2_coverage_rows if r["target"] == "20" and r["policy"] == "stratified"
                    and r["method"] == method_name]
            right = [r for r in v2_coverage_rows if r["target"] == "20" and r["policy"] == "global"
                     and r["method"] == "behavior_global"]
            for row in left:
                row["method"] = method_name
            for row in right:
                row["method"] = "behavior"
            return op.paired_delta([*left, *right], "coverage", method_name, "behavior", 5000, 20263022)
        name = method_name
        return v2_coverage[name]

    v3_rank = v3["paired_ranking_deltas"]
    effects = {}
    for label, rates, coverage, ranking in [
        ("Sequence alignment", r1d, coverage_pair("behavior_align_global"), v1_ranking),
        ("Mode thresholds", r2d["behavior_stratified"], coverage_pair("behavior_stratified"),
         {m: {"estimate": 0.0, "ci_95": [0.0, 0.0]} for m in op.RANK_METRICS}),
        ("H–A specialist", r2d["behavior_align_ha"], coverage_pair("behavior_align_ha_global"), v2_ha_rank),
        ("Pareto-grid selector", v3["paired_rate_deltas"], v3["coverage"]["paired_delta"], v3_rank),
    ]:
        vals = {}
        for metric in ("false_split_rate", "false_merge_rate", "false_split_human_agent",
                       "false_merge_human_agent", "balanced_accuracy"):
            vals[metric] = rates[metric]["estimate"] * 100
        vals["coverage"] = coverage["estimate"] * 100
        for metric in op.RANK_METRICS:
            vals[metric] = ranking[metric]["estimate"] * 100
        effects[label] = vals
    return effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("zh", "en"), default="en")
    args = parser.parse_args()
    zh = args.language == "zh"
    labels = {
        "baseline": "行为基线",
        "sequence alignment": "序列对齐",
        "multi-scale": "多尺度",
        "24D profile": "24维画像",
        "task-weighted": "任务加权",
        "H–A specialist": "H–A 专项",
        "mode thresholds": "分模式阈值",
        "v2 selector": "第二轮选择",
        "v3 Pareto selector": "第三轮 Pareto",
        "Sequence alignment": "序列对齐",
        "Mode thresholds": "分模式阈值",
        "H–A specialist": "H–A 专用排序",
        "Pareto-grid selector": "Pareto 网格选择",
    }
    metric_labels = (("漏关联", "误合并", "H–A 漏关联", "H–A 误合并", "平衡准确率",
                      "查询覆盖率", "首位命中", "MRR", "Recall@5", "NDCG@5") if zh else
                     ("Missed links", "False merges", "H–A missed links", "H–A false merges",
                      "Balanced accuracy", "Query coverage", "Hit@1", "MRR", "Recall@5", "NDCG@5"))
    family = "Arial Unicode MS" if zh else "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.size": 9,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig = plt.figure(figsize=(14, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(0.93, 1.35))
    ax = fig.add_subplot(grid[0, 0])
    for label, color, fmr, fsr, offset, baseline in rate_points():
        ax.errorbar(fmr["estimate"] * 100, fsr["estimate"] * 100,
                    xerr=[[max(0, (fmr["estimate"] - fmr["ci_95"][0]) * 100)],
                          [max(0, (fmr["ci_95"][1] - fmr["estimate"]) * 100)]],
                    yerr=[[max(0, (fsr["estimate"] - fsr["ci_95"][0]) * 100)],
                          [max(0, (fsr["ci_95"][1] - fsr["estimate"]) * 100)]],
                    fmt="o" if not baseline else "*", ms=9 if baseline else 6,
                    capsize=2.5, elinewidth=1, color=color,
                    zorder=4 if baseline else 3)
        display_label = labels.get(label, label) if zh else label
        ax.annotate(display_label, (fmr["estimate"] * 100, fsr["estimate"] * 100),
                    xytext=offset, textcoords="offset points", fontsize=7.5, color=color,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.55,
                                "shrinkA": 0, "shrinkB": 4})
    ax.set(xlabel=("误合并率（%）· 越低越好" if zh else "False merges (%) · lower is better"),
           ylabel=("漏关联率（%）· 越低越好" if zh else "Missed links (%) · lower is better"),
           title=("训练折误合并目标：20%" if zh else "20% training false-merge target"))
    ax.set_xlim(0, 39); ax.set_ylim(25, 98); ax.grid(alpha=.2)
    ax.text(.03, .03, ("左下更好。误差线：任务簇 bootstrap 95% 区间。" if zh else
                        "Lower-left is better. Bars: task-bootstrap 95% CIs."),
            transform=ax.transAxes, fontsize=7.2, color="#475569")

    effects = build_effects()
    metric_order = ("false_split_rate", "false_merge_rate", "false_split_human_agent",
                    "false_merge_human_agent", "balanced_accuracy", "coverage", "hit_at_1",
                    "mrr", "recall_at_5", "ndcg_at_5")
    method_keys = list(effects)
    method_labels = [labels[name] for name in method_keys] if zh else method_keys
    values = np.asarray([[effects[method][metric] for metric in metric_order]
                         for method in method_keys])
    oriented = values.copy()
    oriented[:, :4] *= -1
    heat = fig.add_subplot(grid[0, 1])
    vmax = max(4, float(np.nanpercentile(np.abs(oriented), 95)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    image = heat.imshow(oriented, cmap="RdYlGn", norm=norm, aspect="auto")
    heat.set_xticks(np.arange(len(metric_labels)), metric_labels, rotation=40, ha="right")
    heat.set_yticks(np.arange(len(method_labels)), method_labels)
    heat.set_title("相对行为基线的变化（百分点）" if zh else
                   "Change from behavior baseline (percentage points)")
    heat.tick_params(axis="both", length=0)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            color = "white" if abs(oriented[i, j]) > vmax * .63 else "#1f2937"
            heat.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8, color=color)
    cb = fig.colorbar(image, ax=heat, fraction=.035, pad=.02)
    cb.set_label(("绿色 = 改善，红色 = 变差（已按指标方向校正）" if zh else
                  "Green = improvement, red = deterioration (direction adjusted)"), fontsize=8)
    fig.suptitle("看清收益与代价" if zh else "What improved, and what got worse",
                 fontsize=15, fontweight="bold")
    png = OUT / f"recall_optimization_comparison_{args.language}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    if not zh:
        fig.savefig(OUT / "recall_optimization_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(png.relative_to(ROOT))


if __name__ == "__main__":
    main()
