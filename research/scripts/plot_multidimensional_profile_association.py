#!/usr/bin/env python3
"""Create publication-style tensor and error-tradeoff figures from the frozen run."""
from __future__ import annotations

import json
import pathlib
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/multidimensional_profile_association_v1/results.json"
OUT = ROOT / "results/multidimensional_profile_association_v1"
FIGURES = ROOT / "paper/figures"
SITE_ASSETS = ROOT.parent / "site/assets"

METHOD_ORDER = (
    "command_cosine", "behavior", "structure_aware", "mode_aware", "length_only",
    "profile_only", "tensor_flat_only", "tensor_tucker_only",
    "behavior_profile", "behavior_tensor_flat", "behavior_tensor_tucker",
    "behavior_mode_tucker",
)
METHOD_LABELS = {
    "command_cosine": "Command cosine", "behavior": "Behavior ranker",
    "structure_aware": "Behavior + structure", "mode_aware": "Behavior + mode",
    "length_only": "Length control", "profile_only": "24D profile",
    "tensor_flat_only": "Flat tensor", "tensor_tucker_only": "Tucker core",
    "behavior_profile": "Behavior + 24D profile",
    "behavior_tensor_flat": "Behavior + flat tensor",
    "behavior_tensor_tucker": "Behavior + Tucker core",
    "behavior_mode_tucker": "Behavior + mode + Tucker",
}
COLORS = {
    "command_cosine": "#0072B2", "behavior": "#E69F00",
    "structure_aware": "#56B4E9", "mode_aware": "#D55E00",
    "length_only": "#7D8790", "profile_only": "#009E73",
    "tensor_flat_only": "#CC79A7", "tensor_tucker_only": "#008B7A",
    "behavior_profile": "#65A67A", "behavior_tensor_flat": "#B2678A",
    "behavior_tensor_tucker": "#004D40", "behavior_mode_tucker": "#A54416",
}


def fmt_interval(record):
    return float(record["estimate"]), np.asarray(record["ci_95"], dtype=float)


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def tensor_figure(result):
    labels = result["evaluation"]["profile"]["tensor_axes"]["action_family"]
    descriptive = result["descriptive_tensor_means_by_record_label"]
    counts = [descriptive[label]["trace_count"] for label in ("human", "agent")]
    h_mean = np.asarray(descriptive["human"]["mean_tensor"], dtype=float)
    a_mean = np.asarray(descriptive["agent"]["mean_tensor"], dtype=float)
    mean = (h_mean * counts[0] + a_mean * counts[1]) / sum(counts)
    titles = ["Command share", "Vocabulary diversity", "Adjacent repeat rate"]
    titles_zh = ["命令占比", "词汇多样度", "相邻重复率"]
    phase_names = ["Early", "Early-mid", "Late-mid", "Late"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.0), constrained_layout=True)
    tensor_cmap = plt.get_cmap("cividis")
    image = None
    for channel, ax in enumerate(axes):
        image = ax.imshow(mean[:, :, channel].T, cmap="cividis", vmin=0, vmax=1,
                          aspect="auto", interpolation="nearest")
        ax.set_title(f"{chr(65 + channel)}  {titles[channel]}", loc="left", pad=10, fontweight="bold")
        ax.set_xticks(range(4), phase_names, rotation=22, ha="right")
        ax.set_yticks(range(8), labels)
        ax.set_xlabel("Ordered phase")
        if channel == 0:
            ax.set_ylabel("Action family")
        else:
            ax.set_ylabel("")
        for row in range(8):
            for col in range(4):
                value = mean[col, row, channel]
                rgb = tensor_cmap(value)[:3]
                luminance = .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2]
                color = "white" if luminance < .47 else "#172331"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6.2, color=color)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.3)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(image, ax=axes, shrink=.77, pad=.018, label="Mean channel value")
    fig.suptitle("Cohort-mean phase × action × channel tensor (n = 177)", x=.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(.02, -.018, "Descriptive pooled mean; Human/Agent records are not separated. Capture pathway is confounded with record label.", fontsize=7, color="#596A7B")
    path = OUT / "multidimensional_profile_tensor_v1"
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path.with_suffix(".pdf"), FIGURES / "multidimensional_profile_tensor_v1.pdf")
    shutil.copy2(path.with_suffix(".png"), SITE_ASSETS / "multidimensional_profile_tensor_v1.png")


def tradeoff_figure(result):
    summary = result["false_merge_targets"]["20"]["summary"]
    methods = [name for name in METHOD_ORDER if name in summary]
    fig = plt.figure(figsize=(12.0, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.2, .85))
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1]); legend_ax.axis("off")
    x_limits, y_limits = [], []
    for index, method in enumerate(methods):
        fmr, fmr_ci = fmt_interval(summary[method]["false_merge_rate"])
        fsr, fsr_ci = fmt_interval(summary[method]["false_split_rate"])
        color = COLORS[method]
        marker = "D" if method in {"behavior_profile", "behavior_tensor_flat",
                                   "behavior_tensor_tucker", "behavior_mode_tucker"} else "o"
        ax.errorbar(fmr, fsr, xerr=[[fmr - fmr_ci[0]], [fmr_ci[1] - fmr]],
                    yerr=[[fsr - fsr_ci[0]], [fsr_ci[1] - fsr]], fmt=marker,
                    markersize=5.5, capsize=2.2, linewidth=1.15, color=color,
                    ecolor=color, alpha=.95, zorder=3)
        x_limits.extend(fmr_ci); y_limits.extend(fsr_ci)
    ax.set_xlim(0, max(.35, min(.75, max(x_limits) + .07)))
    ax.set_ylim(0, min(1, max(y_limits) + .08))
    ax.set_xlabel("False-merge rate (lower is safer)")
    ax.set_ylabel("False-split rate (lower preserves matches)")
    ax.set_title("A  Fixed 20% training false-merge target", loc="left", fontweight="bold")
    ax.grid(axis="both", color="#DCE3E9", linewidth=.65)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#AEB8C2")
    legend_ax.text(0, 1.0, "Methods", transform=legend_ax.transAxes, va="top", fontweight="bold", fontsize=10)
    for index, method in enumerate(methods):
        yy = .92 - index * .075
        legend_ax.scatter([.04], [yy], s=25, color=COLORS[method], transform=legend_ax.transAxes, clip_on=False)
        legend_ax.text(.10, yy, METHOD_LABELS[method], transform=legend_ax.transAxes,
                       va="center", fontsize=8)
    legend_ax.text(0, .02, "Points are task-macro estimates; bars are paired 95% task-cluster bootstrap intervals.",
                   transform=legend_ax.transAxes, fontsize=7, color="#596A7B", wrap=True)
    fig.suptitle("Multidimensional profiles: association error trade-off", x=.02, ha="left", fontsize=12, fontweight="bold")
    path = OUT / "multidimensional_profile_tradeoff_v1"
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path.with_suffix(".pdf"), FIGURES / "multidimensional_profile_tradeoff_v1.pdf")
    shutil.copy2(path.with_suffix(".png"), SITE_ASSETS / "multidimensional_profile_tradeoff_v1.png")


def effects_figure(result):
    comparisons = result["paired_deltas_at_20_percent_target_vs_behavior"]
    methods = [name for name in METHOD_ORDER if name in comparisons.get("false_merge_human_agent", {})]
    panels = [
        ("false_merge_human_agent", "Different-task H–A false merges", "Lower is better"),
        ("false_split_human_agent", "Same-task H–A false splits", "Lower is better"),
        ("hit_at_1", "Hit@1", "Higher is better"),
        ("mrr", "MRR", "Higher is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    axes = axes.flat
    for panel_index, (ax, (metric, title, direction)) in enumerate(zip(axes, panels)):
        estimates, low, high = [], [], []
        for method in methods:
            record = comparisons[metric][method]
            estimate, interval = fmt_interval(record)
            estimates.append(estimate); low.append(interval[0]); high.append(interval[1])
        y = np.arange(len(methods))
        for index, method in enumerate(methods):
            ax.errorbar(estimates[index], y[index], xerr=[[estimates[index] - low[index]],
                        [high[index] - estimates[index]]], fmt="o", markersize=4.5,
                        capsize=2, color=COLORS[method], linewidth=1.1)
        ax.axvline(0, color="#4F5F6F", linewidth=1, linestyle="--")
        ax.set_yticks(y, [METHOD_LABELS[name] for name in methods])
        ax.invert_yaxis()
        ax.set_title(f"{chr(65 + panel_index)}  {title}\n{direction} relative to behavior ranker",
                     loc="left", fontsize=9, fontweight="bold")
        ax.set_xlabel("Paired change (method − behavior)")
        ax.grid(axis="x", color="#DCE3E9", linewidth=.65)
        ax.tick_params(axis="y", labelsize=6.7)
    fig.suptitle("Cross-mode association and retrieval changes at the 20% target", x=.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(.02, -.015, "Negative values reduce errors in the first two panels; positive values improve ranking in the last two. Intervals do not encode multiplicity-adjusted tests.", fontsize=7, color="#596A7B")
    path = OUT / "multidimensional_profile_effects_v1"
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path.with_suffix(".pdf"), FIGURES / "multidimensional_profile_effects_v1.pdf")
    shutil.copy2(path.with_suffix(".png"), SITE_ASSETS / "multidimensional_profile_effects_v1.png")


def main():
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    tensor_figure(result)
    tradeoff_figure(result)
    effects_figure(result)


if __name__ == "__main__":
    main()
