#!/usr/bin/env python3
"""Render the AIT feature-control diagnostic from the frozen result JSON."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
INPUT = ROOT / "research/results/behavioral_fingerprint_v2/ait_ablation/metrics.json"
PAPER = ROOT / "research/paper/figures"
SITE = ROOT / "site/assets/behavioral-fingerprint-v2"

NAVY = "#17324f"
BLUE = "#2676bd"
TEAL = "#238b8e"
ORANGE = "#d57936"
MUTED = "#64758a"
GRID = "#d9e2ec"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.color": NAVY,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "axes.linewidth": .7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})


def main() -> None:
    payload = json.loads(INPUT.read_text())
    rows = [r for r in payload["metrics"] if r["dataset"] == "extended_all_eligible_min4"]
    methods = [
        ("B_plus_S", "Full B+S", BLUE),
        ("B_plus_S_window_order_shuffled", "Window order shuffled", MUTED),
        ("first_window_BS", "First 8 events only", ORANGE),
        ("B_plus_S_plus_T", "B+S+T", TEAL),
        ("B_plus_S_plus_T_gap_order_shuffled", "Gap order shuffled", "#7c69a7"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharex=True, sharey=True,
                             gridspec_kw={"wspace": .14})
    panels = [
        ("task_holdout", "Task-group holdout"),
        ("source_holdout", "Source-group holdout"),
    ]
    y = np.arange(len(methods))[::-1]
    for ax, (panel, title) in zip(axes, panels):
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        panel_rows = {r["method"]: r for r in rows if r["panel"] == panel}
        for yi, (method, label, color) in zip(y, methods):
            item = panel_rows[method]
            value = item["task_macro_ba"] * 100
            low, high = [v * 100 for v in item["task_cluster_95ci_ba"]]
            ax.errorbar(value, yi, xerr=[[value - low], [high - value]],
                        fmt="o", color=color, ecolor=color, markersize=6,
                        elinewidth=1.7, capsize=3, zorder=3)
            ax.text(min(high + 1.4, 96), yi + .13, f"{value:.1f}%",
                    fontsize=7.5, color=color, va="center")
        ax.axvline(50, color=ORANGE, linewidth=1, linestyle=(0, (3, 3)), alpha=.85)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Task-macro balanced accuracy")
        ax.grid(axis="x", color=GRID, linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [label for _, label, _ in methods])
    axes[1].tick_params(labelleft=False)
    fig.suptitle("AIT feature controls · expanded 109-run cohort", x=.03, y=1.02,
                 ha="left", fontsize=10, fontweight="bold", color=NAVY)
    fig.subplots_adjust(left=.25, right=.98, top=.82, bottom=.24)
    PAPER.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(PAPER / f"fingerprint-v2-ait-ablation.{ext}", dpi=260 if ext == "png" else None)
        fig.savefig(SITE / f"fingerprint-v2-ait-ablation.{ext}", dpi=260 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
