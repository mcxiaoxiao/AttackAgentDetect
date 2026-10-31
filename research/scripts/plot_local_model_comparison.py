#!/usr/bin/env python3
"""Render the grouped local-model comparison from its machine-readable result."""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "local_model_comparison.json"
OUTPUTS = (ROOT / "results" / "figures", ROOT / "paper" / "figures")
INK = "#172331"
MUTED = "#5F7180"
GRID = "#DCE5ED"
BLUE = "#2585D8"
DEEP = "#1769A8"
PALE = "#E8F3FC"


def interval_text(values):
    return f"{values[0]:.2f}–{values[1]:.2f}"


def render(data):
    order = [
        "command_tfidf_logistic",
        "action_tfidf_logistic",
        "count_action_logistic",
        "count_action_extra_trees",
    ]
    labels = ["Command TF–IDF + LR", "Action TF–IDF + LR", "Counts + LR", "Counts + ExtraTrees"]
    columns = ["balanced_accuracy", "macro_f1", "roc_auc"]
    titles = ["Balanced accuracy", "Macro-F1", "ROC-AUC"]
    matrix = np.asarray([[data["models"][key][metric] for metric in columns] for key in order])

    fig = plt.figure(figsize=(7.2, 3.6), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.16, 1), left=.17, right=.98,
                            top=.84, bottom=.23, wspace=.64)
    ax = fig.add_subplot(grid[0, 0])
    img = ax.imshow(matrix, cmap="Blues", vmin=.5, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(titles)), titles, fontsize=7.2, color=INK)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7.0, color=INK)
    ax.set_xticks(np.arange(-.5, len(titles), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0, pad=5)
    for row, key in enumerate(order):
        for col, metric in enumerate(columns):
            value = data["models"][key][metric]
            ci = data["models"][key]["95_task_cluster_ci"][metric]
            color = "white" if value >= .80 else INK
            ax.text(col, row, f"{value:.2f}\n[{ci[0]:.2f}, {ci[1]:.2f}]",
                    ha="center", va="center", color=color, fontsize=6.5, linespacing=1.28)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("A  Task-held-out trace-label prediction", loc="left", pad=10,
                 color=INK, fontsize=8.4, fontweight="semibold")

    bx = fig.add_subplot(grid[0, 1])
    y = np.arange(len(labels))[::-1]
    for offset, (metric, color, marker) in enumerate((
        ("brier", DEEP, "o"),
        ("ece_10_bin", BLUE, "s"),
    )):
        dy = -.11 if offset == 0 else .11
        for row, key in enumerate(order):
            value = data["models"][key][metric]
            low, high = data["models"][key]["95_task_cluster_ci"][metric]
            bx.errorbar(value, y[row] + dy, xerr=[[value - low], [high - value]],
                        fmt=marker, color=color, ecolor=color, capsize=2.2,
                        markersize=3.3, linewidth=.95, zorder=3)
    bx.set_yticks(y, labels, fontsize=6.8, color=INK)
    bx.set_xlim(0, .28)
    bx.set_xticks(np.arange(0, .281, .05), [f"{x:.2f}" for x in np.arange(0, .281, .05)],
                  fontsize=6.4, color=MUTED)
    bx.set_xlabel("● Brier  ·  ■ ECE (10 bins)  ·  lower is better",
                  fontsize=6.6, color=MUTED, labelpad=6)
    bx.set_title("B  Raw probability error", loc="left", pad=10,
                 color=INK, fontsize=8.4, fontweight="semibold")
    bx.grid(axis="x", color=GRID, linewidth=.55)
    bx.tick_params(axis="y", length=0, pad=5)
    bx.tick_params(axis="x", length=0, pad=3)
    for spine in bx.spines.values():
        spine.set_visible(False)
    fig.text(.17, .075,
             "40 task groups · equal-weight benchmark macro · 95% task-cluster bootstrap · raw scores, no calibration",
             fontsize=6.3, color=MUTED)
    fig.text(.17, .035,
             "Human labels use terminal capture; Agent labels use structured tool-call capture. Scores are confounded by source.",
             fontsize=6.3, color=MUTED)

    for folder in OUTPUTS:
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / "figureA7_local_model_comparison.pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(folder / "figureA7_local_model_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    if data["data"]["records"] != 177 or data["data"]["task_groups"] != 40:
        raise SystemExit("Unexpected dataset shape; refusing to draw from stale results")
    render(data)
    print("Rendered Figure A7 from results/local_model_comparison.json")


if __name__ == "__main__":
    main()
