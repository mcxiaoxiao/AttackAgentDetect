#!/usr/bin/env python3
"""Publication figures for the AIT execution-mode recognition section.

Read-only on the frozen ait_recognition/metrics.json (seed 20261103). Produces a
generalization forest plot (task-group vs source-group holdout) and a source-
holdout confusion-matrix companion. Style matches build_figures.py. The figure
makes the paper's point: under a common auditd capture contract, command
vocabulary recognizes the recorded Human/Agent label and persists across held-out
source groups, while the B/S/T behavioral fingerprint falls back toward chance.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[3]
PAPER = ROOT / "research/paper/figures"
SITE = ROOT / "site/assets/behavioral-fingerprint-v2"
METRICS = ROOT / "research/results/behavioral_fingerprint_v2/ait_recognition/metrics.json"
PAPER.mkdir(parents=True, exist_ok=True)
SITE.mkdir(parents=True, exist_ok=True)

NAVY = "#17324f"
BLUE = "#2676bd"
TEAL = "#238b8e"
ORANGE = "#e38b45"
MUTED = "#64758a"
GRID = "#d9e2ec"
DATASET = "extended_all_eligible_min4"  # 109 runs; the cohort used on the site page

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": NAVY, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.linewidth": .7,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight",
})


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(PAPER / f"{name}.{ext}", dpi=220 if ext == "png" else None)
        fig.savefig(SITE / f"{name}.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)


def get(metrics, panel, method):
    return next(r for r in metrics if r["dataset"] == DATASET
                and r["panel"] == panel and r["method"] == method)


def generalization(metrics):
    # top-to-bottom: the robust command baseline first, then fingerprint readouts
    rows = [
        ("command_tfidf", "Command TF–IDF", NAVY),
        ("B_plus_S_plus_T", "B + S + T fingerprint", ORANGE),
        ("B_plus_S", "B + S fingerprint", TEAL),
        ("core8", "Legacy core-8", BLUE),
        ("length", "Log-length baseline", MUTED),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    y = np.arange(len(rows))[::-1]
    for yi, (method, label, color) in zip(y, rows):
        t = get(metrics, "task_holdout", method)
        s = get(metrics, "source_holdout", method)
        tv, tci = 100 * t["task_macro_ba"], 100 * np.array(t["task_cluster_95ci_ba"])
        sv, sci = 100 * s["task_macro_ba"], 100 * np.array(s["task_cluster_95ci_ba"])
        ax.plot([tv, sv], [yi, yi], color=color, lw=1.0, alpha=.30, zorder=1)
        ax.errorbar(tv, yi, xerr=[[tv - tci[0]], [tci[1] - tv]], fmt="o", ms=5.5,
                    capsize=3, lw=1.7, color=color, zorder=3)
        ax.errorbar(sv, yi, xerr=[[sv - sci[0]], [sci[1] - sv]], fmt="s", ms=5.5,
                    mfc="white", mec=color, mew=1.6, capsize=3, lw=1.7,
                    color=color, zorder=3)
    ax.axvline(50, color=MUTED, lw=.9, ls=(0, (4, 3)), alpha=.8, zorder=0)
    ax.text(50, len(rows) - .35, " chance", color=MUTED, fontsize=7, va="bottom")
    # annotate the two source-holdout anchors that carry the message
    cs = get(metrics, "source_holdout", "command_tfidf")
    bs = get(metrics, "source_holdout", "B_plus_S_plus_T")
    ax.annotate(f"{100*cs['task_macro_ba']:.1f}%", (100*cs['task_macro_ba'], y[0]),
                xytext=(6, 7), textcoords="offset points", fontsize=7.5,
                color=NAVY, weight="bold")
    ax.annotate(f"{100*bs['task_macro_ba']:.1f}%", (100*bs['task_macro_ba'], y[1]),
                xytext=(6, -11), textcoords="offset points", fontsize=7.5, color=ORANGE)
    ax.set_yticks(y, [r[1] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Task-macro balanced accuracy (%)")
    ax.set_title("AIT execution-mode recognition · 109 runs · common auditd contract",
                 loc="left", fontsize=9.5, weight="bold", pad=8)
    ax.grid(axis="x", color=GRID, linewidth=.65)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=8)
    ax.tick_params(axis="x", length=0, pad=5)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NAVY, markersize=6,
               label="Task-group holdout"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
               markeredgecolor=NAVY, markeredgewidth=1.5, markersize=6,
               label="Source-group holdout"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.3,
              bbox_to_anchor=(0.0, 1.0), handlelength=1.4, borderaxespad=0)
    fig.text(.01, .015, "Bars: 95% task-cluster intervals. Task holdout reuses actors; source holdout leaves actor groups unseen. Format, actor, task, app excluded.",
             fontsize=6.6, color=MUTED)
    fig.subplots_adjust(left=.27, right=.985, top=.90, bottom=.20)
    save(fig, "ait-mode-recognition-generalization")


def confusion(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    panels = [("command_tfidf", "Command TF–IDF", NAVY),
              ("B_plus_S_plus_T", "B + S + T fingerprint", ORANGE)]
    for ax, (method, title, color) in zip(axes, panels):
        r = get(metrics, "source_holdout", method)
        cm = np.array(r["confusion_human_agent"], dtype=float)
        row = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([0, 1], ["Human", "Agent"])
        ax.set_yticks([0, 1], ["Human", "Agent"])
        ax.set_xlabel("Predicted label")
        ax.set_title(f"{title}\nBA {100*r['task_macro_ba']:.1f}% · AUROC {r['auroc']:.2f}",
                     loc="left", fontsize=8.5, weight="bold", pad=6, color=color)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i,j])}\n({100*row[i,j]:.0f}%)", ha="center",
                        va="center", fontsize=8,
                        color="white" if row[i, j] > .55 else NAVY)
        ax.set_xticks([.5], minor=True); ax.set_yticks([.5], minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Recorded label")
    fig.suptitle("Source-group holdout confusion · 109 AIT runs", fontsize=9.5,
                 weight="bold", x=.02, ha="left", y=1.02)
    fig.subplots_adjust(left=.09, right=.99, top=.82, bottom=.16, wspace=.28)
    save(fig, "ait-mode-recognition-confusion")


def main():
    metrics = json.loads(METRICS.read_text())
    generalization(metrics)
    confusion(metrics)
    prov = {"source_metrics_sha256": sha256(METRICS),
            "script_sha256": sha256(pathlib.Path(__file__)),
            "dataset": DATASET, "seed": 20261103,
            "figures": ["ait-mode-recognition-generalization",
                        "ait-mode-recognition-confusion"]}
    out = ROOT / "research/results/behavioral_fingerprint_v2/ait_recognition/figure_provenance.json"
    out.write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n")
    print("wrote figures to", PAPER, "and", SITE)
    print("provenance:", out)


if __name__ == "__main__":
    main()