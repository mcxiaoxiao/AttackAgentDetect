#!/usr/bin/env python3
"""Build the sole new main-text figure from frozen matched predictions."""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/experiments/behavioral_fingerprint_v2"))
import paper_style

R = json.loads((ROOT / "research/results/matched_fingerprint_v1/metrics.json").read_text())
DATA = R["cth"]["methods"]
PAPER = ROOT / "research/paper/figures"
SITE = ROOT / "site/assets/behavioral-fingerprint-v2"
paper_style.apply()
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.titlesize": 8.5})


def lookup(learner, variant, fraction):
    return next(r for r in DATA if r["learner"] == learner and r["variant"] == variant
                and r["fraction"] == str(fraction) and r["pair_type"] == "ALL")


fig, axes = plt.subplots(3, 3, figsize=(paper_style.PAGE_WIDTH, 4.65), sharex=True,
                         sharey="row", gridspec_kw={"wspace": .14, "hspace": .18})
fractions = [0.0, .25, .5, .75, 1.0]
x = np.asarray(fractions) * 100
for col, (learner, title) in enumerate((("svm", "SVM"), ("tree", "Decision tree"),
                                         ("xgb", "XGBoost"))):
    for row, (metric, ylabel) in enumerate((("ba", "Balanced accuracy (%)"),
                                             ("fmr", "False merges (%)"),
                                             ("fsr", "Missed links (%)"))):
        ax = axes[row, col]
        for variant, color, marker, linestyle in (
                ("evidence", paper_style.BLUE, "o", (0, (3, 2))),
                ("BS", paper_style.TEAL, "s", "-")):
            vals = [lookup(learner, variant, p)["metrics"][metric] for p in fractions]
            y = np.asarray([v["point"] for v in vals]) * 100
            lo = np.asarray([v["ci95"][0] for v in vals]) * 100
            hi = np.asarray([v["ci95"][1] for v in vals]) * 100
            ax.fill_between(x, lo, hi, color=color, alpha=.12, linewidth=0)
            ax.plot(x, y, color=color, marker=marker, markersize=3,
                    linewidth=1.2, linestyle=linestyle, markerfacecolor="white"
                    if variant == "evidence" else color)
        ax.grid(axis="y", linewidth=.45, alpha=.75)
        # Common limits across each metric's three learners; BA and FSR axes
        # are explicitly truncated so small changes remain legible.
        limits = {"ba": (44, 76), "fmr": (-1, 31), "fsr": (44, 101)}
        ticks = {"ba": [45, 55, 65, 75], "fmr": [0, 10, 20, 30],
                 "fsr": [50, 75, 100]}
        ax.set_ylim(*limits[metric])
        ax.set_yticks(ticks[metric])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", length=2, pad=2)
        if col == 0:
            ax.set_ylabel(ylabel)
        if row == 0:
            ax.set_title(title, weight="bold", pad=5)
        if row == 2:
            ax.set_xlabel("Agent role share (%)")
handles = [Line2D([0], [0], color=paper_style.BLUE, lw=1.2, marker="o",
                  markerfacecolor="white", markersize=3.5, linestyle=(0, (3, 2)),
                  label="Original evidence"),
           Line2D([0], [0], color=paper_style.TEAL, lw=1.2, marker="s",
                  markersize=3.5, label="Evidence + B/S")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.005),
           ncol=2, frameon=False, fontsize=7, handlelength=2)
fig.subplots_adjust(left=.095, right=.995, top=.91, bottom=.10)
paper_style.save(fig, "matched-mixture-v1", (PAPER, SITE))
print("Created matched-mixture-v1 PDF, PNG, and SVG")
