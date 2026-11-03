#!/usr/bin/env python3
"""Render complementary evidence views from the unchanged experimental cohorts.

No models, predictions, eligibility gates, or evaluation draws are changed here.
The timing illustration reuses the two trace IDs in build_figures.matrix_example;
its distribution panel includes all 109 eligible AIT runs, one value per run.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "research/results/behavioral_fingerprint_v2"
OUT = ROOT / "site/assets/behavioral-fingerprint-v2"
PAPER = ROOT / "research/paper/figures"
NAVY, BLUE, TEAL = "#17324f", "#2676bd", "#238b8e"
MUTED, GRID, PALE = "#64758a", "#d9e2ec", "#f1f6fa"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": NAVY, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.linewidth": .7,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none",
})


def read(path):
    return json.loads(path.read_text())


def save(fig, name):
    for dest in (PAPER, OUT):
        dest.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(dest / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def association():
    results = read(ROOT / "research/results/association_fingerprint_v3/ba_tree_nested_multisource.json")["results"]
    panels = [("cth", "CTH", "40 held-out tasks"),
              ("kypo", "KYPO", "97 held-out participants"),
              ("antieval", "Antieval", "31 held-out tasks")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    for ax, (key, title, unit) in zip(axes, panels):
        r = results[key]
        values = np.array([r["baseline_ba"], r["ba"]]) * 100
        ax.plot([0, 1], values, color=TEAL, linewidth=2.1, zorder=2)
        ax.scatter(0, values[0], s=42, facecolor="white", edgecolor=BLUE, linewidth=1.6, zorder=3)
        ax.scatter(1, values[1], s=42, marker="D", facecolor=TEAL, edgecolor="white", linewidth=.6, zorder=3)
        for x, v in enumerate(values):
            ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9, color=BLUE if x == 0 else TEAL, weight="bold")
        ax.set_title(title, loc="left", fontsize=10, weight="bold", pad=23)
        ax.text(0, 1.045, unit, transform=ax.transAxes, color=MUTED, fontsize=7.8)
        ax.set(xlim=(-.35, 1.35), ylim=(45, 82), xticks=[0, 1],
               xticklabels=["Evidence", "+ B/S"], yticks=[50, 60, 70, 80])
        ax.grid(axis="y", color=GRID, linewidth=.6)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(length=0, pad=5)
        delta = r["delta_ba"] * 100
        lo, hi = np.asarray(r["paired_cluster_ci95"]) * 100
        ax.text(.5, -.27, f"Δ {delta:+.2f} pp", transform=ax.transAxes,
                ha="center", fontsize=9, color=TEAL, weight="bold")
        ax.text(.5, -.39, f"95% CI [{lo:+.2f}, {hi:+.2f}]", transform=ax.transAxes,
                ha="center", fontsize=7.8, color=MUTED)
    axes[0].set_ylabel("Task-macro balanced accuracy (%)")
    fig.subplots_adjust(left=.09, right=.99, top=.80, bottom=.31, wspace=.28)
    save(fig, "fingerprint-association-comparison")


def recognition():
    data = [x for x in read(RESULTS / "ait_recognition/metrics.json")
            if x["dataset"] == "extended_all_eligible_min4"]
    panels = [("task_holdout", "Task-group holdout"), ("source_holdout", "Source-group holdout")]
    methods = [("length", "Length"), ("core8", "Legacy K1–K8"), ("B", "B"),
               ("B_plus_S", "B + S"), ("B_plus_S_plus_T", "B + S + T (linear)"),
               ("extra_trees_B_plus_S_plus_T", "B + S + T (trees)"),
               ("command_tfidf", "Command TF–IDF")]
    lookup = {(r["panel"], r["method"]): r for r in data}
    a = np.array([[lookup[(p, m)]["task_macro_ba"] * 100 for p, _ in panels] for m, _ in methods])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    image = ax.imshow(a, vmin=25, vmax=100, cmap="Blues", aspect="auto", interpolation="nearest")
    for i, (method, _) in enumerate(methods):
        for j, (panel, _) in enumerate(panels):
            r = lookup[(panel, method)]
            lo, hi = np.asarray(r["task_cluster_95ci_ba"]) * 100
            color = "white" if a[i, j] >= 70 else NAVY
            ax.text(j, i-.12, f"{a[i,j]:.1f}%", ha="center", va="center", fontsize=10,
                    color=color, weight="bold")
            ax.text(j, i+.20, f"[{lo:.1f}, {hi:.1f}]", ha="center", va="center",
                    fontsize=7.7, color=color)
    ax.set_yticks(range(len(methods)), [label for _, label in methods])
    ax.set_xticks([0, 1], [label for _, label in panels])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=9)
    ax.set_yticks(np.arange(-.5, len(methods), 1), minor=True)
    ax.set_xticks([.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.add_patch(Rectangle((-.5, 3.5), 2, 1, fill=False, edgecolor=TEAL, linewidth=1.5))
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(image, cax=fig.add_axes([.93, .25, .017, .50]), ticks=[25, 50, 75, 100])
    cb.set_label("BA (%)", fontsize=8)
    cb.outline.set_visible(False)
    deltas = read(RESULTS / "ait_recognition/paired_deltas.json")
    gains = []
    for panel, _ in panels:
        r = next(x for x in deltas if x["dataset"] == "extended_all_eligible_min4"
                 and x["panel"] == panel and x["baseline"] == "B_plus_S"
                 and x["comparison"] == "B_plus_S_plus_T")
        lo, hi = np.asarray(r["task_cluster_95ci"]) * 100
        gains.append(f"+T: {r['task_macro_delta_ba']*100:+.2f} pp\n95% CI [{lo:+.2f}, {hi:+.2f}]")
    for j, label in enumerate(gains):
        ax.text(j, 7.17, label, ha="center", va="top", fontsize=8, color=TEAL, linespacing=1.6)
    fig.subplots_adjust(left=.28, right=.90, top=.90, bottom=.19)
    save(fig, "fingerprint-recognition-matrix")


def timing():
    path = ROOT / "research/experiments/log_fingerprint_pilot_v1/run.py"
    spec = importlib.util.spec_from_file_location("timing_story_pilot", path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    rows = []
    for run in pilot.load_runs():
        events = [e for e in run["events"] if e.get("login")][:128]
        if len(events) < 4:
            continue
        timestamps = np.asarray([e["time"] for e in events], dtype=float)
        gaps = np.diff(timestamps)
        gaps = gaps[np.isfinite(gaps) & (gaps >= 0)]
        assert len(gaps) > 0
        rows.append({"id": run["id"], "task": run["task"], "label": run["y"],
                     "events": len(events), "time": timestamps - timestamps[0],
                     "median_gap_seconds": float(np.median(gaps))})
    assert len(rows) == 109
    assert [sum(r["label"] == y for r in rows) for y in (0, 1)] == [61, 48]
    examples = [next(r for r in rows if r["id"] == rid)
                for rid in ("78d4c642dcc3b79a", "bd959913c2c5293d")]
    assert examples[0]["task"] == examples[1]["task"] == "Nextcloud/exp1"
    fig = plt.figure(figsize=(7.2, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.13, 1], wspace=.36)
    left, right = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    left.set_title("a  Launches in two retained traces", loc="left", fontsize=9, weight="bold", pad=16)
    for y, (r, label, color) in enumerate(zip(examples, ("Human", "Agent"), (BLUE, TEAL))):
        pos = 1-y
        left.hlines(pos, 0, r["time"][-1], color=GRID, linewidth=1.3)
        left.vlines(r["time"], pos-.14, pos+.14, color=color, linewidth=1.1)
        left.text(2, pos+.25, f"{label}: {r['events']} launches", fontsize=8.1, color=color)
    left.set(ylim=(-.4, 1.65), xlim=(-5, 400), yticks=[], xticks=[0, 100, 200, 300, 400],
             xlabel="Seconds since the first observed launch")
    left.spines[["left", "right", "top"]].set_visible(False)
    left.text(0, -.36, "Same task: Nextcloud/exp1", transform=left.transAxes, fontsize=8, color=MUTED)
    right.set_title("b  All 109 eligible runs", loc="left", fontsize=9, weight="bold", pad=16)
    medians = {}
    for y, label, color, style in ((0, "Human (61)", BLUE, "-"), (1, "Agent (48)", TEAL, "--")):
        x = np.sort([r["median_gap_seconds"] for r in rows if r["label"] == y])
        yy = np.arange(1, len(x)+1) / len(x)
        right.step(np.r_[0, x, 100], np.r_[0, yy, 1], where="post", color=color, linestyle=style,
                   linewidth=1.8, label=label)
        medians[label] = float(np.median(x))
    right.set_xscale("symlog", linthresh=.1, linscale=.5)
    right.set(xlim=(0, 100), ylim=(0, 1.02), yticks=[0, .5, 1], yticklabels=["0", "50", "100"],
              xticks=[0, .1, 1, 10, 100], xticklabels=["0", ".1", "1", "10", "100"],
              xlabel="Per-run median launch gap (seconds)", ylabel="Cumulative share of runs (%)")
    right.legend(loc="lower right", frameon=False, fontsize=8)
    right.grid(axis="y", color=GRID, linewidth=.6)
    right.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.04, right=.99, top=.84, bottom=.27)
    save(fig, "fingerprint-observed-rhythm")
    audit = {"contract": "AIT login-associated audit launches; first 128 events, minimum four; zero gaps retained",
             "run_count": len(rows), "distribution_unit": "one median per run",
             "illustration_selection": "same trace IDs as the existing matrix illustration; no score-based selection",
             "examples": [{k: v for k, v in r.items() if k != "time"} for r in examples],
             "median_of_per_run_medians_seconds": medians,
             "runs": [{k: v for k, v in r.items() if k != "time"} for r in rows]}
    (RESULTS / "evidence_story_timing.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    association()
    recognition()
    timing()
    print("Rendered association slopes, recognition matrix, and observed launch rhythm without changing cohorts.")
