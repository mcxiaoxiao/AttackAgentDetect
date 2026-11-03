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
from paper_style import apply, save as save_publication, PAGE_WIDTH
from paper_style import INK as NAVY, BLUE, TEAL, MUTED, GRID, PALE
apply()


def read(path):
    return json.loads(path.read_text())


def save(fig, name):
    save_publication(fig, name, (PAPER, OUT))


def association():
    results = read(ROOT / "research/results/association_fingerprint_v3/ba_tree_nested_multisource.json")["results"]
    panels = [("cth", "CTH (40 tasks)"), ("kypo", "KYPO (97 participants)"),
              ("antieval", "Antieval (31 tasks)")]
    fig, (left, right) = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.45),
                                     gridspec_kw={"width_ratios": [1.15, 1]})
    for y, (key, label) in zip([2, 1, 0], panels):
        r = results[key]
        baseline, full = r["baseline_ba"] * 100, r["ba"] * 100
        left.plot([baseline, full], [y, y], color=GRID, lw=1.4, zorder=1)
        left.scatter(baseline, y, s=22, facecolor="white", edgecolor=BLUE, lw=1.1,
                     label="Evidence" if y == 2 else None, zorder=3)
        left.scatter(full, y, s=22, marker="s", color=TEAL,
                     label="Evidence + B/S" if y == 2 else None, zorder=3)
        left.annotate(f"{baseline:.2f}", (baseline, y), xytext=(-2, 7), textcoords="offset points",
                      ha="center", fontsize=7.2, color=BLUE)
        left.annotate(f"{full:.2f}", (full, y), xytext=(2, -12), textcoords="offset points",
                      ha="center", fontsize=7.2, color=TEAL)
        delta = r["delta_ba"] * 100
        lo, hi = np.asarray(r["paired_cluster_ci95"]) * 100
        right.errorbar(delta, y, xerr=[[delta - lo], [hi - delta]], fmt="s", ms=3.5,
                       capsize=2.5, color=TEAL, lw=1.05)
        right.annotate(f"{delta:+.2f} [{lo:+.2f}, {hi:+.2f}]", (delta, y),
                       xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7)
    left.set_title("a  Association balanced accuracy", loc="left", weight="bold", pad=16)
    right.set_title("b  Paired fingerprint increment", loc="left", weight="bold", pad=16)
    left.set(xlim=(50, 76), ylim=(-.55, 2.55), yticks=[2, 1, 0],
             yticklabels=[label for _, label in panels], xticks=[50, 60, 70], xlabel="BA (%)")
    right.set(xlim=(-4, 10), ylim=(-.55, 2.55), yticks=[], xticks=[-4, 0, 4, 8],
              xlabel="ΔBA (percentage points)")
    right.axvline(0, color=MUTED, lw=.75, ls=(0, (3, 3)))
    for ax in (left, right):
        ax.grid(axis="x", color=GRID, lw=.45)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.set_axisbelow(True)
    fig.legend(*left.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.57, -.01),
               ncol=2, fontsize=7.2, frameon=False, handlelength=1.2, columnspacing=2)
    fig.subplots_adjust(left=.20, right=.985, top=.83, bottom=.27, wspace=.25)
    save(fig, "fingerprint-association-comparison")


def recognition():
    data = [x for x in read(RESULTS / "ait_recognition/metrics.json")
            if x["dataset"] == "extended_all_eligible_min4"]
    panels = [("task_holdout", "Task-group holdout"), ("source_holdout", "Source-group holdout")]
    methods = [("length", "Length"), ("core8", "Legacy K1–K8"), ("B", "B"),
               ("B_plus_S", "B + S"), ("B_plus_S_plus_T", "B/S/T, linear"),
               ("extra_trees_B_plus_S_plus_T", "B/S/T, trees"), ("command_tfidf", "Command TF–IDF")]
    lookup = {(r["panel"], r["method"]): r for r in data}
    a = np.array([[lookup[(panel, method)]["task_macro_ba"] * 100 for panel, _ in panels]
                  for method, _ in methods])
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3.2))
    image = ax.imshow(a, vmin=0, vmax=100, cmap="Blues", aspect="auto", interpolation="nearest")
    for i, (method, _) in enumerate(methods):
        for j, (panel, _) in enumerate(panels):
            r = lookup[(panel, method)]
            lo, hi = np.asarray(r["task_cluster_95ci_ba"]) * 100
            color = "white" if a[i, j] >= 66 else NAVY
            ax.text(j, i-.12, f"{a[i,j]:.1f}%", ha="center", va="center", fontsize=8.2,
                    color=color, weight="bold")
            ax.text(j, i+.23, f"[{lo:.1f}, {hi:.1f}]", ha="center", va="center", fontsize=6.8, color=color)
    ax.set_yticks(range(len(methods)), [label for _, label in methods])
    ax.set_xticks([0, 1], [label for _, label in panels]);ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=6, labelsize=7.3)
    ax.set_yticks(np.arange(-.5, len(methods), 1), minor=True);ax.set_xticks([.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5);ax.tick_params(which="minor", length=0)
    ax.add_patch(Rectangle((-.5, 3.5), 2, 1, fill=False, edgecolor=TEAL, linewidth=1.15))
    for spine in ax.spines.values():spine.set_visible(False)
    cb = fig.colorbar(image, cax=fig.add_axes([.93, .30, .014, .53]), ticks=[0, 50, 100])
    cb.set_label("BA (%)", fontsize=7);cb.ax.tick_params(labelsize=6.5, length=2);cb.outline.set_visible(False)
    deltas = read(RESULTS / "ait_recognition/paired_deltas.json")
    for j, (panel, _) in enumerate(panels):
        r = next(x for x in deltas if x["dataset"] == "extended_all_eligible_min4"
                 and x["panel"] == panel and x["baseline"] == "B_plus_S"
                 and x["comparison"] == "B_plus_S_plus_T")
        lo, hi = np.asarray(r["task_cluster_95ci"]) * 100
        ax.text(j, 7.05, f"+T: {r['task_macro_delta_ba']*100:+.2f} pp [{lo:+.2f}, {hi:+.2f}]",
                ha="center", va="top", fontsize=7, color=TEAL)
    fig.subplots_adjust(left=.23, right=.90, top=.90, bottom=.17)
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
    fig = plt.figure(figsize=(PAGE_WIDTH, 2.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=.35)
    left, right = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    left.set_title("a  Same-task launch sequences", loc="left", weight="bold", pad=11)
    for y, (r, label, color) in enumerate(zip(examples, ("Human", "Agent"), (BLUE, TEAL))):
        pos = 1-y
        left.hlines(pos, 0, r["time"][-1], color=GRID, linewidth=.7)
        left.vlines(r["time"], pos-.12, pos+.12, color=color, linewidth=.8)
        left.text(r["time"][-1] + 8, pos, str(r["events"]), fontsize=7, color=color, va="center")
    left.set(ylim=(-.45, 1.45), xlim=(-4, 405), yticks=[1, 0], yticklabels=["Human", "Agent"],
             xticks=[0, 100, 200, 300, 400], xlabel="Time since first launch (s)")
    left.tick_params(axis="y", length=0, pad=5)
    left.spines[["left", "right", "top"]].set_visible(False)
    left.text(0, -.41, "Nextcloud/exp1; 40 and 43 launches", transform=left.transAxes, fontsize=7, color=MUTED)
    right.set_title("b  Per-run launch rhythm (n = 109)", loc="left", weight="bold", pad=11)
    medians = {}
    for y, label, color, style in ((0, "Human (61)", BLUE, "-"), (1, "Agent (48)", TEAL, "--")):
        x = np.sort([r["median_gap_seconds"] for r in rows if r["label"] == y])
        yy = np.arange(1, len(x)+1) / len(x)
        right.step(np.r_[0, x, 100], np.r_[0, yy, 1], where="post", color=color, linestyle=style,
                   linewidth=1.2, label=label)
        medians[label] = float(np.median(x))
        right.vlines(medians[label], 0, .5, color=color, linewidth=.7, ls=(0, (2, 3)), alpha=.6)
        right.scatter(medians[label], .5, marker="o", s=12, facecolor="white", edgecolor=color, linewidth=.8, zorder=5)
        right.annotate(f"{medians[label]:.3f} s", (medians[label], .5), xytext=(-4, 9), textcoords="offset points",
                       ha="right", fontsize=6.7, color=color)
    right.set_xscale("symlog", linthresh=.1, linscale=.5)
    right.set(xlim=(0, 100), ylim=(0, 1.02), yticks=[0, .25, .5, .75, 1],
              yticklabels=["0", "25", "50", "75", "100"], xticks=[0, .1, 1, 10, 100],
              xticklabels=["0", "0.1", "1", "10", "100"], xlabel="Median launch gap per run (s)", ylabel="Cumulative share (%)")
    right.legend(loc="lower right", frameon=False, fontsize=6.8, handlelength=2)
    right.grid(axis="y", color=GRID, linewidth=.45)
    right.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.085, right=.985, top=.83, bottom=.28)
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
