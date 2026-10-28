#!/usr/bin/env python3
"""Render publication figures from the versioned aggregate experiment results.

This script deliberately uses only results/*.json; it never loads or copies raw
terminal or Agent logs.  PDF is the vector master and PNG is a 300-dpi preview.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER_FIGURES = ROOT / "paper" / "figures"
OUTPUTS = [RESULTS / "figures", PAPER_FIGURES]

INK = "#172331"
MUTED = "#667586"
LINE = "#D8E1EA"
PAPER = "#F7F9FC"
BLUE = "#2585D8"
BLUE_LIGHT = "#99CDFC"
GREEN = "#4E9D78"
ORANGE = "#D89442"
GRAY = "#8A96A3"
PURPLE = "#8063A9"


def load_results():
    cross = json.loads((RESULTS / "crossbenchmark_exploration.json").read_text())
    mode = json.loads((RESULTS / "mode_aware_association.json").read_text())
    return cross, mode


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.6,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": LINE,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.facecolor": PAPER,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def export(fig, name):
    for folder in OUTPUTS:
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def add_panel_label(ax, label, title):
    ax.text(0.0, 1.06, label, transform=ax.transAxes, color=BLUE,
            fontsize=9, fontweight="bold", va="bottom")
    ax.text(0.065, 1.06, title, transform=ax.transAxes, color=INK,
            fontsize=8.8, fontweight="semibold", va="bottom")


def draw_figure1(cross, mode):
    base_methods = [
        ("Command unigram", cross["retrieval"]["command_unigram|human_agent"], BLUE),
        ("Ordered command bigram", cross["retrieval"]["ordered_command_bigram|human_agent"], BLUE_LIGHT),
        ("Ordered action bigram", cross["retrieval"]["ordered_action_bigram|human_agent"], GREEN),
        ("ProcGrep", cross["retrieval"]["procgrep_local_patterns|human_agent"], PURPLE),
    ]
    proto_methods = [
        ("Similarity only", mode["retrieval"]["human_agent|similarity_only"], GRAY),
        ("Predicted mode", mode["retrieval"]["human_agent|mode_aware"], ORANGE),
        ("Oracle mode*", mode["retrieval"]["human_agent|oracle_mode_upper_bound"], INK),
    ]
    rows = base_methods + proto_methods

    fig = plt.figure(figsize=(7.18, 3.34), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.42, .83), left=.20, right=.97,
                          top=.81, bottom=.22, wspace=.04)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(rows))[::-1]
    for pos, (label, item, color) in zip(y, rows):
        mean = item["benchmark_macro"]["recall_at_1"]
        low, high = item["task_stratified_95_ci"]["recall_at_1"]
        ax.errorbar(mean, pos, xerr=[[mean-low], [high-mean]], fmt="o",
                    color=color, ecolor=color, markersize=4.5, elinewidth=1.15,
                    capsize=2.3, markeredgecolor="white", markeredgewidth=.6,
                    zorder=3)
    chance = cross["retrieval"]["ordered_action_bigram|human_agent"]["benchmark_macro"]["random_recall_at_1"]
    ax.axvline(chance, color=MUTED, linestyle=(0, (3, 2)), linewidth=.9, zorder=1)
    ax.text(chance + .012, 6.46, f"random {chance:.3f}", color=MUTED,
            fontsize=6.8, va="top")
    ax.axhline(2.5, color=LINE, linewidth=.8)
    ax.set_yticks(y, [label for label, _, _ in rows])
    ax.set_xlim(0, .67)
    ax.set_ylim(-.55, 6.65)
    ax.set_xticks(np.arange(0, .61, .1))
    ax.set_xlabel("Recall@1")
    ax.grid(axis="x", color=LINE, linewidth=.55, alpha=.8)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_title("Human query  →  Agent gallery", loc="left", pad=14,
                 fontsize=9, color=INK, fontweight="semibold")
    ax.text(.98, 1.055, "A", transform=ax.transAxes, color=BLUE,
            fontsize=9, fontweight="bold", ha="right", va="bottom")

    details = fig.add_subplot(gs[0, 1], sharey=ax)
    details.set_xlim(0, 1)
    details.set_ylim(ax.get_ylim())
    details.axis("off")
    details.text(.02, 1.075, "Estimate  [95% task CI]", transform=details.transAxes,
                 color=MUTED, fontsize=7.0, fontweight="semibold", va="bottom")
    for pos, (_, item, _) in zip(y, rows):
        mean = item["benchmark_macro"]["recall_at_1"]
        lo, hi = item["task_stratified_95_ci"]["recall_at_1"]
        details.text(.02, pos, f"{mean:.3f}  [{lo:.3f}, {hi:.3f}]",
                     transform=details.get_yaxis_transform(), va="center",
                     color=INK, fontsize=7.1, family="DejaVu Sans Mono")
    fig.text(.20, .955, "SAME-BENCHMARK CANDIDATE GALLERIES  ·  40 TASKS",
             fontsize=7.1, color=BLUE, fontweight="bold", va="top")
    fig.text(.20, .045, "Whiskers show 95% task-cluster intervals. Oracle mode uses ground-truth labels at inference.",
             color=MUTED, fontsize=6.3, va="bottom")
    export(fig, "figure1_retrieval")


def draw_figure2(mode):
    keys = ["human_agent", "agent_agent_cross_model", "agent_human", "human_human"]
    labels = ["Human → Agent", "Agent → Agent*", "Agent → Human", "Human → Human"]
    colors = [BLUE, BLUE, BLUE, ORANGE]
    items = [mode["paired_delta_mode_aware_minus_similarity_only"][f"{key}|recall_at_1"]
             for key in keys]
    fig = plt.figure(figsize=(7.18, 2.72))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0), left=.19, right=.97,
                          top=.79, bottom=.24, wspace=.04)
    ax = fig.add_subplot(gs[0, 0])
    vals = [item["mean_delta"] for item in items]
    intervals = [item["task_stratified_95_ci"] for item in items]
    y = np.arange(4)[::-1]
    for pos, value, interval, color in zip(y, vals, intervals, colors):
        ax.errorbar(value, pos, xerr=[[value-interval[0]], [interval[1]-value]],
                    fmt="o", color=color, ecolor=color, markersize=5.0,
                    elinewidth=1.3, capsize=2.7, markeredgecolor="white",
                    markeredgewidth=.7, zorder=3)
    ax.axvline(0, color=INK, linewidth=.85)
    ax.set_yticks(y, labels)
    ax.set_xlim(-.22, .22)
    ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2], ["−20", "−10", "0", "+10", "+20"])
    ax.set_xlabel("Δ Recall@1 (percentage points)")
    ax.grid(axis="x", color=LINE, linewidth=.55)
    ax.tick_params(axis="y", length=0, pad=5)
    add_panel_label(ax, "A", "Change by query direction")

    details = fig.add_subplot(gs[0, 1], sharey=ax)
    details.set_xlim(0, 1)
    details.set_ylim(ax.get_ylim())
    details.axis("off")
    details.text(.02, 1.04, "Δ  [95% task CI]", transform=details.transAxes,
                 color=MUTED, fontsize=7.1, fontweight="semibold")
    details.text(.89, 1.04, "n", transform=details.transAxes,
                 color=MUTED, fontsize=7.1, fontweight="semibold", ha="right")
    for pos, item in zip(y, items):
        lo, hi = item["task_stratified_95_ci"]
        details.text(.02, pos, f"{item['mean_delta']*100:+.1f}  [{lo*100:+.1f}, {hi*100:+.1f}]",
                     transform=details.get_yaxis_transform(), va="center",
                     color=INK, fontsize=7.4, family="DejaVu Sans Mono")
        details.text(.89, pos, str(item["task_count"]),
                     transform=details.get_yaxis_transform(), va="center",
                     color=MUTED, fontsize=7.4, ha="right")
    fig.text(.19, .95, "MODE EFFECTS VARY WITH QUERY DIRECTION", color=BLUE,
             fontsize=7.2, fontweight="bold", va="top")
    fig.text(.19, .075, "* Cross-model Agent queries. Intervals resample task groups within each benchmark.",
             color=MUTED, fontsize=6.5, va="bottom")
    export(fig, "figure2_mode_effects")


def draw_figure_a1(cross):
    methods = [
        ("Command n-grams", "tfidf_command_unigram_bigram", BLUE),
        ("Count + action mix", "command_count_and_action_mix", ORANGE),
        ("Benchmark only", "benchmark_only_negative_control", GRAY),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.18, 2.6), sharey=True)
    for ax, metric, title in zip(axes, ["balanced_accuracy", "roc_auc"],
                                 ["Balanced accuracy", "ROC AUC"]):
        for pos, (label, key, color) in zip([2, 1, 0], methods):
            item = cross["classifier"][key]
            mean = item[metric]
            lo, hi = item["task_stratified_95_ci"][metric]
            ax.errorbar(mean, pos, xerr=[[mean-lo], [hi-mean]], fmt="o",
                        color=color, ecolor=color, markersize=5, elinewidth=1.3,
                        capsize=2.7, markeredgecolor="white", markeredgewidth=.7)
        ax.axvline(.5, color=MUTED, linestyle=(0, (3, 2)), linewidth=.85)
        ax.set_xlim(.43, 1.0)
        ax.set_xticks([.5, .6, .7, .8, .9, 1.0])
        ax.set_xlabel(title)
        ax.grid(axis="x", color=LINE, linewidth=.55)
        ax.tick_params(axis="y", length=0, pad=5)
        add_panel_label(ax, "A" if metric == "balanced_accuracy" else "B", title)
    axes[0].set_yticks([2, 1, 0], [x[0] for x in methods])
    fig.text(.08, .96, "EXECUTION-MODE LABEL CLASSIFICATION  ·  n=177 TRAJECTORIES",
             color=BLUE, fontsize=7.2, fontweight="bold", va="top")
    fig.text(.08, .045, "Dashed line: chance. Labels distinguish expert terminal recordings from selected Agent tool-call logs.",
             color=MUTED, fontsize=6.4, va="bottom")
    fig.subplots_adjust(left=.26, right=.98, top=.78, bottom=.25, wspace=.15)
    export(fig, "figureA1_mode_classifier")


def draw_figure_a2(mode):
    directions = ["human_agent", "agent_agent_cross_model", "agent_human", "human_human"]
    labels = ["H → A", "A → A*", "A → H", "H → H"]
    metrics = [
        ("recall_at_1", "Δ Recall@1", (-.22, .22), [-.2, -.1, 0, .1, .2]),
        ("mrr", "Δ MRR", (-.17, .13), [-.15, -.075, 0, .075, .15]),
        ("query_auc", "Δ query AUC", (-.10, .08), [-.10, -.05, 0, .05]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.18, 3.45), sharey=True)
    y = np.arange(4)[::-1]
    for ax, (metric, title, bounds, ticks), letter in zip(axes, metrics, "ABC"):
        for pos, direction, label in zip(y, directions, labels):
            item = mode["paired_delta_mode_aware_minus_similarity_only"][f"{direction}|{metric}"]
            value = item["mean_delta"] * 100
            lo, hi = np.array(item["task_stratified_95_ci"]) * 100
            color = ORANGE if direction == "human_human" else BLUE
            ax.errorbar(value, pos, xerr=[[value-lo], [hi-value]], fmt="o",
                        color=color, ecolor=color, markersize=4.0, elinewidth=1.0,
                        capsize=2.0, markeredgecolor="white", markeredgewidth=.5)
        ax.axvline(0, color=INK, linewidth=.75)
        ax.set_xlim(np.array(bounds) * 100)
        ax.set_xticks(np.array(ticks) * 100)
        ax.set_xticklabels([f"{v*100:+.0f}" if v else "0" for v in ticks])
        ax.set_title(f"{letter}   {title} (pp)", loc="left", fontsize=8.1,
                     pad=12, color=INK, fontweight="semibold")
        ax.grid(axis="x", color=LINE, linewidth=.5)
        ax.tick_params(axis="y", length=0, pad=4)
    axes[0].set_yticks(y, labels)
    fig.text(.09, .96, "COMPLETE PAIRED EFFECTS  ·  MODE-AWARE MINUS SIMILARITY-ONLY",
             color=BLUE, fontsize=7.1, fontweight="bold", va="top")
    fig.text(.09, .045, "A→A uses different source models. H→H has 17 independent tasks; other directions have 40.",
             color=MUTED, fontsize=6.4, va="bottom")
    fig.subplots_adjust(left=.16, right=.99, top=.78, bottom=.25, wspace=.45)
    export(fig, "figureA2_full_directional_effects")


def draw_figure_a3(cross):
    benchmarks = [
        ("CyBench", "cybench"),
        ("InterCode-CTF", "intercode-ctf"),
        ("NYUCTF", "nyuctf"),
    ]
    methods = [
        ("Command unigram", "command_unigram|human_agent"),
        ("Command bigram", "ordered_command_bigram|human_agent"),
        ("Action bigram", "ordered_action_bigram|human_agent"),
        ("ProcGrep", "procgrep_local_patterns|human_agent"),
    ]
    values = np.array([[cross["retrieval"][key]["by_benchmark"][benchmark]["recall_at_1"]
                        for _, key in methods] for _, benchmark in benchmarks])
    fig, ax = plt.subplots(figsize=(7.18, 2.45))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=.75, aspect="auto")
    row_labels = []
    for name, benchmark in benchmarks:
        item = cross["retrieval"]["command_unigram|human_agent"]["by_benchmark"][benchmark]
        row_labels.append(f"{name}  ·  {item['task_count']} tasks  ·  random {item['random_recall_at_1']:.3f}")
    ax.set_yticks(range(len(benchmarks)), row_labels)
    ax.set_xticks(range(len(methods)), [name for name, _ in methods])
    ax.tick_params(axis="both", length=0, pad=6)
    ax.set_xticks(np.arange(-.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(benchmarks), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i, (_, benchmark) in enumerate(benchmarks):
        chance = cross["retrieval"]["command_unigram|human_agent"]["by_benchmark"][benchmark]["random_recall_at_1"]
        for j, (_, key) in enumerate(methods):
            value = values[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if value > .43 else INK,
                    fontsize=8.2, fontweight="semibold" if value >= chance else "normal")
    ax.set_frame_on(False)
    cbar = fig.colorbar(image, ax=ax, fraction=.026, pad=.015)
    cbar.set_label("Recall@1", fontsize=7.7, color=MUTED)
    cbar.ax.tick_params(labelsize=7, colors=MUTED, length=0)
    cbar.outline.set_visible(False)
    fig.text(.07, .955, "HUMAN-TO-AGENT SAME-TASK RETRIEVAL  ·  RECALL@1",
             color=BLUE, fontsize=7.1, fontweight="bold", va="top")
    fig.subplots_adjust(left=.36, right=.965, top=.86, bottom=.16)
    export(fig, "figureA3_benchmark_retrieval")


def main():
    setup_style()
    cross, mode = load_results()
    assert cross["audit"]["rows"] == 177
    assert cross["audit"]["matched_task_groups"] == 40
    draw_figure1(cross, mode)
    draw_figure2(mode)
    draw_figure_a1(cross)
    draw_figure_a2(mode)
    draw_figure_a3(cross)
    print("Rendered 5 figures (PDF vector masters + 300-dpi PNG) from aggregate JSON.")


if __name__ == "__main__":
    main()
