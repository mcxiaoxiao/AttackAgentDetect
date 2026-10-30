#!/usr/bin/env python3
"""Render compact blue publication figures from feature-ablation outputs."""
from __future__ import annotations
import csv
import json
import pathlib
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = pathlib.Path(__file__).resolve().parents[1]
JSON = ROOT / "results" / "feature_ablation_v1.json"
CSV = ROOT / "results" / "feature_ablation_task_metrics_v1.csv"
OUT = ROOT / "results" / "figures" / "feature_ablation_v1"
SITE = ROOT / "site" / "assets" / "feature_ablation_v1.png"
FEATURES = ["F0_count_action", "F1_command_tfidf", "F2_sequence_structure", "F3_command_plus_structure"]
SHORT = ["Count + action", "Command TF-IDF", "Sequence structure", "Command + structure"]
BLUE = ["#F5FAFE", "#D9EAF8", "#9BC9EB", "#4B98D1", "#173F68"]


def read_rows():
    with CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_delta(records, *, feature_a="F3_command_plus_structure", feature_b="F1_command_tfidf", metric="balanced_accuracy"):
    by_key = {}
    for row in records:
        key = (row["benchmark"], row["task"], row["feature"], row["classifier"], row["seed"], row["cohort"])
        by_key[key] = float(row[metric])
    per_task = collections.defaultdict(list)
    for (benchmark, task, feature, classifier, seed, cohort), value in by_key.items():
        if feature != feature_a:
            continue
        other = by_key.get((benchmark, task, feature_b, classifier, seed, cohort))
        if other is not None:
            per_task[(benchmark, task)].append(value - other)
    by_benchmark = collections.defaultdict(list)
    for (benchmark, task), values in per_task.items():
        by_benchmark[benchmark].append(float(np.mean(values)))
    if not by_benchmark:
        return None, None, 0
    estimate = float(np.mean([np.mean(v) for v in by_benchmark.values()]))
    rng = np.random.default_rng(20261030)
    draws = []
    for _ in range(5000):
        draws.append(np.mean([np.mean(rng.choice(v, len(v), replace=True)) for v in by_benchmark.values()]))
    return estimate, np.quantile(draws, [0.025, 0.975]), sum(map(len, by_benchmark.values()))


def main():
    result = json.loads(JSON.read_text(encoding="utf-8"))
    rows = read_rows()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 320,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    cmap = LinearSegmentedColormap.from_list("study-blue", BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1.05, 1.45]})
    summary = {(r["feature"], r["classifier"]): r for r in result["primary_summary"]}
    values = np.array([[summary[(feature, model)]["metrics"]["balanced_accuracy"] for model in ("logistic", "extra_trees")] for feature in FEATURES])
    image = axes[0].imshow(values, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
    axes[0].set_xticks([0, 1], ["Logistic", "ExtraTrees"])
    axes[0].set_yticks(np.arange(len(SHORT)), SHORT)
    axes[0].set_title("Task-held-out source-label separability", loc="left", fontweight="semibold")
    axes[0].set_xlabel("Classifier")
    for i, feature in enumerate(FEATURES):
        for j, model in enumerate(("logistic", "extra_trees")):
            ci = summary[(feature, model)]["ci95_task_cluster"]["balanced_accuracy"]
            color = "white" if values[i, j] > 0.76 else "#142A3B"
            axes[0].text(j, i, f"{values[i,j]:.2f}\n[{ci[0]:.2f}, {ci[1]:.2f}]", ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.035)
    colorbar.set_label("Balanced accuracy")

    conditions = [
        ("Full · 177", "full_177", None),
        ("Full · ≥2 cohort", "matched_ge2_full_same_cohort", None),
        ("Prefix 2", "matched_ge2_prefix_2", None),
        ("Full · ≥4 cohort", "matched_ge4_full_same_cohort", None),
        ("Prefix 4", "matched_ge4_prefix_4", None),
        ("Shuffled prefix 4", "prefix4_order_shuffle_", "shuffle"),
    ]
    estimates, intervals, labels, counts = [], [], [], []
    for label, cohort, kind in conditions:
        if kind == "shuffle":
            selected = [r for r in rows if r["cohort"].startswith(cohort) and r["classifier"] == "logistic" and r["feature"] in {"F1_command_tfidf", "F3_command_plus_structure"}]
        else:
            selected = [r for r in rows if r["cohort"] == cohort and r["classifier"] == "logistic" and r["feature"] in {"F1_command_tfidf", "F3_command_plus_structure"}]
        estimate, ci, n = paired_delta(selected)
        if estimate is None:
            continue
        estimates.append(estimate); intervals.append(ci); labels.append(label); counts.append(n)
    y = np.arange(len(labels))
    low = np.asarray([v - ci[0] for v, ci in zip(estimates, intervals)])
    high = np.asarray([ci[1] - v for v, ci in zip(estimates, intervals)])
    colors = [
        "#173F68" if label == "Full · 177" else
        "#45A6FC" if label.startswith("Full") else
        "#99CDFC" if label.startswith("Prefix") else "#74A9D8"
        for label in labels
    ]
    axes[1].barh(y, estimates, color=colors, height=0.64, edgecolor="white", linewidth=0.8)
    axes[1].errorbar(estimates, y, xerr=[low, high], fmt="none", ecolor="#142A3B", capsize=2.5, linewidth=1)
    axes[1].axvline(0, color="#536476", linewidth=0.8)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Balanced-accuracy change (F3 − F1)")
    axes[1].set_title("Structure increment · Logistic", loc="left", fontweight="semibold")
    axes[1].set_xlim(-0.16, 0.16)
    axes[1].grid(axis="x", color="#E1EAF2", linewidth=0.7)
    axes[1].set_axisbelow(True)
    for yi, value in zip(y, estimates):
        axes[1].text(0.09, yi, f"{value:+.3f}", ha="left", va="center", fontsize=7.5, color="#142A3B")
    fig.suptitle("Do sequence-structure statistics add signal beyond command vocabulary?", x=0.02, ha="left", fontsize=12, fontweight="bold", color="#142A3B")
    fig.text(0.02, 0.005, "Right panel shows logistic regression; classifier-specific full-cohort results are in the heatmap. Paired task intervals; CTF labels are capture-path confounded.", fontsize=7.5, color="#536476")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93), w_pad=2.0)
    OUT.parent.mkdir(parents=True, exist_ok=True); SITE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(SITE, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT.with_suffix('.pdf')} and {SITE}")


if __name__ == "__main__":
    main()
