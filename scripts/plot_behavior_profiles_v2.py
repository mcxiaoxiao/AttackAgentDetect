#!/usr/bin/env python3
"""Plot the fixed v2 behavior-profile comparison from task-level results."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/behavior_profiles_v2/task_metrics.csv"
OUTPUT = ROOT / "results/behavior_profiles_v2/profile_metric_deltas.json"
PAPER_FIGURE = ROOT / "paper/figures/behavior-profile-v2.pdf"
SITE_EN = ROOT / "site/assets/behavior-profile-v2-en.png"
SITE_ZH = ROOT / "site/assets/behavior-profile-v2-zh.png"
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 20261030

METRICS = (
    ("balanced_accuracy", "Balanced accuracy", "平衡准确率", 1),
    ("macro_f1", "Macro F1", "宏平均 F1", 1),
    ("auroc", "AUROC", "AUROC", 1),
    ("human_recall", "Human recall", "Human 召回率", 1),
    ("agent_recall", "Agent recall", "Agent 召回率", 1),
    ("brier", "Brier score", "Brier 分数", -1),
)


def task_level_deltas(frame: pd.DataFrame) -> dict:
    selected = frame[(frame["cohort"] == "full") & frame["method"].isin(("command_tfidf", "inner_selected"))]
    metrics = [item[0] for item in METRICS]
    by_task = selected.groupby(["method", "benchmark", "task"], sort=True)[metrics].mean().reset_index()
    baseline = by_task[by_task["method"] == "command_tfidf"].set_index(["benchmark", "task"])
    profile = by_task[by_task["method"] == "inner_selected"].set_index(["benchmark", "task"])
    if not baseline.index.equals(profile.index):
        raise ValueError("Baseline and profile methods do not have identical task coverage")

    task_deltas = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    benchmarks = sorted(set(profile.index.get_level_values("benchmark")))
    for metric, _label_en, _label_zh, direction in METRICS:
        differences = (profile[metric] - baseline[metric]) * direction
        by_benchmark = {
            benchmark: differences.xs(benchmark, level="benchmark").to_numpy()
            for benchmark in benchmarks
        }
        estimate = float(np.mean([values.mean() for values in by_benchmark.values()]))
        draws = np.empty(BOOTSTRAP_DRAWS)
        for draw in range(BOOTSTRAP_DRAWS):
            draws[draw] = np.mean([
                rng.choice(values, size=len(values), replace=True).mean()
                for values in by_benchmark.values()
            ])
        lower, upper = np.quantile(draws, [0.025, 0.975])
        task_deltas[metric] = {
            "delta_percentage_points_direction_adjusted": float(estimate * 100),
            "ci95_percentage_points": [float(lower * 100), float(upper * 100)],
        }

    def benchmark_macro(method: str, metric: str) -> float:
        values = by_task[by_task["method"] == method]
        return float(values.groupby("benchmark")[metric].mean().mean())

    fold_audit = json.loads((ROOT / "results/behavior_profiles_v2/results.json").read_text())
    folds = [row for row in fold_audit["fold_audit"] if row["cohort"] == "full"]
    nonzero_profile_weights = sum(row.get("selected_weight", 0) > 0 for row in folds)
    return {
        "status": "exploratory reuse of the previously inspected cohort",
        "records": 177,
        "task_groups": int(profile.shape[0]),
        "split_seeds": 6,
        "outer_fold_count": len(folds),
        "outer_folds_with_nonzero_profile_weight": int(nonzero_profile_weights),
        "methods": {
            "baseline": "command_tfidf",
            "comparison": "inner_selected",
            "description": "Inner-fold selection among command baseline and profile weights 0.1, 0.3, 1.0.",
        },
        "baseline_metrics": {metric: benchmark_macro("command_tfidf", metric) for metric in metrics},
        "profile_metrics": {metric: benchmark_macro("inner_selected", metric) for metric in metrics},
        "direction_adjusted_paired_task_cluster_deltas": task_deltas,
        "uncertainty": f"{BOOTSTRAP_DRAWS:,} paired task-cluster bootstrap draws, stratified by benchmark",
        "note": "All intervals are exploratory; no independent confirmation was performed.",
    }


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot(summary: dict, *, chinese: bool, output: pathlib.Path) -> None:
    configure_style()
    rows = list(METRICS)
    labels = [row[2 if chinese else 1] for row in rows]
    values = summary["direction_adjusted_paired_task_cluster_deltas"]
    estimates = [values[row[0]]["delta_percentage_points_direction_adjusted"] for row in rows]
    intervals = [values[row[0]]["ci95_percentage_points"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    y_positions = np.arange(len(rows))[::-1]
    ax.axvline(0, color="#334155", linewidth=1.0, zorder=1)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E5EAF0", linewidth=0.8)
    for y, estimate, (lower, upper) in zip(y_positions, estimates, intervals):
        color = "#0072B2" if estimate >= 0 else "#D55E00"
        ax.errorbar(
            estimate, y,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="o", color=color, ecolor="#64748B", elinewidth=1.4,
            capsize=3, markersize=5.5, markeredgecolor="white",
            markeredgewidth=0.7, zorder=3,
        )
        ax.text(upper + 0.13, y, f"{estimate:+.2f}", va="center", ha="left", fontsize=7.5, color="#334155")

    ax.set_yticks(y_positions, labels)
    ax.set_xlim(-6.7, 3.2)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xlabel("方向校正变化（百分点；正值表示改善）" if chinese else
                  "Direction-adjusted change (percentage points; positive is better)")
    title = "24 维行为画像未带来可靠增益" if chinese else "24-feature behavior profile adds no reliable gain"
    subtitle = (
        f"命令 TF–IDF + 内层筛选画像 · 40 个任务组 · 探索性复用 · 非零画像权重 {summary['outer_folds_with_nonzero_profile_weight']}/{summary['outer_fold_count']} 折"
        if chinese else
        f"Command TF–IDF + inner-selected profile · 40 task groups · exploratory reuse · nonzero profile weight in {summary['outer_folds_with_nonzero_profile_weight']}/{summary['outer_fold_count']} folds"
    )
    fig.suptitle(title, x=0.14, y=0.98, ha="left", fontsize=13, fontweight="bold", color="#172331")
    ax.set_title(subtitle, loc="left", fontsize=8, color="#526170", pad=10)
    footer = "95% 任务簇区间；所有区间均跨 0" if chinese else "95% task-cluster intervals; every interval includes zero"
    fig.text(0.14, 0.015, footer, ha="left", va="bottom", fontsize=8, color="#526170")
    fig.subplots_adjust(left=0.20, right=0.90, top=0.84, bottom=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = task_level_deltas(pd.read_csv(INPUT))
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    plot(summary, chinese=False, output=SITE_EN)
    plot(summary, chinese=True, output=SITE_ZH)
    plot(summary, chinese=False, output=PAPER_FIGURE)
    print(json.dumps(summary["direction_adjusted_paired_task_cluster_deltas"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
