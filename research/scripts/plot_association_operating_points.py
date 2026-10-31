#!/usr/bin/env python3
"""Publication-oriented association error and retrieval figure."""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "association_operating_points_v1.json"
OUT = ROOT / "results" / "figures" / "association_operating_points_v1"
SITE = ROOT.parent / "site" / "assets" / "association_operating_points_v1.png"
METHODS = ["command_cosine", "behavior", "structure_aware", "mode_aware", "length_only"]
LABELS = ["Command cosine", "Behavior ranker", "Structure-aware", "Mode-aware", "Length control"]
COLORS = ["#99CDFC", "#45A6FC", "#74A9D8", "#173F68", "#8798A8"]


def point_interval(entry):
    value = entry.get("estimate")
    ci = entry.get("ci_95")
    if value is None or ci is None:
        return np.nan, np.nan, np.nan
    return value, value - ci[0], ci[1] - value


def main():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    errors = result["false_merge_targets"]["20"]["summary"]
    ranking = result["ranking_summary"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 320,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    y = np.arange(len(METHODS))
    for ax, metric, title in [
        (axes[0], "false_split_rate", "Same-task proxy · false splits"),
        (axes[1], "false_merge_rate", "Different-task proxy · false merges"),
    ]:
        for i, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
            value, low, high = point_interval(errors[method][metric])
            ax.errorbar(value * 100, y[i], xerr=[[low * 100], [high * 100]], fmt="o", markersize=5,
                        capsize=2.8, linewidth=1.1, color=color, markeredgecolor="white", markeredgewidth=0.5)
        ax.set_yticks(y, LABELS)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xlabel("Held-out task-macro error (%)")
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.grid(axis="x", color="#E1EAF2", linewidth=0.7)
        ax.set_axisbelow(True)

    rank_metrics = [("hit_at_1", "Hit@1"), ("mrr", "MRR"), ("recall_at_5", "Recall@5")]
    x = np.arange(len(rank_metrics))
    offsets = np.linspace(-0.25, 0.25, len(METHODS))
    for index, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
        values, lows, highs = [], [], []
        for metric, _ in rank_metrics:
            v, low, high = point_interval(ranking[method][metric])
            values.append(v); lows.append(low); highs.append(high)
        axes[2].errorbar(x + offsets[index], values,
                         yerr=[lows, highs], fmt="o", markersize=4.5, capsize=2,
                         linewidth=1.1, color=color, label=label,
                         markeredgecolor="white", markeredgewidth=0.4)
    axes[2].set_xticks(x, [label for _, label in rank_metrics])
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks(np.arange(0, 1.01, 0.2))
    axes[2].set_ylabel("Task-macro retrieval score")
    axes[2].set_title("Same candidate gallery · ranking", loc="left", fontweight="semibold")
    axes[2].grid(axis="y", color="#E1EAF2", linewidth=0.7)
    axes[2].set_axisbelow(True)
    axes[2].legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2, columnspacing=0.8, handletextpad=0.35)
    fig.suptitle("Execution-mode features at a 20% training false-merge ceiling", x=0.02, ha="left", fontsize=12, fontweight="bold", color="#142A3B")
    fig.text(0.02, 0.005, "Positive labels are shared CTF task IDs, not shared attack activity. Error bars are 95% task-cluster bootstrap intervals.", fontsize=7.5, color="#536476")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92), w_pad=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True); SITE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(SITE, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT.with_suffix('.pdf')} and {SITE}")


if __name__ == "__main__":
    main()
