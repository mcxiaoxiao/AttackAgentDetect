#!/usr/bin/env python3
"""Render a compact category-sequence and score-attribution case figure."""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[3]
INPUT = ROOT / "research/results/behavioral_fingerprint_v2/ait_interpretable_mode_20261112/case.json"
OUT = ROOT / "research/paper/figures/ait-interpretable-ah-case"
sys.path.insert(0, str(ROOT))
from research.experiments.behavioral_fingerprint_v2.command_categories import CATEGORY_LIST
from research.experiments.behavioral_fingerprint_v2.paper_style import BLUE, TEAL

# Keep the explanation bars on the same B/S/T palette as the fingerprint
# overview and dimension figures: B blue, S teal, and observed T gray.
TIME = "#7a8798"

COLORS = {
    "system": "#8B5CF6", "navigation": "#14B8A6", "inspection": "#3B82F6",
    "editing": "#F59E0B", "network": "#EF4444", "process": "#EC4899",
    "file_ops": "#84CC16", "agent_tools": "#64748B", "unknown": "#CBD5E1",
    "intercept": "#9AA8B8", "B": BLUE, "S": TEAL, "T": TIME,
}


def friendly(name: str) -> str:
    labels = {
        "gap_ge_10s_share": "gap>=10s",
        "log_gap_autocorrelation": "gap-autocorr",
        "gap_le_1s_share": "gap<=1s",
        "action_return_share": "category-return",
        "long_lag_revisit_share": "long-revisit",
        "category_diversity": "category-diversity",
        "longest_run_share": "long-run",
        "new_category_share": "new-category",
        "median_log_launch_gap": "median-gap",
        "same_category_neighbor_share": "adjacent-repeat",
    }
    pieces = name.split(":")
    feature = labels.get(pieces[1], pieces[1].replace("_", " "))
    return f"{feature} ({pieces[0]})"


def main():
    case = json.loads(INPUT.read_text())
    examples = case["examples"]
    fig = plt.figure(figsize=(7.15, 4.0), constrained_layout=False)
    outer = fig.add_gridspec(2, 2, left=.08, right=.98, top=.84, bottom=.29,
                             hspace=.88, wspace=.26, width_ratios=[.40, .60])
    palette = COLORS
    max_abs = max(abs(float(item["blocks"][block]["logit_contribution"]))
                  for item in examples for block in ("B", "S", "T"))
    xlim = max(1.0, max_abs * 1.22)

    for row_index, item in enumerate(examples):
        mode = item["recorded_mode"]
        score = float(item["agent_direction_score"])
        title = f"Recorded {mode} · {item['event_count']} events · score {score:.2f}"
        ax_seq = fig.add_subplot(outer[row_index, 0])
        ax_seq.set_title(title, loc="left", fontsize=7.3, fontweight="bold",
                         color="#243447", pad=7)
        ax_seq.set_xlim(0, 16)
        ax_seq.set_ylim(0, 1)
        ax_seq.axis("off")
        seq = item["categories"]["first_16_category_events"]
        for i, category in enumerate(seq):
            rect = Rectangle((i + .035, .34), .91, .30,
                             facecolor=palette.get(category, palette["unknown"]),
                             edgecolor="white", linewidth=.35)
            ax_seq.add_patch(rect)
            ax_seq.text(i + .49, .49, str(i + 1), ha="center", va="center",
                        fontsize=4.6, color="white" if category != "unknown" else "#405064")
        ax_seq.text(0, .17, "Category sequence (first 16 events)", ha="left", va="top",
                    fontsize=5.8, color="#556575")

        contributions = {"B": float(item["blocks"]["B"]["logit_contribution"]),
                         "S": float(item["blocks"]["S"]["logit_contribution"]),
                         "T": float(item["blocks"]["T"]["logit_contribution"])}
        ax_bar = fig.add_subplot(outer[row_index, 1])
        y_values = [2, 1, 0]
        block_names = ["B · local behavior", "S · sequence", "T · timing"]
        bar_values = [contributions[block] for block in ("B", "S", "T")]
        bar_colors = [palette[block] for block in ("B", "S", "T")]
        bars = ax_bar.barh(y_values, bar_values, color=bar_colors, height=.52, zorder=2)
        ax_bar.axvline(0, color="#758395", linewidth=.7, linestyle=(0, (2, 2)), zorder=1)
        ax_bar.set_xlim(-xlim, xlim)
        ax_bar.set_ylim(-.55, 2.58)
        ax_bar.set_yticks(y_values, block_names, fontsize=5.5, color="#405064")
        ax_bar.tick_params(axis="x", labelsize=4.5, colors="#657486", length=2)
        ax_bar.set_xlabel("Contribution to Agent-direction logit", fontsize=5.5,
                          color="#556575", labelpad=1)
        ax_bar.spines[["top", "right", "left"]].set_visible(False)
        ax_bar.spines["bottom"].set_color("#C5CDD5")
        ax_bar.grid(axis="x", color="#E5EAF0", linewidth=.45, zorder=0)
        ax_bar.tick_params(axis="y", length=0, pad=2)
        for bar, value in zip(bars, bar_values):
            offset = .11 if abs(value) < .2 else .06
            x = value + (offset if value >= 0 else -offset)
            ax_bar.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}",
                        ha="left" if value >= 0 else "right", va="center",
                        fontsize=5.6, color="#233244")
        top_effects = []
        for block in ("B", "S", "T"):
            top = item["blocks"][block]["top_features"][0]
            top_effects.append(f"{block}: {friendly(top['feature'])} {top['logit_contribution']:+.2f}")
        ax_bar.text(0, -.34, "Top terms: " + "; ".join(top_effects),
                    transform=ax_bar.transAxes, ha="left", va="top", fontsize=4.7,
                    color="#526274", wrap=False)

    legend_names = CATEGORY_LIST + ["unknown"]
    legend_handles = [Rectangle((0, 0), 1, 1, facecolor=palette[name]) for name in legend_names]
    legend_labels = [name.replace("_", " ") for name in legend_names]
    fig.legend(legend_handles, legend_labels, loc="lower left", bbox_to_anchor=(.08, .075),
               ncol=9, frameon=False, fontsize=4.9, handlelength=.72, handleheight=.55,
               columnspacing=.85, handletextpad=.3)
    fig.suptitle(f"One preselected same-task pair: {case['task']} (task-held-out)",
                 x=.08, y=.965, ha="left", fontsize=9.2, fontweight="bold", color="#192736")
    fig.text(.98, .025,
             "Unknown executables collapse to ‘unknown’; contributions are model descriptions, not causal effects. Scores are uncalibrated.",
             ha="right", va="center", fontsize=4.8, color="#667586")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(json.dumps({"figure_pdf": str(OUT.with_suffix('.pdf')),
                      "figure_png": str(OUT.with_suffix('.png')),
                      "case_task": case["task"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
