#!/usr/bin/env python3
"""Plot the locked external false-merge / false-split operating points."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "multidimensional_rerun_v1" / "results.json"
FIG_DIR = ROOT / "results" / "figures"
SITE_ASSETS = ROOT.parent / "site" / "assets"
METHODS = (
    ("command_tfidf", "Command TF–IDF", "命令 TF–IDF", "#0072B2"),
    ("profile_24d", "24D profile", "24 维画像", "#E69F00"),
    ("tensor_4x8x3", "4×8×3 tensor", "4×8×3 张量", "#009E73"),
    ("tensor_tucker", "Tucker tensor", "Tucker 张量", "#CC79A7"),
    ("partial_sequence", "IDF partial sequence", "IDF 局部序列", "#56B4E9"),
    ("learned_fusion", "Learned fusion", "学习式融合", "#D55E00"),
)
OFFSETS = {
    "command_tfidf": (-92, 12),
    "profile_24d": (10, -16),
    "tensor_4x8x3": (8, 8),
    "tensor_tucker": (8, -16),
    "partial_sequence": (-108, -12),
    "learned_fusion": (8, 10),
}


def draw(language: str, summaries: dict) -> None:
    chinese = language == "zh"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Heiti SC", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8.1, 5.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for key, label_en, label_zh, color in METHODS:
        values = summaries[key]
        fmr = values["false_merge_rate"]
        fsr = values["false_split_rate"]
        x, y = fmr["estimate"] * 100, fsr["estimate"] * 100
        xerr = np.array([[x - fmr["ci_95"][0] * 100], [fmr["ci_95"][1] * 100 - x]])
        yerr = np.array([[y - fsr["ci_95"][0] * 100], [fsr["ci_95"][1] * 100 - y]])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", markersize=7.3,
                    color=color, ecolor=color, elinewidth=1.4, capsize=3.2,
                    markeredgecolor="white", markeredgewidth=.8, zorder=3)
        ax.annotate(label_zh if chinese else label_en, (x, y),
                    xytext=OFFSETS[key], textcoords="offset points",
                    fontsize=9, color=color, weight="semibold" if key == "profile_24d" else "normal",
                    arrowprops={"arrowstyle": "-", "color": color, "lw": .8, "alpha": .7}
                    if key in {"command_tfidf", "partial_sequence"} else None)

    ax.set_xlim(-2, 110)
    ax.set_ylim(-2, 105)
    ax.set_xlabel("False-merge rate · FMR (%)" if not chinese else "误合并率 · FMR（%）")
    ax.set_ylabel("Missed-link rate · FSR (%)" if not chinese else "漏关联率 · FSR（%）")
    ax.set_title(
        "Locked external operating points" if not chinese else "锁定外部集：各方法的误差工作点",
        loc="left", pad=12, weight="semibold",
    )
    ax.grid(True, color="#d9e2e9", linewidth=.7, alpha=.85)
    ax.text(.015, .025, "better ↓" if not chinese else "左下方误差较低",
            transform=ax.transAxes, color="#526575", fontsize=8)
    fig.text(
        .01, -.025,
        "Thresholds were selected on development tasks. Error bars: 95% task-cluster bootstrap; n = 14 task groups. Same-task labels measure retrieval relevance, not actor identity."
        if not chinese else
        "阈值只由开发任务选定。误差线为 95% 任务簇 Bootstrap；n = 14 个任务组。“同题”衡量检索相关性，不代表操作者身份。",
        ha="left", va="top", fontsize=7.3, color="#526575",
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    if chinese:
        fig.savefig(FIG_DIR / "external-error-tradeoff-v1-zh.png", dpi=320, bbox_inches="tight", facecolor="white")
        fig.savefig(SITE_ASSETS / "external-error-tradeoff-v1-zh.png", dpi=240, bbox_inches="tight", facecolor="white")
        fig.savefig(FIG_DIR / "external-error-tradeoff-v1.pdf", bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(FIG_DIR / "external-error-tradeoff-v1-en.png", dpi=320, bbox_inches="tight", facecolor="white")
        fig.savefig(SITE_ASSETS / "external-error-tradeoff-v1-en.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    summaries = result["metrics"]["primary_error_and_ranking_summaries"]
    draw("zh", summaries)
    draw("en", summaries)
    print("Wrote bilingual external FMR–FSR comparison figures.")


if __name__ == "__main__":
    main()
