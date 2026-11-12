#!/usr/bin/env python3
"""Update the original three-stage overview to the current feature contract."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/paper/figures"

NAVY = "#17324f"
BLUE = "#2676bd"
TEAL = "#238b8e"
MUTED = "#64758a"
GRID = "#d9e2ec"
BOX = "#f0f5fa"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.06",
        facecolor=BOX, edgecolor=color, linewidth=1.05,
    ))
    ax.text(x + .12, y + h - .18, title, fontsize=8.2, weight="bold",
            color=NAVY, va="top")
    ax.text(x + .12, y + .38, body, fontsize=6.9, color=MUTED,
            va="center", linespacing=1.45)


def arrow(ax, start, end, color=BLUE, rad=0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9,
        connectionstyle=f"arc3,rad={rad}", color=color, linewidth=1.15,
    ))


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(7.2, 3.25), facecolor="white")
    ax.set(xlim=(0, 7.2), ylim=(0, 3.25))
    ax.axis("off")

    box(ax, .10, 1.90, 1.78, .92, "Retained logs",
        "ordered commands + actions\nevent times if recorded", BLUE)
    box(ax, 2.24, 1.90, 1.82, .92, "Observation adapter",
        "source-specific field mapping\nexplicit observed-field mask", BLUE)
    box(ax, 4.42, 1.90, 2.68, .92, "Windowed B/S/T profile",
        "B: repertoire + organization\nS: order + revisits\nT1–T6: measured launch gaps", TEAL)

    box(ax, 2.24, .30, 2.08, .92, "Auxiliary H/A score",
        "category-normalized B/S/T\nAgent-direction score", BLUE)
    box(ax, 4.72, .30, 2.38, .92, "Primary pair retrieval",
        "E + pairwise B/S/T differences\nsame-task, same-mode relation", TEAL)

    arrow(ax, (1.92, 2.36), (2.20, 2.36))
    arrow(ax, (4.10, 2.36), (4.38, 2.36))
    arrow(ax, (4.90, 1.85), (3.75, 1.26), rad=-.04)
    arrow(ax, (5.96, 1.85), (5.92, 1.26), rad=.02)

    ax.text(.10, 3.06, "From observed logs to pairwise trace retrieval",
            fontsize=10, weight="bold", color=NAVY)
    ax.text(.10, .04,
            "H/A is a separate readout; T is included only when launch timestamps are recorded.",
            fontsize=6.8, color=MUTED)
    fig.subplots_adjust(left=.01, right=.99, bottom=.03, top=.99)
    fig.savefig(OUT / "fingerprint-v3-architecture.pdf", bbox_inches="tight", pad_inches=.025)
    svg_path = OUT / "fingerprint-v3-architecture.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=.025)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(OUT / "fingerprint-v3-architecture.png", dpi=300, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)


if __name__ == "__main__":
    main()
