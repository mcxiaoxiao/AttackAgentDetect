"""Shared typography and exports for the manuscript's measured figures."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

INK, BLUE, TEAL = "#17324f", "#2676bd", "#238b8e"
MUTED, GRID, PALE = "#64758a", "#d9e2ec", "#f1f6fa"
AMBER, MISSING = "#bb733c", "#e0e6ec"
PAGE_WIDTH = 7.05


def apply():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 7.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": MUTED, "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": .45,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
        "svg.fonttype": "none", "hatch.linewidth": .35,
    })


def save(fig, name, folders):
    for folder in folders:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png", "svg"):
            path = folder / f"{name}.{ext}"
            fig.savefig(path, dpi=360, bbox_inches="tight", pad_inches=.035)
            if ext == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
