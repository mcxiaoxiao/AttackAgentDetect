"""Make small, transparent line icons matching the overview's visual language."""
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from paper_style import BLUE, TEAL

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "research/paper/figures"
SITE = ROOT / "site/assets/behavioral-fingerprint-v2"


def draw(kind):
    color = BLUE if kind == "human" else TEAL
    fig = plt.figure(figsize=(.3, .3))
    ax = fig.add_axes([0, 0, 1, 1]);ax.set(xlim=(0, 24), ylim=(0, 24), aspect="equal");ax.axis("off")
    if kind == "human":
        ax.add_patch(Circle((12, 17.5), 3.6, fill=False, edgecolor=color, lw=1.35))
        path = MplPath([(4.5, 2.2), (4.5, 4.2), (4.5, 11.7), (12, 11.7),
                        (19.5, 11.7), (19.5, 4.2), (19.5, 2.2)],
                       [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
                        MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO])
        ax.add_patch(PathPatch(path, fill=False, edgecolor=color, lw=1.35, capstyle="round"))
    else:
        ax.add_patch(FancyBboxPatch((3.5, 3.5), 17, 13, boxstyle="round,pad=0,rounding_size=3",
                                   fill=False, edgecolor=color, lw=1.35))
        ax.plot([12, 12], [16.5, 20], color=color, lw=1.35)
        ax.add_patch(Circle((12, 21.1), 1.15, fill=False, edgecolor=color, lw=1.2))
        for x in (8.5, 15.5):ax.add_patch(Circle((x, 10.5), .85, facecolor=color, edgecolor="none"))
        for xs in ([1, 3.5], [20.5, 23]):ax.plot(xs, [10, 10], color=color, lw=1.35, solid_capstyle="round")
    for ext in ("pdf", "svg", "png"):
        out = PAPER / f"mode-{kind}.{ext}"
        fig.savefig(out, dpi=500, transparent=True, pad_inches=0)
        if ext == "svg":out.write_text("\n".join(line.rstrip() for line in out.read_text().splitlines())+"\n")
        shutil.copy2(out, SITE/out.name)
    plt.close(fig)


if __name__ == "__main__":
    draw("human");draw("agent")
