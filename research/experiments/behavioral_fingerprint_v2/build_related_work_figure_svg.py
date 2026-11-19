#!/usr/bin/env python3
"""Build the editable Figure 2 using the visual palette from Figure 1."""
from __future__ import annotations

import base64
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/paper/figures"
OUT.mkdir(parents=True, exist_ok=True)
SVG = OUT / "fingerprint-related-work-routes-editable.svg"
PNG = OUT / "fingerprint-related-work-routes-editable.png"
ICON_DIR = OUT / "tool-icons"
NAVY = "#06172f"
INK = "#06172f"
MUTED = "#536274"
RULE = "#9fb4c9"
WHITE = "#ffffff"
GREEN = "#218a64"
BLUE = "#1760d5"
GRAY = "#858a92"
RUST = "#a65337"
TEAL = BLUE
PURPLE = "#6845bd"
BLUE_LIGHT = "#e7f1fd"
BLUE_EDGE = "#a9c8f3"
GREEN_LIGHT = "#e6f4ee"
GREEN_EDGE = "#afd7c5"
GRAY_LIGHT = "#f0f1f2"
GRAY_EDGE = "#c7cbd0"

parts = ["""<svg xmlns="http://www.w3.org/2000/svg" width="3344" height="1360" viewBox="0 0 1672 680" role="img" aria-labelledby="title desc">
<title id="title">Model identification and behavioral profiling routes</title>
<desc id="desc">Agent traces jointly reflect tool, model, and skill effects. The figure contrasts reference-limited model identification with fixed-feature behavioral profiling across model and version variants.</desc>
<defs>
 <linearGradient id="headA" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#e4f1fc"/><stop offset="1" stop-color="#d4e9fb"/></linearGradient>
 <linearGradient id="headB" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#e7f5f3"/><stop offset=".52" stop-color="#e9f1fb"/><stop offset="1" stop-color="#edf1f5"/></linearGradient>
 <linearGradient id="matrixFill" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stop-color="#eff7fc"/><stop offset="1" stop-color="#dcecf7"/></linearGradient>
 <linearGradient id="ecosystemFill" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#e4f7f0"/><stop offset="1" stop-color="#e2f1ec"/></linearGradient>
 <marker id="arrowTip" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#174b78"/></marker>
</defs>
<rect width="1672" height="680" fill="#ffffff"/>
"""]


def text(x: float, y: float, value: str, size: float = 24, color: str = INK,
         weight: int = 400, anchor: str = "start", italic: bool = False) -> None:
    style = "italic" if italic else "normal"
    parts.append(
        f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" font-style="{style}" '
        f'fill="{color}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str = WHITE,
         stroke: str = RULE, sw: float = 1.6, radius: float = 9) -> None:
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = NAVY,
         sw: float = 2.3, arrow: bool = False) -> None:
    marker = ' marker-end="url(#arrowTip)"' if arrow else ""
    parts.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round"{marker}/>'
    )


def poly(points: str, fill: str, stroke: str = "none", sw: float = 1.5) -> None:
    parts.append(f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>')


def check(x: float, y: float, size: float = 25) -> None:
    parts.append(
        f'<path d="M{x} {y + size * .54} l{size * .28} {size * .29} l{size * .61} {-size * .72}" '
        f'fill="none" stroke="{GREEN}" stroke-width="4.6" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def cross(x: float, y: float, size: float = 22) -> None:
    line(x, y, x + size, y + size, RUST, 4)
    line(x + size, y, x, y + size, RUST, 4)


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#174b78", sw: float = 2.6) -> None:
    if abs(y2 - y1) < 0.01:
        direction = 1 if x2 >= x1 else -1
        line(x1, y1, x2 - direction * 9, y2, color, sw)
        poly(f"{x2},{y2} {x2-direction*10},{y2-6} {x2-direction*10},{y2+6}", color)
    elif abs(x2 - x1) < 0.01:
        direction = 1 if y2 >= y1 else -1
        line(x1, y1, x2, y2 - direction * 9, color, sw)
        poly(f"{x2},{y2} {x2-6},{y2-direction*10} {x2+6},{y2-direction*10}", color)
    else:
        line(x1, y1, x2, y2, color, sw)


ET.register_namespace("", "http://www.w3.org/2000/svg")


def brand_icon(name: str, x: float, y: float, size: float,
               monochrome: str = INK) -> None:
    if name == "codex":
        encoded = base64.b64encode((ICON_DIR / "codex.png").read_bytes()).decode("ascii")
        parts.append(
            f'<image x="{x}" y="{y}" width="{size}" height="{size}" '
            'preserveAspectRatio="xMidYMid meet" '
            f'href="data:image/png;base64,{encoded}"/>'
        )
        return
    root = ET.parse(ICON_DIR / f"{name}.svg").getroot()
    view_box = escape(root.attrib["viewBox"])
    inner = "".join(ET.tostring(child, encoding="unicode") for child in root)
    inner = inner.replace("currentColor", monochrome)
    fill = root.attrib.get("fill", "")
    fill_attr = f' fill="{monochrome}"' if fill == "currentColor" else ""
    parts.append(
        f'<svg x="{x}" y="{y}" width="{size}" height="{size}" '
        f'viewBox="{view_box}"{fill_attr} preserveAspectRatio="xMidYMid meet">{inner}</svg>'
    )


def brand_entry(x: float, y: float, label: str, icon: str | None,
                size: float = 24, font_size: float = 18) -> None:
    if icon:
        brand_icon(icon, x, y, size)
        label_x = x + size + 8
    else:
        label_x = x
    text(label_x, y + size * .77, label, font_size, INK, 700)


# Two method routes with matched log inputs and a compact capability comparison.
# Both log boxes use the same background; route colors distinguish the analyses.
tools = [("Codex", "codex"), ("DSH", "dsh"), ("OpenCode", "opencode"),
         ("Claude Code", "claude-code"), ("ChatGPT", "chatgpt"),
         ("Kimi", "kimi"), ("Qwen", "qwen"), ("…", None)]
LOG_FILL = "#edf3f8"
LOG_EDGE = "#698ba5"
BLUE_FILL = "#e1edf5"
GREEN_FILL = "#e5f0eb"

# Panel A: identity assignment is bounded by the labeled reference set L.
text(45, 40, "A", 23, "#315f7d", 700)
text(78, 40, "Prior work: model identification", 22, INK, 700)
line(45, 54, 1627, 54, "#9fb4c9", 1.0)

rect(45, 69, 720, 130, LOG_FILL, LOG_EDGE, 1.0, 2)
text(68, 94, "Agent-generated execution logs", 20, INK, 700)
for i, (label, icon) in enumerate(tools):
    row, col = divmod(i, 4)
    brand_entry(68 + col * 164, 108 + row * 28, label, icon, 18, 19)
text(68, 173, "Tool/harness · model · skills → observed trace", 16, MUTED)
text(68, 191, "Labeled references + target trace", 16, MUTED)

arrow(768, 134, 795, 134, "#315f7d", 1.7)
rect(802, 69, 352, 130, BLUE_FILL, "#698ba5", 1.0, 2)
text(824, 96, "Reference-based classification", 19, INK, 700)
text(824, 123, "match target within label set L", 17, MUTED)
text(824, 157, "ŷ = argmaxₘ∈L s(x, m)", 18, INK, 600)

arrow(1157, 134, 1184, 134, "#315f7d", 1.7)
rect(1191, 69, 436, 130, "#f8fbfd", "#698ba5", 1.0, 2)
text(1214, 99, "Model / harness label", 20, INK, 700)
text(1214, 137, "ŷ ∈ L", 23, INK, 700)
text(1214, 164, "Joint effects obscure component attribution", 16, MUTED)
text(1214, 186, "Unseen configuration → add references", 16, "#a65337", 600)

# Panel B: diverse model and version logs share one event feature set.
text(45, 243, "B", 23, "#287b70", 700)
text(78, 243, "Behavioral profiling", 22, INK, 700)
line(45, 257, 1627, 257, "#9fb4c9", 1.0)

rect(45, 272, 720, 130, LOG_FILL, LOG_EDGE, 1.0, 2)
text(68, 297, "Agent-generated execution logs", 20, INK, 700)
for i, (label, icon) in enumerate(tools):
    row, col = divmod(i, 4)
    brand_entry(68 + col * 164, 311 + row * 28, label, icon, 18, 19)
text(68, 375, "Models  m₁, m₂, …    ·    Versions  v₁, v₂, …", 16, MUTED)
text(68, 394, "Skills / prompts vary across traces", 16, MUTED)

arrow(768, 340, 795, 340, "#315f7d", 1.7)
rect(802, 272, 352, 136, GREEN_FILL, "#7ca99d", 1.0, 2)
text(824, 299, "Shared K1–K8 profile", 20, INK, 700)
text(824, 322, "fixed definitions for compatible logs", 18, MUTED)
const_col_x = 828
cell_gap = 38
for i in range(8):
    text(const_col_x + i * cell_gap + 14, 348, f"K{i+1}", 15, INK, 700, "middle")
matrix = [
    ["#84b4cb", "#c7dde7", "#edf3f8", "#92bfd3", "#d9e8ee", "#aacbd9", "#edf3f8", "#7daec6"],
    ["#dbe9ef", "#8cb8cd", "#c8dde7", "#edf3f8", "#91bed2", "#d7e7ed", "#a6c8d7", "#edf3f8"],
]
for row_index, row in enumerate(matrix):
    y = 356 + row_index * 15
    for col_index, shade in enumerate(row):
        rect(const_col_x + col_index * cell_gap, y, 28, 10, shade, "#9db8ca", .45, .5)

line(1158, 340, 1170, 340, "#315f7d", 1.5)
line(1170, 316, 1170, 376, "#315f7d", 1.1)
arrow(1170, 316, 1184, 316, "#315f7d", 1.5)
arrow(1170, 376, 1184, 376, "#315f7d", 1.5)
rect(1191, 279, 436, 66, "#ffffff", "#9eb8cc", .9, 2)
text(1214, 306, "Recorded-mode score", 19, INK, 700)
text(1214, 330, "Human / Agent estimate", 16, MUTED)
rect(1191, 350, 436, 66, "#ffffff", "#9eb8cc", .9, 2)
text(1214, 377, "Trace-relation score", 19, INK, 700)
text(1214, 401, "pairwise comparison", 16, MUTED)

# Checks and crosses compare the two routes' primary capabilities.
text(45, 450, "Capability comparison", 20, INK, 700)
rect(45, 462, 1582, 34, "#f2f6f9", "none", 0, 0)
line(800, 462, 800, 634, "#d7e0e7", .8)
line(1200, 462, 1200, 634, "#d7e0e7", .8)
text(65, 485, "Capability", 18, MUTED, 700)
text(1000, 485, "Model identification", 18, "#315f7d", 700, "middle")
text(1414, 485, "Behavioral profiling", 18, "#287b70", 700, "middle")
capabilities = [
    ("Assign model / harness identity", True, False),
    ("Apply fixed features across tool/model/version/skill variants", False, True),
    ("Estimate mode and score trace relations", False, True),
]
row_top = 496
row_height = 46
for i, (label, model_id, behavior) in enumerate(capabilities):
    y = row_top + i * row_height
    line(45, y, 1627, y, "#d7e0e7", .8)
    text(65, y + 29, label, 19, INK)
    (check if model_id else cross)(990, y + 10, 20)
    (check if behavior else cross)(1404, y + 10, 20)
line(45, row_top + len(capabilities) * row_height, 1627,
     row_top + len(capabilities) * row_height, "#d7e0e7", .8)
vector_elements = "\n".join(parts[1:])
svg = (
    parts[0]
    + '<g id="editable-vector-redraw">\n'
    + vector_elements
    + '\n</g>\n'
    + '<metadata>Editable vector artwork. Brand marks are bundled in figures/tool-icons; DSH denotes DeepSeek Harness.</metadata>\n'
    + '</svg>\n'
)
SVG.write_text(svg, encoding="utf-8")
subprocess.run(["sips", "-s", "format", "png", str(SVG), "--out", str(PNG)], check=True, capture_output=True, text=True)
print(SVG)
print(PNG)
