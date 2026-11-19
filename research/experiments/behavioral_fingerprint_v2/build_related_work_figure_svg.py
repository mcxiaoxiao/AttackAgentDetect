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

parts = ["""<svg xmlns="http://www.w3.org/2000/svg" width="3344" height="1840" viewBox="0 0 1672 920" role="img" aria-labelledby="title desc">
<title id="title">Model identification and behavioral profiling routes</title>
<desc id="desc">A comparison of model identification and behavioral profiling from Agent-generated logs, with a capability matrix.</desc>
<defs>
 <linearGradient id="headA" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#e4f1fc"/><stop offset="1" stop-color="#d4e9fb"/></linearGradient>
 <linearGradient id="headB" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#e7f5f3"/><stop offset=".52" stop-color="#e9f1fb"/><stop offset="1" stop-color="#edf1f5"/></linearGradient>
 <linearGradient id="matrixFill" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stop-color="#eff7fc"/><stop offset="1" stop-color="#dcecf7"/></linearGradient>
 <linearGradient id="ecosystemFill" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#e4f7f0"/><stop offset="1" stop-color="#e2f1ec"/></linearGradient>
 <marker id="arrowTip" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#174b78"/></marker>
</defs>
<rect width="1672" height="920" fill="#ffffff"/>
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


# A and B show the input and output of each method class. Brand marks are
# inline without app-style cards; restrained fills retain the original palette.
tools = [("Codex", "codex"), ("DSH", "dsh"), ("OpenCode", "opencode"),
         ("Claude Code", "claude-code"), ("ChatGPT", "chatgpt"),
         ("Kimi", "kimi"), ("Qwen", "qwen"), ("…", None)]

# Panel A: model identification from labeled Agent traces.
rect(23, 20, 1626, 230, WHITE, "#829ab0", 1.1, 4)
parts.append('<path d="M33 20 H1639 Q1649 20 1649 30 V60 H23 V30 Q23 20 33 20 Z" fill="url(#headA)"/>')
text(43, 49, "A", 25, NAVY, 700)
text(94, 49, "Prior work: model identification", 23, INK, 700)

rect(64, 72, 765, 142, "#f8fbfe", "#698ba5", 1.0, 3)
text(88, 101, "Agent-generated execution logs", 22, INK, 700)
text(88, 127, "labeled references + target execution trace", 19, MUTED)
for i, (label, icon) in enumerate(tools):
    row, col = divmod(i, 4)
    brand_entry(96 + col * 176, 137 + row * 32, label, icon, 19, 17)

arrow(833, 143, 893, 143, "#315f7d", 1.9)
rect(900, 72, 711, 142, "#ffffff", "#698ba5", 1.0, 3)
text(922, 101, "Model-family identification", 22, INK, 700)
text(922, 127, "labeled references → target label", 19, MUTED)
rect(922, 148, 667, 48, "#e9f3fb", "#bfd5e6", .8, 2)
text(1255, 179, "Model / harness label", 20, INK, 700, "middle")

# Panel B: behavioral profiling from the same Agent log sources.
rect(23, 268, 1626, 358, WHITE, "#829ab0", 1.1, 4)
parts.append('<path d="M33 268 H1639 Q1649 268 1649 278 V308 H23 V278 Q23 268 33 268 Z" fill="url(#headB)"/>')
text(43, 297, "B", 25, NAVY, 700)
text(94, 297, "Behavioral profiling", 23, INK, 700)

rect(85, 321, 1500, 80, "url(#ecosystemFill)", "#b9d8cb", .8, 4)
text(111, 354, "Agent-generated logs", 20, INK, 700)
text(111, 379, "one trace per Agent", 17, MUTED)
for i, (label, icon) in enumerate(tools):
    row, col = divmod(i, 4)
    brand_entry(438 + col * 276, 333 + row * 33, label, icon, 19, 17)

# The source logs are pooled into an event representation, then profiled.
for x in (566, 842, 1118, 1394):
    line(x, 401, x, 416, TEAL, 1.4)
line(251, 416, 1394, 416, TEAL, 1.4)
arrow(251, 416, 251, 439, TEAL, 1.7)

rect(49, 446, 405, 74, "#ffffff", "#698ba5", 1.0, 3)
text(75, 477, "Execution-event log", 20, INK, 700)
text(75, 502, "ordered executable events", 17, MUTED)
arrow(456, 483, 512, 483, "#315f7d", 1.9)

rect(521, 434, 561, 151, "url(#matrixFill)", "#527b99", 1.1, 3)
text(545, 460, "K1–K8 behavioral profile", 20, INK, 700)
text(545, 483, "one row per consecutive event window", 17, MUTED)
const_col_x = 657
cell_gap = 47
for i in range(8):
    text(const_col_x + i * cell_gap + 18, 509, f"K{i+1}", 17, INK, 700, "middle")
matrix = [
    ["#c7dce8", "#83b1d0", "#e7f1f6", "#cbdfea", "#91bcd7", "#e4eef4", "#c5dbe7", "#8fb8d1"],
    ["#f7fbfd", "#d8e7ef", "#d0e2ec", "#f7fbfd", "#d7e7ef", "#c5dce8", "#f7fbfd", "#d8e7ef"],
    ["#8db9d4", "#d1e3ed", "#f7fbfd", "#7eafd0", "#c6dce8", "#f7fbfd", "#8ab7d3", "#c8dfe9"],
]
for row_index, row in enumerate(matrix):
    y = 519 + row_index * 19
    for col_index, shade in enumerate(row):
        rect(const_col_x + col_index * cell_gap, y, 38, 15, shade, "#9db8ca", .55, .5)

line(1085, 510, 1138, 510, "#315f7d", 1.7)
line(1138, 469, 1138, 545, "#315f7d", 1.35)
arrow(1138, 469, 1171, 469, "#315f7d", 1.7)
arrow(1138, 545, 1171, 545, "#315f7d", 1.7)
rect(1178, 438, 444, 63, "#ffffff", "#9eb8cc", 1.0, 3)
text(1200, 465, "Recorded-mode estimate", 20, INK, 700)
text(1200, 487, "Human / Agent score", 17, MUTED)
rect(1178, 514, 444, 63, "#ffffff", "#9eb8cc", 1.0, 3)
text(1200, 541, "Trace-relation score", 20, INK, 700)
text(1200, 563, "pairwise comparison", 17, MUTED)

# Capability matrix: checks and crosses compare the methods directly.
rect(23, 645, 1626, 252, WHITE, "#9fb4c9", 1.0, 3)
rect(24, 646, 1624, 47, "#f2f6f9", "none", 0, 4)
line(850, 646, 850, 897, "#d7e0e7", .9)
line(1245, 646, 1245, 897, "#d7e0e7", .9)
text(52, 677, "Capability", 19, MUTED, 700)
text(1065, 677, "Model identification", 19, "#315f7d", 700, "middle")
text(1420, 677, "Behavioral profiling", 19, "#287b70", 700, "middle")
capabilities = [
    ("Assign a model / harness label", True, False),
    ("Estimate recorded execution mode", False, True),
    ("Construct a K1–K8 behavioral profile", False, True),
    ("Score relations between traces", False, True),
]
row_top = 694
row_height = 50
for i, (label, model_id, behavior) in enumerate(capabilities):
    y = row_top + i * row_height
    if i:
        line(42, y, 1630, y, "#d7e0e7", 1.0)
    text(52, y + 32, label, 20, INK)
    (check if model_id else cross)(1052, y + 12, 20)
    (check if behavior else cross)(1408, y + 12, 20)
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
