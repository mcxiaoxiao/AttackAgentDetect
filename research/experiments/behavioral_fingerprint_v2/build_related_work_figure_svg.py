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

parts = ["""<svg xmlns="http://www.w3.org/2000/svg" width="3204" height="1360" viewBox="35 0 1602 680" role="img" aria-labelledby="title desc">
<title id="title">Model identification and behavioral profiling routes</title>
<desc id="desc">Agent traces jointly reflect tool, model, task, and skill effects. Route A assigns only within known identity references. Route B reuses a shared behavioral fingerprint across compatible logs and offers relative similarity cues for analyst-led association review.</desc>
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


def hacker_icon(x: float, y: float) -> None:
    """Small illustrative hooded-computer-user glyph for the out-of-set box."""
    parts.append(
        f'<g id="unknown-hacker-icon" transform="translate({x} {y})" aria-label="Illustrative hacker icon">'
        '<path d="M22 2 C11 2 4 10 4 21 L4 28 C8 21 14 18 22 18 '
        'C30 18 36 21 40 28 L40 21 C40 10 33 2 22 2 Z" fill="#06172f"/>'
        '<path d="M12 21 C12 14 16 9 22 9 C28 9 32 14 32 21 L32 24 '
        'C26 21 18 21 12 24 Z" fill="#f0c45b"/>'
        '<circle cx="18" cy="18" r="1.4" fill="#06172f"/>'
        '<circle cx="26" cy="18" r="1.4" fill="#06172f"/>'
        '<path d="M2 53 C4 39 11 31 22 31 C33 31 40 39 42 53 Z" fill="#06172f"/>'
        '<rect x="8" y="38" width="28" height="15" rx="2" fill="#fff8e7" stroke="#06172f" stroke-width="1.2"/>'
        '<rect x="11" y="41" width="22" height="8" rx="1" fill="#dceaf7"/>'
        '<path d="M6 54 H38" fill="none" stroke="#06172f" stroke-width="2" stroke-linecap="round"/>'
        '</g>'
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


# Two complementary routes with matched log inputs and a capability comparison.
# The route labels describe different targets rather than ranking effectiveness.
tools = [("Codex", "codex"), ("DSH", "dsh"), ("OpenCode", "opencode"),
         ("Claude Code", "claude-code"), ("ChatGPT", "chatgpt"),
         ("Kimi", "kimi"), ("Qwen", "qwen"), ("…", None)]
LOG_FILL = "#edf3f8"
LOG_EDGE = "#698ba5"
BLUE_FILL = "#e1edf5"
GREEN_FILL = "#e5f0eb"

# Panel A: identity assignment is bounded by the labeled reference set L.
text(45, 40, "A", 23, "#315f7d", 700)
text(78, 40, "Route A · reference-based identity", 22, INK, 700)
line(45, 54, 1627, 54, "#9fb4c9", 1.0)

rect(45, 69, 540, 130, LOG_FILL, LOG_EDGE, 1.0, 8)
text(68, 94, "Agent-generated execution logs", 20, INK, 700)
for i, (label, icon) in enumerate(tools):
    row, col = divmod(i, 4)
    brand_entry(68 + col * 118, 108 + row * 28, label, icon, 18, 16)
text(68, 173, "Tool/harness · model · skills → observed trace", 16, MUTED)
text(68, 191, "Labeled references + target trace", 16, MUTED)

arrow(595, 134, 650, 134, "#315f7d", 1.7)
rect(660, 69, 270, 130, BLUE_FILL, "#698ba5", 1.0, 8)
text(682, 96, "Reference-based match", 18, INK, 700)
text(682, 123, "within known label set L", 15, MUTED)
text(682, 157, "ŷ = argmaxₘ∈L s(x, m)", 16, INK, 600)

arrow(940, 134, 995, 134, "#315f7d", 1.7)
rect(1005, 69, 622, 130, "#f8fbfd", "#698ba5", 1.0, 8)
text(1028, 96, "Known-set identity assignment", 19, INK, 700)
parts.append('<rect x="1028" y="111" width="350" height="74" rx="8" fill="#ffffff" stroke="#8ea6ba" stroke-width="1.2" stroke-dasharray="6 4"/>')
text(1044, 132, "L · labeled model / harness references", 14, MUTED, 600)
for i, (label, shade) in enumerate([
    ("m₁", "#d8e8f7"), ("m₂", "#b8d2ef"), ("m₃", "#e7f1fd"),
    ("m₄", "#c5daf0"), ("m₅", "#e0edf9"), ("m₆", "#d0e2f5"), ("…", "#e7f1fd"),
]):
    x = 1044 + i * 45
    rect(x, 145, 40, 26, shade, BLUE_EDGE, .8, 13)
    text(x + 20, 163, label, 12, BLUE, 700, "middle")
parts.append('<rect x="1392" y="111" width="216" height="74" rx="8" fill="#fff8f3" stroke="#d8a18b" stroke-width="1.1"/>')
hacker_icon(1396, 126)
text(1528, 132, "UNKNOWN", 15, RUST, 700, "middle")
text(1528, 151, "outside L", 13, MUTED, 600, "middle")
text(1528, 167, "e.g., unseen task / challenge", 9, "#9b7200", 600, "middle")
text(1528, 180, "model / Agent not registered", 9, "#9b7200", 600, "middle")

# Panel B: diverse model and version logs share one event feature set.
text(45, 243, "B", 23, "#287b70", 700)
text(78, 243, "Route B (this study) · behavioral fingerprint", 22, INK, 700)
line(45, 257, 1627, 257, "#9fb4c9", 1.0)

rect(45, 272, 540, 130, LOG_FILL, LOG_EDGE, 1.0, 8)
text(68, 297, "Agent-generated execution logs", 20, INK, 700)
for i, (label, icon) in enumerate(tools):
    row, col = divmod(i, 4)
    brand_entry(68 + col * 118, 311 + row * 28, label, icon, 18, 16)
text(68, 375, "Models, versions, tools, and skills vary", 15, MUTED)
text(68, 394, "Model / harness identity may be unknown", 15, MUTED)

arrow(595, 340, 650, 340, "#315f7d", 1.7)
rect(660, 272, 270, 136, GREEN_FILL, "#7ca99d", 1.0, 8)
text(682, 299, "Shared B/S/T fingerprint", 17, INK, 700)
text(682, 321, "same feature space across configurations", 12, MUTED)
text(682, 341, "identity may remain unknown", 13, GREEN, 600)
blocks = [
    ("B", ["#8eb7cf", "#b7d0dd", "#dce8ee"]),
    ("S", ["#83b8a7", "#b4d4c9", "#deece6"]),
    ("T", ["#aab0b7", "#c8cdd2", "#e4e7ea"]),
]
for block_index, (name, shades) in enumerate(blocks):
    x = 685 + block_index * 79
    text(x + 21, 359, name, 14, INK, 700, "middle")
    for row_index in range(3):
        for col_index in range(2):
            rect(x + col_index * 22, 365 + row_index * 11, 17, 7,
                 shades[(row_index + col_index) % len(shades)], "#9db8ca", .45, 1.2)
line(930, 340, 950, 340, "#315f7d", 1.5)
line(950, 304, 950, 374, "#315f7d", 1.1)
arrow(950, 304, 995, 304, "#315f7d", 1.5)
arrow(950, 374, 995, 374, "#315f7d", 1.5)
rect(1005, 272, 622, 64, "#ffffff", "#9eb8cc", .9, 8)
text(1028, 298, "Recorded-mode cue", 19, INK, 700)
text(1028, 321, "Human / Agent context", 16, MUTED)
rect(1340, 285, 92, 29, BLUE, BLUE, .8, 15)
text(1386, 304, "Human", 13, WHITE, 700, "middle")
rect(1444, 285, 92, 29, BLUE_LIGHT, BLUE_EDGE, .8, 15)
text(1490, 304, "Agent", 13, BLUE, 700, "middle")

# Relative trace similarity is illustrative evidence for analyst review.
rect(1005, 340, 622, 68, "#ffffff", "#9eb8cc", .9, 8)
text(1028, 367, "Trace similarity & relation cues", 17, INK, 700)
text(1028, 391, "Candidate links for analyst review", 14, MUTED)
query_node = (1350, 374)
candidate_nodes = [
    (1420, 352, "A", .92, 3.1),
    (1485, 360, "B", .72, 2.7),
    (1550, 352, "C", .52, 2.3),
    (1420, 396, "D", .40, 2.1),
    (1485, 388, "E", .27, 1.9),
    (1550, 396, "F", .18, 1.7),
]
for node_x, node_y, _, opacity, width in candidate_nodes:
    parts.append(
        f'<path d="M{query_node[0]} {query_node[1]} L{node_x} {node_y}" '
        f'fill="none" stroke="#1760d5" stroke-opacity="{opacity}" '
        f'stroke-width="{width}" stroke-linecap="round"/>'
    )
nodes = [(query_node[0], query_node[1], "q", BLUE, BLUE, WHITE, 1.5)]
nodes.extend((x, y, label, GREEN, GREEN, WHITE, 3.5)
             for x, y, label, _, _ in candidate_nodes)
for node_x, node_y, label, fill, stroke, color, baseline_offset in nodes:
    parts.append(f'<circle cx="{node_x}" cy="{node_y}" r="9" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    parts.append(
        f'<text x="{node_x - 1.0}" y="{node_y + baseline_offset}" font-family="Arial,Helvetica,sans-serif" '
        f'font-size="10" font-weight="700" fill="{color}" text-anchor="middle">{escape(label)}</text>'
    )

# Checks and crosses compare the two routes' primary capabilities.
text(45, 450, "Capability comparison", 20, INK, 700)
rect(45, 462, 1582, 34, "#f2f6f9", "none", 0, 0)
line(800, 462, 800, 634, "#d7e0e7", .8)
line(1200, 462, 1200, 634, "#d7e0e7", .8)
text(65, 485, "Capability", 18, MUTED, 700)
text(1000, 485, "Route A · identity", 18, "#315f7d", 700, "middle")
text(1414, 485, "Route B (this study)", 18, "#287b70", 700, "middle")
capabilities = [
    ("Assign model / harness identity", True, False),
    ("Use common B/S/T measures on compatible logs", False, True),
    ("Support similarity and trace-relation review", False, True),
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
