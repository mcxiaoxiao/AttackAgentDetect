#!/usr/bin/env python3
"""Build an explanatory, source-backed stacked fingerprint comparison SVG."""
from __future__ import annotations

import html
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from attackagent.fingerprints import (fit_column_scale, matrix,
                                      transform_matrix)

DATA = ROOT / "site/data/behavior-trace-dataset-v2.json"
OUT = ROOT / "site/assets/behavior-fingerprint-pair.svg"
TRACE_A = "7c332b5d14dfbfab6bb8"
TRACE_B = "95cac0293dc1779ac77e"
WIDTH = 1440
GROUPS = [(0, 3, "工具使用", "Tool use"),
          (3, 6, "动作组织", "Action organization"),
          (6, 8, "局部重复", "Local repetition")]


def local_costs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    costs = np.full((left.shape[0], right.shape[0]), np.nan)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            shared = np.isfinite(a) & np.isfinite(b)
            if shared.sum() >= 3:
                costs[i, j] = float(np.mean((a[shared] - b[shared]) ** 2))
    return costs


def dtw_path(costs: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    n, m = costs.shape
    total = np.full((n + 1, m + 1), np.inf)
    length = np.zeros((n + 1, m + 1), dtype=int)
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    total[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local = costs[i - 1, j - 1]
            if not np.isfinite(local):
                continue
            candidates = [(total[i - 1, j - 1], length[i - 1, j - 1], i - 1, j - 1),
                          (total[i - 1, j], length[i - 1, j], i - 1, j),
                          (total[i, j - 1], length[i, j - 1], i, j - 1)]
            best = min(candidates, key=lambda x: (x[0], x[1]))
            if np.isfinite(best[0]):
                total[i, j] = best[0] + local
                length[i, j] = best[1] + 1
                prev[(i, j)] = (best[2], best[3])
    if not length[n, m]:
        raise ValueError("Selected matrices cannot be aligned with the frozen mask rule")
    path, point = [], (n, m)
    while point != (0, 0):
        path.append((point[0] - 1, point[1] - 1))
        point = prev[point]
    path.reverse()
    return path, float(np.sqrt(total[n, m] / length[n, m]))


def lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def color(value: float, limit: float = 1.5) -> str:
    # Shared, centered display scale. Values are not changed for analysis.
    t = max(-1.0, min(1.0, value / limit))
    if t < 0:
        f = -t
        lo, hi = (243, 246, 250), (35, 102, 172)
    else:
        f = t
        lo, hi = (243, 246, 250), (198, 107, 56)
    return "#%02x%02x%02x" % tuple(lerp(lo[i], hi[i], f) for i in range(3))


def polygon(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def cell_title(window: int, feature: int, value: float) -> str:
    text = f"Window {window + 1}, K{feature + 1}: "
    return html.escape(text + (f"z={value:.3f}" if np.isfinite(value) else "missing"))


def draw_layers(parts: list[str], values: np.ndarray, x: float) -> None:
    """Transpose windows onto x and explode the three measured feature groups.

    Both cell corners and layer bounds use the same affine projection. There is
    no empty depth plate and every colored cell corresponds to an observed value.
    """
    windows = len(values)
    cw, dx, dy = 28, 12, 22
    for edge in (0, windows * cw):
        parts.append(f'<path d="M{x + edge},62 L{x + edge + 16},352" class="guide"/>')
    for layer, (start, stop, _, _) in enumerate(GROUPS):
        ox, oy = x + layer * 8, 62 + layer * 145
        depth = stop - start
        def point(w: float, feature: float):
            return ox + w * cw + feature * dx, oy + feature * dy
        corners = [point(0, 0), point(windows, 0),
                   point(windows, depth), point(0, depth)]
        bottom = [corners[3], corners[2],
                  (corners[2][0], corners[2][1] + 5),
                  (corners[3][0], corners[3][1] + 5)]
        parts.append(f'<polygon points="{polygon(bottom)}" class="edge"/>')
        for row, feature in enumerate(range(start, stop)):
            for window in range(windows):
                value = values[window, feature]
                fill = color(float(value)) if np.isfinite(value) else 'url(#missing)'
                corners = [point(window, row), point(window + 1, row),
                           point(window + 1, row + 1), point(window, row + 1)]
                parts.append(f'<polygon points="{polygon(corners)}" fill="{fill}" class="cell"><title>{cell_title(window, feature, value)}</title></polygon>')
            lx, ly = point(0, row + .7)
            parts.append(f'<text x="{lx - 16}" y="{ly}" text-anchor="end" class="axis">K{feature + 1}</text>')
        bounds = [point(0, 0), point(windows, 0), point(windows, depth), point(0, depth)]
        parts.append(f'<polygon points="{polygon(bounds)}" class="outline"/>')
        if layer == 0:
            for window in sorted({0, 3, 7, 11, windows - 1}):
                if window < windows:
                    lx, ly = point(window + .5, 0)
                    parts.append(f'<text x="{lx}" y="{ly - 14}" text-anchor="middle" class="axis">{window + 1:02d}</text>')


def draw_flat(parts: list[str], values: np.ndarray, x: float) -> None:
    cw, ch, y = 28, 29, 53
    for feature in range(8):
        parts.append(f'<text x="{x - 15}" y="{y + feature * ch + 19}" text-anchor="end" class="axis">K{feature + 1}</text>')
        for window in range(len(values)):
            value = values[window, feature]
            fill = color(float(value)) if np.isfinite(value) else 'url(#missing)'
            parts.append(f'<rect x="{x + window * cw}" y="{y + feature * ch}" width="{cw}" height="{ch}" fill="{fill}" class="cell"><title>{cell_title(window, feature, value)}</title></rect>')
    for boundary in (3, 6):
        yy = y + boundary * ch
        parts.append(f'<path d="M{x},{yy} H{x + len(values) * cw}" class="group-rule"/>')
    for window in sorted({0, 3, 7, 11, len(values) - 1}):
        if window < len(values):
            parts.append(f'<text x="{x + (window + .5) * cw}" y="{y - 14}" text-anchor="middle" class="axis">{window + 1:02d}</text>')


def write_view(left: np.ndarray, right: np.ndarray, *, flat: bool, english: bool) -> pathlib.Path:
    height = 350 if flat else 490
    suffix = ('-flat' if flat else '') + ('-en' if english else '')
    output = OUT.with_stem(OUT.stem + suffix)
    title = 'Two source-backed behavioral fingerprint matrices'
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">
<title id="title">{title}</title>
<desc id="desc">Each colored cell is one feature in one consecutive command window. The same values appear in the layered and flat views. Three groups organize K1–K8; they are not attack stages. Trace A has 16 windows and trace B has 13.</desc>
<defs><style>
:root {{ --ink: #18283b; --muted: #596779; --rule: #bac9d8; --guide: #d5e0eb; --paper: #f5f9fd; --edge: #d5e0eb; --label-font: 'IBM Plex Sans', 'PingFang SC', 'Microsoft YaHei', sans-serif; }}
text {{ font-family: var(--label-font); fill: var(--ink); font-variant-numeric: tabular-nums; }}
.axis {{ font-size: 14px; fill: var(--muted); }}
.group {{ font-size: 20px; font-weight: 600; text-anchor: middle; }}
.group-id {{ font-size: 13px; fill: var(--muted); text-anchor: middle; }}
.cell {{ stroke: var(--paper); stroke-width: 1.4; }}
.outline {{ fill: none; stroke: var(--rule); stroke-width: 1; }}
.edge {{ fill: var(--edge); }}
.guide {{ fill: none; stroke: var(--guide); stroke-width: 1; }}
.group-rule {{ fill: none; stroke: var(--paper); stroke-width: 4; }}
</style>
<pattern id="missing" width="6" height="6" patternUnits="userSpaceOnUse"><rect width="6" height="6" fill="#e0e7ef"/><path d="M0,6 L6,0" stroke="#bac9d8" stroke-width="1"/></pattern>
</defs>''']
    draw = draw_flat if flat else draw_layers
    draw(parts, left, 68)
    draw(parts, right, 798)
    for index, (start, stop, zh, en) in enumerate(GROUPS):
        yy = (53 + ((start + stop) / 2) * 29) if flat else (62 + index * 145 + 29)
        parts.append(f'<text x="687" y="{yy}" class="group">{en if english else zh}</text>')
        parts.append(f'<text x="687" y="{yy + 23}" class="group-id">K{start + 1}–K{stop}</text>')
    axis_label = 'Consecutive command windows' if english else '连续命令窗口'
    foot = 330 if flat else 461
    for xx in (292, 1022):
        parts.append(f'<text x="{xx}" y="{foot}" text-anchor="middle" class="axis">{axis_label}</text>')
    parts.append('</svg>')
    output.write_text('\n'.join(parts) + '\n')
    return output


def main() -> None:
    doc = json.loads(DATA.read_text())
    cth = [r for r in doc["records"]
           if r["source_dataset"] == "cyber-task-horizons"
           and len(r.get("commands") or []) >= 32]
    matrices = {}
    for row in cth:
        value, _ = matrix(row["commands"], min_commands=16, max_commands=128)
        if value is not None:
            matrices[row["trace_id"]] = value
    by_id = {r["trace_id"]: r for r in cth}
    if TRACE_A not in matrices or TRACE_B not in matrices:
        raise RuntimeError("The two source-backed example traces are no longer eligible")
    center, scale = fit_column_scale(list(matrices.values()))
    left = transform_matrix(matrices[TRACE_A], center, scale)
    right = transform_matrix(matrices[TRACE_B], center, scale)
    costs = local_costs(left, right)
    path, distance = dtw_path(costs)

    a, b = by_id[TRACE_A], by_id[TRACE_B]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    outputs = [str(write_view(left, right, flat=flat, english=english))
               for flat in (False, True) for english in (False, True)]
    print(json.dumps({"outputs": outputs, "trace_a": TRACE_A, "trace_b": TRACE_B,
                      "task_a": a["task_id"], "task_b": b["task_id"],
                      "shape_a": list(left.shape), "shape_b": list(right.shape),
                      "display_pair_distance": distance, "window_pairs_on_path": len(path)}, indent=2))


if __name__ == "__main__":
    main()
