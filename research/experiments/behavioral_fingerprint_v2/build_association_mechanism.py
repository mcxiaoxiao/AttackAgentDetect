#!/usr/bin/env python3
"""Explain conditioned association with one traceable, already-scored pair.

No models are refit and no aggregate results are changed. The example is the
median acceptance margin among cross-mode CTH positives corrected from a missed
link to an accepted link in the purged source-held-out comparison.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "research/results/association_fingerprint_v3"
PAPER = ROOT / "research/paper/figures"
SITE = ROOT / "site/assets/behavioral-fingerprint-v2"


def select_case() -> dict:
    registry_path = ROOT / "site/data/behavior-trace-dataset-v2.json"
    registry = {r["trace_id"]: r for r in json.loads(registry_path.read_text())["records"]}
    with (RESULTS / "ba_leave_source_out_purged_pairs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    pairs: dict = {}
    for row in rows:
        if row["target"] == "cth":
            pairs.setdefault((row["query"], row["candidate"]), {})[row["method"]] = row
    eligible = []
    for key, models in pairs.items():
        evidence, full = models.get("evidence"), models.get("gated_components")
        if not evidence or not full:
            continue
        if evidence["same_task"] != "1" or evidence["predicted_link"] != "0" or full["predicted_link"] != "1":
            continue
        if registry[key[0]]["mode"] == registry[key[1]]["mode"]:
            continue
        eligible.append((float(full["score"]) - float(full["threshold"]), key, evidence, full))
    eligible.sort(key=lambda item: (item[0], item[1]))
    _, key, evidence, full = eligible[len(eligible) // 2]
    decomposition = sum(float(full[k]) for k in (
        "root_probability", "evidence_contribution", "B_contribution", "S_contribution"
    ))
    assert math.isclose(decomposition, float(full["score"]), abs_tol=1e-10)
    return {
        "selection": "Median full-model acceptance margin among corrected same-task, cross-mode CTH pairs; tie-break by trace IDs",
        "eligible_corrected_pairs": len(eligible),
        "predictions_file": "research/results/association_fingerprint_v3/ba_leave_source_out_purged_pairs.csv",
        "query": {k: registry[key[0]][k] for k in ("trace_id", "task_group", "mode", "commands", "command_count")},
        "candidate": {k: registry[key[1]][k] for k in ("trace_id", "task_group", "mode", "commands", "command_count")},
        "evidence_model": evidence,
        "fingerprint_model": full,
    }


def render(case: dict) -> None:
    from paper_style import apply, save, PAGE_WIDTH, INK, BLUE, TEAL, MUTED, GRID
    apply()
    fig, (left, right) = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.55),
                                      gridspec_kw={"width_ratios": [1.63, 1]})
    fig.subplots_adjust(left=.015, right=.985, top=.88, bottom=.16, wspace=.29)
    left.set(xlim=(0, 1), ylim=(0, 1))
    left.axis("off")
    left.set_title("a  Construction of the pair representation", loc="left", weight="bold", pad=9)
    for y, key, label in [(.94, "query", r"$x_q$"), (.81, "candidate", r"$x_c$")]:
        names = ", ".join(case[key]["commands"][:8]) + ", …"
        left.text(.015, y, label, fontsize=9, va="center")
        left.text(.105, y, names, fontsize=7.1, family="DejaVu Sans Mono", va="center")
    for x in (.24, .73):
        left.annotate("", xy=(x, .65), xytext=(x, .73),
                      arrowprops={"arrowstyle": "->", "lw": .7, "color": MUTED})
    left.text(.24, .57, r"$E=(E_{\rm cmd},E_{\rm act})$", ha="center", fontsize=9)
    left.text(.73, .57, r"$d=|z_q-z_c|$", ha="center", fontsize=9)
    for x in (.24, .73):
        left.annotate("", xy=(x, .425), xytext=(x, .52),
                      arrowprops={"arrowstyle": "->", "lw": .7, "color": MUTED})
    left.text(.5, .37, r"$\phi(q,c)=[E,\ E_{\rm cmd}d_B,\ E_{\rm cmd}d_S]$",
              ha="center", fontsize=10)
    # The 50 cells encode dimensions, not measured feature values.
    for j in range(50):
        color = MUTED if j < 2 else BLUE if j < 26 else TEAL
        left.add_patch(Rectangle((.025 + j * .019, .225), .019, .092,
                                 facecolor=color, alpha=.24, edgecolor="white", lw=.3))
    for x, label, color in [(.044, "E (2)", MUTED), (.29, "B interactions (24)", BLUE),
                            (.747, "S interactions (24)", TEAL)]:
        left.text(x, .16, label, ha="center", fontsize=6.8, color=color)
    left.text(.5, -.065, r"$\phi\in\mathbb{R}^{50}\quad\longrightarrow\quad f_\theta(\phi)=s\quad\longrightarrow\quad s\geq\tau$",
              ha="center", fontsize=9)

    right.set_title("b  One held-out decision", loc="left", weight="bold", pad=9)
    full = case["fingerprint_model"]
    root, score, threshold = (float(full[k]) for k in ("root_probability", "score", "threshold"))
    right.plot([-.3, .3], [root, root], color=INK, lw=1.25)
    prev = root
    for j, (field, label, color) in enumerate([
        ("evidence_contribution", "E", INK), ("B_contribution", "B", BLUE),
        ("S_contribution", "S", TEAL)], start=1):
        delta = float(full[field]); new = prev + delta
        right.plot([j - .7, j - .34], [prev, prev], color=GRID, lw=.75)
        right.bar(j, delta, bottom=prev, width=.65, color=color, alpha=.8, linewidth=0)
        right.text(j, new + .007, f"{delta:+.3f}", ha="center", fontsize=7.2, color=color)
        prev = new
    right.axhline(threshold, color=MUTED, ls=(0, (3, 2)), lw=.8)
    right.text(-.38, threshold + .003, f"τ = {threshold:.3f}", fontsize=7, color=MUTED)
    right.scatter(3.55, score, s=17, marker="D", color=TEAL, zorder=4)
    right.text(3.55,  .669, f"s = {score:.3f}", ha="center", fontsize=7.0, weight="bold", color=TEAL)
    right.set(xlim=(-.5, 3.9), ylim=(.48, .68), xticks=[0, 1, 2, 3],
              xticklabels=["Root", "E", "B", "S"], yticks=[.50, .55, .60, .65], ylabel="Tree score")
    right.tick_params(length=2.5, width=.6)
    right.grid(axis="y", color=GRID, lw=.4, alpha=.6)
    right.set_axisbelow(True)
    save(fig, "fingerprint-conditioned-association", (PAPER, SITE))


if __name__ == "__main__":
    selected = select_case()
    out = ROOT / "research/results/behavioral_fingerprint_v2/association_mechanism_case.json"
    out.write_text(json.dumps(selected, indent=2) + "\n")
    render(selected)
    print("Rendered conditioned association schematic with a traceable held-out example")
