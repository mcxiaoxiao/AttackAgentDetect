#!/usr/bin/env python3
"""Render manuscript and site summaries from the frozen matched-v2 analyses."""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "research/results/matched_fingerprint_v2"
FIVE_PATH = RESULTS / "five_share/five_share_analysis.json"
ABLATION_PATH = RESULTS / "unified_ablation/unified_bst_ablation.json"
NYU_PATH = RESULTS / "independent_nyu_agent/analysis.json"
OUT = RESULTS / "paper_assets"
PAPER_FIGS = ROOT / "research/paper/figures"
SITE_ASSETS = ROOT / "site/assets/matched-fingerprint-v2"
SITE_PAGE = ROOT / "site/behavior-based-trace-association.html"

COLORS = {"E": "#526579", "BS": "#C36B39", "positive": "#19766F", "negative": "#A94C55", "muted": "#697786"}
SHARES = [0, 25, 50, 75, 100]
MODELS = [("svm", "RBF-SVM"), ("tree", "Decision tree"), ("xgb", "XGBoost")]
METRICS = [("ba", "Balanced accuracy (%)", (42, 70)), ("fmr", "False merge rate (%)", (0, 35)), ("fsr", "False split rate (%)", (45, 100))]


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(x):
    return 100 * float(x)


def fmt(x, digits=2):
    return f"{pct(x):.{digits}f}"


def fmt_pp(x, digits=2, signed=True):
    value = pct(x)
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def latex_num(x, digits=2):
    value = pct(x)
    return f"{value:+.{digits}f}" if value > 0 else f"{value:.{digits}f}"


def latex_ci(entry, key="ba", is_delta=False):
    m = entry["metrics"][key]
    ci = entry["ci95"][key]
    point = latex_num(m) if is_delta else f"{pct(m):.2f}"
    lo, hi = ((latex_num(ci[0]), latex_num(ci[1])) if is_delta
              else (f"{pct(ci[0]):.2f}", f"{pct(ci[1]):.2f}"))
    return rf"${point}\;[{lo},{hi}]$"


def html_ci(entry, key="ba", is_delta=False, unit="%"):
    point = pct(entry["metrics"][key])
    lo, hi = (pct(v) for v in entry["ci95"][key])
    ps = f"{point:+.2f}" if is_delta else f"{point:.2f}"
    return f'<span class="effect-value">{ps}{unit}</span> <span class="effect-interval">[{lo:+.2f}, {hi:+.2f}]</span>'


def delta_ci_html(entry, key):
    value = pct(entry["metrics"][key])
    css = "effect-positive" if value > 0 else "effect-negative" if value < 0 else ""
    lo, hi = (pct(v) for v in entry["ci95"][key])
    return f'<span class="{css}">{value:+.2f}</span> <span class="effect-interval">[{lo:+.2f}, {hi:+.2f}]</span>'


def absolute_html(entry, key):
    return f'{pct(entry["metrics"][key]):.2f}'


def normalized_inputs(five, ablation, nyu):
    return {
        "sources": {
            "five_share": {"path": str(FIVE_PATH.relative_to(ROOT)), "sha256": sha(FIVE_PATH)},
            "unified_ablation": {"path": str(ABLATION_PATH.relative_to(ROOT)), "sha256": sha(ABLATION_PATH)},
            "independent_nyu": {"path": str(NYU_PATH.relative_to(ROOT)), "sha256": sha(NYU_PATH)},
        },
        "cth_five_share_equal_weight": five["five_share_equal_weight"],
        "cth_by_share": five["metrics_by_share_and_pair_type"],
        "cth_pair_type_equal_weight": five["pair_type_equal_weight_over_available_shares"],
        "cth_error_transitions": five["error_transitions"],
        "ablation": ablation["results"],
        "ablation_counts": ablation["task_pair_counts"],
        "nyu_absolute": nyu["absolute_metrics"],
        "nyu_paired_delta": nyu["paired_E_BS_minus_E"],
        "nyu_error_transitions": nyu["error_transitions"],
        "protocols": {
            "cth": "17 task groups; five exact role shares; 54,400 candidate-pair decisions per learner and feature arm across shares; same-task retrieval proxy.",
            "ait": "109 runs across 12 task groups; 43,600 candidate-pair decisions per configuration; same troubleshooting task proxy; observed launch times only.",
            "nyu": "160 previously unused challenge groups and 585 Agent runs from four D-CIPHER snapshots; AA only; 234,000 candidate-pair decisions per learner and feature arm; one frozen evaluation.",
        },
    }


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A9B3BD")
    ax.spines["bottom"].set_color("#A9B3BD")
    ax.tick_params(axis="both", labelsize=8, colors="#344454", length=3)
    ax.grid(axis="y", color="#D8DEE4", linewidth=0.55, zorder=0)


def save_figure(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=320 if ext == "png" else None, facecolor="white")
        if ext == "svg":
            # Matplotlib uses trailing spaces to separate continued SVG path
            # commands. XML whitespace includes the following newline, so trim
            # the redundant spaces while keeping the vector paths unchanged.
            svg = path.read_text(encoding="utf8")
            path.write_text("\n".join(line.rstrip(" \t") for line in svg.splitlines()) + "\n", encoding="utf8")
        shutil.copy2(path, PAPER_FIGS / path.name)
        shutil.copy2(path, SITE_ASSETS / path.name)
    plt.close(fig)


def plot_cth_share(five):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 3, figsize=(7.25, 5.75), sharex=True)
    x = np.asarray(SHARES, dtype=float)
    share_keys = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    for r, (model, title) in enumerate(MODELS):
        for c, (metric, metric_title, ylim) in enumerate(METRICS):
            ax = axes[r, c]
            for version, label, color, marker in (("evidence", "E", COLORS["E"], "o"), ("BS", "E + B/S", COLORS["BS"], "s")):
                vals, low, high = [], [], []
                for sk in share_keys:
                    item = five["metrics_by_share_and_pair_type"][model]["ALL"][sk][version]
                    vals.append(pct(item["metrics"][metric]))
                    low.append(pct(item["ci95"][metric][0]))
                    high.append(pct(item["ci95"][metric][1]))
                vals, low, high = map(np.asarray, (vals, low, high))
                ax.fill_between(x, low, high, color=color, alpha=0.11, linewidth=0, zorder=1)
                ax.plot(x, vals, color=color, marker=marker, markersize=3.4, linewidth=1.5, label=label, zorder=3)
            ax.set_ylim(*ylim)
            ax.set_xticks(SHARES)
            if r == 0:
                ax.set_title(metric_title, fontsize=8.2, weight="semibold", pad=8, color="#253746")
            if c == 0:
                ax.set_ylabel(f"{title}\n%", fontsize=8.1, weight="semibold", color="#253746", labelpad=4)
            if r == 2:
                ax.set_xlabel("Agent roles (%)", fontsize=7.5, labelpad=4)
            style_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.018), ncol=2, frameon=False, fontsize=8, handlelength=2.1, columnspacing=2.0)
    fig.subplots_adjust(left=.105, right=.995, top=.92, bottom=.11, wspace=.33, hspace=.29)
    save_figure(fig, "cth-five-share-matched")


def plot_delta_by_share(five):
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45), sharey=True)
    x = np.asarray(SHARES, dtype=float)
    keys = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    for ax, (model, title) in zip(axes, MODELS):
        d = [five["metrics_by_share_and_pair_type"][model]["ALL"][k]["delta_BS_minus_E"] for k in keys]
        y = np.asarray([pct(v["metrics"]["ba"]) for v in d])
        lo = np.asarray([pct(v["ci95"]["ba"][0]) for v in d])
        hi = np.asarray([pct(v["ci95"]["ba"][1]) for v in d])
        ax.axhline(0, color="#697786", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
        ax.fill_between(x, lo, hi, color=COLORS["positive"], alpha=.10, linewidth=0, zorder=1)
        ax.errorbar(x, y, yerr=[y-lo, hi-y], color=COLORS["positive"], marker="o", markersize=3.5, linewidth=1.4, capsize=2.4, zorder=3)
        ax.set_title(title, fontsize=8.5, weight="semibold", color="#253746")
        ax.set_xticks(SHARES)
        ax.set_xlabel("Agent roles (%)", fontsize=7.5)
        style_axes(ax)
    axes[0].set_ylabel("Δ balanced accuracy (pp)\nE + B/S minus E", fontsize=7.6, labelpad=6)
    fig.subplots_adjust(left=.12, right=.99, top=.87, bottom=.22, wspace=.30)
    save_figure(fig, "cth-five-share-delta-ba")


def plot_pair_type(five):
    entry = five["pair_type_equal_weight_over_available_shares"]["xgb"]
    types = ["HH", "HA", "AA"]
    labels = ["Human–Human", "Human–Agent", "Agent–Agent"]
    metrics = [("ba", "Δ BA (pp)"), ("fmr", "Δ FMR (pp)"), ("fsr", "Δ FSR (pp)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.9))
    for ax, (metric, title) in zip(axes, metrics):
        y = np.arange(len(types))
        for i, typ in enumerate(types):
            block = entry[typ]["delta_BS_minus_E"]
            p = pct(block["metrics"][metric]); lo, hi = (pct(v) for v in block["ci95"][metric])
            color = COLORS["positive"] if p > 0 else COLORS["negative"] if p < 0 else COLORS["muted"]
            ax.errorbar(p, i, xerr=[[p-lo], [hi-p]], fmt="o", color=color, markersize=4, capsize=2.5, linewidth=1.1, zorder=3)
        ax.axvline(0, color="#697786", linewidth=.9, linestyle=(0, (3, 2)), zorder=1)
        ax.set_yticks(y, labels if ax is axes[0] else ["", "", ""])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=8.5, weight="semibold", color="#253746")
        ax.set_xlabel("B/S minus E", fontsize=7.5)
        style_axes(ax)
        ax.grid(axis="x", color="#D8DEE4", linewidth=.55)
        ax.grid(axis="y", visible=False)
    fig.suptitle("CTH XGBoost by observed pair type", fontsize=9, weight="semibold", y=1.02, color="#253746")
    fig.subplots_adjust(left=.19, right=.99, top=.84, bottom=.23, wspace=.30)
    save_figure(fig, "cth-xgb-pair-type-deltas")


def plot_nyu(nyu):
    models = [("svm", "RBF-SVM"), ("tree", "Decision tree"), ("xgb", "XGBoost")]
    metrics = [("ba", "Δ BA (pp)"), ("fmr", "Δ FMR (pp)"), ("fsr", "Δ FSR (pp)")]
    delta = nyu["paired_E_BS_minus_E"]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.9))
    for ax, (metric, title) in zip(axes, metrics):
        y = np.arange(len(models))
        for i, (model, _) in enumerate(models):
            block = delta[model]
            p = pct(block["metrics"][metric]); lo, hi = (pct(v) for v in block["ci95"][metric])
            color = COLORS["positive"] if p > 0 else COLORS["negative"] if p < 0 else COLORS["muted"]
            ax.errorbar(p, i, xerr=[[p-lo], [hi-p]], fmt="o", color=color, markersize=4, capsize=2.5, linewidth=1.1, zorder=3)
        ax.axvline(0, color="#697786", linewidth=.9, linestyle=(0, (3, 2)), zorder=1)
        ax.set_yticks(y, [m[1] for m in models] if ax is axes[0] else ["", "", ""])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=8.5, weight="semibold", color="#253746")
        ax.set_xlabel("E + B/S minus E", fontsize=7.5)
        style_axes(ax)
        ax.grid(axis="x", color="#D8DEE4", linewidth=.55)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Frozen external AA confirmation: paired error and BA changes", fontsize=9, weight="semibold", y=1.02, color="#253746")
    fig.subplots_adjust(left=.19, right=.99, top=.84, bottom=.23, wspace=.30)
    save_figure(fig, "nyu-aa-paired-deltas")


def build_tex_main_tables(five, ablation):
    configs = {
        "cth": [("BS", "Full"), ("S", "Remove B"), ("B", "Remove S"), ("evidence", "E only")],
        "ait": [("BST", "Full"), ("ST", "Remove B"), ("BT", "Remove S"), ("BS", "Remove T"), ("evidence", "E only")],
    }
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.1}",
        r"\caption{Matched CTH classifier comparison averaged equally over five Agent-role shares. Every classifier uses the same paired candidate decisions. Changes are B/S minus evidence-only E in percentage points; 95\% intervals are paired task-cluster bootstrap intervals.}",
        r"\label{tab:cth-matched-classifiers}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}", r"\toprule",
        r"Classifier & BA: E & BA: E+B/S & $\Delta$BA [95\% CI] & $\Delta$FMR & $\Delta$FSR \\ \midrule",
    ]
    for model, label in MODELS:
        e = five["five_share_equal_weight"][model]
        delta = e["delta_BS_minus_E"]
        row = [label, f"{pct(e['evidence']['metrics']['ba']):.2f}", f"{pct(e['BS']['metrics']['ba']):.2f}",
               latex_ci(delta, "ba", is_delta=True), f"{pct(delta['metrics']['fmr']):+.2f}", f"{pct(delta['metrics']['fsr']):+.2f}"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    lines += [
        r"\begin{table}[ht]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{2.6pt}\renewcommand{\arraystretch}{1.12}",
        r"\caption{Matched B/S/T ablation on the same candidate pairs. BA is task-macro balanced accuracy in percent; FMR and FSR are error rates in percent; $\Delta$BA compares with the full fingerprint for that dataset. Brackets are 95\% paired task-cluster bootstrap intervals (10,000 resamples). CTH averages the five role shares equally; AIT averages 12 tasks. Repeated pair decisions are not independent observations.}",
        r"\label{tab:matched-bst-ablation}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc rrr r@{}}", r"\toprule",
        r"Configuration & B & S & T & BA (\%) [95\% CI] & FMR (\%) & FSR (\%) & $\Delta$BA (pp) [95\% CI] \\ \midrule",
    ]
    for dataset, rows in configs.items():
        heading = r"CTH: 17 task groups; five exact role shares" if dataset == "cth" else r"AIT: 12 task groups; observed launch times"
        lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textbf{{{heading}}}}} \\")
        for config, label in rows:
            e = ablation["results"][dataset][config]
            b = r"\present" if "B" in config else "--"
            s = r"\present" if "S" in config else "--"
            t = r"\present" if "T" in config else "--"
            ba = latex_ci(e, "ba")
            fmr, fsr = f"{pct(e['metrics']['fmr']):.2f}", f"{pct(e['metrics']['fsr']):.2f}"
            if config == ("BS" if dataset == "cth" else "BST"):
                delta = "Reference"
            else:
                delta = latex_ci(e["delta_vs_full"], "ba", is_delta=True)
            lines.append(rf"{label} & {b} & {s} & {t} & {ba} & {fmr} & {fsr} & {delta} \\")
        if dataset == "cth":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def build_tex_appendix_tables(five, nyu):
    lines = [r"\begin{table}[ht]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{2.5pt}\renewcommand{\arraystretch}{1.1}",
             r"\caption{Frozen NYU D-CIPHER AA challenge-retrieval comparison. The challenge filename defines the same-task proxy. Changes are E+B/S minus E; intervals cluster by 160 task groups.}",
             r"\label{tab:nyu-aa-confirmation}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}", r"\toprule",
             r"Classifier & BA: E & BA: E+B/S & $\Delta$BA [95\% CI] & FMR: E+B/S & $\Delta$FMR & $\Delta$FSR \\ \midrule"]
    for model, label in MODELS:
        base = nyu["absolute_metrics"][model]["E"]["metrics"]
        full = nyu["absolute_metrics"][model]["E_BS"]["metrics"]
        delta = nyu["paired_E_BS_minus_E"][model]
        lines.append(" & ".join([label, f"{pct(base['ba']):.2f}", f"{pct(full['ba']):.2f}", latex_ci(delta,"ba",True),
                                  f"{pct(full['fmr']):.2f}", f"{pct(delta['metrics']['fmr']):+.2f}", f"{pct(delta['metrics']['fsr']):+.2f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    lines += [r"\begin{table}[ht]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.1}",
              r"\caption{CTH XGBoost error changes by pair type, averaged equally over the role shares where the type exists. HH, HA, and AA are trace-role pairs, not activity or actor labels.}",
              r"\label{tab:cth-pair-type-deltas}",
              r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}", r"\toprule",
              r"Pair type & $\Delta$BA [95\% CI] & $\Delta$FMR & $\Delta$FSR \\ \midrule"]
    for pair, label in (("HH", "Human--Human"), ("HA", "Human--Agent"), ("AA", "Agent--Agent")):
        d = five["pair_type_equal_weight_over_available_shares"]["xgb"][pair]["delta_BS_minus_E"]
        lines.append(" & ".join([label, latex_ci(d,"ba",True), f"{pct(d['metrics']['fmr']):+.2f}", f"{pct(d['metrics']['fsr']):+.2f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    lines += [r"\begin{table}[ht]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{2.6pt}\renewcommand{\arraystretch}{1.08}",
              r"\caption{CTH XGBoost results by exact Agent-role share. Values are task-macro means over the same paired candidates; $\Delta$BA intervals preserve task clustering.}",
              r"\label{tab:cth-per-share-xgb}",
              r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rrrrrr@{}}", r"\toprule",
              r"Agent roles & BA: E & BA: E+B/S & $\Delta$BA [95\% CI] & FMR: E+B/S & FSR: E+B/S \\ \midrule"]
    for share in ("0.0", "0.25", "0.5", "0.75", "1.0"):
        x = five["metrics_by_share_and_pair_type"]["xgb"]["ALL"][share]
        lines.append(" & ".join([f"{int(float(share)*100)}\%", f"{pct(x['evidence']['metrics']['ba']):.2f}", f"{pct(x['BS']['metrics']['ba']):.2f}",
                                  latex_ci(x["delta_BS_minus_E"],"ba",True), f"{pct(x['BS']['metrics']['fmr']):.2f}", f"{pct(x['BS']['metrics']['fsr']):.2f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table_html(headers, rows, cls="association-results-table"):
    thead = "".join(f"<th scope=\"col\">{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join((f"<th scope=\"row\">{cell}</th>" if j == 0 else f"<td>{cell}</td>") for j, cell in enumerate(row)) + "</tr>" for row in rows)
    return f'<div class="association-table-scroll" tabindex="0" role="region" aria-label="结果表" data-en-aria-label="Results table"><table class="{cls}"><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table></div>'


def build_site_section(five, ablation, nyu):
    cth_rows = []
    for model, label in MODELS:
        entry = five["five_share_equal_weight"][model]["delta_BS_minus_E"]
        cth_rows.append([label, delta_ci_html(entry, "ba"), delta_ci_html(entry, "fmr"), delta_ci_html(entry, "fsr")])
    ablation_rows = []
    names = {"cth": [("BS", '<span data-en="Full">完整</span>'), ("S", '<span data-en="Remove B">去掉 B</span>'), ("B", '<span data-en="Remove S">去掉 S</span>'), ("evidence", '<span data-en="Evidence only">仅原证据 E</span>')],
             "ait": [("BST", '<span data-en="Full">完整</span>'), ("ST", '<span data-en="Remove B">去掉 B</span>'), ("BT", '<span data-en="Remove S">去掉 S</span>'), ("BS", '<span data-en="Remove T">去掉 T</span>'), ("evidence", '<span data-en="Evidence only">仅原证据 E</span>')]}
    for dataset in ("cth", "ait"):
        data_rows = []
        for config, label in names[dataset]:
            e = ablation["results"][dataset][config]
            checks = ["✓" if feature in config else "—" for feature in ("B", "S", "T")]
            d = e["delta_vs_full"]
            delta = "Reference" if config == ("BS" if dataset == "cth" else "BST") else delta_ci_html(d, "ba")
            data_rows.append([label, *checks, absolute_html(e, "ba"), absolute_html(e, "fmr"), absolute_html(e, "fsr"), delta])
        ablation_rows.append((dataset.upper(), data_rows))
    nyu_rows=[]
    for model,label in MODELS:
        d=nyu["paired_E_BS_minus_E"][model]
        e=nyu["absolute_metrics"][model]
        base=e["E"]["metrics"]; full=e["E_BS"]["metrics"]
        nyu_rows.append([label, f"{pct(base['ba']):.2f}", f"{pct(full['ba']):.2f}", delta_ci_html(d,"ba"), delta_ci_html(d,"fmr"), delta_ci_html(d,"fsr")])
    nyu_delta = nyu["paired_E_BS_minus_E"]
    xgb_delta = nyu_delta["xgb"]["metrics"]
    xgb_fmr = absolute_html(nyu["absolute_metrics"]["xgb"]["E_BS"], "fmr")
    hit1_gain = fmt(nyu["ranking"]["xgb"]["delta_E_BS_minus_E"]["hit1"]["delta"])
    pair_rows=[]
    pair_data=five["pair_type_equal_weight_over_available_shares"]["xgb"]
    pair_labels={"HH":"人–人 <span data-en=\"Human–Human\">Human–Human</span>", "HA":"人–Agent <span data-en=\"Human–Agent\">Human–Agent</span>", "AA":"智能体–智能体 <span data-en=\"Agent–Agent\">Agent–Agent</span>"}
    for pair in ("HH","HA","AA"):
        d=pair_data[pair]["delta_BS_minus_E"]
        supported=", ".join(str(int(float(s)*100)) for s in pair_data[pair]["valid_shares"])
        pair_rows.append([pair_labels[pair], f"{supported}%", delta_ci_html(d,"ba"), delta_ci_html(d,"fmr"), delta_ci_html(d,"fsr")])
    return f'''<!-- MATCHED-FINGERPRINT-V2-RESULTS:BEGIN -->
<section class="section association-section association-results-section" id="matched-v2-results">
<div class="association-section-head"><h2 data-en="Matched fingerprint evaluation">同一批配对上的指纹评估</h2><p data-en="On external NYU Agent-only challenge retrieval, adding B/S raised balanced accuracy for XGBoost and the decision tree. Both models missed fewer same-challenge links and falsely merged more different-challenge pairs. Across the five CTH role shares, the mean change in balanced accuracy remained uncertain for all three classifiers.">在外部 NYU 的 Agent–Agent 同题关联实验中，加入 B／S 提高了 XGBoost 和决策树的平衡准确率。两种模型都减少了同题漏连，同时也增加了异题误连。在 CTH 五档配比实验中，三个分类器的平衡准确率平均变化仍不确定。</p></div>
<p class="association-transfer-note" data-en="CTH contains 17 task groups and five exact shares (0/25/50/75/100% of query-plus-candidate roles); each query has one same-task and three different-task candidates. All methods share the same 54,400 pair decisions across shares. The fraction describes pair roles in this evaluation, not dataset resampling.">CTH 含 17 道任务、五档精确角色比例（查询与候选角色中的 0／25／50／75／100% Agent）；每个查询有一个同题和三个异题候选。所有方法共用跨五档的 54,400 个配对判断。比例指本实验的配对角色，不是对整个数据集按比例重采样。</p>
<figure class="association-wide-figure scientific-evidence-figure"><div class="scientific-figure-viewport" tabindex="0" role="region" aria-label="CTH 五档精确配比结果，按分类器和指标分面" data-en-aria-label="CTH exact five-share results by classifier and metric"><a class="scientific-figure-source" href="assets/matched-fingerprint-v2/cth-five-share-matched.svg"><img src="assets/matched-fingerprint-v2/cth-five-share-matched.png" width="2320" height="1840" alt="三行分别为 RBF-SVM、决策树与 XGBoost；三列为 BA、误合并率、漏关联率。每格比较原证据 E 与 E 加 B/S，横轴是配对角色中 Agent 的精确比例，阴影为 95% 任务聚类区间。" data-en-alt="Three classifier rows (RBF-SVM, decision tree, XGBoost) and three metric columns (BA, false merge rate, false split rate). Each panel compares E with E plus B/S across exact Agent-role shares; bands are 95% task-cluster intervals." loading="lazy"/></a></div><figcaption data-en="All panels use the same paired candidate decisions. Thresholds were selected using inner training tasks under the FMR-at-most-10% rule and then held fixed for each outer fold; the rule does not guarantee 10% FMR in test tasks.">各面板使用同一组候选配对。阈值依据内层训练任务按 FMR 不超过 10% 规则选择，再固定用于对应外层折；测试任务的 FMR 不保证不超过 10%。</figcaption><a class="scientific-figure-open" href="assets/matched-fingerprint-v2/cth-five-share-matched.svg" data-en="Open vector figure">打开矢量图</a></figure>
<h3 data-en="CTH: average change across the five shares">CTH：五档平均变化</h3>
{table_html(['<span data-en="Classifier">分类器</span>', '<span data-en="Δ BA (pp), 95% CI">Δ BA（百分点，95% CI）</span>', '<span data-en="Δ FMR (pp)">Δ FMR（百分点）</span>', '<span data-en="Δ FSR (pp)">Δ FSR（百分点）</span>'], cth_rows)}
<p class="association-transfer-note" data-en="All three intervals for mean BA change span zero. With B/S, XGBoost and SVM miss fewer same-task links and falsely merge more different-task pairs on average. The decision tree with B/S rejects every candidate, giving zero false merges and a 100% missed-link rate.">三个分类器的五档平均 BA 变化区间都跨零。加入 B／S 后，XGBoost 和 SVM 平均少漏了一些同题配对，同时多误连了一些异题配对。加入 B／S 的决策树拒绝了全部候选，误合并率为零，漏关联率为 100%。</p>
<h3 data-en="One ablation table across CTH and AIT">CTH 与 AIT 的统一消融表</h3>
<p data-en="Check marks show the included fingerprint blocks. BA and Δ BA have 95% paired task-cluster intervals; FMR/FSR are point estimates. The full configuration is the reference within each dataset.">勾选表示该配置包含的指纹维度。BA 与 ΔBA 附 95% 配对任务聚类区间；FMR／FSR 为点估计。每个数据集的完整配置作为本数据集参照。</p>
<div class="association-table-scroll" tabindex="0" role="region" aria-label="CTH 与 AIT 的 B/S/T 统一消融表" data-en-aria-label="Unified CTH and AIT B/S/T ablation"><table class="association-results-table association-ablation-table"><thead><tr><th scope="col"><span data-en="Configuration">配置</span></th><th scope="col">B</th><th scope="col">S</th><th scope="col">T</th><th scope="col">BA %<br><span data-en="95% CI">95% CI</span></th><th scope="col">FMR %</th><th scope="col">FSR %</th><th scope="col">Δ BA（pp）<br><span data-en="95% CI">95% CI</span></th></tr></thead><tbody>
{''.join(f'<tr class="table-group"><th colspan="8" scope="rowgroup"><span data-en="{dataset}: {("17 tasks; five exact shares" if dataset=="CTH" else "12 tasks; observed timing")}">{dataset}：{("17 道任务；五档精确配比" if dataset=="CTH" else "12 道任务；含实测时间")}</span></th></tr>' + ''.join('<tr>' + ''.join((f'<th scope="row">{cell}</th>' if j==0 else f'<td>{cell}</td>') for j,cell in enumerate(row)) + '</tr>' for row in data_rows) for dataset,data_rows in ablation_rows)}
</tbody></table></div>
<p class="association-transfer-note" data-en="On CTH, the complete E+B/S model is below E alone in BA; removing B or S does not show a reliable independent contribution under this protocol. On AIT, adding T to E+B/S changes BA by +0.98 pp [−0.21, +2.10], an interval spanning zero. The AIT same-task relation and timing fields are real, but task and collection workflow co-vary with labels.">CTH 的完整 E+B/S BA 低于单独 E；在该协议下，移除 B 或 S 都没有显示出可靠的独立贡献。AIT 中给 E+B/S 加入 T 后，BA 变化为 +0.98 个百分点［−0.21，+2.10］，区间跨零。AIT 使用真实任务关联与时间字段，但任务和采集流程与标签共同变化。</p>
<h3 data-en="External NYU evaluation on Agent–Agent pairs">外部 NYU Agent–Agent 配对评估</h3>
<p data-en="The frozen NYU D-CIPHER evaluation uses 160 previously unused challenge groups and 585 runs from four snapshots of one Agent framework. Both traces in each pair come from Agent runs. The label records whether the two traces come from the same challenge.">NYU D-CIPHER 外部评估包含 160 个此前未用的题组和 585 次运行，来自同一 Agent 框架的四个快照。每对轨迹的两侧都来自 Agent 运行，标签记录它们是否对应同一道题。</p>
{table_html(['<span data-en="Classifier">分类器</span>', 'E BA %', 'E+B/S BA %', '<span data-en="Δ BA (pp), 95% CI">Δ BA（百分点，95% CI）</span>', '<span data-en="Δ FMR (pp)">Δ FMR（百分点）</span>', '<span data-en="Δ FSR (pp)">Δ FSR（百分点）</span>'], nyu_rows)}
<p class="association-transfer-note" data-en="With B/S, balanced accuracy rises by {fmt(xgb_delta['ba'])} percentage points for XGBoost and {fmt(nyu_delta['tree']['metrics']['ba'])} for the decision tree. XGBoost misses about {fmt(-xgb_delta['fsr'], 0)} fewer links per 100 same-challenge pairs and falsely merges about {fmt(xgb_delta['fmr'], 0)} more per 100 different-challenge pairs. SVM loses {fmt(-nyu_delta['svm']['metrics']['ba'])} points of balanced accuracy.">加入 B／S 后，XGBoost 和决策树的平衡准确率分别提高 {fmt(xgb_delta['ba'])} 和 {fmt(nyu_delta['tree']['metrics']['ba'])} 个百分点。以 XGBoost 为例，每 100 对同题轨迹约少漏 {fmt(-xgb_delta['fsr'], 0)} 对，每 100 对异题轨迹约多误连 {fmt(xgb_delta['fmr'], 0)} 对。SVM 的平衡准确率下降 {fmt(-nyu_delta['svm']['metrics']['ba'])} 个百分点。</p>
<p class="association-transfer-note" data-en="For candidate review, XGBoost's top-ranked candidate is correct more often, with Hit@1 up {hit1_gain} percentage points. For automatic linking, the threshold selected to keep validation FMR at or below 10% on CTH yields {xgb_fmr}% FMR on NYU with B/S.">如果用于排查候选，XGBoost 的首位命中率提高 {hit1_gain} 个百分点。如果用于自动关联，在 CTH 验证中按误合并率不超过 10% 选出的阈值，用到 NYU 的加指纹模型上时，实测误合并率为 {xgb_fmr}%。</p>
<h3 data-en="What changes within each CTH pair type?">CTH 各类配对内部发生了什么？</h3>
<p data-en="This XGBoost breakdown reuses the same predictions, averaging only shares where each pair type exists. HH, HA and AA mean the trace roles in a candidate pair; they do not identify attacks or actors.">以下 XGBoost 分层复用同一批预测，只对该类配对存在的比例档等权平均。HH、HA、AA 表示候选对两侧的执行角色，不表示攻击活动或攻击者身份。</p>
{table_html(['<span data-en="Pair type">配对类型</span>', '<span data-en="Agent shares in scope">适用的 Agent 比例（%）</span>', '<span data-en="Δ BA (pp), 95% CI">Δ BA（百分点，95% CI）</span>', '<span data-en="Δ FMR (pp)">Δ FMR（百分点）</span>', '<span data-en="Δ FSR (pp)">Δ FSR（百分点）</span>'], pair_rows)}
<p class="association-transfer-note" data-en="No CTH pair-type BA interval establishes a reliable gain. HA and AA show point increases in BA but also more false merges; HH shows a point BA decline. Pair-type differences are descriptive and share partially overlapping task and candidate support.">CTH 任一配对类型的 BA 区间都未确认可靠增益。HA 与 AA 的 BA 点估计提高，但误合并也增加；HH 的 BA 点估计下降。分层是描述性结果，任务和候选支持部分重叠。</p>
</section>
<!-- MATCHED-FINGERPRINT-V2-RESULTS:END -->'''


def install_site(fragment):
    if not SITE_PAGE.exists():
        return
    page = SITE_PAGE.read_text()
    start = "<!-- MATCHED-FINGERPRINT-V2-RESULTS:BEGIN -->"
    end = "<!-- MATCHED-FINGERPRINT-V2-RESULTS:END -->"
    if start in page and end in page:
        page = re.sub(re.escape(start) + r".*?" + re.escape(end), fragment, page, flags=re.S)
    else:
        transfer = re.search(r'<section\b(?=[^>]*\bid="transfer")', page)
        if not transfer:
            raise RuntimeError("Could not find transfer section insertion point")
        page = page[:transfer.start()] + fragment + "\n" + page[transfer.start():]
    nav_old = '<a data-en="Association gain" href="#association-ba-v3">关联增益</a>'
    nav_new = '<a data-en="Association gain" href="#association-ba-v3">关联增益</a>\n<a data-en="Matched evaluation" href="#matched-v2-results">匹配评估</a>'
    if 'href="#matched-v2-results"' not in page[:page.find(start) if start in page else len(page)]:
        page = page.replace(nav_old, nav_new, 1)
    SITE_PAGE.write_text(page)


def install_original_tables():
    """Append the ten pre-existing table blocks unchanged to the draft appendix."""
    draft_path = ROOT / "research/paper/human-agent-fingerprint-draft.tex"
    snapshot_path = ROOT / "research/results/matched_fingerprint_v1/original_table_snapshot.json"
    draft = draft_path.read_text()
    begin = "% MATCHED-FINGERPRINT-V2-LEGACY-TABLES:BEGIN"
    end = "% MATCHED-FINGERPRINT-V2-LEGACY-TABLES:END"
    if begin not in draft or end not in draft:
        return
    snap = load(snapshot_path)
    blocks = [x["text"] for x in snap["tables"]]
    expected = [x["sha256"] for x in snap["tables"]]
    actual = [hashlib.sha256(b.encode()).hexdigest() for b in blocks]
    if actual != expected or len(blocks) != 10:
        raise RuntimeError("Original table snapshot no longer matches its stored hashes")
    groups = [
        ("\n\\subsection{Earlier source and cohort tables}\n", blocks[:3]),
        ("\n\\clearpage\n\\subsection{Earlier association and ablation tables}\n", blocks[3:6]),
        ("\n\\clearpage\n\\subsection{Earlier feature, transfer, and recognition tables}\n", blocks[6:]),
    ]
    parts = []
    for heading, selected in groups:
        parts.append(heading)
        for block in selected:
            if block.startswith(r"\begin{longtable}"):
                parts.append("\\begingroup\\compacttable\n" + block + "\n\\endgroup\n")
            else:
                parts.append(block + "\n")
    content = begin + "\n" + "\n".join(parts) + "\n" + end
    draft = re.sub(re.escape(begin) + r".*?" + re.escape(end), lambda _: content, draft, flags=re.S)
    draft_path.write_text(draft)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    five, ablation, nyu = load(FIVE_PATH), load(ABLATION_PATH), load(NYU_PATH)
    data = normalized_inputs(five, ablation, nyu)
    (OUT / "paper_display_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "research/paper/matched-v2-main-tables.tex").write_text(build_tex_main_tables(five, ablation))
    (ROOT / "research/paper/matched-v2-appendix-tables.tex").write_text(build_tex_appendix_tables(five, nyu))
    install_original_tables()
    plot_cth_share(five)
    plot_delta_by_share(five)
    plot_pair_type(five)
    plot_nyu(nyu)
    install_site(build_site_section(five, ablation, nyu))
    print(json.dumps({"display_data": str((OUT / 'paper_display_data.json').relative_to(ROOT)),
                      "figures": sorted(p.name for p in OUT.glob("*.*") if p.suffix in {".pdf", ".svg", ".png"}),
                      "site": str(SITE_PAGE.relative_to(ROOT)),
                      "source_hashes": data["sources"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
