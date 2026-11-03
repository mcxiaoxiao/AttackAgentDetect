#!/usr/bin/env python3
"""Build publication and site figures from versioned v2 aggregate results."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from attackagent.fingerprint_v2 import fingerprint_matrix, terminal_action  # noqa: E402

PAPER = ROOT / "research/paper/figures"
SITE = ROOT / "site/assets/behavioral-fingerprint-v2"
RESULTS = ROOT / "research/results/behavioral_fingerprint_v2"
PAPER.mkdir(parents=True, exist_ok=True)
SITE.mkdir(parents=True, exist_ok=True)

NAVY = "#17324f"
BLUE = "#2676bd"
TEAL = "#238b8e"
ORANGE = "#e38b45"
MUTED = "#64758a"
GRID = "#d9e2ec"
PAPER_COLOR = "#f5f9fd"
PALETTE = {"command_cosine": BLUE, "command_plus_BS": TEAL}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.color": NAVY,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "axes.linewidth": .7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})


def json_read(path):
    return json.loads(path.read_text())


def save(fig, name, site_name=None):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(PAPER / f"{name}.{ext}", dpi=220 if ext == "png" else None)
        if site_name:
            fig.savefig(SITE / f"{site_name}.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    ax.set(xlim=(0, 7.2), ylim=(0, 3.25)); ax.axis("off")
    items = [(.1,1.88,1.8,.95,"Retained logs","audit / shell / HTTP\nordered events + time"),
             (2.25,1.88,1.8,.95,"Observation adapter","event names and actions\nfield availability"),
             (4.4,1.88,2.7,.95,"Window fingerprint","B8: local behavior   S8: sequence\nT8: observed timing"),
             (2.25,.28,2.05,.95,"Execution-mode readout","B/S/T summaries\nHuman / Agent score"),
             (4.7,.28,2.4,.95,"Association readout","command / request evidence\n+ fingerprint distances")]
    for i,(x,y,w,h,title,body) in enumerate(items):
        color=TEAL if i==2 else BLUE
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.05",facecolor="#f0f5fa",edgecolor=color,linewidth=1))
        ax.text(x+.10,y+h-.17,title,fontsize=8.7,weight="bold",va="top")
        ax.text(x+.10,y+.39,body,fontsize=7.6,color=MUTED,va="center",linespacing=1.5)
    for a,b in [((1.94,2.35),(2.2,2.35)),((4.09,2.35),(4.35,2.35)),((5.0,1.84),(3.28,1.27)),((5.9,1.84),(5.9,1.27))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=10,color=BLUE,lw=1.2))
    ax.text(.1,3.07,"From log observations to two analytical readouts",fontsize=10,weight="bold")
    ax.text(.1,.04,"Window count J follows trace length; unavailable measurements retain an explicit mask.",fontsize=7,color=MUTED)
    fig.subplots_adjust(left=.01,right=.99,bottom=.03,top=.99)
    save(fig,"fingerprint-v2-architecture","fingerprint-v2-architecture")

def coverage():
    fig,ax=plt.subplots(figsize=(7.2,2.9))
    labels=["CTH mixed tasks (175 traces)","AIT audit logs (109 runs)","Honey (13,080 sessions)","CyBiasBench (450 sessions)"]
    columns=["Order","Start time","Duration","Human","Agent"]
    data=np.array([[1,0,0,1,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1]])
    from matplotlib.colors import ListedColormap
    ax.imshow(data,cmap=ListedColormap(["#edf1f5",BLUE]),vmin=0,vmax=1,aspect="auto")
    for i in range(4):
        for j in range(5): ax.text(j,i,"yes" if data[i,j] else "—",ha="center",va="center",fontsize=8,color="white" if data[i,j] else MUTED)
    ax.set_xticks(range(5),columns,fontsize=8);ax.xaxis.tick_top();ax.set_yticks(range(4),labels,fontsize=8)
    ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True);ax.grid(which="minor",color="white",lw=3)
    ax.tick_params(which="both",length=0,pad=8)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.02,.96,"Observation coverage of the evaluated cohorts",fontsize=10,weight="bold")
    fig.text(.02,.035,"Honey duration is source-reported execution time. Each source has its own task/condition relation label.",fontsize=7,color=MUTED)
    fig.subplots_adjust(left=.36,right=.98,top=.76,bottom=.16)
    save(fig,"fingerprint-v2-cohort-coverage","fingerprint-v2-cohort-coverage")

def _load_pilot():
    path = ROOT / "research/experiments/log_fingerprint_pilot_v1/run.py"
    spec = importlib.util.spec_from_file_location("fingerprint_v2_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_runs()


def matrix_example():
    runs = _load_pilot()
    by_id = {r["id"]: r for r in runs}
    chosen = [("78d4c642dcc3b79a", "Human · Eve"), ("bd959913c2c5293d", "Agent · GPT-4.1 V2")]
    raw = []
    matrices = []
    for rid, label in chosen:
        run = by_id[rid]
        events = [e for e in run["events"] if e.get("login")][:128]
        symbols = [str(e["command"]) for e in events]
        actions = [terminal_action(c) for c in symbols]
        times = [e.get("time") for e in events]
        blocks, audit = fingerprint_matrix(symbols, actions=actions, timestamps=times,
                                           window=8, min_window_events=4, max_events=128)
        if blocks is None:
            raise RuntimeError(f"Selected illustration trace {rid} has no usable window")
        raw.append((run, label, audit))
        matrices.append(blocks)
    all_runs = []
    for run in runs:
        events = [e for e in run["events"] if e.get("login")][:128]
        symbols = [str(e["command"]) for e in events]
        if len(symbols) < 4:
            continue
        blocks, _ = fingerprint_matrix(symbols,
            actions=[terminal_action(c) for c in symbols],
            timestamps=[e.get("time") for e in events], window=8,
            min_window_events=4, max_events=128)
        if blocks is not None:
            all_runs.append(blocks)
    global_stats = {}
    for block in ("B", "S", "T"):
        data = np.vstack([r[block] for r in all_runs])
        # Compute moments column-wise without emitting all-NaN warnings for
        # unavailable timing dimensions in shorter source records.
        valid = np.isfinite(data)
        count = valid.sum(axis=0)
        mean = np.divide(np.where(valid, data, 0.0).sum(axis=0), count,
                         out=np.zeros(data.shape[1], dtype=float), where=count > 0)
        centered = np.where(valid, data - mean, 0.0)
        variance = np.divide((centered ** 2).sum(axis=0), count,
                             out=np.ones(data.shape[1], dtype=float), where=count > 0)
        global_stats[block] = (mean, np.sqrt(variance))
    z_matrices = []
    for blocks in matrices:
        pieces = []
        for block in ("B", "S", "T"):
            mean, sd = global_stats[block]
            sd = np.where(sd > 1e-6, sd, 1.)
            z = (blocks[block] - mean) / sd
            pieces.append(np.clip(z, -2.5, 2.5))
        z_matrices.append(np.hstack(pieces))

    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle, Patch
    from paper_style import apply, save as save_publication, PAGE_WIDTH, MISSING
    apply()
    fig, axes = plt.subplots(2, 1, figsize=(PAGE_WIDTH, 3.15), sharex=True)
    names = [f"{block}{j}" for block in ("B", "S", "T") for j in range(1, 9)]
    cmap = LinearSegmentedColormap.from_list("fingerprint_cobalt", [BLUE, "#f7fafd", "#c66b38"])
    cmap.set_bad(MISSING)
    last = None
    for i, ax in enumerate(axes):
        matrix = z_matrices[i]
        last = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", interpolation="nearest",
                         cmap=cmap, vmin=-2.5, vmax=2.5)
        for row, col in zip(*np.where(~np.isfinite(matrix))):
            ax.add_patch(Rectangle((col-.5, row-.5), 1, 1, facecolor=MISSING,
                                   edgecolor="#afbac7", linewidth=0, hatch="///"))
        ax.set_yticks(np.arange(matrix.shape[0]), [str(j+1) for j in range(matrix.shape[0])])
        ax.set_ylabel("Window", fontsize=7)
        ax.set_xticks(np.arange(24), names, fontsize=6.4)
        ax.tick_params(length=0, labelbottom=(i==1), labelsize=6.4)
        label = "Human" if i == 0 else "Agent"
        ax.set_title(f"{'ab'[i]}  {label} ({raw[i][2]['events']} launches)", loc="left",
                     fontsize=8.3, weight="bold", pad=6)
        for border in (7.5, 15.5):ax.axvline(border, color="white", linewidth=1.7)
        for side in ax.spines.values():side.set_visible(False)
    for x, label, color in [(3.5, "B  Local behavior", BLUE), (11.5, "S  Sequence evolution", TEAL),
                             (19.5, "T  Observed timing", ORANGE)]:
        axes[1].text(x, -.35, label, ha="center", va="top", color=color, fontsize=7.2,
                     transform=axes[1].get_xaxis_transform())
    cb = fig.colorbar(last, cax=fig.add_axes([.27, .075, .42, .025]), orientation="horizontal",
                      ticks=[-2.5, 0, 2.5])
    cb.set_label("Standardized feature (z; clipped ±2.5)", fontsize=6.8, labelpad=2)
    cb.ax.tick_params(labelsize=6.3, length=2);cb.outline.set_linewidth(.4)
    fig.legend(handles=[Patch(facecolor=MISSING, edgecolor="#afbac7", hatch="///", label="Missing")],
               loc="lower right", bbox_to_anchor=(.99, .06), fontsize=6.8, frameon=False)
    fig.subplots_adjust(left=.055, right=.99, top=.92, bottom=.30, hspace=.35)
    save_publication(fig, "fingerprint-v2-matrix-example", (PAPER, SITE))


def mixture_figure():
    metrics=json_read(RESULTS/"cth_mixture/metrics.json"); levels=[0,25,50,75,100]
    selected={m:{round(x["fraction"]*100):x for x in metrics if x["method"]==m} for m in PALETTE}
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.75),sharex=True,sharey=True)
    for i,(ax,metric,title) in enumerate(zip(axes,("fmr","fsr","ba"),("False merges (FMR)","Missed links (FSR)","Balanced accuracy (BA)"))):
        for method,style,marker in (("command_cosine","-","o"),("command_plus_BS","--","s")):
            rows=[selected[method][x] for x in levels]; y=np.array([r["task_macro_"+metric]*100 for r in rows]); ci=np.array([r["task_cluster_95ci_"+metric] for r in rows])*100
            ax.plot(levels,y,ls=style,marker=marker,ms=3.5,lw=1.5,color=PALETTE[method],label="Command" if method=="command_cosine" else "Command + B/S")
            ax.fill_between(levels,ci[:,0],ci[:,1],color=PALETTE[method],alpha=.10,lw=0)
        ax.set_title(chr(97+i)+"  "+title,loc="left",fontsize=8.4,weight="bold",pad=9)
        ax.set(ylim=(0,100),yticks=[0,25,50,75,100],xticks=levels,xlabel="Agent share (%)")
        ax.grid(axis="y",color=GRID,lw=.6);ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(length=0,labelsize=7)
        ax.xaxis.label.set_size(7.5)
    axes[0].set_ylabel("Task-macro rate (%)",fontsize=7.5)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.075,right=.99,top=.78,bottom=.21,wspace=.19)
    save(fig,"fig4-cth-mixture-bst","fig4-cth-mixture-bst")

def timing_increment():
    deltas = json_read(RESULTS / "ait_recognition/paired_deltas.json")
    wanted = [
        ("frozen_v1_89", "task_holdout", "Frozen 89 · task holdout"),
        ("frozen_v1_89", "source_holdout", "Frozen 89 · source holdout"),
        ("extended_all_eligible_min4", "task_holdout", "Expanded 109 · task holdout"),
        ("extended_all_eligible_min4", "source_holdout", "Expanded 109 · source holdout"),
    ]
    rows = []
    for dataset, panel, label in wanted:
        r = next(x for x in deltas if x["dataset"] == dataset and x["panel"] == panel
                 and x["comparison"] == "B_plus_S_plus_T" and x["baseline"] == "B_plus_S")
        rows.append((label, r["task_macro_delta_ba"] * 100,
                     np.array(r["task_cluster_95ci"]) * 100))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    y = np.arange(len(rows))[::-1]
    for yi, (label, value, ci) in zip(y, rows):
        color = TEAL if ci[0] > 0 else ORANGE
        ax.errorbar(value, yi, xerr=[[value-ci[0]], [ci[1]-value]], fmt="o", ms=6,
                    capsize=3.5, lw=1.8, color=color)
    ax.axvline(0, color=NAVY, lw=1, ls=(0, (3, 3)), alpha=.7)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Δ task-macro BA from adding T to B+S (percentage points)")
    ax.set_xlim(-16, 32)
    ax.grid(axis="x", color=GRID, linewidth=.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=5)
    fig.subplots_adjust(left=.28, right=.98, top=.96, bottom=.28)
    save(fig, "fingerprint-v2-ait-timing-increment", "fingerprint-v2-ait-timing-increment")


def recognition_summary():
    metrics = json_read(RESULTS / "ait_recognition/metrics.json")
    panels = [
        ("frozen_v1_89", "task_holdout", "Frozen 89 · task holdout"),
        ("frozen_v1_89", "source_holdout", "Frozen 89 · source holdout"),
        ("extended_all_eligible_min4", "task_holdout", "Expanded 109 · task holdout"),
        ("extended_all_eligible_min4", "source_holdout", "Expanded 109 · source holdout"),
    ]
    methods = ["core8", "B", "B_plus_S", "B_plus_S_plus_T", "command_tfidf"]
    labels = {"core8": "Frozen core 8", "B": "B only", "B_plus_S": "B + S",
              "B_plus_S_plus_T": "B + S + T", "command_tfidf": "Command TF–IDF"}
    colors = {"core8": MUTED, "B": BLUE, "B_plus_S": TEAL,
              "B_plus_S_plus_T": ORANGE, "command_tfidf": NAVY}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey=True)
    for ax, (dataset, panel, title) in zip(axes.flat, panels):
        rows = [next(r for r in metrics if r["dataset"] == dataset and r["panel"] == panel and r["method"] == m)
                for m in methods]
        y = np.arange(len(methods))
        values = np.array([r["task_macro_ba"] * 100 for r in rows])
        low = np.array([r["task_cluster_95ci_ba"][0] * 100 for r in rows])
        high = np.array([r["task_cluster_95ci_ba"][1] * 100 for r in rows])
        for i, method in enumerate(methods):
            ax.errorbar(values[i], y[i], xerr=[[values[i]-low[i]], [high[i]-values[i]]],
                        fmt="o", ms=4.5, capsize=2.5, lw=1.5, color=colors[method])
        ax.set_yticks(y, [labels[m] for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.axvline(50, color=MUTED, lw=.8, ls="--")
        ax.set_title(title, loc="left", fontsize=9, weight="bold", pad=6)
        ax.grid(axis="x", color=GRID, linewidth=.65)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=7.5)
        ax.tick_params(axis="x", length=0, labelsize=7)
    axes[0, 0].set_ylabel("Classifier", fontsize=8, color=MUTED)
    axes[1, 0].set_ylabel("Classifier", fontsize=8, color=MUTED)
    fig.supxlabel("Task-macro balanced accuracy (%)", fontsize=8, y=.025)
    fig.subplots_adjust(left=.25, right=.98, top=.94, bottom=.13, hspace=.33, wspace=.18)
    save(fig, "fingerprint-v2-ait-recognition", "fingerprint-v2-ait-recognition")


def retrieval_increment():
    cth = json_read(RESULTS / "cth_cross_mode/metrics.json")
    cth_delta = next(x for x in cth["paired_deltas_vs_command"] if x["method"] == "command_plus_BS")
    short_trace = json_read(ROOT / "research/results/short_trace_extension_v1/metrics.json")
    short_metrics = [metric for group in short_trace for metric in group["metrics"]]
    cth_legacy = next(x for x in short_metrics
                      if x["source"] == "cyber-task-horizons" and x["method"] == "matrix24" and x["metric"] == "hit1")
    antieval_legacy = next(x for x in short_metrics
                           if x["source"] == "antieval/cybench-trajectories" and x["method"] == "matrix24" and x["metric"] == "hit1")
    honey = json_read(RESULTS / "honey/metrics.json")
    honey_delta = honey["paired_B_plus_S_minus_command_hit1"]
    cybias = json_read(RESULTS / "cybiasbench/metrics.json")
    kypo = json_read(RESULTS / "kypo_human/metrics.json")
    kypo_delta = next(x for x in kypo["metrics"] if x["method"] == "command_plus_BS" and x["metric"] == "hit_at_1")
    route = next(x for x in cybias if x["method"] == "route_cosine")
    fused = next(x for x in cybias if x["method"] == "route_plus_BS")
    per = {x["target"]: x for x in route["per_target"]}
    per_fused = {x["target"]: x for x in fused["per_target"]}
    target_deltas = [(per_fused[t]["task_macro_hit_at_1"] - per[t]["task_macro_hit_at_1"]) * 100
                     for t in sorted(per)]
    cybias_mean = (fused["target_macro_hit_at_1"] - route["target_macro_hit_at_1"]) * 100
    rows = [
        ("CTH mixed · 40 tasks", (cth_delta["delta_hit_at_1"] * 100),
         np.asarray(cth_delta["delta_hit_at_1_task_cluster_95ci"]) * 100, "ci"),
        ("CTH Agent-only · 86 tasks", cth_legacy["paired_delta"] * 100,
         np.asarray(cth_legacy["paired_95ci"]) * 100, "ci"),
        ("Honey · 49 conditions", honey_delta["mean"] * 100,
         np.asarray(honey_delta["ci95"]) * 100, "ci"),
        ("CyBiasBench · target holdout", cybias_mean,
         np.asarray([min(target_deltas), max(target_deltas)]), "range"),
        ("KYPO Human · 8 levels", kypo_delta["paired_delta"] * 100,
         np.asarray(kypo_delta["paired_delta_ci95"]) * 100, "ci"),
        ("Antieval · 27 tasks", antieval_legacy["paired_delta"] * 100,
         np.asarray(antieval_legacy["paired_95ci"]) * 100, "ci"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.05))
    y = np.arange(len(rows))[::-1]
    for yi, (label, value, interval, kind) in zip(y, rows):
        color = BLUE if kind == "ci" else ORANGE
        ax.errorbar(value, yi, xerr=[[max(0, value-interval[0])], [max(0, interval[1]-value)]],
                    fmt="o" if kind == "ci" else "D", ms=5.5, capsize=3.3,
                    lw=1.7, color=color)
        if kind == "range":
            ax.text(interval[1] + .15, yi + .15, "3-target range", fontsize=7, color=MUTED)
    ax.axvline(0, color=NAVY, lw=1, ls=(0, (3, 3)), alpha=.7)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_title("Paired Hit@1 change across held-out samples", loc="left", fontsize=9, weight="bold", pad=8)
    ax.set_xlabel("Change in Hit@1 after adding the named behavior profile (percentage points)")
    ax.set_xlim(-8, 7)
    ax.grid(axis="x", color=GRID, linewidth=.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=.36, right=.98, top=.91, bottom=.21)
    fig.text(.36, .025, "Cohorts use source-specific task/condition labels; CTH and Antieval Agent-only rows use the legacy 24-value profile.", fontsize=6.3, color=MUTED)
    save(fig, "fingerprint-v2-retrieval-increment", "fingerprint-v2-retrieval-increment")


def main():
    architecture()
    coverage()
    matrix_example()
    mixture_figure()
    recognition_summary()
    retrieval_increment()
    timing_increment()
    print(f"Figures written to {PAPER} and {SITE}")


if __name__ == "__main__":
    main()
