#!/usr/bin/env python3
"""Render publication/site figures from the fixed experiment outputs."""
from __future__ import annotations

import collections
import csv
import importlib.util
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "research/results/overnight_fingerprint_v1"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140, "savefig.dpi": 260,
                     "axes.titleweight": "semibold", "axes.labelcolor": "#26384d",
                     "xtick.color": "#52667d", "ytick.color": "#52667d"})
PALETTE = ["#54799a", "#e39b57", "#5b9d89", "#bd6b66", "#8176a8", "#9a8d76"]


def load_json(path):
    return json.loads(path.read_text())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def recognition_figures():
    rec = OUT / "recognition"
    methods = ["length", "command_tfidf", "core8", "window_summary", "matrix_dtw_3nn"]
    labels = ["Length", "Command TF–IDF", "Static K1–K8", "Window summary", "Masked DTW 3-NN"]
    values = [load_json(rec / f"primary__task_holdout__{m}.json") for m in methods]
    scores = np.array([x["task_macro_balanced_accuracy"] for x in values])
    lows = np.array([x["task_macro_ba_cluster_bootstrap_95ci"][0] for x in values])
    highs = np.array([x["task_macro_ba_cluster_bootstrap_95ci"][1] for x in values])
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8.1, 4.1))
    for i, (score, low, high) in enumerate(zip(scores, lows, highs)):
        ax.errorbar(score, i, xerr=[[score-low], [high-score]], fmt="o", color=PALETTE[i],
                    markersize=7, capsize=3, linewidth=1.8)
    ax.axvline(.5, linestyle="--", color="#9aa8b6", linewidth=1, label="0.50 reference")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(.25, 1.03)
    ax.set_xlabel("Task-macro balanced accuracy (95% task-cluster interval)")
    ax.set_title("Human/Agent label prediction on 89 audit-log runs")
    ax.grid(axis="x", color="#e7edf3", linewidth=.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "ait-recognition-task-holdout")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, method, title in zip(axes, ("command_tfidf", "window_summary"),
                                  ("Command-name TF–IDF", "8-feature window readout")):
        d = load_json(rec / f"primary__task_holdout__{method}.json")
        cm = np.asarray(d["confusion_human_agent"], float)
        normalized = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for row in range(2):
            for col in range(2):
                color = "white" if normalized[row, col] > .58 else "#19334d"
                ax.text(col, row, f"{int(cm[row, col])}", ha="center", va="center", color=color, fontsize=11)
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_xticks([0, 1], ["Human", "Agent"])
        ax.set_yticks([0, 1], ["Human", "Agent"])
        ax.set_xlim(-.5, 1.5)
        ax.set_ylim(1.5, -.5)
        ax.set_ylabel("True release label")
    fig.suptitle("Confusion matrices (count; row-normalized shading)", y=1.03, fontsize=10)
    fig.tight_layout()
    save(fig, "ait-recognition-confusion")

    spec = importlib.util.spec_from_file_location("overnight_runner", ROOT / "research/experiments/overnight_fingerprint_v1/run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows, _ = runner.ait_rows()
    finite = np.concatenate([r["matrix"].reshape(-1, 8) for r in rows], axis=0)
    center = np.nanmean(finite, axis=0)
    scale = np.nanstd(finite, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    fig, axes = plt.subplots(2, 1, figsize=(8.3, 4.8), sharex=True, sharey=True)
    for ax, label, title in zip(axes, (0, 1), ("Human-labeled", "Agent-labeled")):
        values = [r["matrix"] for r in rows if r["y"] == label]
        summary = np.full((16, 8), np.nan)
        for window in range(16):
            cells = [((m[window] - center) / scale) for m in values if m.shape[0] > window]
            if cells:
                summary[window] = np.nanmean(np.vstack(cells), axis=0)
        masked = np.ma.masked_invalid(summary.T)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#edf1f5")
        im = ax.imshow(masked, cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto", interpolation="nearest")
        if ax is axes[-1]:
            fig.colorbar(im, ax=ax, fraction=.025, pad=.02, label="Cohort-standardized feature value")
        ax.set_ylabel("Fingerprint dimension")
        ax.set_yticks(np.arange(8), [f"K{i}" for i in range(1, 9)])
        ax.set_xticks(np.arange(16))
        ax.set_title(title, loc="left", fontsize=9)
    axes[-1].set_xlabel("Consecutive 8-command window (maximum 16; no phase labels)")
    axes[-1].set_xticks(list(range(0, 16, 2)), [str(x) for x in range(0, 16, 2)])
    fig.suptitle("Descriptive mean window matrices; not model input scaling", y=1.01, fontsize=10)
    fig.tight_layout()
    save(fig, "ait-window-fingerprint-matrix")


def mixture_type_metrics():
    path = OUT / "mixture/pair_predictions.csv"
    groups = collections.defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            positive = row["same_task"].strip().lower() == "true"
            groups[(float(row["nominal_agent_fraction"]), row["method"], row["query_task"],
                    row["pair_type"], positive)].append(int(row["pred_link"]))
    fractions = sorted({k[0] for k in groups})
    methods = sorted({k[1] for k in groups})
    types = ("HH", "HA", "AA")
    output = []
    for fraction in fractions:
        for method in methods:
            type_metrics = {}
            for typ in types:
                per_task = collections.defaultdict(dict)
                for (frac, meth, task, pair_type, positive), values in groups.items():
                    if frac != fraction or meth != method or pair_type != typ:
                        continue
                    error = float(np.mean([1-x if positive else x for x in values]))
                    per_task[task]["fsr" if positive else "fmr"] = error
                metrics = {}
                for error_name in ("fsr", "fmr"):
                    task_values = [x[error_name] for x in per_task.values() if error_name in x]
                    if len(task_values) >= 2:
                        rng = np.random.default_rng(20261102 + int(fraction * 1000) + len(typ) + len(method))
                        arr = np.asarray(task_values)
                        ci = np.quantile(arr[rng.integers(0, len(arr), size=(2000, len(arr)))].mean(axis=1), [.025, .975])
                    else:
                        ci = [np.nan, np.nan]
                    metrics[error_name] = float(np.mean(task_values)) if task_values else None
                    metrics[error_name + "_task_cluster_95ci"] = [float(ci[0]), float(ci[1])] if np.isfinite(ci).all() else None
                    metrics[error_name + "_tasks"] = len(task_values)
                type_metrics[typ] = metrics
            output.append({"agent_fraction": fraction, "method": method, "pair_types": type_metrics})
    (OUT / "mixture/pair_type_task_metrics.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    return output


def mixture_composite_metrics(draws=20000):
    """Task-cluster uncertainty for the composite thresholded retrieval score.

    Each task is first averaged over the 100 repeated role allocations; tasks,
    rather than repeated draws or query/candidate pairs, are the bootstrap unit.
    """
    grouped = collections.defaultdict(list)
    with (OUT / "mixture/pair_predictions.csv").open() as f:
        for row in csv.DictReader(f):
            fraction = float(row["nominal_agent_fraction"])
            replicate = int(row["replicate"])
            positive = row["same_task"].strip().lower() == "true"
            grouped[(fraction, row["method"], replicate, row["query_task"])].append(
                (positive, int(row["pred_link"]))
            )

    per_task_draws = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for (fraction, method, replicate, task), rows in grouped.items():
        positives = [pred for positive, pred in rows if positive]
        negatives = [pred for positive, pred in rows if not positive]
        if positives and negatives:
            fsr = float(np.mean([1 - pred for pred in positives]))
            fmr = float(np.mean(negatives))
            per_task_draws[fraction][method][task].append(1 - (fmr + fsr) / 2)

    rng_seed = 20261102
    stats = []
    for fraction in sorted(per_task_draws):
        for method_index, method in enumerate(("command_cosine", "action_bigram_cosine")):
            task_means_by_task = per_task_draws[fraction][method]
            tasks = sorted(task_means_by_task)
            values = np.asarray([np.mean(task_means_by_task[task]) for task in tasks], dtype=float)
            if len(values) < 2:
                continue
            rng = np.random.default_rng(rng_seed + int(fraction * 1000) + method_index * 10000)
            sample_idx = rng.integers(0, len(values), size=(draws, len(values)))
            ci = np.quantile(values[sample_idx].mean(axis=1), [.025, .975])
            stats.append({
                "nominal_agent_fraction": fraction,
                "method": method,
                "task_macro_ba": float(values.mean()),
                "task_cluster_95ci": [float(ci[0]), float(ci[1])],
                "tasks": len(values),
            })

    endpoint_changes = []
    for method_index, method in enumerate(("command_cosine", "action_bigram_cosine")):
        endpoint_by_fraction = {
            fraction: {task: float(np.mean(draw_values)) for task, draw_values in methods[method].items()}
            for fraction, methods in per_task_draws.items()
            if method in methods
        }
        common_tasks = sorted(set(endpoint_by_fraction.get(0.0, {})) & set(endpoint_by_fraction.get(1.0, {})))
        deltas = np.asarray([
            endpoint_by_fraction[1.0][task] - endpoint_by_fraction[0.0][task]
            for task in common_tasks
        ], dtype=float)
        paired_rng = np.random.default_rng(rng_seed + 5000 + method_index * 10000)
        paired_idx = paired_rng.integers(0, len(deltas), size=(draws, len(deltas)))
        paired_ci = np.quantile(deltas[paired_idx].mean(axis=1), [.025, .975])
        endpoint_changes.append({
            "method": method,
            "contrast": "100% target minus 0% target",
            "delta": float(deltas.mean()),
            "task_cluster_95ci": [float(paired_ci[0]), float(paired_ci[1])],
            "tasks": len(deltas),
        })

    result = {
        "metric": "task-macro balanced accuracy = 1 - (FMR + FSR) / 2",
        "bootstrap_unit": "challenge task; each task is averaged over 100 repeated role allocations first",
        "bootstrap_draws": draws,
        "bootstrap_seed": rng_seed,
        "nominal_agent_fractions": [0.0, 0.25, 0.5, 0.75, 1.0],
        "realized_role_shares": {"0.00": 0.0, "0.25": 8 / 34, "0.50": 0.5, "0.75": 26 / 34, "1.00": 1.0},
        "results": stats,
        "paired_endpoint_changes": endpoint_changes,
    }
    (OUT / "mixture/composite-balance.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def mixture_figures(type_metrics, composite_metrics):
    data = load_json(OUT / "mixture/metrics.json")
    targets = sorted({x["nominal_agent_fraction"] for x in data})
    xlabels = [f"{x:.0%}" for x in targets]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True)
    for ax, metric, ci_key, title in [(axes[0], "task_macro_fmr_mean", "task_macro_fmr_task_cluster_95ci", "False merge rate"),
                                      (axes[1], "task_macro_fsr_mean", "task_macro_fsr_task_cluster_95ci", "False split rate")]:
        for method, color, label in [("command_cosine", PALETTE[0], "Command TF–IDF cosine"),
                                     ("action_bigram_cosine", PALETTE[1], "Action-bigram cosine")]:
            rows = sorted((x for x in data if x["method"] == method), key=lambda x: x["nominal_agent_fraction"])
            xx = np.asarray([x["nominal_agent_fraction"] for x in rows])
            yy = np.asarray([x[metric] for x in rows])
            ci = [x[ci_key] for x in rows]
            lower = np.asarray([max(0, y-v[0]) if v else 0 for y, v in zip(yy, ci)])
            upper = np.asarray([max(0, v[1]-y) if v else 0 for y, v in zip(yy, ci)])
            ax.errorbar(xx, yy, yerr=[lower, upper], color=color, marker="o", linewidth=1.8,
                        capsize=2.5, label=label)
        ax.set_title(title)
        ax.set_xlabel("Target Agent-role share")
        ax.set_ylim(-.03, 1.03)
        ax.set_xticks(targets, xlabels)
        ax.grid(axis="y", color="#e7edf3", linewidth=.8)
    axes[0].set_ylabel("Task-macro error rate (95% task-cluster interval)")

    ba_by_method = collections.defaultdict(list)
    for row in composite_metrics["results"]:
        ba_by_method[row["method"]].append(row)
    for method, color, label in [("command_cosine", PALETTE[0], "Command TF–IDF cosine"),
                                 ("action_bigram_cosine", PALETTE[1], "Action-bigram cosine")]:
        rows = sorted(ba_by_method[method], key=lambda x: x["nominal_agent_fraction"])
        xx = np.asarray([x["nominal_agent_fraction"] for x in rows])
        yy = np.asarray([x["task_macro_ba"] for x in rows])
        ci = [x["task_cluster_95ci"] for x in rows]
        lower = np.asarray([y - v[0] for y, v in zip(yy, ci)])
        upper = np.asarray([v[1] - y for y, v in zip(yy, ci)])
        axes[2].errorbar(xx, yy, yerr=[lower, upper], color=color, marker="o", linewidth=1.8,
                         capsize=2.5, label=label)
    axes[2].axhline(.5, linestyle="--", color="#9aa8b6", linewidth=1, label="0.50 reference")
    axes[2].set_title("Task-macro balanced accuracy")
    axes[2].set_xlabel("Target Agent-role share")
    axes[2].set_ylim(.35, .68)
    axes[2].set_xticks(targets, xlabels)
    axes[2].set_ylabel("BA (95% task-cluster interval)")
    axes[2].grid(axis="y", color="#e7edf3", linewidth=.8)
    axes[2].legend(frameon=False, fontsize=7, loc="lower left")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("CTH same-task retrieval across target Human/Agent mixtures (17 tasks)", y=1.02, fontsize=10)
    fig.tight_layout()
    save(fig, "ctf-mixture-errors")

    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.0), sharex=True)
    by_frac = {x["agent_fraction"]: x for x in type_metrics if x["method"] == "command_cosine"}
    for ax, field, ci_field, title in [(axes[0], "fmr", "fmr_task_cluster_95ci", "Different-task pairs: false merges"),
                                      (axes[1], "fsr", "fsr_task_cluster_95ci", "Same-task pairs: false splits")]:
        for typ, color in zip(("HH", "HA", "AA"), PALETTE):
            xs, ys, lo, hi = [], [], [], []
            for frac in sorted(by_frac):
                value = by_frac[frac]["pair_types"][typ]
                y = value[field]
                ci = value[ci_field]
                if y is None:
                    continue
                xs.append(frac)
                ys.append(y)
                lo.append(max(0, y-ci[0]) if ci else 0)
                hi.append(max(0, ci[1]-y) if ci else 0)
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker="o", linewidth=1.7,
                            capsize=2, label=typ)
        ax.set_title(title)
        ax.set_xlabel("Target Agent-role share")
        ax.set_ylim(-.03, 1.03)
        ax.set_xticks(targets, xlabels)
        ax.grid(axis="y", color="#e7edf3", linewidth=.8)
    axes[0].set_ylabel("Error rate (task-cluster 95% interval)")
    axes[1].legend(title="Pair type", frameon=False)
    fig.suptitle("Pair-type decomposition for the command-similarity baseline", y=1.02, fontsize=10)
    fig.tight_layout()
    save(fig, "ctf-pair-type-errors")


def retrieval_figures():
    data = load_json(OUT / "retrieval/metrics.json")
    sources = [("cyber-task-horizons", "CTH\n5 candidates", .2),
               ("antieval/cybench-trajectories", "HF Cybench\n3 candidates", 1/3)]
    methods = ["command_cosine", "full_matrix", "ablate_action_organization", "window_order_shuffle"]
    labels = ["Command", "Full matrix", "Remove K4–K7", "Shuffle windows"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
    for ax, (source, title, chance) in zip(axes, sources):
        rows = {x["method"]: x for x in data if x["source"] == source and x["method"] in methods}
        xs = np.arange(len(methods))
        vals = [rows[m]["task_macro_hit_at_1"] for m in methods]
        cis = [rows[m]["task_cluster_bootstrap_95ci"] for m in methods]
        err = [[max(0, v-ci[0]) for v, ci in zip(vals, cis)], [max(0, ci[1]-v) for v, ci in zip(vals, cis)]]
        ax.bar(xs, vals, color=[PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]], width=.66)
        ax.errorbar(xs, vals, yerr=err, fmt="none", color="#273a4c", capsize=3, linewidth=1)
        ax.axhline(chance, linestyle="--", color="#738496", linewidth=1, label=f"Chance = {chance:.0%}")
        ax.set_title(title)
        ax.set_xticks(xs, labels, rotation=18, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e7edf3", linewidth=.8)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("Task-macro Hit@1 (95% task-cluster interval)")
    fig.suptitle("Agent-only task retrieval; sources and candidate counts kept separate", y=1.03, fontsize=10)
    fig.tight_layout()
    save(fig, "agent-retrieval-ablation")

    cross = load_json(OUT / "retrieval/cross_source_metrics.json")
    if cross:
        rows = [metric for direction in cross for metric in direction.get("metrics", [])
                if metric.get("method") in ("command_cosine", "command_plus_matrix")]
        if rows:
            fig, ax = plt.subplots(figsize=(7.6, 3.6))
            labels = [f"{x['train_source'].split('/')[0]} → {x['test_source'].split('/')[0]}\n{x['method'].replace('_', ' ')}" for x in rows]
            vals = [x["task_macro_hit_at_1"] for x in rows]
            cis = [x["task_cluster_bootstrap_95ci"] for x in rows]
            y = np.arange(len(rows))
            colors = [PALETTE[0] if "command_cosine" == x["method"] else PALETTE[1] for x in rows]
            for i, (v, ci, color) in enumerate(zip(vals, cis, colors)):
                ax.errorbar(v, i, xerr=[[max(0, v-ci[0])], [max(0, ci[1]-v)]], fmt="o", color=color, capsize=3)
            ax.set_yticks(y, labels)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_xlabel("Task-macro Hit@1 (95% task-cluster interval)")
            ax.set_title("Exploratory source-held-out transfer; relation remains source-native")
            ax.grid(axis="x", color="#e7edf3", linewidth=.8)
            fig.tight_layout()
            save(fig, "agent-retrieval-cross-source")


def main():
    recognition_figures()
    types = mixture_type_metrics()
    composite = mixture_composite_metrics()
    mixture_figures(types, composite)
    retrieval_figures()
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
