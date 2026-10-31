#!/usr/bin/env python3
"""Nested task-held-out analysis of proxy association false splits and merges.

The unit of inference is a benchmark-qualified CTF task. A shared task ID is a
same-challenge proxy, never an attack-activity or actor label. Thresholds are
selected from inner task-held-out predictions inside each outer training fold.
No raw commands are written to the outputs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "scripts"))
import run_mixed_gallery as gallery  # noqa: E402

OUT_JSON = ROOT / "results" / "association_error_analysis.json"
PAIR_CSV = ROOT / "results" / "association_error_pair_scores.csv"
TASK_CSV = ROOT / "results" / "association_error_task_metrics.csv"
SUMMARY_CSV = ROOT / "results" / "association_error_summary.csv"
THRESHOLD_CSV = ROOT / "results" / "association_error_thresholds.csv"
FIG_DIR = ROOT / "results" / "figures"
METHODS = ("command_cosine", "behavior_only", "mode_aware", "length_only")
RATE_METRICS = (
    "false_split_rate",
    "false_split_human_human",
    "false_split_human_agent",
    "false_split_agent_agent",
    "false_merge_rate",
    "false_merge_human_human",
    "false_merge_human_agent",
    "false_merge_agent_agent",
)
METRICS = (*RATE_METRICS, "balanced_accuracy")


def fit_pair_scorers(train_rows, score_rows, seed):
    """Fit the existing behavior and mode-aware rankers without test labels."""
    all_rows = [*train_rows, *score_rows]
    n_train = len(train_rows)
    base = {
        "rows": all_rows,
        "command": gallery.cosine_matrix(train_rows, all_rows, gallery.text_commands, (1, 2)),
        "action": gallery.cosine_matrix(train_rows, all_rows, gallery.action_bigram_text, (1, 1)),
    }
    base["procgrep"], proc_vocab = gallery.procgrep_matrix(train_rows, all_rows)
    mode_raw, _, _ = gallery.fit_mode_scores(train_rows, all_rows, seed)
    train_pairs, pair_y = gallery.training_pairs(train_rows, seed)
    X_behavior = np.asarray([
        gallery.pair_features(left, right, base, mode_raw, with_mode=False)
        for left, right in train_pairs
    ])
    X_mode = np.asarray([
        gallery.pair_features(left, right, base, mode_raw, with_mode=True)
        for left, right in train_pairs
    ])
    behavior = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=seed)
    mode_aware = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=seed)
    behavior.fit(X_behavior, pair_y)
    mode_aware.fit(X_mode, pair_y)
    indices = {row.trace_id: index for index, row in enumerate(all_rows)}

    def score_many(row_pairs):
        left_indices = [indices[left.trace_id] for left, _ in row_pairs]
        right_indices = [indices[right.trace_id] for _, right in row_pairs]
        features = np.asarray([
            gallery.pair_features(left, right, base, mode_raw, with_mode=False)
            for left, right in zip(left_indices, right_indices)
        ])
        mode_features = np.asarray([
            gallery.pair_features(left, right, base, mode_raw, with_mode=True)
            for left, right in zip(left_indices, right_indices)
        ])
        return {
            "command_cosine": np.asarray([base["command"][left, right] for left, right in zip(left_indices, right_indices)]),
            "behavior_only": behavior.predict_proba(features)[:, 1],
            "mode_aware": mode_aware.predict_proba(mode_features)[:, 1],
            "length_only": features[:, 3],
        }

    return score_many, proc_vocab, len(train_pairs)


def make_pairs(rows, scorer, *, fold, split):
    """Build directed query-candidate pairs within each benchmark only."""
    edge_rows = []
    row_pairs = []
    by_benchmark = collections.defaultdict(list)
    for row in rows:
        by_benchmark[row.benchmark].append(row)
    for benchmark in sorted(by_benchmark):
        traces = sorted(by_benchmark[benchmark], key=lambda row: row.trace_id)
        for query in traces:
            for candidate in traces:
                if query.trace_id == candidate.trace_id:
                    continue
                same_task = int(query.task == candidate.task)
                labels = {query.label, candidate.label}
                pair_type = "H-A" if labels == {"human", "agent"} else ("H-H" if labels == {"human"} else "A-A")
                row = {
                    "split": split,
                    "fold": fold,
                    "benchmark": benchmark,
                    "query_task": query.task,
                    "candidate_task": candidate.task,
                    "query_trace_id": query.trace_id,
                    "candidate_trace_id": candidate.trace_id,
                    "query_mode_for_evaluation_only": query.label,
                    "candidate_mode_for_evaluation_only": candidate.label,
                    "same_task_proxy_for_evaluation_only": same_task,
                    "pair_type_for_evaluation_only": pair_type,
                }
                edge_rows.append(row)
                row_pairs.append((query, candidate))
    score_arrays = scorer(row_pairs)
    for index, row in enumerate(edge_rows):
        row.update({method: float(scores[index]) for method, scores in score_arrays.items()})
    return edge_rows


def _query_rates(pair_rows, method, threshold):
    """Return query-level error rates, first averaging candidate pairs per query."""
    query_groups = collections.defaultdict(list)
    for pair in pair_rows:
        query_groups[(pair["benchmark"], pair["query_task"], pair["query_trace_id"])].append(pair)
    task_values = collections.defaultdict(lambda: collections.defaultdict(list))
    task_counts = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for (benchmark, task, query_id), pairs in query_groups.items():
        grouped = collections.defaultdict(list)
        for pair in pairs:
            grouped[(pair["same_task_proxy_for_evaluation_only"], pair["pair_type_for_evaluation_only"])].append(pair)
        for (same_task, pair_type), values in grouped.items():
            accepted = np.asarray([pair[method] >= threshold for pair in values], dtype=float)
            if same_task:
                # The all-pair metric is recorded once below by merging all positive groups.
                task_values[(benchmark, task)][f"positive:{query_id}:{pair_type}"].append(1.0 - float(accepted.mean()))
                metric_name = {"H-H": "false_split_human_human", "H-A": "false_split_human_agent", "A-A": "false_split_agent_agent"}[pair_type]
                task_values[(benchmark, task)][f"query:{query_id}:{metric_name}"].append(1.0 - float(accepted.mean()))
                task_counts[(benchmark, task)][metric_name][0] += len(values)
                task_counts[(benchmark, task)][metric_name][1] += 1
            else:
                metric_name = {"H-H": "false_merge_human_human", "H-A": "false_merge_human_agent", "A-A": "false_merge_agent_agent"}[pair_type]
                task_values[(benchmark, task)][f"query:{query_id}:{metric_name}"].append(float(accepted.mean()))
                task_counts[(benchmark, task)][metric_name][0] += len(values)
                task_counts[(benchmark, task)][metric_name][1] += 1
        # Query-level overall rates average all eligible candidates, not the three strata.
        positive = [p for p in pairs if p["same_task_proxy_for_evaluation_only"] == 1]
        negative = [p for p in pairs if p["same_task_proxy_for_evaluation_only"] == 0]
        if positive:
            rate = float(np.mean([pair[method] < threshold for pair in positive]))
            task_values[(benchmark, task)][f"query:{query_id}:false_split_rate"].append(rate)
            task_counts[(benchmark, task)]["false_split_rate"][0] += len(positive)
            task_counts[(benchmark, task)]["false_split_rate"][1] += 1
        if negative:
            rate = float(np.mean([pair[method] >= threshold for pair in negative]))
            task_values[(benchmark, task)][f"query:{query_id}:false_merge_rate"].append(rate)
            task_counts[(benchmark, task)]["false_merge_rate"][0] += len(negative)
            task_counts[(benchmark, task)]["false_merge_rate"][1] += 1

    # Collapse query values to a task-level macro rate.
    rows = []
    for benchmark, task in sorted(task_values):
        row = {"benchmark": benchmark, "task": task}
        for metric in RATE_METRICS:
            per_query = collections.defaultdict(list)
            prefix = f"query:"
            for key, values in task_values[(benchmark, task)].items():
                if key.startswith(prefix) and key.endswith(metric):
                    query_id = key[len(prefix):].rsplit(":", 1)[0]
                    per_query[query_id].extend(values)
            if per_query:
                row[metric] = float(np.mean([np.mean(values) for values in per_query.values()]))
                row[f"{metric}_query_count"] = len(per_query)
                row[f"{metric}_pair_count"] = int(task_counts[(benchmark, task)][metric][0])
            else:
                row[metric] = None
                row[f"{metric}_query_count"] = 0
                row[f"{metric}_pair_count"] = 0
        if row["false_split_rate"] is not None and row["false_merge_rate"] is not None:
            row["balanced_accuracy"] = 1.0 - 0.5 * (row["false_split_rate"] + row["false_merge_rate"])
            row["balanced_accuracy_query_count"] = min(
                row["false_split_rate_query_count"], row["false_merge_rate_query_count"]
            )
            row["balanced_accuracy_pair_count"] = (
                row["false_split_rate_pair_count"] + row["false_merge_rate_pair_count"]
            )
        else:
            row["balanced_accuracy"] = None
            row["balanced_accuracy_query_count"] = 0
            row["balanced_accuracy_pair_count"] = 0
        rows.append(row)
    return rows


def _macro_metric(task_rows, metric):
    by_benchmark = collections.defaultdict(list)
    for row in task_rows:
        value = row.get(metric)
        if value is not None:
            by_benchmark[row["benchmark"]].append(float(value))
    per_benchmark = {benchmark: float(np.mean(values)) for benchmark, values in by_benchmark.items()}
    return (float(np.mean(list(per_benchmark.values()))) if per_benchmark else None,
            len(per_benchmark), sum(len(values) for values in by_benchmark.values()))


def select_threshold(pair_rows, method, grid_size=101):
    """Tune a method-specific threshold on nested, task-held-out training pairs."""
    scores = np.asarray([pair[method] for pair in pair_rows], dtype=float)
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, grid_size)))
    candidates = np.unique(np.concatenate(([np.nextafter(scores.min(), -np.inf)], candidates,
                                            [np.nextafter(scores.max(), np.inf)])))
    best = None
    for threshold in candidates:
        rates = _query_rates(pair_rows, method, float(threshold))
        false_split, split_benchmarks, _ = _macro_metric(rates, "false_split_rate")
        false_merge, merge_benchmarks, _ = _macro_metric(rates, "false_merge_rate")
        if false_split is None or false_merge is None:
            continue
        score = 1.0 - 0.5 * (false_split + false_merge)
        candidate = (score, float(threshold), false_split, false_merge, split_benchmarks, merge_benchmarks)
        # Conservative deterministic tie break: prefer the higher threshold.
        if best is None or score > best[0] + 1e-12 or (abs(score - best[0]) <= 1e-12 and threshold > best[1]):
            best = candidate
    if best is None:
        raise RuntimeError(f"Could not select a threshold for {method}")
    return {
        "threshold": best[1],
        "training_macro_balanced_accuracy": best[0],
        "training_macro_false_split_rate": best[2],
        "training_macro_false_merge_rate": best[3],
        "training_benchmarks_for_split": best[4],
        "training_benchmarks_for_merge": best[5],
        "threshold_fit_pair_rows": len(pair_rows),
        "threshold_grid_size": grid_size,
        "tie_rule": "highest threshold among equal task-macro balanced-accuracy candidates",
    }


def task_cluster_interval(task_rows, metric, *, n_boot=5000, seed=20261030):
    by_benchmark = collections.defaultdict(list)
    for row in task_rows:
        if row.get(metric) is not None:
            by_benchmark[row["benchmark"]].append(float(row[metric]))
    point_by_benchmark = {key: float(np.mean(values)) for key, values in by_benchmark.items()}
    point = float(np.mean(list(point_by_benchmark.values()))) if point_by_benchmark else None
    if point is None:
        return {"estimate": None, "ci_95": None, "task_count": 0, "benchmark_count": 0}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for index in range(n_boot):
        sampled_means = []
        for benchmark in sorted(by_benchmark):
            values = np.asarray(by_benchmark[benchmark])
            sampled_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        draws[index] = np.mean(sampled_means)
    return {
        "estimate": point,
        "ci_95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "task_count": sum(len(values) for values in by_benchmark.values()),
        "benchmark_count": len(by_benchmark),
        "bootstrap_unit": "benchmark-qualified task, resampled within benchmark",
        "bootstrap_draws": n_boot,
    }


def paired_delta_interval(task_rows, metric, *, baseline="behavior_only", n_boot=5000, seed=20261030):
    by_method_task = {(row["benchmark"], row["task"], row["method"]): row.get(metric) for row in task_rows}
    by_benchmark = collections.defaultdict(list)
    for row in task_rows:
        if row["method"] != "mode_aware":
            continue
        mode_value = row.get(metric)
        base_value = by_method_task.get((row["benchmark"], row["task"], baseline))
        if mode_value is not None and base_value is not None:
            by_benchmark[row["benchmark"]].append(float(mode_value) - float(base_value))
    point = float(np.mean([np.mean(values) for values in by_benchmark.values()])) if by_benchmark else None
    if point is None:
        return {"mode_aware_minus_behavior_only": None, "ci_95": None, "task_count": 0}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for index in range(n_boot):
        vals = []
        for benchmark in sorted(by_benchmark):
            values = np.asarray(by_benchmark[benchmark])
            vals.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        draws[index] = np.mean(vals)
    return {
        f"mode_aware_minus_{baseline}": point,
        "ci_95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "task_count": sum(len(values) for values in by_benchmark.values()),
        "bootstrap_unit": "paired benchmark-qualified task, resampled within benchmark",
        "bootstrap_draws": n_boot,
    }


def draw_figure(summary, path_base):
    """Draw task-cluster intervals for the error rates that expose the mechanism."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = {
        "command_cosine": "#99CDFC",
        "behavior_only": "#45A6FC",
        "mode_aware": "#173F68",
        "length_only": "#A9B8C8",
    }
    labels = {
        "command_cosine": "Command cosine",
        "behavior_only": "Behavior ranker",
        "mode_aware": "Mode-aware ranker",
        "length_only": "Length control",
    }
    group_metrics = [
        ("false_split_human_human", "H–H"),
        ("false_split_human_agent", "H–A"),
        ("false_split_agent_agent", "A–A"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True, constrained_layout=True)
    methods = list(METHODS)
    offsets = np.linspace(-0.27, 0.27, len(methods))
    x = np.arange(len(group_metrics))
    for ax, prefix, title in [
        (axes[0], "false_split", "Same-task proxy: false splits"),
        (axes[1], "false_merge", "Different-task proxy: false merges"),
    ]:
        for method_index, method in enumerate(methods):
            ys, lows, highs = [], [], []
            for metric, _ in group_metrics:
                target = metric if prefix == "false_split" else metric.replace("false_split", "false_merge")
                entry = summary[method][target]
                value = entry["estimate"]
                interval = entry["ci_95"]
                ys.append(np.nan if value is None else value * 100)
                lows.append(np.nan if interval is None else (value - interval[0]) * 100)
                highs.append(np.nan if interval is None else (interval[1] - value) * 100)
            ax.errorbar(x + offsets[method_index], ys, yerr=[lows, highs], fmt="o", markersize=5,
                        capsize=2.5, linewidth=1.2, color=colors[method], label=labels[method],
                        markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.set_xticks(x, [label for _, label in group_metrics])
        ax.set_title(title, loc="left", pad=10, fontweight="semibold")
        ax.set_ylim(-4, 104)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", color="#D8E2EB", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Task-macro error rate (%)")
    axes[0].legend(frameon=False, loc="upper left", ncol=2, fontsize=8, handletextpad=0.5,
                   columnspacing=1.0, borderaxespad=0.2)
    fig.suptitle("Execution-mode context under a training-only match threshold", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color="#142A3B")
    fig.text(0.01, -0.015,
             "Pairs are labeled by shared CTF task only; this is not attack-activity ground truth. Error bars: 95% task-cluster bootstrap CI.",
             ha="left", va="top", fontsize=7.7, color="#536476")
    for ext in ("pdf", "png"):
        fig.savefig(path_base.with_suffix(f".{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(n_boot=5000, threshold_grid=101):
    rows = gallery.load_rows()
    if len(rows) != 177 or len({(row.benchmark, row.task) for row in rows}) != 40:
        raise SystemExit(f"Pinned trace audit changed: got {len(rows)} rows / {len({(r.benchmark, r.task) for r in rows})} task groups")
    outer_folds = gallery.split_tasks(rows, seed=gallery.BASE_SEED, n_folds=3)
    pair_rows, task_rows, threshold_rows, fold_records = [], [], [], []

    for outer in outer_folds:
        outer_train = [row for row in rows if row.task in set(outer["train_tasks"])]
        outer_test = [row for row in rows if row.task in set(outer["test_tasks"])]
        calibration_pairs = []
        inner_folds = gallery.split_tasks(
            outer_train,
            seed=gallery.BASE_SEED + 7000 + outer["fold"],
            n_folds=3,
        )
        calibration_task_count = set()
        calibration_vocab = []
        calibration_pair_fits = 0
        for inner in inner_folds:
            inner_train = [row for row in outer_train if row.task in set(inner["train_tasks"])]
            inner_valid = [row for row in outer_train if row.task in set(inner["test_tasks"])]
            scorer, vocab, fit_pair_count = fit_pair_scorers(
                inner_train, inner_valid,
                gallery.BASE_SEED + outer["fold"] * 100 + inner["fold"],
            )
            calibration_pairs.extend(make_pairs(inner_valid, scorer, fold=outer["fold"], split="threshold_oof"))
            calibration_task_count.update((row.benchmark, row.task) for row in inner_valid)
            calibration_vocab.append(vocab)
            calibration_pair_fits += fit_pair_count
        if calibration_task_count & {(row.benchmark, row.task) for row in outer_test}:
            raise AssertionError("outer test task leaked into threshold calibration")

        thresholds = {}
        for method in METHODS:
            fitted = select_threshold(calibration_pairs, method, grid_size=threshold_grid)
            thresholds[method] = fitted["threshold"]
            threshold_rows.append({"outer_fold": outer["fold"], "method": method,
                                   "validation_task_groups": len(calibration_task_count), **fitted})

        final_scorer, proc_vocab, final_pair_count = fit_pair_scorers(
            outer_train, outer_test, gallery.BASE_SEED + outer["fold"]
        )
        current_pairs = make_pairs(outer_test, final_scorer, fold=outer["fold"], split="outer_test")
        for row in current_pairs:
            for method in METHODS:
                row[f"threshold_{method}"] = thresholds[method]
                row[f"predicted_same_task_proxy_{method}"] = int(row[method] >= thresholds[method])
        pair_rows.extend(current_pairs)

        for method in METHODS:
            rates = _query_rates(current_pairs, method, thresholds[method])
            for rate_row in rates:
                task_rows.append({
                    "outer_fold": outer["fold"],
                    "method": method,
                    "threshold": thresholds[method],
                    **rate_row,
                })
        fold_records.append({
            "outer_fold": outer["fold"],
            "train_task_groups": len({(row.benchmark, row.task) for row in outer_train}),
            "test_task_groups": len({(row.benchmark, row.task) for row in outer_test}),
            "threshold_oof_task_groups": len(calibration_task_count),
            "threshold_oof_directed_pair_rows": len(calibration_pairs),
            "threshold_oof_pair_fit_count_sum": calibration_pair_fits,
            "outer_training_pair_count": final_pair_count,
            "outer_test_directed_pair_rows": len(current_pairs),
            "procgrep_vocabulary": proc_vocab,
            "thresholds": thresholds,
        })

    summary = {}
    for method in METHODS:
        summary[method] = {}
        method_rows = [row for row in task_rows if row["method"] == method]
        for index, metric in enumerate(METRICS):
            summary[method][metric] = task_cluster_interval(
                method_rows, metric, n_boot=n_boot,
                seed=gallery.BASE_SEED + index + METHODS.index(method) * 101,
            )

    paired = {
        baseline: {metric: paired_delta_interval(
            task_rows, metric, baseline=baseline, n_boot=n_boot,
            seed=gallery.BASE_SEED + 2000 + index + 100 * METHODS.index(baseline),
        ) for index, metric in enumerate(METRICS)}
        for baseline in ("behavior_only", "command_cosine")
    }

    pair_counts = collections.defaultdict(collections.Counter)
    for row in pair_rows:
        key = ("same_task_proxy" if row["same_task_proxy_for_evaluation_only"] else "different_task_proxy",
               row["pair_type_for_evaluation_only"])
        pair_counts[key][row["benchmark"]] += 1
    counts = [
        {"pair_label": label, "pair_type": pair_type,
         "directed_pair_count": int(sum(per_benchmark.values())),
         "task_count": len({(row["benchmark"], row["query_task"]) for row in pair_rows
                             if ("same_task_proxy" if row["same_task_proxy_for_evaluation_only"] else "different_task_proxy") == label
                             and row["pair_type_for_evaluation_only"] == pair_type}),
         "pairs_by_benchmark": dict(sorted(per_benchmark.items()))}
        for (label, pair_type), per_benchmark in sorted(pair_counts.items())
    ]

    source_manifest = json.loads((ROOT / "data" / "manifests" / "crossbenchmark_public_subset.json").read_text())
    script_hash = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    result = {
        "status": "exploratory proxy association-error evaluation; nested task-held-out thresholds",
        "protocol_version": "association-error-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "Cyber Task Horizons public traces",
            "revision": source_manifest["revision"],
            "license": source_manifest["data_license"],
            "records": len(rows),
            "benchmark_qualified_task_groups": len({(row.benchmark, row.task) for row in rows}),
            "records_by_mode": dict(collections.Counter(row.label for row in rows)),
            "mode_capture_confound": "Human terminal-input captures and Agent structured tool-call records; source pathway is confounded with the release mode label.",
        },
        "evaluation": {
            "outer_protocol": "three benchmark-stratified folds, grouped by benchmark-qualified task; seed 20261030",
            "threshold_protocol": "inside each outer training fold, three additional task-grouped folds produce out-of-fold pairs; method-specific threshold maximizes task-macro balanced accuracy on these pairs only",
            "threshold_grid": f"{threshold_grid} empirical score quantiles plus observed score bounds",
            "threshold_tie_rule": "choose the highest threshold among tied validation balanced-accuracy candidates",
            "test_pairs": "all directed query-candidate pairs within the same benchmark and outer held-out fold; no mode filtering",
            "positive_proxy": "same benchmark-qualified CTF task; not same attack activity",
            "negative_proxy": "different tasks in the same benchmark; not verified different operators or campaigns",
            "decision_rule": "score >= outer-fold-specific threshold means associate",
            "methods": {
                "command_cosine": "existing command TF-IDF cosine baseline",
                "behavior_only": "existing task-supervised pair ranker using command, action-bigram, ProcGrep, and length similarity",
                "mode_aware": "same task-supervised ranker with out-of-fold execution-mode agreement and gap features",
                "length_only": "command-count similarity negative control",
            },
            "metric_aggregation": "candidate pairs averaged per query; queries averaged per task; tasks averaged within benchmark; benchmarks equally weighted",
            "uncertainty": "5,000 (configurable) benchmark-stratified task-cluster bootstrap draws, conditional on fitted folds; paired intervals for mode-aware minus behavior-only",
            "error_definitions": "false split = reject a same-task proxy pair; false merge = accept a different-task proxy pair; H-A denotes a cross-mode pair",
        },
        "pair_counts": counts,
        "thresholds": fold_records,
        "summary": summary,
        "paired_deltas": paired,
        "paired_deltas_mode_aware_minus_behavior_only": paired["behavior_only"],
        "outputs": {
            "pair_scores": "results/association_error_pair_scores.csv (held-out IDs, proxy labels, scores, and decisions; no raw commands)",
            "task_metrics": "results/association_error_task_metrics.csv",
            "summary": "results/association_error_summary.csv",
            "thresholds": "results/association_error_thresholds.csv",
            "figure": "scripts/run_association_error_analysis.py -> results/figures/association_error_rates.pdf/png",
        },
        "source_sha256": script_hash,
        "interpretation_guard": "This experiment quantifies same-challenge proxy errors in a dataset whose mode labels are confounded with capture pathway. It cannot establish real APT activity-level false merges/splits, causality, or general actor identity recognition.",
    }
    for row in pair_rows:
        for method in METHODS:
            row[method] = round(float(row[method]), 10)
            row[f"threshold_{method}"] = round(float(row[f"threshold_{method}"]), 10)
    write_csv(PAIR_CSV, pair_rows)
    write_csv(TASK_CSV, task_rows)
    summary_rows = []
    for method in METHODS:
        for metric in METRICS:
            summary_rows.append({"method": method, "metric": metric, **summary[method][metric]})
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(THRESHOLD_CSV, threshold_rows)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    draw_figure(summary, FIG_DIR / "association_error_rates")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--threshold-grid", type=int, default=101)
    args = parser.parse_args()
    if args.bootstraps < 100:
        parser.error("--bootstraps must be at least 100")
    if args.threshold_grid < 21:
        parser.error("--threshold-grid must be at least 21")
    result = run(n_boot=args.bootstraps, threshold_grid=args.threshold_grid)
    print(json.dumps({
        "output": str(OUT_JSON),
        "dataset": result["dataset"],
        "summary": result["summary"],
        "paired_deltas_mode_aware_minus_behavior_only": result["paired_deltas_mode_aware_minus_behavior_only"],
        "paired_deltas_mode_aware_minus_command_cosine": result["paired_deltas"]["command_cosine"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
