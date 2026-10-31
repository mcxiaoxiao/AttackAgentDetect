#!/usr/bin/env python3
"""Task-disjoint generalization to Agent model sources withheld from training.

For each Agent source in turn, hold that model's traces out of every training
fold and evaluate on its traces paired with human traces from held-out tasks.
This asks whether a command-derived Human/Agent classifier transfers to an
unseen model source within this public CTF corpus. It does not remove the
human-terminal vs Agent-tool-call collection-path confound.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from scripts.run_mode_aware_association import FOLDS, SEED, load_rows  # noqa: E402
from attackagent.traces import ATOM_GROUP  # noqa: E402

OUT_JSON = ROOT / "results" / "unseen_agent_model_generalization.json"
OUT_CSV = ROOT / "results" / "unseen_agent_model_task_metrics.csv"
FIG_DIR = ROOT / "results" / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "figures"
N_BOOTSTRAPS = 5000
FEATURES = ("command_tfidf", "action_tfidf", "count_action_mix")


def group_key(row):
    return f"{row.benchmark}/{row.activity_id}"


def count_action_features(records):
    groups = tuple(sorted(set(ATOM_GROUP.values())))
    result = np.zeros((len(records), 1 + len(groups)), dtype=float)
    for i, row in enumerate(records):
        result[i, 0] = np.log1p(len(row.commands))
        counts = collections.Counter(row.actions)
        if row.actions:
            for j, name in enumerate(groups, start=1):
                result[i, j] = counts[name] / len(row.actions)
    return result


def fit_predict(feature_name, records, train_indices, test_indices, seed):
    y = np.asarray([int(row.mode == "agent") for row in records], dtype=int)
    if feature_name == "command_tfidf":
        texts = [" ".join(row.commands) for row in records]
        vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1
        )
        vectorizer.fit([texts[i] for i in train_indices])
        matrix = vectorizer.transform(texts)
    elif feature_name == "action_tfidf":
        texts = [" ".join(row.actions) for row in records]
        vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1
        )
        vectorizer.fit([texts[i] for i in train_indices])
        matrix = vectorizer.transform(texts)
    elif feature_name == "count_action_mix":
        matrix = count_action_features(records)
    else:
        raise ValueError(feature_name)

    if len(np.unique(y[train_indices])) != 2:
        raise ValueError("The training partition must contain Human and Agent traces")
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=seed,
    )
    classifier.fit(matrix[train_indices], y[train_indices])
    return classifier.predict_proba(matrix[test_indices])[:, 1]


def task_metrics(rows, indices, scores, score_indices):
    score_map = {int(idx): float(score) for idx, score in zip(score_indices, scores)}
    by_task = collections.defaultdict(list)
    for idx in indices:
        by_task[group_key(rows[idx])].append(idx)
    result = []
    for task, task_indices in sorted(by_task.items()):
        labels = np.asarray([int(rows[i].mode == "agent") for i in task_indices])
        probabilities = np.asarray([score_map[i] for i in task_indices])
        if set(labels) != {0, 1}:
            continue
        prediction = probabilities >= 0.5
        benchmark = rows[task_indices[0]].benchmark
        result.append({
            "task": task,
            "benchmark": benchmark,
            "records": len(task_indices),
            "human_records": int(np.sum(labels == 0)),
            "agent_records": int(np.sum(labels == 1)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
            "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
        })
    return result


def macro_by_benchmark(rows, metric):
    by_benchmark = collections.defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(float(row[metric]))
    if not by_benchmark:
        return None
    return float(np.mean([np.mean(values) for values in by_benchmark.values()]))


def bootstrap_ci(rows, metric, seed, n_boot=N_BOOTSTRAPS):
    by_benchmark = collections.defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row)
    if not by_benchmark:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        benchmark_means = []
        for benchmark in sorted(by_benchmark):
            tasks = by_benchmark[benchmark]
            selected = rng.integers(0, len(tasks), size=len(tasks))
            benchmark_means.append(float(np.mean([tasks[i][metric] for i in selected])))
        draws.append(float(np.mean(benchmark_means)))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def aggregate(rows, key_fields, metrics, seed_offset=0):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    output = []
    for number, (key, subset) in enumerate(sorted(grouped.items()), start=1):
        item = {field: value for field, value in zip(key_fields, key)}
        item["task_count"] = len({row["task"] for row in subset})
        for metric in metrics:
            item[metric] = macro_by_benchmark(subset, metric)
            item[f"{metric}_95_task_cluster_ci"] = bootstrap_ci(
                subset, metric, SEED + seed_offset + number * 31 + len(metric)
            )
        output.append(item)
    return output


def aggregate_across_models(rows, metrics, seed_offset=0):
    """Macro over held-out models while resampling task clusters, not model-task rows."""
    output = []
    for feature_number, feature in enumerate(FEATURES):
        subset = [row for row in rows if row["feature"] == feature]
        if not subset:
            continue
        item = {"heldout_agent_model": "macro across three held-out models", "feature": feature}
        item["task_count"] = len({row["task"] for row in subset})
        for metric in metrics:
            grouped = collections.defaultdict(list)
            for row in subset:
                grouped[(row["benchmark"], row["task"])].append(float(row[metric]))
            by_benchmark = collections.defaultdict(list)
            for (benchmark, _task), values in grouped.items():
                by_benchmark[benchmark].append(float(np.mean(values)))
            item[metric] = float(np.mean([np.mean(values) for values in by_benchmark.values()]))

            rng = np.random.default_rng(SEED + seed_offset + feature_number * 101 + len(metric))
            by_benchmark_task = collections.defaultdict(lambda: collections.defaultdict(list))
            for row in subset:
                by_benchmark_task[row["benchmark"]][row["task"]].append(float(row[metric]))
            draws = []
            for _ in range(N_BOOTSTRAPS):
                benchmark_means = []
                for benchmark in sorted(by_benchmark_task):
                    tasks = sorted(by_benchmark_task[benchmark])
                    selected = rng.choice(tasks, size=len(tasks), replace=True)
                    task_means = [
                        np.mean(by_benchmark_task[benchmark][task]) for task in selected
                    ]
                    benchmark_means.append(float(np.mean(task_means)))
                draws.append(float(np.mean(benchmark_means)))
            item[f"{metric}_95_task_cluster_ci"] = [
                float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
            ]
        output.append(item)
    return output


def provenance_audit(records):
    rows = []
    for row in records:
        source_class = "human" if row.source == "expert" else "agent"
        rows.append({"label": row.mode, "source_class": source_class, "source": row.source})
    mismatches = sum(row["label"] != row["source_class"] for row in rows)
    counts = collections.Counter((row["label"], row["source_class"]) for row in rows)
    return {
        "records": len(rows),
        "mismatches": mismatches,
        "label_equals_capture_source_class": mismatches == 0,
        "counts": {f"{a}|{b}": n for (a, b), n in sorted(counts.items())},
        "interpretation": "This is a provenance audit, not a model score: the trace labels are assigned from collection files whose capture paths distinguish expert terminal sessions from Agent tool-call records.",
    }


def build_predictions(raw, records):
    groups = np.asarray([group_key(row) for row in records])
    agent_models = sorted({row.source for row in records if row.mode == "agent"})
    fold_summaries = []
    task_rows = []
    splitter = GroupKFold(n_splits=FOLDS)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(records)), groups=groups), start=1
    ):
        train_tasks = set(groups[train_idx])
        test_tasks = set(groups[test_idx])
        if train_tasks & test_tasks:
            raise AssertionError("Task leakage across the outer fold")
        y = np.asarray([int(row.mode == "agent") for row in records])
        fold_entry = {"fold": fold, "train_tasks": sorted(train_tasks), "test_tasks": sorted(test_tasks)}

        for agent_model in agent_models:
            eligible_tasks = {
                task for task in test_tasks
                if any(group_key(row) == task and row.mode == "human" for row in records)
                and any(group_key(row) == task and row.mode == "agent" and row.source == agent_model for row in records)
            }
            test_eval = np.asarray([
                i for i in test_idx
                if groups[i] in eligible_tasks
                and (records[i].mode == "human" or (records[i].mode == "agent" and records[i].source == agent_model))
            ], dtype=int)
            train_all = np.asarray(train_idx, dtype=int)
            train_unseen = np.asarray([
                i for i in train_idx
                if records[i].mode == "human" or records[i].source != agent_model
            ], dtype=int)
            if len(test_eval) == 0:
                continue
            if set(y[test_eval]) != {0, 1}:
                raise AssertionError(f"Unpaired test classes for {agent_model} in fold {fold}")
            for feature_number, feature_name in enumerate(FEATURES):
                all_scores = fit_predict(feature_name, records, train_all, test_eval, SEED + fold)
                unseen_scores = fit_predict(feature_name, records, train_unseen, test_eval, SEED + fold)
                all_rows = task_metrics(records, test_eval, all_scores, test_eval)
                unseen_rows = task_metrics(records, test_eval, unseen_scores, test_eval)
                all_by_task = {row["task"]: row for row in all_rows}
                unseen_by_task = {row["task"]: row for row in unseen_rows}
                if all_by_task.keys() != unseen_by_task.keys():
                    raise AssertionError("All-model and unseen-model metrics use different tasks")
                for task in sorted(all_by_task):
                    all_item, unseen_item = all_by_task[task], unseen_by_task[task]
                    task_rows.append({
                        "fold": fold,
                        "task": task,
                        "benchmark": all_item["benchmark"],
                        "heldout_agent_model": agent_model,
                        "feature": feature_name,
                        "records": all_item["records"],
                        "human_records": all_item["human_records"],
                        "agent_records": all_item["agent_records"],
                        "all_models_balanced_accuracy": all_item["balanced_accuracy"],
                        "unseen_model_balanced_accuracy": unseen_item["balanced_accuracy"],
                        "delta_balanced_accuracy": unseen_item["balanced_accuracy"] - all_item["balanced_accuracy"],
                        "all_models_macro_f1": all_item["macro_f1"],
                        "unseen_model_macro_f1": unseen_item["macro_f1"],
                        "delta_macro_f1": unseen_item["macro_f1"] - all_item["macro_f1"],
                        "all_models_roc_auc": all_item["roc_auc"],
                        "unseen_model_roc_auc": unseen_item["roc_auc"],
                        "delta_roc_auc": unseen_item["roc_auc"] - all_item["roc_auc"],
                    })
            fold_entry.setdefault("evaluated_tasks_by_model", {})[agent_model] = sorted(eligible_tasks)
            fold_entry.setdefault("training_records_by_model", {})[agent_model] = {
                "all_models": len(train_all),
                "target_model_excluded": len(train_unseen),
            }
        fold_summaries.append(fold_entry)
    return agent_models, fold_summaries, task_rows


def write_figure(aggregated, task_rows):
    selected = [
        row for row in aggregated
        if row["feature"] == "command_tfidf" and "benchmark" in row
    ]
    models = sorted({row["heldout_agent_model"] for row in selected})
    benchmarks = sorted({row["benchmark"] for row in selected})
    lookup = {(row["heldout_agent_model"], row["benchmark"]): row for row in selected}
    values = np.full((len(models), len(benchmarks)), np.nan)
    counts = np.zeros_like(values)
    for i, model in enumerate(models):
        for j, benchmark in enumerate(benchmarks):
            item = lookup.get((model, benchmark))
            if item:
                values[i, j] = item["unseen_model_balanced_accuracy"]
                counts[i, j] = item["task_count"]

    fig = plt.figure(figsize=(7.15, 3.55), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.12), left=.16, right=.92,
                            top=.82, bottom=.23, wspace=.62)
    ax = fig.add_subplot(grid[0, 0])
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, cmap="Blues", vmin=.5, vmax=1.0, aspect="auto")
    benchmark_labels = {
        "cybench": "CyBench",
        "intercode-ctf": "InterCode",
        "nyuctf": "NYU CTF",
    }
    model_labels = {
        "claude-opus-4.6": "Claude Opus 4.6",
        "claude-sonnet-4.6": "Claude Sonnet 4.6",
        "gpt-5.5": "GPT-5.5",
    }
    ax.set_xticks(np.arange(len(benchmarks)), [benchmark_labels[x] for x in benchmarks], fontsize=7.2)
    ax.set_yticks(np.arange(len(models)), [model_labels[x] for x in models], fontsize=7.2)
    ax.set_title("Unseen-model balanced accuracy", loc="left", pad=9, fontsize=8.1, fontweight="semibold")
    ax.set_xticks(np.arange(-.5, len(benchmarks), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(models)):
        for j in range(len(benchmarks)):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i,j]:.2f}\nn={int(counts[i,j])}",
                        ha="center", va="center", fontsize=7.2,
                        color="white" if values[i, j] >= .82 else "#172331")
            else:
                ax.text(j, i, "—", ha="center", va="center", color="#6E7781", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    cbar.ax.tick_params(labelsize=6.2, length=2)
    ax.text(-.20, 1.12, "A", transform=ax.transAxes, fontsize=9, color="#0072B2", fontweight="bold")

    bx = fig.add_subplot(grid[0, 1])
    paired = [row for row in task_rows if row["feature"] == "command_tfidf"]
    # Summarize paired task deltas per model, then bootstrap whole tasks within
    # benchmark so the three model-specific evaluations do not inflate n.
    delta_rows = []
    for row in paired:
        delta_rows.append({
            "task": row["task"],
            "benchmark": row["benchmark"],
            "heldout_agent_model": row["heldout_agent_model"],
            "delta": row["unseen_model_balanced_accuracy"] - row["all_models_balanced_accuracy"],
        })
    model_means = []
    for model in models:
        subset = [r for r in delta_rows if r["heldout_agent_model"] == model]
        # Equal-weight benchmark macro; aggregate each model separately.
        point = macro_by_benchmark(subset, "delta")
        ci = bootstrap_ci(subset, "delta", SEED + len(model) * 11)
        model_means.append((model, point, ci))
    y = np.arange(len(model_means))[::-1]
    for idx, (model, point, ci) in enumerate(model_means):
        bx.errorbar(point, y[idx], xerr=[[point - ci[0]], [ci[1] - point]],
                    fmt="o", color="#2878B5", ecolor="#2878B5", capsize=2.4,
                    markersize=4.2, linewidth=1.1)
    bx.axvline(0, color="#69727C", linewidth=.8, linestyle=(0, (3, 2)))
    bx.set_yticks(y, [model_labels[m] for m, _, _ in model_means], fontsize=7.2)
    bx.set_xlim(-.20, .20)
    bx.set_xlabel("Δ balanced accuracy (unseen − all)", fontsize=7.3, labelpad=5)
    bx.tick_params(axis="x", labelsize=6.8)
    bx.set_title("Paired transfer change", loc="left", pad=9, fontsize=8.1, fontweight="semibold")
    bx.grid(axis="x", color="#D9DEE4", linewidth=.55)
    bx.spines[["top", "right"]].set_visible(False)
    bx.text(-.10, 1.12, "B", transform=bx.transAxes, fontsize=9, color="#0072B2", fontweight="bold")
    fig.text(.16, .055, "Five task-grouped folds; points and 95% task-cluster intervals. n counts task groups.",
             fontsize=6.2, color="#5E6873")
    for directory in (FIG_DIR, PAPER_FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "figureA6_unseen_model_generalization.pdf", bbox_inches="tight")
        fig.savefig(directory / "figureA6_unseen_model_generalization.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def main():
    raw, records, matched = load_rows()
    if len(records) != 177 or len(matched) != 40:
        raise AssertionError(f"Unexpected frozen dataset slice: {len(records)} traces, {len(matched)} tasks")
    models, folds, rows = build_predictions(raw, records)
    if not rows:
        raise AssertionError("No paired task/model evaluation rows were created")

    metrics = (
        "all_models_balanced_accuracy", "unseen_model_balanced_accuracy", "delta_balanced_accuracy",
        "all_models_macro_f1", "unseen_model_macro_f1", "delta_macro_f1",
        "all_models_roc_auc", "unseen_model_roc_auc", "delta_roc_auc",
    )
    summaries = []
    for feature in FEATURES:
        summaries.extend(aggregate(
            [row for row in rows if row["feature"] == feature],
            ("heldout_agent_model", "benchmark", "feature"), metrics,
            seed_offset=len(summaries) * 100,
        ))
    summaries.extend(aggregate(
        rows,
        ("heldout_agent_model", "feature"), metrics,
        seed_offset=50000,
    ))
    summaries.extend(aggregate_across_models(rows, metrics, seed_offset=90000))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    csv_hash = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()
    source_manifest = sorted(
        (row.benchmark, row.activity_id, row.trace_id, row.mode, row.source)
        for row in records
    )
    result = {
        "status": "post-hoc grouped generalization stress test on fixed public CTF traces",
        "data": {
            "dataset_revision": "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae",
            "traces": len(records),
            "matched_task_groups": len(matched),
            "agent_models": models,
            "trace_manifest_sha256": hashlib.sha256(
                json.dumps(source_manifest, separators=(",", ":")).encode()
            ).hexdigest(),
            "capture_path_label_audit": provenance_audit(records),
        },
        "protocol": {
            "outer_split": f"{FOLDS}-fold GroupKFold by benchmark/task_id; complete task groups are held out",
            "unseen_model_condition": "For each target model, remove all of its Agent traces from training, then evaluate only held-out-task human traces and that model's Agent traces; target model identifiers are not features.",
            "all_models_reference": "Same held-out task folds and evaluation records, with all Agent models represented in training.",
            "features": {
                "command_tfidf": "TF-IDF command-name unigram and bigram features; vocabulary fit on the corresponding training partition.",
                "action_tfidf": "TF-IDF coarse action-category unigram and bigram features; vocabulary fit on training partition.",
                "count_action_mix": "log(1 + command count) plus within-trace proportions of fixed action categories.",
            },
            "classifier": "LogisticRegression(C=1.0, class_weight=balanced, max_iter=3000); fixed threshold 0.5; no hyperparameter search.",
            "unit": "benchmark/task_id; each per-task score uses one target Agent trace and all available human traces for that task",
            "aggregation": "task-macro within each benchmark, then equal-weight macro over benchmarks; 5,000 task-cluster bootstrap resamples stratified by benchmark",
            "seed": SEED,
        },
        "folds": folds,
        "task_level_metrics": rows,
        "summary": summaries,
        "outputs": {
            "task_metrics_csv": str(OUT_CSV.relative_to(ROOT)),
            "task_metrics_csv_sha256": csv_hash,
            "figure_pdf": "results/figures/figureA6_unseen_model_generalization.pdf",
            "figure_png": "results/figures/figureA6_unseen_model_generalization.png",
        },
        "interpretation_limits": [
            "This measures transfer across three benchmarked Agent model sources within one public CTF release, not transfer to real enterprise telemetry or unseen agent frameworks.",
            "Human labels correspond to expert terminal-session captures while Agent labels correspond to structured tool-call captures; transfer results cannot isolate behavior from capture modality.",
            "The 40 tasks are development data already used in earlier exploratory analyses; this is a stress test, not an independent confirmation set.",
            "The source-to-label mapping is recorded as a provenance audit and is not supplied to the classifier.",
            "Confidence intervals resample task groups conditional on the selected benchmarks and fixed predictions; they do not include retraining or benchmark-population uncertainty.",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
        },
        "source_script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_figure(summaries, rows)
    print(json.dumps({
        "data": result["data"],
        "fold_summary": [
            {"fold": item["fold"], "test_task_count": len(item["test_tasks"]),
             "heldout_model_task_counts": {name: len(tasks) for name, tasks in item.get("evaluated_tasks_by_model", {}).items()}}
            for item in folds
        ],
        "command_tfidf_summary": [row for row in summaries if row["feature"] == "command_tfidf"],
        "outputs": result["outputs"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
