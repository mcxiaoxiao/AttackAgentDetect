#!/usr/bin/env python3
"""Nested task-held-out evaluation of the research association prototype."""

import collections
import csv
import itertools
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attackagent.prototype import ModeAwareAssociationPrototype, from_observation  # noqa: E402
from attackagent.traces import load_crossbenchmark_observations, load_observations  # noqa: E402

SEED = 20261029
FOLDS = 5
BOOTSTRAPS = 5000
OUT_JSON = ROOT / "results" / "mode_aware_association.json"
OUT_CSV = ROOT / "results" / "mode_aware_task_metrics.csv"
FIG_DIR = ROOT / "results" / "figures"

DIRECTIONS = {
    "human_agent": ("human", "agent"),
    "human_human": ("human", "human"),
    "agent_human": ("agent", "human"),
    "agent_agent_cross_model": ("agent", "agent"),
}


def load_rows():
    rows = [row for row in load_observations() + load_crossbenchmark_observations() if row.names]
    labels_by_task = collections.defaultdict(set)
    for row in rows:
        labels_by_task[row.task].add(row.label)
    matched = {task for task, labels in labels_by_task.items() if labels == {"human", "agent"}}
    rows = [row for row in rows if row.task in matched]
    return rows, [from_observation(row) for row in rows], matched


def ranking_metrics(scores, target):
    if target not in scores or len(scores) < 2:
        return None
    positive = scores[target]
    negatives = [score for task, score in scores.items() if task != target]
    tie = lambda x, y: np.isclose(x, y, rtol=1e-12, atol=1e-12)
    above = sum(score > positive and not tie(score, positive) for score in negatives)
    tied = sum(tie(score, positive) for score in negatives)
    # Expected reciprocal rank under a uniform random order within exact ties.
    mrr = float(np.mean([1.0 / (above + offset) for offset in range(1, tied + 2)]))
    best = max(scores.values())
    top_ties = sum(tie(score, best) for score in scores.values())
    recall = 1.0 / top_ties if tie(positive, best) else 0.0
    auc = (sum(score < positive and not tie(score, positive) for score in negatives) + 0.5 * tied) / len(negatives)
    return {"recall_at_1": float(recall), "mrr": mrr, "query_auc": float(auc),
            "candidate_tasks": len(scores), "random_recall_at_1": 1.0 / len(scores)}


def retrieval_for_fold(rows, scores_by_pair, test_tasks):
    by_task = collections.defaultdict(list)
    task_benchmark = {}
    for i, row in enumerate(rows):
        by_task[(row.benchmark, row.task, row.label)].append(i)
        task_benchmark[row.task] = row.benchmark
    output = []
    methods = list(next(iter(scores_by_pair.values())).keys())
    for task in sorted(test_tasks):
        benchmark = task_benchmark[task]
        candidate_tasks = sorted(t for t, b in task_benchmark.items() if b == benchmark)
        for direction, (query_mode, gallery_mode) in DIRECTIONS.items():
            query_rows = by_task.get((benchmark, task, query_mode), [])
            for method in methods:
                query_results = []
                for query in query_rows:
                    candidate_scores = {}
                    for candidate_task in candidate_tasks:
                        candidates = by_task.get((benchmark, candidate_task, gallery_mode), [])
                        if direction == "human_human" and query_mode == "human":
                            candidates = [j for j in candidates if j != query]
                        if direction == "agent_agent_cross_model":
                            candidates = [j for j in candidates if rows[j].source != rows[query].source]
                        pair_values = []
                        for candidate in candidates:
                            key = (min(query, candidate), max(query, candidate))
                            value = scores_by_pair.get(key)
                            if value is not None:
                                pair_values.append(value[method])
                        if pair_values:
                            candidate_scores[candidate_task] = float(np.mean(pair_values))
                    item = ranking_metrics(candidate_scores, task)
                    if item is not None:
                        query_results.append(item)
                if query_results:
                    output.append({
                        "task": task, "benchmark": benchmark, "direction": direction, "method": method,
                        "query_count": len(query_results),
                        **{metric: float(np.mean([x[metric] for x in query_results])) for metric in (
                            "recall_at_1", "mrr", "query_auc", "random_recall_at_1"
                        )},
                    })
    return output


def macro(rows, metric):
    by_benchmark = collections.defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row[metric])
    return float(np.mean([np.mean(values) for values in by_benchmark.values()]))


def mode_summary(rows, probabilities):
    y = np.asarray([int(row.label == "agent") for row in rows])

    def calculate(indices):
        yy, pp = y[indices], probabilities[indices]
        pred = pp >= .5
        return {
            "balanced_accuracy": float(balanced_accuracy_score(yy, pred)),
            "macro_f1": float(f1_score(yy, pred, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(yy, pp)) if len(set(yy)) == 2 else None,
        }

    all_idx = np.arange(len(rows))
    point = calculate(all_idx)
    by_benchmark = {}
    for benchmark in sorted({row.benchmark for row in rows}):
        indices = np.asarray([i for i, row in enumerate(rows) if row.benchmark == benchmark])
        by_benchmark[benchmark] = {"records": len(indices), **calculate(indices)}
    bootstrap = {}
    task_bench = {}
    for row in rows:
        task_bench[row.task] = row.benchmark
    tasks_by_benchmark = {
        benchmark: sorted(task for task, b in task_bench.items() if b == benchmark)
        for benchmark in sorted(set(task_bench.values()))
    }
    rng = np.random.default_rng(SEED + 404)
    draws = {name: [] for name in point}
    for _ in range(BOOTSTRAPS):
        sampled_indices = []
        for benchmark, tasks in tasks_by_benchmark.items():
            picked = rng.choice(tasks, size=len(tasks), replace=True)
            sampled_indices.extend(i for task in picked for i, row in enumerate(rows) if row.task == task)
        values = calculate(np.asarray(sampled_indices, dtype=int))
        for name, value in values.items():
            if value is not None:
                draws[name].append(value)
    for name, values in draws.items():
        bootstrap[name] = [float(np.quantile(values, .025)), float(np.quantile(values, .975))]
    return {**point, "n": len(rows), "task_stratified_95_ci": bootstrap, "by_benchmark": by_benchmark,
            "interpretation": "expert terminal-session versus selected Agent tool-call records; label is confounded by capture modality"}


def cluster_bootstrap(rows, metric, seed, n_boot=BOOTSTRAPS, other=None):
    by_benchmark = collections.defaultdict(list)
    for row in rows:
        value = row[metric] - row[other] if other else row[metric]
        by_benchmark[row["benchmark"]].append(value)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        benchmark_means = []
        for benchmark in sorted(by_benchmark):
            values = by_benchmark[benchmark]
            picked = rng.integers(0, len(values), size=len(values))
            benchmark_means.append(float(np.mean([values[i] for i in picked])))
        draws.append(float(np.mean(benchmark_means)))
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def summarize(task_metrics):
    groups = collections.defaultdict(list)
    for row in task_metrics:
        groups[(row["direction"], row["method"])].append(row)
    output = {}
    for (direction, method), rows in groups.items():
        output[f"{direction}|{method}"] = {
            "task_count": len(rows),
            "benchmark_macro": {metric: macro(rows, metric) for metric in (
                "recall_at_1", "mrr", "query_auc", "random_recall_at_1"
            )},
            "task_stratified_95_ci": {
                metric: cluster_bootstrap(rows, metric, SEED + len(output) * 13 + len(metric))
                for metric in ("recall_at_1", "mrr", "query_auc")
            },
            "by_benchmark": {
                benchmark: {
                    "task_count": sum(r["benchmark"] == benchmark for r in rows),
                    **{metric: float(np.mean([r[metric] for r in rows if r["benchmark"] == benchmark]))
                       for metric in ("recall_at_1", "mrr", "query_auc", "random_recall_at_1")},
                }
                for benchmark in sorted({r["benchmark"] for r in rows})
            },
        }

    # Paired task-cluster deltas: mode-aware score minus similarity-only score.
    deltas = {}
    for direction in DIRECTIONS:
        for metric in ("recall_at_1", "mrr", "query_auc"):
            by_task = {}
            for row in task_metrics:
                if row["direction"] == direction and row["method"] in ("similarity_only", "mode_aware"):
                    by_task.setdefault(row["task"], {})[row["method"]] = row
            pairs = []
            for task, item in by_task.items():
                if set(item) == {"similarity_only", "mode_aware"}:
                    pairs.append({"task": task, "benchmark": item["mode_aware"]["benchmark"],
                                  "delta": item["mode_aware"][metric] - item["similarity_only"][metric]})
            by_benchmark = collections.defaultdict(list)
            for row in pairs:
                by_benchmark[row["benchmark"]].append(row["delta"])
            point = float(np.mean([np.mean(v) for v in by_benchmark.values()])) if by_benchmark else None
            ci = cluster_bootstrap(
                [{"benchmark": r["benchmark"], "delta": r["delta"]} for r in pairs],
                "delta", SEED + 902 + len(deltas), other=None
            ) if pairs else None
            deltas[f"{direction}|{metric}"] = {"task_count": len(pairs), "mean_delta": point, "task_stratified_95_ci": ci}
    return output, deltas


def write_figure(summary, deltas):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    directions = ["human_agent", "agent_agent_cross_model", "agent_human", "human_human"]
    labels = ["Human → Agent", "Agent → Agent*", "Agent → Human", "Human → Human"]
    means, low, high = [], [], []
    for direction in directions:
        item = deltas.get(f"{direction}|recall_at_1")
        if item is None:
            means.append(0); low.append(0); high.append(0)
        else:
            mean = item["mean_delta"]
            ci = item["task_stratified_95_ci"]
            means.append(mean); low.append(mean - ci[0]); high.append(ci[1] - mean)
    colors = ["#4D6A86" if value < 0 else "#3D8A70" for value in means]
    fig, ax = plt.subplots(figsize=(8.3, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, width=.62)
    ax.errorbar(x, means, yerr=[low, high], fmt="none", ecolor="#20252B", capsize=4, lw=1.2)
    ax.axhline(0, color="#66717C", lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Δ task-macro Recall@1 (mode-aware − similarity-only)")
    ax.set_title("Incremental value of predicted execution-mode information")
    ax.grid(axis="y", alpha=.22)
    ax.text(.99, .02, "* Agent–Agent uses different source models", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#66717C")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mode_aware_association_delta.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "mode_aware_association_delta.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    raw_rows, records, tasks = load_rows()
    task_array = np.asarray([row.task for row in raw_rows])
    splitter = GroupKFold(n_splits=FOLDS)
    task_metrics, fold_audit = [], []
    all_mode_oof = np.full(len(records), np.nan)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(records)), groups=task_array), start=1):
        assert not ({raw_rows[i].task for i in train_idx} & {raw_rows[i].task for i in test_idx})
        model = ModeAwareAssociationPrototype(seed=SEED + fold)
        train_records = [records[i] for i in train_idx]
        model.fit(train_records)
        test_mode = model.predict_mode([records[i] for i in test_idx])
        all_mode_oof[test_idx] = [item["agent_like_score"] for item in test_mode]
        mode_probs = np.full(len(records), np.nan)
        mode_probs[train_idx] = model.training_mode_oof_
        mode_probs[test_idx] = all_mode_oof[test_idx]

        # Score each unordered pair involving an outer-fold test trace exactly once.
        test_index_set = set(int(i) for i in test_idx)
        pairs = [(i, j) for i in range(len(records)) for j in range(i + 1, len(records))
                 if i in test_index_set or j in test_index_set]
        score_arrays = model.predict_pair_scores(
            records, pairs, agent_probabilities=mode_probs,
            oracle_probabilities=np.asarray([int(row.mode == "agent") for row in records], dtype=float),
        )
        score_map = {pair: {name: float(values[k]) for name, values in score_arrays.items()}
                     for k, pair in enumerate(pairs)}
        fold_metrics = retrieval_for_fold(raw_rows, score_map, {raw_rows[i].task for i in test_idx})
        task_metrics.extend(fold_metrics)
        fold_audit.append({
            "fold": fold,
            "train_tasks": len({raw_rows[i].task for i in train_idx}),
            "test_tasks": len({raw_rows[i].task for i in test_idx}),
            "train_task_ids": sorted({raw_rows[i].task for i in train_idx}),
            "test_task_ids": sorted({raw_rows[i].task for i in test_idx}),
            "train_records": len(train_idx),
            "test_records": len(test_idx),
            "association_training_pairs": model.training_pair_counts_,
        })
        print(f"fold {fold}/{FOLDS}: {len(train_idx)} train records; {len(test_idx)} held-out records; "
              f"{model.training_pair_counts_['pairs']} balanced training pairs", flush=True)

    if np.isnan(all_mode_oof).any():
        raise AssertionError("Outer-fold predictions missing")
    summary, deltas = summarize(task_metrics)
    results = {
        "status": "post-hoc exploratory; nested task-grouped evaluation",
        "seed": SEED,
        "independent_unit": "benchmark/task_id",
        "prototype": {
            "mode_head": "TF-IDF command unigrams/bigrams + balanced logistic regression",
            "association_head": "balanced pairwise logistic ranker on command cosine and ordered action-bigram cosine; predicted mode probabilities enter as additive and interaction features",
            "training_pairs": "task-balanced; positives share benchmark/task_id; negatives are different task IDs within the same benchmark; human-human, human-agent, and cross-source Agent-Agent represented when available",
            "evaluation": "5 outer GroupKFold folds by task; training mode features are inner-group out-of-fold; candidate retrieval restricted to same benchmark",
            "score_semantics": "ranking score trained on balanced sampled pairs; not prevalence-calibrated probability and not same-campaign probability",
            "oracle": "oracle_mode_upper_bound uses true human/agent labels at prediction and is diagnostic only, not deployable",
            "script_class": "unsupported: no public script-labeled traces in the selected data",
        },
        "audit": {
            "records": len(records), "tasks": len(tasks),
            "tasks_by_benchmark": dict(collections.Counter({
                benchmark: len({row.task for row in raw_rows if row.benchmark == benchmark})
                for benchmark in {row.benchmark for row in raw_rows}
            })),
            "task_split": "all traces for one benchmark/task_id remain in a single outer fold",
        },
        "folds": fold_audit,
        "mode_head": mode_summary(raw_rows, all_mode_oof),
        "retrieval": summary,
        "paired_delta_mode_aware_minus_similarity_only": deltas,
        "task_metrics": task_metrics,
        "limitations": [
            "The target is the same CTF challenge ID, not the same attack activity, campaign, or APT actor.",
            "Human terminal transcripts and Agent tool-call records have different collection modalities; execution-mode predictions may learn this confound.",
            "No script-labeled traces are available, so the prototype cannot distinguish fixed scripts from people or Agents.",
            "Pair scores are ranking scores under balanced pair sampling; real-world prevalence calibration and decision thresholds are untested.",
            "Only 40 independent task groups across three selected benchmarks are available; intervals are exploratory.",
            "The oracle mode result uses ground-truth labels and is not a usable detector; it bounds diagnostic potential only.",
        ],
    }
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = list(task_metrics[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(task_metrics)
    write_figure(summary, deltas)
    print(json.dumps({
        "audit": results["audit"],
        "mode_head": results["mode_head"],
        "retrieval": {key: {"task_count": value["task_count"], "benchmark_macro": value["benchmark_macro"],
                            "ci": value["task_stratified_95_ci"]} for key, value in summary.items()},
        "paired_delta": deltas,
        "outputs": [str(OUT_JSON.relative_to(ROOT)), str(OUT_CSV.relative_to(ROOT)),
                    "results/figures/mode_aware_association_delta.png"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
