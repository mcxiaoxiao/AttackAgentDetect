#!/usr/bin/env python3
"""Task-grouped retrieval and execution-mode diagnostics on pinned public traces.

Primary outcome is same-challenge retrieval among candidates from the same
benchmark. A challenge ID is only a proxy label; it is not an attack campaign.
"""

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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import cosine_similarity

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "vendor" / "procgrep" / "src"))
from attackagent.traces import ATOM_GROUP, load_crossbenchmark_observations, load_observations  # noqa: E402
from procgrep import Trace, encode, fit_bpe, jsd  # noqa: E402
from procgrep.bpe import ProcedureVocabulary  # noqa: E402

SEED = 20261029
FOLDS = 5
BOOTSTRAPS = 5000
OUT_JSON = ROOT / "results" / "crossbenchmark_exploration.json"
OUT_CSV = ROOT / "results" / "crossbenchmark_task_metrics.csv"
FIG_DIR = ROOT / "results" / "figures"


def matched_rows():
    all_rows = load_observations() + load_crossbenchmark_observations()
    all_rows = [row for row in all_rows if row.names]
    labels_by_task = collections.defaultdict(set)
    for row in all_rows:
        labels_by_task[row.task].add(row.label)
    matched = {task for task, labels in labels_by_task.items() if labels == {"human", "agent"}}
    return [row for row in all_rows if row.task in matched], matched


def group_folds(tasks):
    task_list = sorted(tasks)
    if len(task_list) < FOLDS:
        raise ValueError(f"Need at least {FOLDS} independent task groups; got {len(task_list)}")
    splitter = GroupKFold(n_splits=FOLDS)
    for train_idx, test_idx in splitter.split(np.zeros(len(task_list)), groups=task_list):
        yield set(task_list[i] for i in train_idx), set(task_list[i] for i in test_idx)


def safe_vectorize(vectorizer, train_text, all_text):
    try:
        vectorizer.fit(train_text)
        matrix = vectorizer.transform(all_text)
        if matrix.shape[1]:
            return matrix
    except ValueError:
        pass
    from scipy.sparse import csr_matrix
    return csr_matrix((len(all_text), 0), dtype=np.float64)


def procgrep_similarity(rows, train_indices):
    alphabet = tuple(sorted(set(ATOM_GROUP.values())))
    train_corpus = [list(rows[i].groups) for i in train_indices if rows[i].groups]
    vocab = fit_bpe(train_corpus, vocab_size=32, seed=SEED, fit_corpus="outer-fold training tasks")
    vocab = ProcedureVocabulary(
        atoms=alphabet, merges=vocab.merges, seed=vocab.seed,
        min_pair_frequency=vocab.min_pair_frequency, fit_corpus=vocab.fit_corpus,
    )
    traces = [Trace(trace_id=r.trace_id, agent=r.source, atoms=list(r.groups)) for r in rows]
    fingerprints = encode(traces, vocab=vocab)
    n = len(rows)
    sim = np.eye(n, dtype=np.float64)
    distributions = [fp.distribution() for fp in fingerprints]
    for i, j in itertools.combinations(range(n), 2):
        score = 1.0 - jsd(distributions[i], distributions[j])
        sim[i, j] = sim[j, i] = score
    return sim, vocab.spec.compact()


def representations(rows, train_indices):
    all_names = [" ".join(row.names) for row in rows]
    all_groups = [" ".join(row.groups) for row in rows]
    train_names = [all_names[i] for i in train_indices]
    train_groups = [all_groups[i] for i in train_indices]
    specs = {
        "command_unigram": (CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 1)), train_names, all_names),
        "ordered_command_bigram": (CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(2, 2)), train_names, all_names),
        "ordered_action_bigram": (CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(2, 2)), train_groups, all_groups),
    }
    result = {}
    for name, (vectorizer, train_text, corpus) in specs.items():
        matrix = safe_vectorize(vectorizer, train_text, corpus)
        result[name] = cosine_similarity(matrix) if matrix.shape[1] else np.zeros((len(rows), len(rows)))
    proc, proc_spec = procgrep_similarity(rows, train_indices)
    result["procgrep_local_patterns"] = proc
    return result, proc_spec


def ranking_metrics(candidate_scores, positive_task):
    if positive_task not in candidate_scores or len(candidate_scores) < 2:
        return None
    positive = candidate_scores[positive_task]
    negative = [score for task, score in candidate_scores.items() if task != positive_task]
    ties = sum(np.isclose(score, positive, rtol=1e-12, atol=1e-12) for score in negative)
    above = sum(score > positive and not np.isclose(score, positive, rtol=1e-12, atol=1e-12) for score in negative)
    # Average reciprocal rank over a uniformly random ordering within ties.
    # Taking the reciprocal of the average rank understates tied results.
    mrr = float(np.mean([1.0 / (above + offset) for offset in range(1, ties + 2)]))
    top = max(candidate_scores.values())
    top_ties = sum(np.isclose(score, top, rtol=1e-12, atol=1e-12) for score in candidate_scores.values())
    recall_at_1 = 1.0 / top_ties if np.isclose(positive, top, rtol=1e-12, atol=1e-12) else 0.0
    auc = (sum(score < positive and not np.isclose(score, positive, rtol=1e-12, atol=1e-12)
               for score in negative) + 0.5 * ties) / len(negative)
    return {"recall_at_1": recall_at_1, "mrr": mrr, "query_auc": auc,
            "candidate_tasks": len(candidate_scores), "random_recall_at_1": 1.0 / len(candidate_scores)}


def score_candidate(matrix, rows, query_index, candidate_indices, direction):
    query = rows[query_index]
    valid = [i for i in candidate_indices if i != query_index]
    if direction == "human_human" and query.label == "human":
        valid = [i for i in valid if rows[i].trace_id != query.trace_id]
    if direction == "agent_agent_cross_model":
        valid = [i for i in valid if rows[i].source != query.source]
    if not valid:
        return None
    sims = np.asarray([matrix[query_index, i] for i in valid], dtype=float)
    if direction == "human_agent":
        by_model = collections.defaultdict(list)
        for i, score in zip(valid, sims):
            by_model[rows[i].source].append(float(score))
        return float(np.mean([np.mean(scores) for scores in by_model.values()]))
    return float(np.mean(sims))


DIRECTIONS = {
    "human_agent": ("human", "agent"),
    "human_human": ("human", "human"),
    "agent_human": ("agent", "human"),
    "agent_agent_cross_model": ("agent", "agent"),
}


def retrieval_for_fold(rows, matrices, test_tasks):
    index_by_group = collections.defaultdict(list)
    task_bench = {}
    for i, row in enumerate(rows):
        index_by_group[(row.benchmark, row.task, row.label)].append(i)
        task_bench[row.task] = row.benchmark
    output = []
    for task in sorted(test_tasks):
        benchmark = task_bench[task]
        candidate_tasks = sorted(t for t, b in task_bench.items() if b == benchmark)
        for direction, (query_label, gallery_label) in DIRECTIONS.items():
            query_indices = index_by_group.get((benchmark, task, query_label), [])
            for method, matrix in matrices.items():
                per_query = []
                for q in query_indices:
                    scores = {}
                    for candidate_task in candidate_tasks:
                        candidates = index_by_group.get((benchmark, candidate_task, gallery_label), [])
                        value = score_candidate(matrix, rows, q, candidates, direction)
                        if value is not None:
                            scores[candidate_task] = value
                    metrics = ranking_metrics(scores, task)
                    if metrics is not None:
                        per_query.append(metrics)
                if per_query:
                    output.append({
                        "task": task, "benchmark": benchmark, "direction": direction, "method": method,
                        "query_count": len(per_query),
                        "candidate_tasks": float(np.mean([m["candidate_tasks"] for m in per_query])),
                        "random_recall_at_1": float(np.mean([m["random_recall_at_1"] for m in per_query])),
                        "recall_at_1": float(np.mean([m["recall_at_1"] for m in per_query])),
                        "mrr": float(np.mean([m["mrr"] for m in per_query])),
                        "query_auc": float(np.mean([m["query_auc"] for m in per_query])),
                    })
    return output


def classifier_features(rows, train_indices, test_indices):
    texts = [" ".join(row.names) for row in rows]
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1)
    X = safe_vectorize(vectorizer, [texts[i] for i in train_indices], texts)
    y = np.asarray([int(row.label == "agent") for row in rows])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED)
    clf.fit(X[train_indices], y[train_indices])
    prob = clf.predict_proba(X[test_indices])[:, 1]

    # Negative controls: command volume and action mix; they are still behavioral
    # summaries, not valid identity cues, and are compared on identical task folds.
    groups = tuple(sorted(set(ATOM_GROUP.values())))
    summary = np.zeros((len(rows), 1 + len(groups)), dtype=float)
    for i, row in enumerate(rows):
        summary[i, 0] = np.log1p(len(row.names))
        counts = collections.Counter(row.groups)
        if row.groups:
            for j, group in enumerate(groups, start=1):
                summary[i, j] = counts[group] / len(row.groups)
    coarse = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED)
    coarse.fit(summary[train_indices], y[train_indices])
    coarse_prob = coarse.predict_proba(summary[test_indices])[:, 1]

    # Benchmark-only control estimates whether suite identity alone explains labels.
    benchmarks = sorted({row.benchmark for row in rows})
    bmat = np.zeros((len(rows), len(benchmarks)), dtype=float)
    lookup = {b: j for j, b in enumerate(benchmarks)}
    for i, row in enumerate(rows):
        bmat[i, lookup[row.benchmark]] = 1.0
    bc = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED)
    bc.fit(bmat[train_indices], y[train_indices])
    bench_prob = bc.predict_proba(bmat[test_indices])[:, 1]
    return {
        "tfidf_command_unigram_bigram": prob,
        "command_count_and_action_mix": coarse_prob,
        "benchmark_only_negative_control": bench_prob,
    }, y


def metric_summary(y, probability):
    pred = probability >= 0.5
    return {
        "n": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "human_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "agent_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(set(y)) == 2 else None,
        "confusion_matrix_human_agent": [
            [int(np.sum((y == 0) & (pred == 0))), int(np.sum((y == 0) & (pred == 1)))],
            [int(np.sum((y == 1) & (pred == 0))), int(np.sum((y == 1) & (pred == 1)))],
        ],
    }


def cluster_bootstrap(values, seed, n_boot=BOOTSTRAPS):
    """Stratified task-cluster bootstrap; benchmarks are fixed strata."""
    by_benchmark = collections.defaultdict(list)
    for row in values:
        by_benchmark[row["benchmark"]].append(row)
    rng = np.random.default_rng(seed)
    draws = []
    benchmarks = sorted(by_benchmark)
    if not benchmarks or any(not by_benchmark[b] for b in benchmarks):
        return None
    for _ in range(n_boot):
        means = []
        for benchmark in benchmarks:
            rows = by_benchmark[benchmark]
            selected = rng.integers(0, len(rows), size=len(rows))
            means.append(np.mean([rows[i]["value"] for i in selected]))
        draws.append(float(np.mean(means)))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize_retrieval(task_metrics):
    grouped = collections.defaultdict(list)
    for row in task_metrics:
        grouped[(row["method"], row["direction"])].append(row)
    summary = {}
    for (method, direction), rows in grouped.items():
        item = {"task_count": len(rows), "by_benchmark": {}}
        for benchmark in sorted({r["benchmark"] for r in rows}):
            values = [r for r in rows if r["benchmark"] == benchmark]
            item["by_benchmark"][benchmark] = {
                "task_count": len(values),
                **{metric: float(np.mean([r[metric] for r in values]))
                   for metric in ("recall_at_1", "mrr", "query_auc", "random_recall_at_1")},
            }
        item["benchmark_macro"] = {
            metric: float(np.mean([v[metric] for v in item["by_benchmark"].values()]))
            for metric in ("recall_at_1", "mrr", "query_auc", "random_recall_at_1")
        }
        item["task_stratified_95_ci"] = {}
        for metric in ("recall_at_1", "mrr", "query_auc"):
            ci_rows = [{"benchmark": r["benchmark"], "value": r[metric]} for r in rows]
            item["task_stratified_95_ci"][metric] = cluster_bootstrap(
                ci_rows, SEED + len(summary) * 17 + len(metric)
            )
        summary[f"{method}|{direction}"] = item
    return summary


def summarize_classifier(rows, predictions, labels, tasks):
    y = labels
    out = {}
    for method, scores in predictions.items():
        overall = metric_summary(y, scores)
        overall["task_stratified_95_ci"] = {}
        for metric in ("balanced_accuracy", "macro_f1", "roc_auc"):
            boot = []
            task_ids = sorted(tasks)
            rng = np.random.default_rng(SEED + len(method) + len(metric))
            for _ in range(BOOTSTRAPS):
                selected = rng.choice(task_ids, size=len(task_ids), replace=True)
                idx = [i for task in selected for i, row in enumerate(rows) if row.task == task]
                val = metric_summary(y[idx], scores[idx])[metric]
                if val is not None:
                    boot.append(val)
            overall["task_stratified_95_ci"][metric] = [
                float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
            ] if boot else None
        overall["by_benchmark"] = {
            benchmark: metric_summary(
                np.asarray([y[i] for i, row in enumerate(rows) if row.benchmark == benchmark]),
                np.asarray([scores[i] for i, row in enumerate(rows) if row.benchmark == benchmark]),
            )
            for benchmark in sorted({r.benchmark for r in rows})
        }
        out[method] = overall
    return out


def leave_one_benchmark_out(rows):
    result = {}
    for heldout in sorted({row.benchmark for row in rows}):
        train = [i for i, row in enumerate(rows) if row.benchmark != heldout]
        test = [i for i, row in enumerate(rows) if row.benchmark == heldout]
        texts = [" ".join(row.names) for row in rows]
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1)
        X = safe_vectorize(vec, [texts[i] for i in train], texts)
        y = np.asarray([int(row.label == "agent") for row in rows])
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED)
        model.fit(X[train], y[train])
        score = model.predict_proba(X[test])[:, 1]
        test_rows = [rows[i] for i in test]
        val = metric_summary(y[test], score)
        val["task_count"] = len({row.task for row in test_rows})
        val["task_ids"] = sorted({row.task for row in test_rows})
        result[heldout] = val
    return result


def write_figures(summary, classification):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    methods = ["command_unigram", "ordered_command_bigram", "ordered_action_bigram", "procgrep_local_patterns"]
    labels = ["Command bag", "Ordered command bigrams", "Ordered action bigrams", "ProcGrep"]
    colors = ["#4D6A86", "#D67C42", "#3D8A70", "#8866A3"]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    benchmarks = ["intercode-ctf", "cybench", "nyuctf"]
    for ax, benchmark in zip(axes, benchmarks):
        for method, label, color in zip(methods, labels, colors):
            key = f"{method}|human_agent"
            item = summary.get(key)
            if not item or benchmark not in item["by_benchmark"]:
                continue
            value = item["by_benchmark"][benchmark]["recall_at_1"]
            ax.scatter([label], [value], color=color, s=50, zorder=3)
        ref = summary.get("command_unigram|human_agent", {}).get("by_benchmark", {}).get(benchmark)
        if ref:
            ax.axhline(ref["random_recall_at_1"], color="#6E7781", ls="--", lw=1, label="Random rank")
        ax.set_title(benchmark)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Task-macro Recall@1 (human → Agent)")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Same-challenge retrieval with same-benchmark hard negatives", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "crossbenchmark_human_to_agent_retrieval.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "crossbenchmark_human_to_agent_retrieval.pdf", bbox_inches="tight")
    plt.close(fig)

    names = ["tfidf_command_unigram_bigram", "command_count_and_action_mix", "benchmark_only_negative_control"]
    pretty = ["Command TF-IDF", "Count + action mix", "Benchmark only"]
    fig, ax = plt.subplots(figsize=(8.5, 4.1))
    x = np.arange(len(names))
    means = [classification[n]["balanced_accuracy"] for n in names]
    ci = [classification[n]["task_stratified_95_ci"]["balanced_accuracy"] for n in names]
    low = [max(0, m - c[0]) for m, c in zip(means, ci)]
    high = [max(0, c[1] - m) for m, c in zip(means, ci)]
    ax.bar(x, means, color=["#4D6A86", "#D67C42", "#A9B2BC"], width=.62)
    ax.errorbar(x, means, yerr=[low, high], fmt="none", ecolor="#20252B", capsize=4, lw=1.2)
    ax.axhline(.5, color="#6E7781", ls="--", lw=1)
    ax.set_xticks(x, pretty)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Task-held-out classification; 95% task-cluster bootstrap CI")
    ax.grid(axis="y", alpha=.22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "crossbenchmark_mode_classifier.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "crossbenchmark_mode_classifier.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    rows, tasks = matched_rows()
    task_to_benchmark = {row.task: row.benchmark for row in rows}
    labels = np.asarray([int(row.label == "agent") for row in rows])
    predictions = {name: np.full(len(rows), np.nan) for name in (
        "tfidf_command_unigram_bigram", "command_count_and_action_mix", "benchmark_only_negative_control"
    )}
    task_metrics = []
    fold_records = []
    proc_specs = []

    for fold, (train_tasks, test_tasks) in enumerate(group_folds(tasks), start=1):
        train_indices = [i for i, row in enumerate(rows) if row.task in train_tasks]
        test_indices = [i for i, row in enumerate(rows) if row.task in test_tasks]
        assert not (train_tasks & test_tasks)
        matrices, proc_spec = representations(rows, train_indices)
        proc_specs.append(proc_spec)
        task_metrics.extend(retrieval_for_fold(rows, matrices, test_tasks))
        fold_predictions, _ = classifier_features(rows, train_indices, test_indices)
        for name, score in fold_predictions.items():
            predictions[name][test_indices] = score
        fold_records.append({
            "fold": fold,
            "train_tasks": sorted(train_tasks),
            "test_tasks": sorted(test_tasks),
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "procgrep_vocabulary": proc_spec,
        })
    if any(np.isnan(values).any() for values in predictions.values()):
        raise AssertionError("Some records did not receive a grouped out-of-fold prediction")

    retrieval_summary = summarize_retrieval(task_metrics)
    classifier_summary = summarize_classifier(rows, predictions, labels, tasks)
    audit = {
        "rows": len(rows),
        "matched_task_groups": len(tasks),
        "tasks_by_benchmark": {
            benchmark: sorted(task for task in tasks if task_to_benchmark[task] == benchmark)
            for benchmark in sorted(set(task_to_benchmark.values()))
        },
        "records_by_benchmark_label": {
            f"{benchmark}|{label}": sum(row.benchmark == benchmark and row.label == label for row in rows)
            for benchmark in sorted(set(task_to_benchmark.values())) for label in ("human", "agent")
        },
        "agent_records_by_model": dict(collections.Counter(row.source for row in rows if row.label == "agent")),
        "python_calls_omitted_from_behavior_features": sum(row.skipped_python_calls for row in rows),
        "task_split_unit": "benchmark/task_id; all records from a task stay in one outer fold",
        "outer_fold_count": FOLDS,
        "method_fit_rule": "command vocabulary, TF-IDF weights, ProcGrep BPE merges, and classifiers are fit on training task groups only",
        "retrieval_candidate_rule": "all candidate tasks from the same benchmark; positive is exact benchmark/task_id; no task text or model name is used in similarity features",
    }
    result = {
        "status": "post-hoc exploratory public benchmark study",
        "seed": SEED,
        "dataset_revision": "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae",
        "data_license": "CC BY 4.0; raw logs are not redistributed by this repository",
        "audit": audit,
        "folds": fold_records,
        "retrieval": retrieval_summary,
        "classifier": classifier_summary,
        "leave_one_benchmark_out_classifier": leave_one_benchmark_out(rows),
        "task_metrics": task_metrics,
        "limitations": [
            "Challenge ID is not a shared attack, operator, campaign, or APT attribution label.",
            "Human terminal transcripts and model tool-call logs remain confounded by capture modality and collection pathway.",
            "The classifier distinguishes records in this benchmark release; it does not identify a real-world decision-maker.",
            "The retrieval question is same-challenge search with same-benchmark hard negatives, not campaign association.",
            "Tasks are the statistical clusters; trace pairs and repeated model runs are not treated as independent units.",
            "The selected benchmarks are fixed case studies; uncertainty intervals do not generalize to all cyber tasks.",
        ],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_metrics[0]))
        writer.writeheader()
        writer.writerows(task_metrics)
    write_figures(retrieval_summary, classifier_summary)
    print(json.dumps({
        "audit": audit,
        "retrieval": {
            key: {"task_count": value["task_count"], "benchmark_macro": value["benchmark_macro"],
                  "ci": value["task_stratified_95_ci"]}
            for key, value in retrieval_summary.items()
        },
        "classifier": {
            key: {"balanced_accuracy": value["balanced_accuracy"], "macro_f1": value["macro_f1"],
                  "roc_auc": value["roc_auc"], "ci": value["task_stratified_95_ci"]}
            for key, value in classifier_summary.items()
        },
        "leave_one_benchmark_out": {
            key: {"balanced_accuracy": value["balanced_accuracy"], "macro_f1": value["macro_f1"],
                  "task_count": value["task_count"]}
            for key, value in result["leave_one_benchmark_out_classifier"].items()
        },
        "outputs": [str(OUT_JSON.relative_to(ROOT)), str(OUT_CSV.relative_to(ROOT)),
                    "results/figures/crossbenchmark_human_to_agent_retrieval.png",
                    "results/figures/crossbenchmark_mode_classifier.png"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
