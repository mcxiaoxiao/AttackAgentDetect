#!/usr/bin/env python3
"""Post-hoc controls for mode specificity and candidate-gallery composition.

Fixed before this run: no tuning, original task folds/pairs, fixed C=0.5.
The task identity remains a benchmark proxy, not an intrusion identity.
"""

import collections
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from attackagent.prototype import ModeAwareAssociationPrototype  # noqa: E402
from scripts.run_mode_aware_association import (  # noqa: E402
    DIRECTIONS, FOLDS, SEED, load_rows, macro, ranking_metrics,
)

METHODS = ("similarity_only", "mode_aware", "length_conditioned", "nonlinear_similarity")
METRICS = ("recall_at_1", "mrr", "query_auc")
BOOTSTRAPS = 5000


class EmpiricalLengthCDF:
    """Fit an unlabeled scalar reference on training lengths only."""

    def fit(self, lengths):
        values = np.asarray(lengths, dtype=float)
        if values.size == 0 or np.any(values < 0) or not np.isfinite(values).all():
            raise ValueError("Training lengths must be nonempty, finite and nonnegative")
        self.reference_ = np.sort(np.log1p(values))
        return self

    def transform(self, lengths):
        values = np.asarray(lengths, dtype=float)
        if np.any(values < 0) or not np.isfinite(values).all():
            raise ValueError("Lengths must be finite and nonnegative")
        return np.searchsorted(self.reference_, np.log1p(values), side="right") / len(self.reference_)


def nonlinear_features(base):
    c, a = base[:, 0], base[:, 1]
    return np.column_stack((c, a, c * c, a * a, c * a, np.abs(c - a)))


def evaluate_gallery(rows, score_map, test_tasks, heldout_only=False):
    """Evaluate with identical queries, filtering candidates before scoring."""
    by_task = collections.defaultdict(list)
    task_benchmark = {}
    for i, row in enumerate(rows):
        by_task[(row.benchmark, row.task, row.label)].append(i)
        task_benchmark[row.task] = row.benchmark
    methods = tuple(next(iter(score_map.values())))
    output = []
    for task in sorted(test_tasks):
        benchmark = task_benchmark[task]
        candidate_tasks = sorted(t for t, b in task_benchmark.items()
                                 if b == benchmark and (not heldout_only or t in test_tasks))
        for direction, (query_mode, gallery_mode) in DIRECTIONS.items():
            for method in methods:
                query_results = []
                for query in by_task.get((benchmark, task, query_mode), []):
                    scores = {}
                    for candidate_task in candidate_tasks:
                        candidates = by_task.get((benchmark, candidate_task, gallery_mode), [])
                        if direction == "human_human":
                            candidates = [j for j in candidates if j != query]
                        if direction == "agent_agent_cross_model":
                            candidates = [j for j in candidates if rows[j].source != rows[query].source]
                        values = [score_map[(min(query, j), max(query, j))][method]
                                  for j in candidates if (min(query, j), max(query, j)) in score_map]
                        if values:
                            scores[candidate_task] = float(np.mean(values))
                    item = ranking_metrics(scores, task)
                    if item is not None:
                        query_results.append(item)
                if query_results:
                    output.append({
                        "task": task, "benchmark": benchmark, "direction": direction, "method": method,
                        "query_count": len(query_results),
                        **{metric: float(np.mean([x[metric] for x in query_results]))
                           for metric in (*METRICS, "random_recall_at_1", "candidate_tasks")},
                        "minimum_candidate_tasks": min(x["candidate_tasks"] for x in query_results),
                        "maximum_candidate_tasks": max(x["candidate_tasks"] for x in query_results),
                    })
    return output


def bootstrap_interval(rows, metric, seed):
    """Benchmark-stratified resampling of whole task-level values."""
    rng = np.random.default_rng(seed)
    benchmark_draws = []
    for benchmark in sorted({r["benchmark"] for r in rows}):
        values = np.asarray([r[metric] for r in rows if r["benchmark"] == benchmark])
        samples = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
        benchmark_draws.append(values[samples].mean(axis=1))
    means = np.mean(benchmark_draws, axis=0)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def summarize_controls(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["protocol"], row["direction"], row["method"])].append(row)
    result = {}
    for key, items in sorted(groups.items()):
        result["|".join(key)] = {
            "task_count": len(items),
            "benchmark_macro": {metric: macro(items, metric) for metric in (*METRICS, "random_recall_at_1", "candidate_tasks")},
            "task_stratified_95_ci": {metric: bootstrap_interval(items, metric, SEED + 99 + len(result)) for metric in METRICS},
            "candidate_task_range": [min(r["minimum_candidate_tasks"] for r in items), max(r["maximum_candidate_tasks"] for r in items)],
            "by_benchmark": {
                bench: {"task_count": sum(r["benchmark"] == bench for r in items),
                        **{metric: float(np.mean([r[metric] for r in items if r["benchmark"] == bench]))
                           for metric in (*METRICS, "random_recall_at_1", "candidate_tasks")}}
                for bench in sorted({r["benchmark"] for r in items})
            },
        }
    deltas = {}
    # Compare every candidate to baseline and specifically mode to each rival.
    contrasts = [(m, "similarity_only") for m in METHODS if m != "similarity_only"]
    contrasts += [("mode_aware", "length_conditioned"), ("mode_aware", "nonlinear_similarity")]
    for protocol in ("full_gallery", "heldout_gallery"):
        for direction in DIRECTIONS:
            for candidate, reference in contrasts:
                a = {r["task"]: r for r in groups.get((protocol, direction, candidate), [])}
                b = {r["task"]: r for r in groups.get((protocol, direction, reference), [])}
                if set(a) != set(b):
                    raise AssertionError("Methods have different eligible query tasks")
                if not a:
                    continue
                items = [{"benchmark": a[t]["benchmark"], "task": t,
                          **{m: a[t][m] - b[t][m] for m in METRICS}} for t in sorted(a)]
                key = "|".join((protocol, direction, candidate + "_minus_" + reference))
                deltas[key] = {"task_count": len(items),
                               "mean_delta": {m: macro(items, m) for m in METRICS},
                               "task_stratified_95_ci": {m: bootstrap_interval(items, m, SEED + 703 + len(deltas)) for m in METRICS}}
    return result, deltas


def verify_original_reproduction(task_metrics):
    path = ROOT / "results/mode_aware_association.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    expected = {(r["task"], r["direction"], r["method"]): r for r in old["task_metrics"]
                if r["method"] in ("similarity_only", "mode_aware")}
    actual = {(r["task"], r["direction"], r["method"]): r for r in task_metrics
              if r["protocol"] == "full_gallery" and r["method"] in ("similarity_only", "mode_aware")}
    assert set(actual) == set(expected), "Original eligible query tasks differ"
    largest = max(abs(actual[k][m] - expected[k][m]) for k in actual for m in METRICS)
    assert largest < 1e-12, f"Original results differ by {largest}"
    return {"status": "passed", "task_direction_method_rows": len(actual), "max_absolute_metric_difference": largest,
            "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    raw, records, tasks = load_rows()
    grouped = np.asarray([r.task for r in raw])
    metrics, folds = [], []
    splitter = GroupKFold(n_splits=FOLDS)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(records)), groups=grouped), 1):
        train = [records[i] for i in train_idx]
        test_tasks = {raw[i].task for i in test_idx}
        train_tasks = {raw[i].task for i in train_idx}
        assert not train_tasks & test_tasks
        model = ModeAwareAssociationPrototype(seed=SEED + fold).fit(train)
        train_pairs = model._training_pairs(train)
        labels = np.asarray([p[2] for p in train_pairs])
        mode_prob = np.full(len(records), np.nan)
        mode_prob[train_idx] = model.training_mode_oof_
        mode_prob[test_idx] = [r["agent_like_score"] for r in model.predict_mode([records[i] for i in test_idx])]
        cdf = EmpiricalLengthCDF().fit([len(r.commands) for r in train])
        q_train = cdf.transform([len(r.commands) for r in train])
        q_all = cdf.transform([len(r.commands) for r in records])
        base_train = model._pair_features(train, train_pairs, q_train, model.command_matrix_, model.action_matrix_, False)
        length_train = model._pair_features(train, train_pairs, q_train, model.command_matrix_, model.action_matrix_, True)
        assert length_train.shape[1] == 11 and base_train.shape[1] == 2
        length_model = LogisticRegression(C=.5, class_weight="balanced", max_iter=3000, random_state=SEED + fold).fit(length_train, labels)
        nonlinear_model = LogisticRegression(C=.5, class_weight="balanced", max_iter=3000, random_state=SEED + fold).fit(nonlinear_features(base_train), labels)
        test_set = set(test_idx)
        pairs = [(i, j) for i in range(len(records)) for j in range(i + 1, len(records)) if i in test_set or j in test_set]
        scores = model.predict_pair_scores(records, pairs, agent_probabilities=mode_prob)
        cm = model.command_vectorizer_.transform(model._text(records, "commands"))
        am = model.action_vectorizer_.transform(model._text(records, "actions")) if hasattr(model.action_vectorizer_, "vocabulary_") else csr_matrix((len(records), 0))
        base = model._pair_features(records, pairs, q_all, cm, am, False)
        length = model._pair_features(records, pairs, q_all, cm, am, True)
        scores["length_conditioned"] = length_model.predict_proba(length)[:, 1]
        scores["nonlinear_similarity"] = nonlinear_model.predict_proba(nonlinear_features(base))[:, 1]
        score_map = {pair: {m: float(v[k]) for m, v in scores.items()} for k, pair in enumerate(pairs)}
        for protocol, heldout in (("full_gallery", False), ("heldout_gallery", True)):
            for row in evaluate_gallery(raw, score_map, test_tasks, heldout):
                metrics.append({"protocol": protocol, "fold": fold, **row})
        folds.append({"fold": fold, "train_task_ids": sorted(train_tasks), "test_task_ids": sorted(test_tasks),
                      "train_records": len(train_idx), "test_records": len(test_idx),
                      "training_pairs": model.training_pair_counts_,
                      "heldout_candidate_tasks_by_benchmark": {
                          b: len({raw[i].task for i in test_idx if raw[i].benchmark == b})
                          for b in sorted({raw[i].benchmark for i in test_idx})
                      }})
        print(f"Fold {fold}/{FOLDS}: original models and fixed controls complete", flush=True)
    reproduction = verify_original_reproduction(metrics)
    summary, deltas = summarize_controls(metrics)
    result = {
        "status": "post-hoc exploratory controls; no hyperparameter selection",
        "seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
        "audit": {"records": len(records), "tasks": len(tasks), "task_overlap": 0},
        "methods": {
            "similarity_only": "original 2-feature command/action cosine ranker",
            "mode_aware": "original 11-feature mode-probability ranker; nested OOF mode scores",
            "length_conditioned": "same 11-feature function replacing mode score by training-only empirical CDF of log1p command count; labels are not used to construct this scalar",
            "nonlinear_similarity": "6 features [c,a,c^2,a^2,c*a,abs(c-a)] without mode score or command count",
            "common_fit": "identical original training pairs, balanced logistic regression C=0.5, max_iter=3000; no feature standardization or tuning",
        },
        "gallery_protocols": {
            "full_gallery": "same-benchmark candidates from all 40 tasks; training task traces can be negative candidates; exactly reproduces original protocol",
            "heldout_gallery": "same-benchmark candidates from the outer held-out tasks only; queries with fewer than 2 eligible candidate tasks are excluded",
        },
        "interpretation": [
            "These controls test whether mode-score ranking gains are specific to mode supervision, versus trace length or extra nonlinear flexibility.",
            "The length and nonlinear rankers share the original mode-stratified training-pair sampling for fair comparison; only their input features omit execution-mode scores.",
            "Heldout-only galleries are much smaller and may omit small benchmarks in some folds; raw metrics across gallery protocols are not comparable improvements.",
            "Bootstrap intervals resample task-level predictions conditional on the fixed fitted folds and gallery; they do not include model refitting or candidate-gallery sampling uncertainty.",
            "All data have already been explored; no independent final test set is claimed.",
            "CTF task identity is a proxy, and capture modality remains confounded with execution-mode labels.",
        ],
        "original_reproduction": reproduction,
        "source_script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "folds": folds, "retrieval": summary, "paired_deltas": deltas, "task_metrics": metrics,
    }
    output = ROOT / "results/specificity_controls.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "results/specificity_task_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(metrics)
    print(json.dumps({"reproduction": reproduction,
                      "human_agent": {k: v for k, v in summary.items() if "|human_agent|" in k},
                      "human_agent_deltas": {k: v for k, v in deltas.items() if "|human_agent|" in k}}, indent=2))


if __name__ == "__main__":
    main()
