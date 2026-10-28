#!/usr/bin/env python3
"""Post-hoc artifact audit and task-clustered uncertainty for the first study."""

import collections
import itertools
import json
import pathlib
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from attackagent.traces import ATOM_GROUP, load_observations  # noqa: E402
from run_exploration import SEED, build_representations, split_tasks  # noqa: E402

OUT = ROOT / "results" / "sensitivity.json"
BOOTSTRAPS = 5000


def scores(y, predicted, probability):
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "human_precision": float(precision_score(y, predicted, pos_label=0, zero_division=0)),
        "human_recall": float(recall_score(y, predicted, pos_label=0, zero_division=0)),
        "agent_precision": float(precision_score(y, predicted, pos_label=1, zero_division=0)),
        "agent_recall": float(recall_score(y, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(set(y)) == 2 else None,
        "confusion_matrix_human_agent": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def task_bootstrap_classification(test, truth, predictions, probabilities, seed):
    by_task = collections.defaultdict(list)
    for i, observation in enumerate(test):
        by_task[observation.task].append(i)
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in ("balanced_accuracy", "macro_f1", "roc_auc")}
    for _ in range(BOOTSTRAPS):
        chosen = rng.choice(tasks, size=len(tasks), replace=True)
        indexes = np.concatenate([by_task[task] for task in chosen])
        y = np.asarray(truth)[indexes]
        pred = np.asarray(predictions)[indexes]
        prob = np.asarray(probabilities)[indexes]
        result = scores(y, pred, prob)
        for key in draws:
            if result[key] is not None:
                draws[key].append(result[key])
    return {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "valid_resamples": len(values),
        }
        for key, values in draws.items()
    }


def fit_classifier(train, test, feature):
    if feature == "command_ngrams":
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2))
        x_train = vectorizer.fit_transform([" ".join(x.names) for x in train])
        x_test = vectorizer.transform([" ".join(x.names) for x in test])
    elif feature == "command_count":
        x_train = np.log1p([[len(x.names)] for x in train])
        x_test = np.log1p([[len(x.names)] for x in test])
    elif feature == "action_mix":
        groups = tuple(sorted(set(ATOM_GROUP.values())))

        def matrix(observations):
            rows = []
            for observation in observations:
                counts = collections.Counter(observation.groups)
                total = max(1, sum(counts.values()))
                rows.append([counts[group] / total for group in groups])
            return np.asarray(rows, dtype=float)

        x_train, x_test = matrix(train), matrix(test)
    else:
        raise ValueError(feature)

    model = LogisticRegression(
        C=1.0, class_weight="balanced", random_state=SEED, max_iter=2000
    )
    y_train = np.asarray([x.label == "agent" for x in train], dtype=int)
    y_test = np.asarray([x.label == "agent" for x in test], dtype=int)
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    return y_test, predicted, probability


def pair_records(rows, ids, matrix, pair_type):
    records = []
    for i, j in itertools.combinations(ids, 2):
        kind = "-".join(sorted((rows[i].label, rows[j].label)))
        if kind == pair_type:
            records.append((rows[i].task, rows[j].task,
                            int(rows[i].task == rows[j].task), float(matrix[i, j])))
    return records


def pair_task_bootstrap(records, test_tasks, seed):
    rng = np.random.default_rng(seed)
    values = []
    tasks = sorted(test_tasks)
    for _ in range(BOOTSTRAPS):
        multiplicities = collections.Counter(rng.choice(tasks, size=len(tasks), replace=True))
        y, similarity, weights = [], [], []
        for left, right, same, score in records:
            weight = multiplicities[left] if same else multiplicities[left] * multiplicities[right]
            if weight:
                y.append(same)
                similarity.append(score)
                weights.append(weight)
        if len(set(y)) == 2:
            values.append(float(roc_auc_score(y, similarity, sample_weight=weights)))
    if not values:
        return {"lower_95": None, "upper_95": None, "valid_resamples": 0}
    return {
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "valid_resamples": len(values),
    }


def pair_summary(records, ci):
    same = [row for row in records if row[2] == 1]
    different = [row for row in records if row[2] == 0]
    same_tasks = {row[0] for row in same}
    cross_task_pairs = {tuple(sorted((row[0], row[1]))) for row in different}
    return {
        "candidate_pairs": len(records),
        "same_task_pairs": len(same),
        "different_task_pairs": len(different),
        "same_task_mean_similarity": float(np.mean([row[3] for row in same])) if same else None,
        "different_task_mean_similarity": float(np.mean([row[3] for row in different])) if different else None,
        "independent_positive_tasks": len(same_tasks),
        "independent_negative_task_pairs": len(cross_task_pairs),
        "pair_auc": float(roc_auc_score([row[2] for row in records], [row[3] for row in records]))
        if len({row[2] for row in records}) == 2 else None,
        "task_cluster_bootstrap_95_ci": ci,
    }


def main():
    observations = [x for x in load_observations() if x.names]
    partition = split_tasks(observations)
    train = [x for x in observations if x.task in partition["train"]]
    test = [x for x in observations if x.task in partition["test"]]
    truth = np.asarray([x.label == "agent" for x in test], dtype=int)
    classification = {}
    for offset, feature in enumerate(("majority", "command_count", "action_mix", "command_ngrams")):
        if feature == "majority":
            majority = int(np.mean([x.label == "agent" for x in train]) >= 0.5)
            predicted = np.full(len(test), majority, dtype=int)
            probability = np.full(len(test), float(majority))
        else:
            _, predicted, probability = fit_classifier(train, test, feature)
        result = scores(truth, predicted, probability)
        result["task_cluster_bootstrap_95_ci"] = task_bootstrap_classification(
            test, truth, predicted, probability, SEED + 10 + offset
        )
        result["agent_recall_by_model"] = {}
        for model in sorted({x.source for x in test if x.label == "agent"}):
            indexes = [i for i, x in enumerate(test) if x.label == "agent" and x.source == model]
            result["agent_recall_by_model"][model] = {
                "n": len(indexes),
                "recall": float(np.mean(predicted[indexes] == 1)),
            }
        classification[feature] = result

    all_rows, matrices, _ = build_representations(train, [], test)
    test_ids = [i for i, x in enumerate(all_rows) if x.task in partition["test"]]
    test_tasks = set(partition["test"])
    pair_results = {}
    for method, matrix in matrices.items():
        pair_results[method] = {}
        for offset, kind in enumerate(("human-human", "agent-human", "agent-agent")):
            records = pair_records(all_rows, test_ids, matrix, kind)
            ci = pair_task_bootstrap(records, test_tasks, SEED + 100 + offset)
            pair_results[method][kind] = pair_summary(records, ci)

    output = {
        "analysis_status": "post-hoc diagnostic, after first exploratory result was viewed",
        "seed": SEED,
        "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_unit": "task; pairs are treated as dependent within and across tasks",
        "test_task_count": len(test_tasks),
        "test_observation_count": len(test),
        "test_label_counts": dict(collections.Counter(x.label for x in test)),
        "classification": classification,
        "same_task_retrieval": pair_results,
        "limitations": [
            "Task identity is a benchmark task proxy, not a shared attack actor or campaign label.",
            "Human terminal keystrokes and model tool calls have different collection pipelines.",
            "Only six independent test tasks are available; bootstrap intervals are coarse and exploratory.",
            "A command-count baseline is a confound audit, not a deployable behavior detector.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
