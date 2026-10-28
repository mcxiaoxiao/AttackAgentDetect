#!/usr/bin/env python3
"""Post-hoc split sensitivity: random sessions versus task-held-out sessions.

This is a leakage diagnostic, not a new confirmatory estimate. Repeated split
results are descriptive; their spread is not an inferential confidence interval.
"""

import collections
import json
import pathlib
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attackagent.traces import load_observations  # noqa: E402

OUT = ROOT / "results" / "split-diagnostics.json"
BASE_SEED = 20261029
REPEATS = 50
TEST_SIZE = 0.20


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "repeats": len(values),
        "mean": float(values.mean()),
        "sd_across_splits_descriptive_only": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def evaluate(rows, train_ids, test_ids):
    train = [rows[i] for i in train_ids]
    test = [rows[i] for i in test_ids]
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2)
    )
    x_train = vectorizer.fit_transform([" ".join(x.names) for x in train])
    x_test = vectorizer.transform([" ".join(x.names) for x in test])
    y_train = np.asarray([x.label == "agent" for x in train], dtype=int)
    y_test = np.asarray([x.label == "agent" for x in test], dtype=int)
    model = LogisticRegression(
        C=1.0, class_weight="balanced", random_state=BASE_SEED, max_iter=2000
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "test_sessions": len(test),
        "train_task_count": len({x.task for x in train}),
        "test_task_count": len({x.task for x in test}),
        "overlap_task_count": len({x.task for x in train} & {x.task for x in test}),
    }


def main():
    observations = [x for x in load_observations() if x.names]
    by_task = collections.defaultdict(set)
    for row in observations:
        by_task[row.task].add(row.label)
    matched_tasks = {task for task, labels in by_task.items() if labels == {"human", "agent"}}
    rows = [x for x in observations if x.task in matched_tasks]
    labels = np.asarray([x.label == "agent" for x in rows], dtype=int)
    groups = np.asarray([x.task for x in rows])
    row_results, group_results = [], []

    for offset in range(REPEATS):
        seed = BASE_SEED + offset
        train_ids, test_ids = train_test_split(
            np.arange(len(rows)), test_size=TEST_SIZE, random_state=seed,
            stratify=labels,
        )
        row_results.append(evaluate(rows, train_ids, test_ids))

        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_ids, test_ids = next(splitter.split(np.zeros(len(rows)), labels, groups))
        group_results.append(evaluate(rows, train_ids, test_ids))

    output = {
        "status": "post-hoc split sensitivity diagnostic; descriptive, not confirmatory",
        "dataset_revision": "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae",
        "matched_nonempty_task_count": len(matched_tasks),
        "session_count": len(rows),
        "class_counts": dict(collections.Counter(x.label for x in rows)),
        "features": "TF-IDF command-name unigrams and bigrams",
        "classifier": "LogisticRegression(C=1.0, class_weight=balanced)",
        "test_fraction": TEST_SIZE,
        "repeats": REPEATS,
        "base_seed": BASE_SEED,
        "random_session_split": {
            "metrics": {
                metric: summarize([row[metric] for row in row_results])
                for metric in ("balanced_accuracy", "macro_f1")
            },
            "mean_train_test_task_overlap": float(np.mean([row["overlap_task_count"] for row in row_results])),
            "per_split": row_results,
        },
        "task_group_split": {
            "metrics": {
                metric: summarize([row[metric] for row in group_results])
                for metric in ("balanced_accuracy", "macro_f1")
            },
            "mean_train_test_task_overlap": float(np.mean([row["overlap_task_count"] for row in group_results])),
            "per_split": group_results,
        },
        "protocol_notes": [
            "Each split fits the vocabulary and classifier on its training partition only.",
            "Random session splits may place sessions from the same benchmark task in train and test.",
            "Task-group splits hold out complete task IDs and are the relevant generalization check.",
            "Repeated splits share observations and are not independent; SD and ranges are descriptive, not confidence intervals.",
            "The class label remains confounded with the data collection modality in both conditions.",
            "No hyperparameter search or threshold tuning was performed.",
        ],
    }
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    compact = {
        "status": output["status"],
        "matched_nonempty_task_count": len(matched_tasks),
        "session_count": len(rows),
        "random_session_split": output["random_session_split"]["metrics"],
        "random_overlap": output["random_session_split"]["mean_train_test_task_overlap"],
        "task_group_split": output["task_group_split"]["metrics"],
        "group_overlap": output["task_group_split"]["mean_train_test_task_overlap"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
