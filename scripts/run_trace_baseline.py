#!/usr/bin/env python3
"""Fixed Trace-style TF-IDF + LinearSVC classifier on the same task split."""

import collections
import json
import pathlib
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from attackagent.traces import load_observations  # noqa: E402
from run_exploration import SEED, split_tasks  # noqa: E402
from run_sensitivity import BOOTSTRAPS, task_bootstrap_classification  # noqa: E402

OUT = ROOT / "results" / "trace-baseline.json"


def main():
    rows = [x for x in load_observations() if x.names]
    partition = split_tasks(rows)
    train = [x for x in rows if x.task in partition["train"]]
    test = [x for x in rows if x.task in partition["test"]]
    assert not ({x.task for x in train} & {x.task for x in test})
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2))
    train_matrix = vectorizer.fit_transform([" ".join(x.names) for x in train])
    test_matrix = vectorizer.transform([" ".join(x.names) for x in test])
    model = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)
    model.fit(train_matrix, [int(x.label == "agent") for x in train])
    decision = model.decision_function(test_matrix)
    predicted = (decision >= 0).astype(int)
    truth = np.asarray([int(x.label == "agent") for x in test])
    result = {
        "status": "fixed post-hoc algorithm transfer check; not a Trace replication",
        "train_task_count": len(partition["train"]),
        "test_task_count": len(partition["test"]),
        "test_samples": len(test),
        "class_counts": dict(collections.Counter(x.label for x in test)),
        "features": "TF-IDF command-name unigrams and bigrams",
        "model": "LinearSVC(C=1.0, class_weight=balanced)",
        "calibrated_probabilities": False,
        "metrics": {
            "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "human_precision": float(precision_score(truth, predicted, pos_label=0, zero_division=0)),
            "human_recall": float(recall_score(truth, predicted, pos_label=0, zero_division=0)),
            "agent_precision": float(precision_score(truth, predicted, pos_label=1, zero_division=0)),
            "agent_recall": float(recall_score(truth, predicted, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(truth, decision)),
            "confusion_matrix_human_agent": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
        },
        "task_cluster_bootstrap_95_ci": task_bootstrap_classification(
            test, truth, predicted, decision, SEED + 22
        ),
        "bootstrap_resamples": BOOTSTRAPS,
        "limitations": [
            "Labels distinguish benchmark human sessions from selected model sessions, not attacker identity.",
            "Terminal recording and model tool-call logging remain confounded with labels.",
            "Only six independent test tasks are available.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
