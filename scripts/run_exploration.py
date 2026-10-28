#!/usr/bin/env python3
"""Task-held-out exploratory baselines on public human/model shell traces."""

import collections
import itertools
import json
import pathlib
import sys

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attackagent.traces import ATOM_GROUP, load_observations  # noqa: E402
from procgrep import Trace, encode, fit_bpe, jsd  # noqa: E402
from procgrep.bpe import ProcedureVocabulary  # noqa: E402

SEED = 20261029
OUT = ROOT / "results" / "preliminary.json"


def split_tasks(observations):
    h = {x.task for x in observations if x.label == "human" and x.names}
    a = {x.task for x in observations if x.label == "agent" and x.names}
    common = sorted(h & a)
    tasks = np.random.default_rng(SEED).permutation(common).tolist()
    n_train, n_val = int(len(tasks) * 0.6), int(len(tasks) * 0.2)
    if min(n_train, n_val, len(tasks) - n_train - n_val) < 2:
        raise ValueError("Not enough independent matched tasks")
    return {
        "train": set(tasks[:n_train]),
        "validation": set(tasks[n_train:n_train + n_val]),
        "test": set(tasks[n_train + n_val:]),
    }


def metric(y, predicted, scores):
    labels = [0, 1]
    return {
        "n": len(y),
        "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "macro_f1": f1_score(y, predicted, average="macro", zero_division=0),
        "human_precision": precision_score(y, predicted, pos_label=0, zero_division=0),
        "human_recall": recall_score(y, predicted, pos_label=0, zero_division=0),
        "agent_precision": precision_score(y, predicted, pos_label=1, zero_division=0),
        "agent_recall": recall_score(y, predicted, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y, scores) if len(set(y)) == 2 else None,
        "confusion_matrix_human_agent": confusion_matrix(y, predicted, labels=labels).tolist(),
    }


def classifier(partitions):
    train, valid, test = (partitions[k] for k in ("train", "validation", "test"))
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2))
    train_text = [" ".join(x.names) for x in train]
    model = LogisticRegression(C=1.0, class_weight="balanced", random_state=SEED, max_iter=2000)
    model.fit(vectorizer.fit_transform(train_text), [x.label == "agent" for x in train])
    result = {}
    for name, part in (("validation", valid), ("test", test)):
        X = vectorizer.transform([" ".join(x.names) for x in part])
        y = [int(x.label == "agent") for x in part]
        score = model.predict_proba(X)[:, 1]
        predicted = (score >= 0.5).astype(int)
        result[name] = metric(y, predicted, score)
        result[name]["agent_recall_by_model"] = {}
        for model_name in sorted({x.source for x in part if x.label == "agent"}):
            indexes = [i for i, x in enumerate(part) if x.label == "agent" and x.source == model_name]
            result[name]["agent_recall_by_model"][model_name] = {
                "n": len(indexes),
                "recall": float(np.mean(predicted[indexes] == 1)),
            }
    return result, model, vectorizer


def build_representations(train, valid, test):
    all_traces = train + valid + test
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    vectorizer.fit([" ".join(x.names) for x in train])
    matrices = vectorizer.transform([" ".join(x.names) for x in all_traces])
    cos = cosine_similarity(matrices)

    alphabet = tuple(sorted(set(ATOM_GROUP.values())))
    vocab = fit_bpe((list(x.groups) for x in train), vocab_size=32, seed=SEED, fit_corpus="matched train tasks")
    # Reserve the fixed public category alphabet for unseen test categories.
    vocab = ProcedureVocabulary(atoms=alphabet, merges=vocab.merges, seed=vocab.seed,
                                min_pair_frequency=vocab.min_pair_frequency, fit_corpus=vocab.fit_corpus)
    proc_traces = [Trace(trace_id=x.trace_id, agent=x.source, atoms=list(x.groups)) for x in all_traces]
    fps = encode(proc_traces, vocab=vocab)
    js = np.eye(len(all_traces))
    for i, j in itertools.combinations(range(len(all_traces)), 2):
        score = 1.0 - jsd(fps[i].distribution(), fps[j].distribution())
        js[i, j] = js[j, i] = score
    return all_traces, {"command_cosine": cos, "procgrep_jsd_similarity": js}, vocab.spec.compact()


def pairs(rows, ids, matrix, kind):
    out = []
    for i, j in itertools.combinations(ids, 2):
        pair_type = "-".join(sorted((rows[i].label, rows[j].label)))
        if pair_type != kind:
            continue
        out.append((int(rows[i].task == rows[j].task), float(matrix[i, j])))
    return out


def choose_threshold(samples):
    y = np.array([a for a, _ in samples])
    scores = np.array([b for _, b in samples])
    if len(set(y)) != 2:
        return None
    candidates = np.unique(np.r_[0.0, scores, 1.0, 1.000001])
    options = [(balanced_accuracy_score(y, scores >= threshold), threshold) for threshold in candidates]
    return float(max(options, key=lambda t: (t[0], t[1]))[1])


def summarize(samples, threshold):
    pos = [s for y, s in samples if y]
    neg = [s for y, s in samples if not y]
    if not pos or not neg or threshold is None:
        return {"positive_pairs": len(pos), "negative_pairs": len(neg), "status": "insufficient"}
    all_y = [y for y, _ in samples]
    all_scores = [s for _, s in samples]
    return {
        "positive_pairs": len(pos), "negative_pairs": len(neg),
        "same_task_mean_similarity": float(np.mean(pos)),
        "different_task_mean_similarity": float(np.mean(neg)),
        "false_split_rate": float(np.mean(np.asarray(pos) < threshold)),
        "false_merge_rate": float(np.mean(np.asarray(neg) >= threshold)),
        "pair_auc": float(roc_auc_score(all_y, all_scores)),
    }


def downstream(rows, ids, matrix, global_threshold, thresholds, predicted_labels):
    """Compare fixed policies on the same test pairs; no threshold fits on test."""
    policies = {"common": [], "uniform_half": [], "oracle_groups": [], "predicted_groups": []}
    y = []
    for i, j in itertools.combinations(ids, 2):
        truth_kind = "-".join(sorted((rows[i].label, rows[j].label)))
        pred_kind = "-".join(sorted((predicted_labels[i], predicted_labels[j])))
        score = float(matrix[i, j])
        y.append(int(rows[i].task == rows[j].task))
        policies["common"].append(int(score >= global_threshold))
        policies["uniform_half"].append(int(score * 0.5 >= global_threshold))
        policies["oracle_groups"].append(int(score >= thresholds[truth_kind]))
        policies["predicted_groups"].append(int(score >= thresholds[pred_kind]))
    out = {}
    for name, predicted in policies.items():
        cm = confusion_matrix(y, predicted, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out[name] = {
            "pairs": len(y), "same_task_pairs": int(sum(y)),
            "false_split_rate": float(fn / (fn + tp)) if fn + tp else None,
            "false_merge_rate": float(fp / (fp + tn)) if fp + tn else None,
            "balanced_accuracy": balanced_accuracy_score(y, predicted),
            "confusion_matrix_different_same": cm.tolist(),
        }
    return out


def main():
    observations = load_observations()
    split = split_tasks(observations)
    assert not split["train"] & split["validation"] and not split["train"] & split["test"]
    assert not split["validation"] & split["test"]
    parts = {
        key: [x for x in observations if x.task in tasks and x.names]
        for key, tasks in split.items()
    }
    seen = [x.trace_id for part in parts.values() for x in part]
    assert len(seen) == len(set(seen))
    rows, matrices, vocab = build_representations(parts["train"], parts["validation"], parts["test"])
    val_ids = [i for i, x in enumerate(rows) if x.task in split["validation"]]
    test_ids = [i for i, x in enumerate(rows) if x.task in split["test"]]
    classification, model, vectorizer = classifier(parts)
    labels = (model.predict_proba(vectorizer.transform([" ".join(x.names) for x in rows]))[:, 1] >= 0.5)
    predicted_labels = ["agent" if label else "human" for label in labels]
    association = {}
    for method, matrix in matrices.items():
        kinds = ("human-human", "agent-human", "agent-agent")
        validation = {kind: pairs(rows, val_ids, matrix, kind) for kind in kinds}
        threshold = choose_threshold([item for group in validation.values() for item in group])
        thresholds = {}
        for kind, group in validation.items():
            positive = sum(y for y, _ in group)
            negative = len(group) - positive
            thresholds[kind] = choose_threshold(group) if positive >= 3 and negative >= 3 else threshold
        association[method] = {
            "threshold_selected_on_validation": threshold,
            "group_thresholds_selected_on_validation": thresholds,
            "validation_pair_counts": {
                kind: {"same": sum(y for y, _ in group), "different": sum(1 - y for y, _ in group)}
                for kind, group in validation.items()
            },
            "test": {kind: summarize(pairs(rows, test_ids, matrix, kind), threshold)
                     for kind in kinds},
            "downstream_test": downstream(rows, test_ids, matrix, threshold, thresholds, predicted_labels),
        }
    summary = {
        "seed": SEED,
        "dataset_revision": "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae",
        "matched_nonempty_tasks": sum(len(v) for v in split.values()),
        "splits": {
            name: {
                "task_count": len(tasks), "tasks": sorted(tasks),
                "human_samples": sum(x.label == "human" for x in parts[name]),
                "agent_samples": sum(x.label == "agent" for x in parts[name]),
            } for name, tasks in split.items()
        },
        "excluded_empty": dict(collections.Counter(x.label for x in observations if not x.names)),
        "python_tool_calls_excluded": sum(x.skipped_python_calls for x in observations),
        "classification": classification,
        "association": association,
        "procgrep_vocab_spec": vocab,
        "limitations": [
            "Same task is a proxy and not a verified shared attacker or campaign.",
            "Human keystrokes and model tool calls are differently instrumented.",
            "Pairs share traces and are not independent statistical observations.",
            "No traditional automation control is present.",
            "Group-specific threshold selection has more fitted parameters than the common baseline.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
