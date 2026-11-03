#!/usr/bin/env python3
"""One-time CTH-trained, task-disjoint NYU CTF Agent–Agent confirmation."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import xgboost
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
RESULTS = ROOT / "research/results/matched_fingerprint_v2"
MANIFEST_PATH = RESULTS / "independent_nyu_task_manifest.json"
OUT = RESULTS / "independent_nyu_agent"
SEED = 20261106
REPRESENTATION = (4, "mean_std", "conditional")
PARAMETERS = {"svm": 1.0, "tree": 3, "xgb": 3}
LEARNERS = ("svm", "tree", "xgb")
VERSIONS = ("E", "E_BS")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


M = module(EXP / "run_arguments.py", "run_arguments_nyu")
PAIR = module(EXP / "run_pair_support.py", "pair_support_nyu")
AUDIT = module(EXP / "audit_nyu_dcipher_confirmation.py", "audit_nyu_confirmation")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf8")


def load_external(source):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf8"))
    revision = AUDIT.repository_revision(source)
    assert revision == manifest["source_revision"], (revision, manifest["source_revision"])
    lookup = {}
    for task in manifest["eligible_tasks"]:
        for trace in task["traces"]:
            lookup[trace["trace_id"]] = (task, trace)
    rows = []
    profiles_input_hashes = []
    for submission in AUDIT.SUBMISSIONS:
        folder = Path(source) / "transcripts" / submission
        for path in sorted(folder.glob("*.json")):
            if path.name == "summary.json":
                continue
            raw_hash = AUDIT.sha_file(path)
            task_name = path.stem.lower()
            trace_id = AUDIT.sha_bytes(f"{submission}/{task_name}/{raw_hash}".encode())[:20]
            if trace_id not in lookup:
                continue
            frozen_task, frozen_trace = lookup[trace_id]
            assert frozen_task["task"] == task_name
            assert frozen_trace["raw_file_sha256"] == raw_hash
            commands, calls, missing = AUDIT.extract_commands(path)
            assert len(commands) == frozen_trace["command_count"]
            assert AUDIT.sha_bytes("\n".join(commands).encode()) == frozen_trace["sequence_sha256"]
            rows.append({"id": trace_id, "task": task_name, "mode": "agent",
                         "commands": commands,
                         "actions": [M.ATOM_GROUP[c] for c in commands],
                         "timestamps": None,
                         "locator": f"{submission}/{path.name}",
                         "source_submission": submission})
            profiles_input_hashes.append(raw_hash)
    assert len(rows) == manifest["eligible_trace_count"] == 585
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(len(row["commands"]) >= 2 for row in rows)
    return rows, manifest


def append_external_pair(result, rows, query, candidate, partition, replicate,
                         query_index, position, same_task):
    a, b = rows[query], rows[candidate]
    assert query != candidate
    assert (a["task"] == b["task"]) == bool(same_task)
    result.append({"pair_id": f"{replicate}:{partition}:{query_index}:{position}",
                   "fold": partition, "replicate": replicate, "fraction": 1.0,
                   "query_idx": query, "candidate_idx": candidate,
                   "query_id": a["id"], "candidate_id": b["id"],
                   "query_task": a["task"], "candidate_task": b["task"],
                   "query_mode": "agent", "candidate_mode": "agent",
                   "pair_type": "AA", "same_task": int(same_task),
                   "candidate_position": position})


def external_pairs(rows, manifest, expected_pair_count=234_000):
    by_task = collections.defaultdict(list)
    partitions = {}
    for task in manifest["eligible_tasks"]:
        partitions[task["task"]] = int(task["candidate_partition"])
    for index, row in enumerate(rows):
        by_task[row["task"]].append(index)
    assert all(len(by_task[task]) >= 2 for task in partitions)
    by_partition = {p: sorted(t for t, v in partitions.items() if v == p)
                    for p in range(3)}
    result = []
    for replicate in range(100):
        rng = random.Random(SEED + replicate)
        query_index = 0
        for partition in range(3):
            partition_tasks = by_partition[partition]
            assert len(partition_tasks) >= 4
            for task in partition_tasks:
                for query in sorted(by_task[task], key=lambda i: rows[i]["id"]):
                    positive = rng.choice([i for i in by_task[task] if i != query])
                    negative_tasks = rng.sample([t for t in partition_tasks if t != task], 3)
                    candidates = [positive] + [rng.choice(by_task[t]) for t in negative_tasks]
                    for position, candidate in enumerate(candidates):
                        append_external_pair(result, rows, query, candidate, partition,
                                             replicate, query_index, position,
                                             rows[query]["task"] == rows[candidate]["task"])
                    query_index += 1
    expected = 100 * int(manifest["eligible_trace_count"]) * 4
    assert len(result) == expected == expected_pair_count
    assert all(row["pair_type"] == "AA" for row in result)
    return result


def model(kind, parameter):
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=float(parameter), kernel="rbf",
                                                   gamma="scale", class_weight=None))
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=int(parameter), min_samples_leaf=3,
                                      class_weight=None, random_state=M.SEED)
    if kind == "xgb":
        return M.classifier("xgb", int(parameter))
    raise ValueError(kind)


def fit_scores(kind, parameter, x_train, y_train, weights, x_test):
    estimator = model(kind, parameter)
    if kind == "svm":
        estimator.fit(x_train, y_train, svc__sample_weight=weights)
        return estimator.decision_function(x_test)
    estimator.fit(x_train, y_train, sample_weight=weights)
    return estimator.predict_proba(x_test)[:, 1]


def make_features(rows, train_indices, profiles):
    return M.Features(rows, train_indices, profiles,
                      REPRESENTATION[0], REPRESENTATION[1])


def matrices(features, train_pairs, validation_pairs):
    train_a, train_b, train_y = M.train_arrays(train_pairs)
    test_a, test_b, test_y, test_tasks = M.as_arrays(validation_pairs)
    evidence_train = features.evidence(train_a, train_b)
    evidence_test = features.evidence(test_a, test_b)
    e_train = features.evidence_head_only(train_a, train_b)
    e_test = features.evidence_head_only(test_a, test_b)
    x_train, _ = features.full(e_train, train_a, train_b, ("B", "S"), REPRESENTATION[2])
    x_test, missing = features.full(e_test, test_a, test_b, ("B", "S"), REPRESENTATION[2])
    return {"E": (e_train, e_test, np.zeros(len(test_y), dtype=bool)),
            "E_BS": (x_train, x_test, missing)}, train_y, test_y, test_tasks


def crossfit_thresholds(cth_rows, tasks, task_folds, all_rows, profiles):
    chunks = {learner: {version: {"y": [], "scores": [], "tasks": [], "missing": []}
                        for version in VERSIONS} for learner in LEARNERS}
    fold_audit = {}
    for fold in sorted(set(task_folds.values())):
        validation_tasks = {task for task in tasks if task_folds[task] == fold}
        training_tasks = set(tasks) - validation_tasks
        training_pairs, weights = PAIR.training_pairs(
            all_rows, training_tasks, M.SEED + 1_000_000 + fold, "all_pairs")
        validation_pairs = M.gallery(cth_rows, validation_tasks, fold, 1,
                                     M.SEED + 800_000 + fold)
        training_indices = [i for i, row in enumerate(cth_rows)
                            if row["task"] in training_tasks]
        features = make_features(all_rows, training_indices, profiles)
        x, train_y, val_y, val_tasks = matrices(features, training_pairs, validation_pairs)
        fold_audit[str(fold)] = {
            "training_tasks": sorted(training_tasks),
            "validation_tasks": sorted(validation_tasks),
            "training_trace_count": len(training_indices),
            "training_pair_count": len(training_pairs),
            "positive_pairs": int(train_y.sum()),
            "negative_pairs": int((train_y == 0).sum()),
            "training_weight_mass_by_label": {
                str(label): float(weights[train_y == label].sum()) for label in (0, 1)},
            "validation_pair_count": len(validation_pairs),
            "validation_query_count": len(validation_pairs) // 4}
        for learner in LEARNERS:
            for version in VERSIONS:
                x_train, x_validation, missing = x[version]
                score = fit_scores(learner, PARAMETERS[learner], x_train,
                                   train_y, weights, x_validation)
                chunk = chunks[learner][version]
                chunk["y"].extend(val_y.tolist())
                chunk["scores"].extend(score.tolist())
                chunk["tasks"].extend(val_tasks.tolist())
                chunk["missing"].extend(missing.tolist())
    thresholds, audit = {}, {}
    for learner in LEARNERS:
        thresholds[learner], audit[learner] = {}, {}
        for version in VERSIONS:
            chunk = chunks[learner][version]
            y = np.asarray(chunk["y"], dtype=np.int8)
            score = np.asarray(chunk["scores"], dtype=float)
            task_labels = np.asarray(chunk["tasks"], dtype=object)
            missing = np.asarray(chunk["missing"], dtype=bool)
            if version == "E_BS":
                base = chunks[learner]["E"]
                base_pred = M.decision(np.asarray(base["scores"], dtype=float),
                                       thresholds[learner]["E"])
                threshold = M.threshold(y, score, missing, base_pred)
                prediction = M.decision(score, threshold, missing, base_pred)
            else:
                threshold = M.threshold(y, score)
                prediction = M.decision(score, threshold)
            thresholds[learner][version] = float(threshold)
            metric = M.macro(y, prediction, task_labels)
            audit[learner][version] = {
                "threshold": float(threshold), "threshold_rule": "maximize OOF recall at pooled FMR <= 10%",
                "oof_macro_fmr": float(metric[0]), "oof_macro_fsr": float(metric[1]),
                "oof_macro_ba": float(metric[2]), "oof_pairs": int(len(y)),
                "missing_fingerprint_pairs": int(missing.sum())}
    return thresholds, {"folds": fold_audit, "threshold_metrics": audit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="local checkout of the pinned NYU leaderboard submissions repository")
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    cth_rows, cth_tasks, task_folds = M.build_cth()
    external_rows, manifest = load_external(source)
    assert manifest["protocol_sha256"] == sha(EXP / "independent_nyu_aa_protocol.md")
    assert manifest["status"] == "frozen_before_confirmation_model_fit"
    assert len(cth_tasks) == 17 and len(cth_rows) == 85
    all_rows = cth_rows + external_rows
    profiles = M.profile_cache(all_rows, minimum=2)
    external_test_pairs = external_pairs(all_rows, manifest)
    M.write_gz_csv(OUT / "pair_manifest.csv.gz", external_test_pairs)

    thresholds, calibration = crossfit_thresholds(
        cth_rows, cth_tasks, task_folds, all_rows, profiles)
    training_pairs, weights = PAIR.training_pairs(
        all_rows, set(cth_tasks), M.SEED + 950001, "all_pairs")
    train_a, train_b, train_y = M.train_arrays(training_pairs)
    test_a, test_b, _, _ = M.as_arrays(external_test_pairs)
    training_indices = list(range(len(cth_rows)))
    features = make_features(all_rows, training_indices, profiles)
    e_train = features.evidence_head_only(train_a, train_b)
    e_test = features.evidence_head_only(test_a, test_b)
    x_train, _ = features.full(e_train, train_a, train_b, ("B", "S"), REPRESENTATION[2])
    x_test, missing_test = features.full(e_test, test_a, test_b, ("B", "S"), REPRESENTATION[2])

    records = []
    for learner in LEARNERS:
        scores = {
            "E": fit_scores(learner, PARAMETERS[learner], e_train, train_y,
                            weights, e_test),
            "E_BS": fit_scores(learner, PARAMETERS[learner], x_train, train_y,
                               weights, x_test)}
        base_prediction = M.decision(scores["E"], thresholds[learner]["E"])
        bs_prediction = M.decision(scores["E_BS"], thresholds[learner]["E_BS"],
                                   missing_test, base_prediction)
        for version, prediction in (("E", base_prediction), ("E_BS", bs_prediction)):
            for i, pair in enumerate(external_test_pairs):
                fallback = bool(version == "E_BS" and missing_test[i])
                actual_score = scores["E"][i] if fallback else scores[version][i]
                actual_threshold = (thresholds[learner]["E"] if fallback
                                    else thresholds[learner][version])
                actual_prediction = base_prediction[i] if fallback else prediction[i]
                row = {key: pair[key] for key in (
                    "fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
                    "query_task", "candidate_task", "query_mode", "candidate_mode",
                    "pair_type", "same_task", "candidate_position")}
                row.update(dataset="nyu_ctf_independent_aa", learner=learner,
                           variant=version, representation="/".join(map(str, REPRESENTATION)),
                           parameter=PARAMETERS[learner],
                           score=f"{float(actual_score):.12g}",
                           threshold=f"{float(actual_threshold):.12g}",
                           predicted_link=int(bool(actual_prediction)), fallback=int(fallback))
                records.append(row)

    with gzip.open(OUT / "predictions.csv.gz", "wt", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    training_records = []
    for i, (a, b, label) in enumerate(training_pairs):
        training_records.append({"pair_index": i, "query_id": cth_rows[a]["id"],
                                 "candidate_id": cth_rows[b]["id"],
                                 "query_task": cth_rows[a]["task"],
                                 "candidate_task": cth_rows[b]["task"],
                                 "pair_type": M.pair_type(cth_rows[a]["mode"], cth_rows[b]["mode"]),
                                 "same_task": label, "sample_weight": float(weights[i])})
    M.write_gz_csv(OUT / "training_pairs.csv.gz", training_records)
    metadata = {
        "status": "executed_once_frozen_cross_source_confirmation",
        "source": manifest["source"], "source_revision": manifest["source_revision"],
        "protocol_sha256": manifest["protocol_sha256"],
        "task_manifest_sha256": sha(MANIFEST_PATH),
        "development_dataset": "CTH mixed-role CTF; 17 tasks, 85 traces",
        "confirmation_scope": "160 unseen NYU CTF challenge task groups; 585 D-CIPHER traces; AA only",
        "confirmation_tasks": len(manifest["eligible_tasks"]),
        "confirmation_traces": manifest["eligible_trace_count"],
        "query_galleries_per_repetition": manifest["eligible_trace_count"],
        "repetitions": 100, "candidate_count": 4,
        "pair_manifest_rows": len(external_test_pairs),
        "prediction_rows": len(records),
        "training_pair_count": len(training_pairs),
        "training_positive_pairs": int(train_y.sum()),
        "training_negative_pairs": int((train_y == 0).sum()),
        "training_weight_mass_by_label": {
            str(label): float(weights[train_y == label].sum()) for label in (0, 1)},
        "representation": list(REPRESENTATION), "parameters": PARAMETERS,
        "thresholds": thresholds, "threshold_calibration": calibration,
        "candidate_partition_counts": manifest["candidate_partition_counts"],
        "pair_manifest_sha256": sha(OUT / "pair_manifest.csv.gz"),
        "inputs_sha256": {"nyu_manifest": sha(MANIFEST_PATH),
                          "cth_registry": sha(ROOT / "research/results/behavior_corpus_v1/trace_registry.csv"),
                          "cth_evidence": sha(ROOT / "research/results/behavior_corpus_v1/evidence.jsonl"),
                          "protocol": sha(EXP / "independent_nyu_aa_protocol.md"),
                          "runner": sha(Path(__file__)),
                          "features": sha(ROOT / "attackagent/fingerprint_v2.py"),
                          "command_parser": sha(ROOT / "attackagent/traces.py"),
                          "pair_training": sha(EXP / "run_pair_support.py")},
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                        "seed": SEED},
        "elapsed_seconds": round(time.monotonic() - started, 2)}
    json_write(OUT / "run_manifest.json", metadata)
    print(json.dumps({key: metadata[key] for key in (
        "confirmation_tasks", "confirmation_traces", "pair_manifest_rows",
        "prediction_rows", "training_pair_count", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
