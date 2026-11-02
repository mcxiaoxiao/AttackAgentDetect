#!/usr/bin/env python3
"""Short-trace CTH task-retrieval mixture experiment, fingerprint v2.

This is a post-v1 exploratory rerun. It reuses the frozen 17-task cohort and
outer task folds, admits two-command windows, and keeps all features/labels
source-local. No operator, activity, or campaign inference is made.
"""
from __future__ import annotations

import collections
import csv
import itertools
import json
import math
import pathlib
import random
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from attackagent.fingerprint_v2 import fingerprint_matrix, summarize  # noqa: E402
from attackagent.traces import ATOM_GROUP  # noqa: E402

CATALOG = ROOT / "research/results/behavior_trace_dataset_v2/traces/trace-catalog-v2.csv"
FOLDS = ROOT / "research/results/overnight_fingerprint_v1/task_folds.csv"
OUT = ROOT / "research/results/behavioral_fingerprint_v2/cth_mixture"
SEED = 20261103
MIN_EVENTS = 2
WINDOW = 8
REPLICATES = 100
QUERIES_PER_FOLD = 4
FRACTIONS = (0.0, .25, .5, .75, 1.0)
METHOD_COLUMNS = {
    "B_profile": [2],
    "B_plus_S": [2, 3],
    "command_plus_BS": [0, 2, 3],
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_rows():
    rows = []
    for raw in read_csv(CATALOG):
        if raw["source_dataset"] != "cyber-task-horizons" or raw["track"] != "ctf":
            continue
        commands = json.loads(raw["commands"])
        blocks, audit = fingerprint_matrix(commands, window=WINDOW,
                                           min_window_events=MIN_EVENTS,
                                           max_events=128)
        row = {"trace_id": raw["trace_id"], "task": raw["task_group"],
               "mode": raw["mode"], "commands": commands,
               "actions": [ATOM_GROUP.get(x) for x in commands],
               "command_count": len(commands), "profile_eligible": blocks is not None,
               "audit": audit}
        if blocks is not None:
            row["B"] = blocks["B"]
            row["S"] = blocks["S"]
            row["B_summary"] = summarize(blocks["B"])
            row["S_summary"] = summarize(blocks["S"])
        rows.append(row)
    return rows


def action_bigram_text(actions):
    tokens = [f"{a}__{b}" for a, b in zip(actions, actions[1:])
              if a is not None and b is not None]
    return " ".join(tokens) if tokens else "EMPTYTOKEN"


def sample_training_pairs(rows, train_idx, rng):
    by_task = collections.defaultdict(list)
    for i in train_idx:
        by_task[rows[i]["task"]].append(i)
    positives = []
    for task, members in sorted(by_task.items()):
        pairs = list(itertools.combinations(members, 2))
        rng.shuffle(pairs)
        positives.extend((a, b, 1) for a, b in pairs[:8])
    if len(positives) < 30:
        raise RuntimeError(f"Too few task-positive training pairs: {len(positives)}")
    task_ids = sorted(by_task)
    negative_set = set()
    attempts = 0
    target = len(positives)
    while len(negative_set) < target and attempts < target * 40:
        attempts += 1
        ta, tb = rng.sample(task_ids, 2)
        a = rng.choice(by_task[ta])
        b = rng.choice(by_task[tb])
        negative_set.add((min(a, b), max(a, b), 0))
    negatives = list(negative_set)
    if len(negatives) < max(30, target // 2):
        raise RuntimeError(f"Too few task-negative training pairs: {len(negatives)}")
    if len(negatives) > target:
        rng.shuffle(negatives)
        negatives = negatives[:target]
    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


def sample_validation_pairs(rows, validation_idx, rng):
    """Balanced, task-grouped pair set used only to choose the train-side cutoff."""
    by_task = collections.defaultdict(list)
    for i in validation_idx:
        by_task[rows[i]["task"]].append(i)
    positives = []
    for task, members in sorted(by_task.items()):
        pairs = list(itertools.combinations(members, 2))
        rng.shuffle(pairs)
        positives.extend((a, b, 1) for a, b in pairs[:8])
    task_ids = [t for t, members in by_task.items() if members]
    negatives, seen = [], set()
    attempts = 0
    while len(negatives) < len(positives) and attempts < max(100, len(positives) * 40):
        attempts += 1
        ta, tb = rng.sample(task_ids, 2)
        a, b = rng.choice(by_task[ta]), rng.choice(by_task[tb])
        key = (min(a, b), max(a, b), 0)
        if key not in seen:
            seen.add(key)
            negatives.append(key)
    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


def prepare_features(rows, train_idx, all_idx, seed):
    command_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                  token_pattern=r"(?u)\b[\w.+-]+\b", lowercase=True)
    action_vec = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True,
                                 token_pattern=r"(?u)\b[\w.+-]+\b", lowercase=True)
    command_docs = [" ".join(rows[i]["commands"]) for i in train_idx]
    action_docs = [action_bigram_text(rows[i]["actions"]) for i in train_idx]
    command_vec.fit(command_docs)
    action_vec.fit(action_docs)
    all_commands = [" ".join(r["commands"]) if r["profile_eligible"] else "EMPTYTOKEN" for r in rows]
    all_actions = [action_bigram_text(r["actions"]) if r["profile_eligible"] else "EMPTYTOKEN" for r in rows]
    command_matrix = command_vec.transform(all_commands)
    action_matrix = action_vec.transform(all_actions)

    profiles = {}
    imputers, scalers = {}, {}
    for block in ("B", "S"):
        train_x = np.vstack([rows[i][block + "_summary"] for i in train_idx])
        all_x = np.vstack([rows[i][block + "_summary"] if rows[i]["profile_eligible"]
                           else np.full(train_x.shape[1], np.nan) for i in range(len(rows))])
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        scaler = StandardScaler()
        fit_x = imputer.fit_transform(train_x)
        imputer_x = imputer.transform(all_x)
        scaler.fit(fit_x)
        profiles[block] = scaler.transform(imputer_x)
        imputers[block], scalers[block] = imputer, scaler

    def pair_features(pairs):
        x = np.empty((len(pairs), 4), dtype=float)
        for pos, (a, b, _) in enumerate(pairs):
            x[pos, 0] = command_matrix[a].multiply(command_matrix[b]).sum()
            x[pos, 1] = action_matrix[a].multiply(action_matrix[b]).sum()
            x[pos, 2] = -float(np.mean(np.abs(profiles["B"][a] - profiles["B"][b])))
            x[pos, 3] = -float(np.mean(np.abs(profiles["S"][a] - profiles["S"][b])))
        return x

    return pair_features, {"command_vocabulary": len(command_vec.vocabulary_),
                           "action_bigram_vocabulary": len(action_vec.vocabulary_),
                           "train_trace_count": len(train_idx), "profile_blocks": ["B", "S"],
                           "profile_scaling": "median imputation and standardization fitted on outer training traces only"}


def fit_models(train_x, train_y):
    fitted = {}
    for name, cols in METHOD_COLUMNS.items():
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler(),
                              LogisticRegression(C=1.0, class_weight="balanced",
                                                 solver="liblinear", max_iter=2000,
                                                 random_state=SEED))
        model.fit(train_x[:, cols], train_y)
        fitted[name] = model
    return fitted


def score_threshold(scores, y, max_fmr=.10):
    scores, y = np.asarray(scores, dtype=float), np.asarray(y, dtype=int)
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return math.inf
    candidates = sorted(set(np.r_[pos, neg].tolist()), reverse=True) + [math.inf]
    best_recall, best_threshold = -1.0, math.inf
    for threshold in candidates:
        fmr = float(np.mean(neg >= threshold))
        if fmr <= max_fmr + 1e-12:
            recall = float(np.mean(pos >= threshold))
            if recall > best_recall or (math.isclose(recall, best_recall) and threshold > best_threshold):
                best_recall, best_threshold = recall, threshold
    return float(best_threshold)


def crossfit_thresholds(rows, outer_train_idx, outer_fold):
    """Choose every link cutoff from inner task-held-out predictions only."""
    groups = sorted({rows[i]["task"] for i in outer_train_idx})
    group_fold = {g: pos % 3 for pos, g in enumerate(groups)}
    oof_scores = collections.defaultdict(list)
    oof_labels = []
    oof_audit = []
    for inner_fold in range(3):
        inner_fit = [i for i in outer_train_idx if group_fold[rows[i]["task"]] != inner_fold]
        inner_val = [i for i in outer_train_idx if group_fold[rows[i]["task"]] == inner_fold]
        if not inner_fit or not inner_val:
            continue
        inner_pairs = sample_training_pairs(rows, inner_fit,
                                            random.Random(SEED + outer_fold * 10000 + inner_fold))
        val_pairs = sample_validation_pairs(rows, inner_val,
                                            random.Random(SEED + outer_fold * 20000 + inner_fold + 5))
        if not inner_pairs or not val_pairs:
            continue
        pair_features, _ = prepare_features(rows, inner_fit, inner_val, SEED + inner_fold)
        inner_x, val_x = pair_features(inner_pairs), pair_features(val_pairs)
        inner_y = np.asarray([p[2] for p in inner_pairs], dtype=int)
        val_y = np.asarray([p[2] for p in val_pairs], dtype=int)
        inner_models = fit_models(inner_x, inner_y)
        preds = {"command_cosine": val_x[:, 0],
                 "action_bigram_cosine": val_x[:, 1],
                 **{name: model.predict_proba(val_x[:, cols])[:, 1]
                    for name, cols in METHOD_COLUMNS.items() for model in [inner_models[name]]}}
        for name, values in preds.items():
            oof_scores[name].extend(float(x) for x in values)
        oof_labels.extend(int(x) for x in val_y)
        oof_audit.append({"inner_fold": inner_fold,
                          "fit_task_count": len({rows[i]["task"] for i in inner_fit}),
                          "validation_task_count": len({rows[i]["task"] for i in inner_val}),
                          "fit_pair_count": len(inner_pairs),
                          "validation_pair_count": len(val_pairs),
                          "validation_positive_pairs": int(val_y.sum())})
    if not oof_labels or len(set(oof_labels)) != 2:
        raise RuntimeError(f"Could not create task-held-out threshold scores for outer fold {outer_fold}")
    thresholds = {name: score_threshold(values, oof_labels)
                  for name, values in oof_scores.items()}
    return thresholds, {"policy": "three-fold task-held-out out-of-fold scores; maximize recall subject to FMR<=10%",
                        "oof_pairs": len(oof_labels), "oof_positive_pairs": int(sum(oof_labels)),
                        "oof_negative_pairs": int(len(oof_labels)-sum(oof_labels)),
                        "thresholds": thresholds, "inner_folds": oof_audit}


def tie_credit(scores, positive_idx):
    top = max(scores)
    tied = [i for i, s in enumerate(scores) if math.isclose(s, top, rel_tol=1e-10, abs_tol=1e-12)]
    return 1 / len(tied) if positive_idx in tied else 0.0


def mode_pair(a, b):
    if a == b == "human":
        return "HH"
    if a == b == "agent":
        return "AA"
    return "HA"


def ci_cluster(values, seed, draws=5000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    sims = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(sims, [.025, .975])]


def metric_summary(pair_rows, query_rows, method, fraction):
    selected_pairs = [r for r in pair_rows if r["method"] == method and r["fraction"] == fraction]
    selected_queries = [r for r in query_rows if r["method"] == method and r["fraction"] == fraction]
    by_task = collections.defaultdict(list)
    for r in selected_pairs:
        by_task[r["query_task"]].append(r)
    task_values = {}
    for task, rows in by_task.items():
        pos = [r for r in rows if r["same_task"]]
        neg = [r for r in rows if not r["same_task"]]
        if pos and neg:
            fsr = float(np.mean([1 - r["pred_link"] for r in pos]))
            fmr = float(np.mean([r["pred_link"] for r in neg]))
            task_values[task] = {"fsr": fsr, "fmr": fmr, "ba": 1 - (fsr + fmr) / 2}
    task_query = collections.defaultdict(list)
    for r in selected_queries:
        task_query[r["query_task"]].append(r["hit_at_1"])
    for task, vals in task_query.items():
        if task in task_values:
            task_values[task]["hit_at_1"] = float(np.mean(vals))
    names = sorted(task_values)
    result = {"fraction": fraction, "method": method, "task_count": len(names),
              "replicates": REPLICATES, "queries": len(selected_queries),
              "unique_role_fraction_mean": float(np.mean([r["agent_fraction"] for r in selected_queries])) if selected_queries else None}
    for metric in ("fmr", "fsr", "ba", "hit_at_1"):
        values = [task_values[t][metric] for t in names if metric in task_values[t]]
        result[f"task_macro_{metric}"] = float(np.mean(values)) if values else None
        result[f"task_cluster_95ci_{metric}"] = ci_cluster(values, SEED + int(fraction * 1000) + len(metric))
    type_result = {}
    for typ in ("HH", "HA", "AA"):
        sub = [r for r in selected_pairs if r["pair_type"] == typ]
        pos = [r for r in sub if r["same_task"]]
        neg = [r for r in sub if not r["same_task"]]
        type_result[typ] = {"positive_pairs": len(pos), "fsr": float(np.mean([1-r["pred_link"] for r in pos])) if pos else None,
                            "negative_pairs": len(neg), "fmr": float(np.mean([r["pred_link"] for r in neg])) if neg else None}
    result["pair_type_errors"] = type_result
    return result


def main():
    rows = load_rows()
    pools = collections.defaultdict(lambda: {"human": [], "agent": []})
    raw_pools = collections.defaultdict(lambda: collections.Counter())
    for i, row in enumerate(rows):
        raw_pools[row["task"]][row["mode"]] += 1
        if row["profile_eligible"]:
            pools[row["task"]][row["mode"]].append(i)

    core_tasks = sorted(t for t, c in raw_pools.items() if c["human"] >= 2 and c["agent"] >= 2)
    assert len(core_tasks) == 17, f"expected frozen 17-task cohort, found {len(core_tasks)}"
    mixed_tasks = sorted(t for t, c in raw_pools.items() if c["human"] and c["agent"])
    if any(len(pools[t][m]) < 2 for t in core_tasks for m in ("human", "agent")):
        raise RuntimeError("The short-window profile does not cover the frozen 2H/2A task cohort")

    fold_map = {r["group"]: int(r["fold"]) for r in read_csv(FOLDS)
                if r["endpoint"] == "cth_mixture"}
    if not set(core_tasks) <= set(fold_map):
        raise RuntimeError("The frozen v1 outer task folds do not cover this v2 core cohort")
    folds = sorted(set(fold_map[t] for t in core_tasks))
    test_tasks_by_fold = {fold: sorted(t for t in core_tasks if fold_map[t] == fold) for fold in folds}
    eligible_idx = [i for i, r in enumerate(rows) if r["profile_eligible"]]
    all_pair_rows, all_query_rows, fold_audit, mix_audit = [], [], [], []
    methods = ["command_cosine", "action_bigram_cosine", *METHOD_COLUMNS]

    for fold in folds:
        test_tasks = set(test_tasks_by_fold[fold])
        train_idx = [i for i in eligible_idx if rows[i]["task"] not in test_tasks]
        train_pairs = sample_training_pairs(rows, train_idx,
                                            random.Random(SEED + fold * 1009))
        pair_features, transform_audit = prepare_features(rows, train_idx, eligible_idx,
                                                          SEED + fold)
        train_x = pair_features(train_pairs)
        train_y = np.asarray([p[2] for p in train_pairs], dtype=int)
        fitted = fit_models(train_x, train_y)
        thresholds, threshold_audit = crossfit_thresholds(rows, train_idx, fold)
        fold_audit.append({"fold": fold, "test_task_count": len(test_tasks),
                           "test_tasks": sorted(test_tasks),
                           "train_trace_count": len(train_idx),
                           "train_task_count": len({rows[i]["task"] for i in train_idx}),
                           "train_pair_count": len(train_pairs),
                           "train_positive_pairs": int(train_y.sum()),
                           "train_negative_pairs": int((train_y == 0).sum()),
                           "threshold_selection": threshold_audit, "transform": transform_audit})

        for rep in range(REPLICATES):
            for fraction in FRACTIONS:
                rng = random.Random(SEED + fold * 1000003 + rep * 101 + int(fraction * 1000))
                chosen_tasks = sorted(rng.sample(sorted(test_tasks), QUERIES_PER_FOLD))
                role_slots = [(task, role) for task in chosen_tasks for role in ("query", "gallery")]
                n_agent = int(round(fraction * len(role_slots)))
                agent_slot_ids = set(rng.sample(range(len(role_slots)), n_agent)) if n_agent else set()
                role_mode = {(task, role): ("agent" if slot in agent_slot_ids else "human")
                             for slot, (task, role) in enumerate(role_slots)}
                selected = {}
                for task in chosen_tasks:
                    qm, gm = role_mode[(task, "query")], role_mode[(task, "gallery")]
                    if qm == gm:
                        qid, gid = rng.sample(pools[task][qm], 2)
                    else:
                        qid = rng.choice(pools[task][qm])
                        gid = rng.choice(pools[task][gm])
                    selected[task] = {"query": qid, "gallery": gid,
                                      "query_mode": qm, "gallery_mode": gm}
                actual = sum(x == "agent" for x in role_mode.values()) / len(role_mode)
                if not math.isclose(actual, fraction, abs_tol=1e-12):
                    raise AssertionError(f"non-exact concentration {actual} for {fraction}")
                mix_audit.append({"fold": fold, "replicate": rep, "fraction": fraction,
                                  "agent_role_slots": n_agent, "role_slots": len(role_slots),
                                  "agent_fraction": actual, "query_tasks": chosen_tasks})

                test_pairs = []
                pair_meta = []
                query_meta = []
                for query_task in chosen_tasks:
                    q = selected[query_task]
                    candidate_tasks = list(chosen_tasks)
                    rng.shuffle(candidate_tasks)
                    candidate_ids = [selected[t]["gallery"] for t in candidate_tasks]
                    candidate_truth = [t == query_task for t in candidate_tasks]
                    positive_pos = candidate_truth.index(True)
                    qid = q["query"]
                    for candidate_task, candidate_id, is_positive in zip(candidate_tasks, candidate_ids, candidate_truth):
                        pair = (qid, candidate_id, int(is_positive))
                        test_pairs.append(pair)
                        pair_meta.append({"fold": fold, "replicate": rep, "fraction": fraction,
                                          "query_task": query_task, "candidate_task": candidate_task,
                                          "query_trace_id": rows[qid]["trace_id"],
                                          "candidate_trace_id": rows[candidate_id]["trace_id"],
                                          "query_mode": q["query_mode"],
                                          "candidate_mode": selected[candidate_task]["gallery_mode"],
                                          "pair_type": mode_pair(q["query_mode"], selected[candidate_task]["gallery_mode"]),
                                          "same_task": bool(is_positive), "agent_fraction": actual})
                    query_meta.append({"fold": fold, "replicate": rep, "fraction": fraction,
                                       "query_task": query_task, "query_trace_id": rows[qid]["trace_id"],
                                       "agent_fraction": actual, "positive_candidate_index": positive_pos})
                test_x = pair_features(test_pairs)
                scores_by_method = {
                    "command_cosine": test_x[:, 0],
                    "action_bigram_cosine": test_x[:, 1],
                    **{m: model.predict_proba(test_x[:, cols])[:, 1]
                       for m, (cols, model) in ((name, (METHOD_COLUMNS[name], fitted[name]))
                                                 for name in METHOD_COLUMNS)}
                }
                for method in methods:
                    scores = np.asarray(scores_by_method[method], dtype=float)
                    threshold = thresholds[method]
                    credits = {}
                    for pos, meta in enumerate(pair_meta):
                        pred = int(scores[pos] >= threshold)
                        all_pair_rows.append({**meta, "method": method,
                                              "score": float(scores[pos]), "threshold": float(threshold),
                                              "pred_link": pred})
                        credits.setdefault(meta["query_task"], []).append((scores[pos], meta["same_task"]))
                    for qm in query_meta:
                        values = credits[qm["query_task"]]
                        query_scores = [s for s, _ in values]
                        pos_index = next(i for i, (_, truth) in enumerate(values) if truth)
                        all_query_rows.append({**qm, "method": method,
                                               "hit_at_1": tie_credit(query_scores, pos_index),
                                               "candidate_count": len(query_scores),
                                               "chance_hit_at_1": 1 / len(query_scores)})

        print(f"CTH fold {fold}: train={len(train_idx)} traces/{len({rows[i]['task'] for i in train_idx})} tasks; test={len(test_tasks)} tasks", flush=True)

    summary = []
    for fraction in FRACTIONS:
        for method in methods:
            summary.append(metric_summary(all_pair_rows, all_query_rows, method, fraction))
    # Paired cluster bootstrap of test-task deltas over the command baseline.
    deltas = []
    for fraction in FRACTIONS:
        for method in ("B_profile", "B_plus_S", "command_plus_BS"):
            base_by, new_by = {}, {}
            for r in all_pair_rows:
                if r["fraction"] != fraction:
                    continue
                target = base_by if r["method"] == "command_cosine" else (new_by if r["method"] == method else None)
                if target is not None:
                    target.setdefault(r["query_task"], []).append(r)
            task_delta = []
            for task in sorted(set(base_by) & set(new_by)):
                def task_ba(records):
                    pos = [x for x in records if x["same_task"]]
                    neg = [x for x in records if not x["same_task"]]
                    if not pos or not neg:
                        return None
                    fsr = np.mean([1-x["pred_link"] for x in pos])
                    fmr = np.mean([x["pred_link"] for x in neg])
                    return float(1-(fsr+fmr)/2)
                a, b = task_ba(base_by[task]), task_ba(new_by[task])
                if a is not None and b is not None:
                    task_delta.append((task, b-a))
            vals = [v for _, v in task_delta]
            deltas.append({"fraction": fraction, "method": method,
                           "baseline": "command_cosine", "task_count": len(vals),
                           "delta_task_macro_ba": float(np.mean(vals)) if vals else None,
                           "task_cluster_95ci": ci_cluster(vals, SEED+int(fraction*1000)+len(method)+900)})

    OUTPUT = OUT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "pair_predictions.csv", all_pair_rows)
    write_csv(OUT / "query_predictions.csv", all_query_rows)
    write_csv(OUT / "mixture_slots.csv", mix_audit)
    write_csv(OUT / "folds.csv", fold_audit)
    write_json(OUT / "metrics.json", summary)
    write_json(OUT / "paired_deltas.json", deltas)
    write_json(OUT / "eligibility.json", {
        "status": "exploratory_short_trace_v2_post_v1",
        "source": "Cyber Task Horizons; CTF only",
        "raw_cth_ctf_trace_count": len(rows),
        "raw_cth_ctf_task_count": len(raw_pools),
        "raw_cth_ctf_mode_counts": {m: sum(c[m] for c in raw_pools.values()) for m in ("human", "agent")},
        "mixed_mode_task_count": len(mixed_tasks),
        "mixed_mode_raw_trace_count": sum(sum(raw_pools[t].values()) for t in mixed_tasks),
        "mixed_mode_raw_counts": {m: sum(raw_pools[t][m] for t in mixed_tasks) for m in ("human", "agent")},
        "mixed_mode_profile_eligible_counts": {
            m: sum(r["task"] in set(mixed_tasks) and r["mode"] == m and r["profile_eligible"] for r in rows)
            for m in ("human", "agent")},
        "mixed_mode_tasks_with_at_least_two_profile_eligible_each": sum(
            len(pools[t]["human"]) >= 2 and len(pools[t]["agent"]) >= 2 for t in mixed_tasks),
        "fingerprint_eligible_minimum_two_commands": sum(r["profile_eligible"] for r in rows),
        "minimum_commands": MIN_EVENTS,
        "window_size": WINDOW,
        "core_task_count": len(core_tasks),
        "core_trace_count": sum(r["task"] in set(core_tasks) for r in rows),
        "core_eligible_trace_count": sum(r["task"] in set(core_tasks) and r["profile_eligible"] for r in rows),
        "core_mode_counts": {m: sum(r["task"] in set(core_tasks) and r["mode"] == m for r in rows) for m in ("human", "agent")},
        "core_profile_mode_counts": {m: sum(r["task"] in set(core_tasks) and r["mode"] == m and r["profile_eligible"] for r in rows) for m in ("human", "agent")},
        "all_core_groups_keep_two_profiles_each_mode": all(len(pools[t][m]) >= 2 for t in core_tasks for m in ("human", "agent")),
        "outer_folds_reused_from_v1": {str(f): sorted(v) for f, v in test_tasks_by_fold.items()},
        "test_protocol": "4 query tasks per fold; one same-task gallery and three other-task galleries; 8 unique query/gallery role slots per fold; exact 0/25/50/75/100 Agent shares; four candidates per query; chance Hit@1=25%",
        "training_groups": "all other eligible CTH CTF task groups; relation labels are same source-qualified task group",
        "inputs_excluded": ["mode label", "task ID", "dataset/model/framework IDs", "participant IDs", "source path", "attack outcome"],
        "timing": "not available in normalized CTH records; no timing block used",
    })
    write_json(OUT / "provenance.json", {
        "protocol": "cth-short-trace-mixture-v2",
        "seed": SEED, "replicates": REPLICATES, "fraction_levels": FRACTIONS,
        "minimum_events": MIN_EVENTS, "window_size": WINDOW,
        "association_models": {"pair_lr": "C=1, class_weight=balanced, liblinear",
                               "all_method_thresholds": "three-fold task-held-out out-of-fold scores; maximize train-side recall subject to FMR<=10%"},
        "feature_scaling": "median imputation and StandardScaler fit within outer training traces only",
        "confidence_intervals": "5,000 bootstrap draws over 17 task clusters; candidate draws are not independent units",
        "scope": "task-level retrieval proxy; not operator, attack, activity, or campaign attribution",
    })
    print(json.dumps({"core_tasks": len(core_tasks), "core_records": sum(r["task"] in set(core_tasks) for r in rows),
                      "summary": [{"fraction": x["fraction"], "method": x["method"], "BA": x["task_macro_ba"],
                                   "FMR": x["task_macro_fmr"], "FSR": x["task_macro_fsr"], "Hit@1": x["task_macro_hit_at_1"]}
                                  for x in summary]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
