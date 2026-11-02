#!/usr/bin/env python3
"""Cross-target task-condition retrieval using the public CyBiasBench traces.

Only aggregate results are written. Request bodies, headers, actor identifiers,
task labels, outcomes, response codes and raw per-session rows are excluded from
the feature matrix and the committed outputs.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import itertools
import json
import math
import pathlib
import random
import re
import subprocess
import sys
import urllib.parse

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from attackagent.fingerprint_v2 import combine_summary, fingerprint_matrix  # noqa: E402

SOURCE = ROOT / "research/data/external/cybiasbench-v1"
TRACE_FILE = SOURCE / "bias_injection_traces.parquet"
RECORD_FILE = SOURCE / "bias_injection_records.csv"
REPO_DIR = SOURCE / "repo"
OUT = ROOT / "research/results/behavioral_fingerprint_v2/cybiasbench"
SEED = 20261103
MAX_EVENTS = 128
MIN_EVENTS = 4


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def clean_segment(segment):
    s = urllib.parse.unquote(str(segment)).strip().lower()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):
        return "{int}"
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f-]{20,}", s):
        return "{uuid}"
    if re.fullmatch(r"[0-9a-f]{16,}", s):
        return "{hexid}"
    if len(s) > 40:
        return "{long}"
    return s


def normalize_route(value):
    path = urllib.parse.urlsplit(str(value)).path
    parts = [clean_segment(x) for x in path.split("/") if x]
    parts = [x for x in parts if x is not None]
    return "/" + "/".join(parts) if parts else "/"


def request_action(method):
    method = str(method or "OTHER").upper()
    if method in {"GET", "HEAD", "OPTIONS", "TRACE", "PROPFIND"}:
        return "read"
    if method in {"POST", "PUT", "PATCH", "DELETE", "PROPPATCH", "MKCOL", "COPY", "MOVE", "LOCK", "UNLOCK"}:
        return "write"
    return "other"


def load_sessions():
    if not TRACE_FILE.is_file() or not RECORD_FILE.is_file():
        raise FileNotFoundError("CyBiasBench raw data is unavailable. See the experiment README for the pinned upstream download steps.")
    labels = {}
    with RECORD_FILE.open(newline="") as f:
        for row in csv.DictReader(f):
            labels[row["record_id"]] = {"target": row["target"], "agent": row["agent"],
                                        "family": row["requested_family"], "rep": row["rep"]}
    requests = collections.defaultdict(list)
    table = pq.read_table(TRACE_FILE, columns=["record_id", "request_index", "req_method", "req_path"])
    total_requests = table.num_rows
    for r in table.to_pylist():
        requests[r["record_id"]].append(r)
    sessions = []
    raw_count_quantiles = []
    action_counts, route_count = collections.Counter(), collections.Counter()
    for rid, items in sorted(requests.items()):
        meta = labels.get(rid)
        if not meta:
            raise ValueError(f"Trace without session-level evaluation metadata: {rid}")
        items.sort(key=lambda x: (x["request_index"] is None,
                                  x["request_index"] if x["request_index"] is not None else 0))
        raw_count_quantiles.append(len(items))
        symbols, actions = [], []
        for item in items[:MAX_EVENTS]:
            method = str(item["req_method"] or "OTHER").upper()
            route = normalize_route(item["req_path"] or "/")
            symbol = f"{method}:{route}"
            action = request_action(method)
            symbols.append(symbol)
            actions.append(action)
            action_counts[action] += 1
            route_count[route] += 1
        blocks, audit = fingerprint_matrix(symbols, actions=actions,
                                           window=8, min_window_events=MIN_EVENTS,
                                           max_events=MAX_EVENTS)
        if blocks is None:
            continue
        sessions.append({"record_id": rid, **meta, "symbols": symbols,
                         "actions": actions, "raw_request_count": len(items),
                         "observed_prefix_count": len(symbols),
                         "B": combine_summary(blocks, ("B",)),
                         "BS": combine_summary(blocks, ("B", "S")),
                         "audit": audit})
    return sessions, {"raw_request_rows": total_requests,
                      "metadata_sessions": len(labels),
                      "sessions_with_requests": len(requests),
                      "eligible_sessions_minimum_4_requests": len(sessions),
                      "request_count_quantiles": {str(q): int(np.quantile(raw_count_quantiles, q))
                                                  for q in (0, .25, .5, .75, 1)},
                      "prefix_cap": MAX_EVENTS,
                      "request_action_counts_in_prefix": dict(action_counts),
                      "unique_route_shapes_in_prefix": len(route_count),
                      "normalization": "method + query-stripped path; integer, UUID, long hex, and >40-character segments replaced with type markers",
                      "excluded_fields": ["resp_status_code", "attack_family", "success", "success_evidence", "matched_rules", "CAPEC", "CWE", "request body", "headers", "model, target, and requested-family identifiers"]}


def action_bigram_text(actions):
    pairs = [f"{a}__{b}" for a, b in zip(actions, actions[1:])]
    return " ".join(pairs) if pairs else "EMPTYTOKEN"


def build_pairs(sessions, targets, seed, per_query_negative=1):
    lookup = collections.defaultdict(list)
    agents, families = set(), set()
    for i, s in enumerate(sessions):
        if s["target"] not in targets:
            continue
        lookup[(s["target"], s["family"], s["agent"])].append(i)
        agents.add(s["agent"])
        families.add(s["family"])
    agents, families = sorted(agents), sorted(families)
    rng = random.Random(seed)
    pairs = []
    for target in sorted(targets):
        target_families = sorted(f for t, f, a in lookup if t == target)
        target_agents = sorted(a for a in agents if any((target, f, a) in lookup for f in target_families))
        for q_agent, c_agent in itertools.permutations(target_agents, 2):
            for family in target_families:
                qpool = lookup[(target, family, q_agent)]
                cpool = lookup[(target, family, c_agent)]
                other_families = [f for f in target_families if f != family and lookup[(target, f, c_agent)]]
                if not qpool or not cpool or not other_families:
                    continue
                for qid in qpool:
                    positive = rng.choice(cpool)
                    pairs.append((qid, positive, 1))
                    for _ in range(per_query_negative):
                        neg_family = rng.choice(other_families)
                        negative = rng.choice(lookup[(target, neg_family, c_agent)])
                        pairs.append((qid, negative, 0))
    rng.shuffle(pairs)
    return pairs


def prepare_feature_space(sessions, train_indices):
    route_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                token_pattern=r"(?u)\b[\w.+:/{}-]+\b", lowercase=True)
    action_vec = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True,
                                 token_pattern=r"(?u)\b[\w.+-]+\b", lowercase=True)
    route_vec.fit([" ".join(sessions[i]["symbols"]) for i in train_indices])
    action_vec.fit([action_bigram_text(sessions[i]["actions"]) for i in train_indices])
    route_docs = route_vec.transform([" ".join(s["symbols"]) for s in sessions])
    action_docs = action_vec.transform([action_bigram_text(s["actions"]) for s in sessions])
    summaries = {}
    for block in ("B", "BS"):
        train_x = np.vstack([sessions[i][block] for i in train_indices])
        all_x = np.vstack([s[block] for s in sessions])
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        scaler = StandardScaler()
        train_i = imputer.fit_transform(train_x)
        scaler.fit(train_i)
        summaries[block] = scaler.transform(imputer.transform(all_x))

    def pair_features(pairs):
        x = np.empty((len(pairs), 4), dtype=float)
        for pos, (a, b, _) in enumerate(pairs):
            x[pos, 0] = route_docs[a].multiply(route_docs[b]).sum()
            x[pos, 1] = action_docs[a].multiply(action_docs[b]).sum()
            x[pos, 2] = -float(np.mean(np.abs(summaries["B"][a] - summaries["B"][b])))
            x[pos, 3] = -float(np.mean(np.abs(summaries["BS"][a] - summaries["BS"][b])))
        return x

    return pair_features, {"route_vocab": len(route_vec.vocabulary_),
                           "action_bigram_vocab": len(action_vec.vocabulary_),
                           "scaling": "train sessions only; median imputation and standardization"}


def fit_models(x, y):
    specs = {"route_plus_B": [0, 2], "route_plus_BS": [0, 3]}
    models = {}
    for name, cols in specs.items():
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler(),
                              LogisticRegression(C=1.0, class_weight="balanced",
                                                 solver="liblinear", max_iter=2000,
                                                 random_state=SEED))
        model.fit(x[:, cols], y)
        models[name] = (cols, model)
    return models


def score_threshold(scores, y, max_fmr=.10):
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    pos, neg = scores[y == 1], scores[y == 0]
    candidates = sorted(set(np.r_[pos, neg].tolist()), reverse=True) + [math.inf]
    best = (-1.0, math.inf)
    for t in candidates:
        fmr = float(np.mean(neg >= t)) if len(neg) else 1.
        if fmr <= max_fmr + 1e-12:
            recall = float(np.mean(pos >= t)) if len(pos) else 0.
            if recall > best[0] or (math.isclose(recall, best[0]) and t > best[1]):
                best = (recall, t)
    return float(best[1])


def rank_metrics(scores, truths):
    positive = next(i for i, t in enumerate(truths) if t)
    best = max(scores)
    tied = [i for i, s in enumerate(scores) if math.isclose(float(s), float(best), rel_tol=1e-10, abs_tol=1e-12)]
    hit = 1/len(tied) if positive in tied else 0.
    rank = sum(s > scores[positive] + 1e-12 for s in scores)
    if any(math.isclose(float(s), float(scores[positive]), rel_tol=1e-10, abs_tol=1e-12) for s in scores):
        tie_n = sum(math.isclose(float(s), float(scores[positive]), rel_tol=1e-10, abs_tol=1e-12) for s in scores)
        mrr = float(np.mean([1/(rank+j+1) for j in range(tie_n)]))
    else:
        mrr = 0.
    return hit, mrr


def evaluate_target(sessions, target, train_targets, seed):
    train_indices = [i for i, s in enumerate(sessions) if s["target"] in train_targets]
    train_pairs = build_pairs(sessions, train_targets, seed, per_query_negative=1)
    if not train_pairs or {p[2] for p in train_pairs} != {0, 1}:
        raise RuntimeError("Could not form both task-relation classes for training")
    pair_features, transform_audit = prepare_feature_space(sessions, train_indices)
    train_x = pair_features(train_pairs)
    train_y = np.asarray([p[2] for p in train_pairs], int)
    models = fit_models(train_x, train_y)

    # Calibrate every threshold with leave-one-training-target-out predictions.
    oof_scores = collections.defaultdict(list)
    oof_labels = []
    for inner_target in sorted(train_targets):
        inner_seed = seed + sum(ord(ch) for ch in inner_target) * 17
        inner_train_targets = set(train_targets) - {inner_target}
        inner_train_indices = [i for i, s in enumerate(sessions) if s["target"] in inner_train_targets]
        inner_train_pairs = build_pairs(sessions, inner_train_targets, inner_seed)
        inner_val_pairs = build_pairs(sessions, {inner_target}, inner_seed + 1)
        if not inner_train_pairs or not inner_val_pairs:
            continue
        inner_features, _ = prepare_feature_space(sessions, inner_train_indices)
        xx, vx = inner_features(inner_train_pairs), inner_features(inner_val_pairs)
        yy = np.asarray([p[2] for p in inner_train_pairs], int)
        vy = np.asarray([p[2] for p in inner_val_pairs], int)
        inner_models = fit_models(xx, yy)
        scores = {"route_cosine": vx[:, 0], "action_bigram_cosine": vx[:, 1],
                  **{name: model.predict_proba(vx[:, cols])[:, 1]
                     for name, (cols, model) in inner_models.items()}}
        for name, vals in scores.items():
            oof_scores[name].extend(float(v) for v in vals)
        oof_labels.extend(int(v) for v in vy)
    thresholds = {name: score_threshold(vals, oof_labels) for name, vals in oof_scores.items()}
    if set(thresholds) != {"route_cosine", "action_bigram_cosine", "route_plus_B", "route_plus_BS"}:
        raise RuntimeError("The inner target-holdout calibration did not cover all models")

    session_by = collections.defaultdict(list)
    all_agents = sorted({s["agent"] for s in sessions if s["target"] == target})
    families = sorted({s["family"] for s in sessions if s["target"] == target})
    for i, s in enumerate(sessions):
        if s["target"] == target:
            session_by[(s["family"], s["agent"])].append(i)
    rng = random.Random(seed + 101)
    test_pairs, queries = [], []
    for query_agent, candidate_agent in itertools.permutations(all_agents, 2):
        for family in families:
            for qid in session_by[(family, query_agent)]:
                positive_pool = session_by[(family, candidate_agent)]
                positive = rng.choice(positive_pool)
                negative_families = [f for f in families if f != family and session_by[(f, candidate_agent)]]
                chosen_negative_families = rng.sample(negative_families, 4)
                candidate_ids = [positive]
                truths = [True]
                candidate_ids.extend(rng.choice(session_by[(f, candidate_agent)]) for f in chosen_negative_families)
                truths.extend([False] * 4)
                order = list(range(len(candidate_ids)))
                rng.shuffle(order)
                candidate_ids = [candidate_ids[i] for i in order]
                truths = [truths[i] for i in order]
                pairs = [(qid, cid, int(truth)) for cid, truth in zip(candidate_ids, truths)]
                test_pairs.extend(pairs)
                queries.append({"query_id": qid, "family": family, "query_agent": query_agent,
                                "candidate_agent": candidate_agent, "pairs": pairs,
                                "truths": truths})
    test_x = pair_features(test_pairs)
    methods = {"route_cosine": test_x[:, 0], "action_bigram_cosine": test_x[:, 1],
               **{name: model.predict_proba(test_x[:, cols])[:, 1]
                  for name, (cols, model) in models.items()}}
    result = []
    cursor = 0
    for method, scores in methods.items():
        threshold = thresholds[method]
        pair_scores = scores
        pair_rows = []
        query_rows = []
        cursor = 0
        for q in queries:
            q_scores = pair_scores[cursor:cursor+len(q["pairs"])]
            for meta, score in zip(q["pairs"], q_scores):
                pair_rows.append({"label": meta[2], "pred": int(score >= threshold)})
            hit, mrr = rank_metrics(list(q_scores), q["truths"])
            query_rows.append({"family": q["family"], "hit_at_1": hit, "mrr": mrr})
            cursor += len(q["pairs"])
        pos = [r for r in pair_rows if r["label"] == 1]
        neg = [r for r in pair_rows if r["label"] == 0]
        fmr = float(np.mean([r["pred"] for r in neg]))
        fsr = float(np.mean([1-r["pred"] for r in pos]))
        by_family = collections.defaultdict(list)
        for r in query_rows:
            by_family[r["family"]].append(r)
        family_hit = [float(np.mean([x["hit_at_1"] for x in v])) for _, v in sorted(by_family.items())]
        family_mrr = [float(np.mean([x["mrr"] for x in v])) for _, v in sorted(by_family.items())]
        result.append({"target": target, "method": method,
                       "query_count": len(queries), "candidate_count": 5,
                       "chance_hit_at_1": .2,
                       "task_macro_hit_at_1": float(np.mean(family_hit)),
                       "task_macro_mrr": float(np.mean(family_mrr)),
                       "fmr": fmr, "fsr": fsr, "balanced_accuracy": 1-(fmr+fsr)/2,
                       "threshold": threshold, "positive_pairs": len(pos), "negative_pairs": len(neg)})
    audit = {"target": target, "training_targets": sorted(train_targets),
             "training_trace_count": len(train_indices), "training_pair_count": len(train_pairs),
             "training_positive_pairs": int(train_y.sum()),
             "training_negative_pairs": int((train_y == 0).sum()),
             "threshold_oof_pairs": len(oof_labels),
             "threshold_policy": "leave-one-training-target-out scores; max recall under FMR<=10%",
             "thresholds": thresholds, "transform": transform_audit,
             "test_family_count": len(families), "test_agent_count": len(all_agents)}
    return result, audit


def main():
    sessions, coverage = load_sessions()
    targets = sorted({s["target"] for s in sessions})
    if len(targets) != 3 or len(sessions) != 450:
        raise RuntimeError(f"Unexpected pinned CyBias cohort: {len(sessions)} sessions; targets={targets}")
    all_results, fold_audit = [], []
    for target in targets:
        train_targets = set(targets) - {target}
        results, audit = evaluate_target(sessions, target, train_targets, SEED + targets.index(target) * 1237)
        all_results.extend(results)
        fold_audit.append(audit)
        print(f"CyBias held-out target {target}: queries={results[0]['query_count']}", flush=True)
    methods = sorted({r["method"] for r in all_results})
    summary = []
    for method in methods:
        vals = [r for r in all_results if r["method"] == method]
        summary.append({"method": method, "targets": [r["target"] for r in vals],
                        "target_macro_hit_at_1": float(np.mean([r["task_macro_hit_at_1"] for r in vals])),
                        "target_range_hit_at_1": [float(min(r["task_macro_hit_at_1"] for r in vals)),
                                                   float(max(r["task_macro_hit_at_1"] for r in vals))],
                        "target_macro_mrr": float(np.mean([r["task_macro_mrr"] for r in vals])),
                        "target_macro_fmr": float(np.mean([r["fmr"] for r in vals])),
                        "target_macro_fsr": float(np.mean([r["fsr"] for r in vals])),
                        "target_macro_ba": float(np.mean([r["balanced_accuracy"] for r in vals])),
                        "per_target": vals,
                        "uncertainty_note": "only three held-out targets; report target range, not a pseudo-precise query-level interval"})
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "metrics.json", summary)
    write_json(OUT / "folds.json", fold_audit)
    write_json(OUT / "coverage.json", coverage)
    repo_revision = None
    if (REPO_DIR / ".git").exists():
        try:
            repo_revision = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            pass
    write_json(OUT / "provenance.json", {
        "protocol": "cybias-task-condition-retrieval-v2",
        "source": "Harry24k/CyBiasBench data/bias_injection_traces.parquet and bias_injection_records.csv",
        "repository_revision": repo_revision,
        "trace_file_sha256": sha256(TRACE_FILE), "records_file_sha256": sha256(RECORD_FILE),
        "raw_trace_rows": coverage["raw_request_rows"], "sessions": len(sessions),
        "prefix_cap": MAX_EVENTS, "minimum_events": MIN_EVENTS,
        "outer_validation": "leave one target environment out; train and threshold selection use the other two targets",
        "relation_proxy": "same target and requested attack family across different Agent snapshots; negatives use the same target and fixed query-to-candidate Agent pair with a different requested family",
        "feature_contract": "request method and query-stripped normalized path shape; order preserved; no status/body/header/target/family/outcome/model field enters the feature matrix",
        "models": {"route_cosine": "TF-IDF method/path token cosine", "action_bigram_cosine": "TF-IDF read/write action bigrams",
                   "route_plus_B": "balanced logistic regression over request cosine and B profile similarity",
                   "route_plus_BS": "balanced logistic regression over request cosine and B/S profile similarities"},
        "intervals": "three target clusters are reported individually and as a range; requests/sessions are not resampled as independent targets",
        "license_handling": "aggregate metrics only; no modified session rows redistributed",
        "outputs": "aggregate result files only; no per-session predictions or derived traces written",
    })
    print(json.dumps({"coverage": coverage, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
