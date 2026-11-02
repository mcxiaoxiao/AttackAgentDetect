#!/usr/bin/env python3
"""Reproducible public-source evaluation for the B/S/T fingerprint v2."""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import pathlib
import random
import shlex
import statistics
import sys
import time
import zipfile

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

EXP = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "research/results/behavioral_fingerprint_v2"
HONEY_ZIP = ROOT / "research/data/external/honey-for-agent-v1/Logs_final.zip"
SEED = 20261103
MAX_TRAIN_PAIRS_PER_CLASS = 2500
QUERIES_PER_CONDITION_PAIR = 3
BOOTSTRAPS = 5000


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: pathlib.Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def as_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) and x >= 0 else None
    except (TypeError, ValueError):
        return None


def command_symbol(value: str) -> str:
    """Keep only the executable token; arguments are outside this fingerprint view."""
    try:
        parts = shlex.split(str(value), posix=True)
    except ValueError:
        parts = str(value).strip().split()
    if not parts:
        return ""
    token = parts[0].strip().lower()
    if token in {"sudo", "doas"} and len(parts) > 1:
        token = parts[1].strip().lower()
    return pathlib.PurePosixPath(token).name or token


def parse_honey_session(member: str, payload: bytes):
    obj = json.loads(payload)
    parts = pathlib.PurePosixPath(member).parts
    if len(parts) < 6 or len(obj) < 2:
        return None
    prompt, turn_cap, environment, model = parts[1:5]
    symbols, actions, execution_s, response_s = [], [], [], []
    for turn in obj[1:]:
        if not isinstance(turn, list) or len(turn) < 3:
            continue
        command_record = turn[2]
        if not isinstance(command_record, list) or len(command_record) < 2 or command_record[0] != "raw_command":
            continue
        symbol = command_symbol(command_record[1])
        if not symbol:
            continue
        symbols.append(symbol)
        actions.append(ATOM_GROUP.get(symbol))
        response_s.append(None)
        execution_s.append(None)
        if isinstance(turn[0], list) and len(turn[0]) > 1:
            response_s[-1] = as_float(turn[0][1])
        if isinstance(turn[1], list) and len(turn[1]) > 1:
            execution_s[-1] = as_float(turn[1][1])
    if len(symbols) < 4:
        return {"member": member, "prompt": prompt, "turn_cap": turn_cap,
                "environment": environment, "model": model, "commands": symbols,
                "responses_ms": response_s, "execution_ms": execution_s,
                "eligible": False, "blocks": None, "audit": {"events": len(symbols)}}
    blocks, audit = fingerprint_matrix(symbols, actions=actions,
                                       durations=[None if x is None else x / 1000 for x in execution_s],
                                       window=8, min_window_events=4, max_events=128)
    return {"member": member, "prompt": prompt, "turn_cap": turn_cap,
            "environment": environment, "model": model, "commands": symbols,
            "responses_ms": response_s, "execution_ms": execution_s,
            "eligible": blocks is not None, "blocks": blocks, "audit": audit}


def load_honey():
    if not HONEY_ZIP.is_file():
        raise FileNotFoundError(f"Download the source archive first: {HONEY_ZIP}")
    sessions, total = [], 0
    with zipfile.ZipFile(HONEY_ZIP) as archive:
        for member in sorted(x for x in archive.namelist() if x.endswith(".json")):
            total += 1
            try:
                row = parse_honey_session(member, archive.read(member))
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, IndexError):
                continue
            if row is not None:
                row["condition"] = "/".join((row["prompt"], row["turn_cap"], row["environment"]))
                if row["eligible"]:
                    row["summary"] = {b: summarize(row["blocks"][b]) for b in ("B", "S", "T")}
                sessions.append(row)
    return sessions, total


def fit_block_vectors(sessions, train_idx, block):
    train = np.vstack([sessions[i]["summary"][block] for i in train_idx])
    all_rows = np.vstack([sessions[i]["summary"][block] if sessions[i]["eligible"]
                          else np.full(train.shape[1], np.nan) for i in range(len(sessions))])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(imputer.fit_transform(train))
    xall = scaler.transform(imputer.transform(all_rows))
    return xall, imputer, scaler


def pair_dist(vectors, i, j):
    return float(np.sqrt(np.mean((vectors[i] - vectors[j]) ** 2)))


def model_condition_index(sessions, indices):
    out = collections.defaultdict(list)
    for i in indices:
        if sessions[i]["eligible"]:
            out[(sessions[i]["model"], sessions[i]["condition"])].append(i)
    return out


def available_conditions(sessions, models):
    per_model = [set(s["condition"] for s in sessions if s["eligible"] and s["model"] == m)
                 for m in models]
    common = set.intersection(*per_model) if per_model else set()
    return sorted(common)


def make_train_pairs(sessions, train_models, n_per_class, rng):
    idx = [i for i, s in enumerate(sessions) if s["eligible"] and s["model"] in train_models]
    by_mc = model_condition_index(sessions, idx)
    conditions = sorted(set(c for _, c in by_mc))
    condition_models = {c: {m for m in train_models if by_mc.get((m, c))} for c in conditions}
    model_pairs = list(itertools.combinations(sorted(train_models), 2))
    positive_options, negative_options = [], []
    for m1, m2 in model_pairs:
        for c in conditions:
            if m1 in condition_models[c] and m2 in condition_models[c]:
                positive_options.append((m1, m2, c))
        for c1 in conditions:
            if m1 not in condition_models[c1]:
                continue
            p1, cap1, env1 = c1.split("/")
            for c2 in conditions:
                if m2 not in condition_models[c2] or c1 == c2:
                    continue
                p2, cap2, env2 = c2.split("/")
                if p1 != p2 and cap1 == cap2 and env1 == env2:
                    negative_options.append((m1, m2, c1, c2))
    pairs = []
    for label, options in ((1, positive_options), (0, negative_options)):
        if not options:
            continue
        seen = set()
        attempts = 0
        while len(seen) < n_per_class and attempts < n_per_class * 50:
            attempts += 1
            option = rng.choice(options)
            if label:
                m1, m2, cond = option
                a = rng.choice(by_mc[(m1, cond)])
                b = rng.choice(by_mc[(m2, cond)])
            else:
                m1, m2, c1, c2 = option
                a = rng.choice(by_mc[(m1, c1)])
                b = rng.choice(by_mc[(m2, c2)])
            seen.add((a, b, label))
        pairs.extend(sorted(seen))
    rng.shuffle(pairs)
    return pairs


def make_pair_features(pairs, docs, vectors):
    result = np.empty((len(pairs), 4), dtype=float)
    for k, (a, b, _) in enumerate(pairs):
        result[k, 0] = float(docs[a].multiply(docs[b]).sum())
        result[k, 1] = -pair_dist(vectors["B"], a, b)
        result[k, 2] = -pair_dist(vectors["S"], a, b)
        result[k, 3] = -pair_dist(vectors["T"], a, b)
    return result


def fit_pair_model(x, y, columns):
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(),
                          LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear",
                                             max_iter=2000, random_state=SEED))
    model.fit(x[:, columns], y)
    return model


def rank_credit(scores):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        return 0.0, 0.0
    best = float(np.max(scores))
    winners = [i for i, s in enumerate(scores) if math.isclose(float(s), best, rel_tol=1e-10, abs_tol=1e-12)]
    hit = 1 / len(winners) if 0 in winners else 0.0
    greater = int(np.sum(scores > scores[0] + 1e-12))
    tied = [i for i, s in enumerate(scores) if math.isclose(float(s), float(scores[0]), rel_tol=1e-10, abs_tol=1e-12)]
    if not tied:
        return hit, 0.0
    mrr = float(np.mean([1 / (greater + r + 1) for r in range(len(tied))])) if 0 in tied else 0.0
    return hit, mrr


def bootstrap_macro(rows, field, seed=SEED):
    by_condition = collections.defaultdict(list)
    for r in rows:
        by_condition[r["condition"]].append(r[field])
    ids = sorted(by_condition)
    vals = np.asarray([np.mean(by_condition[x]) for x in ids], dtype=float)
    if len(vals) < 2:
        return {"mean": float(np.mean(vals)) if len(vals) else None, "ci95": None, "clusters": len(vals)}
    rng = np.random.default_rng(seed)
    sims = vals[rng.integers(0, len(vals), (BOOTSTRAPS, len(vals)))].mean(axis=1)
    return {"mean": float(vals.mean()), "ci95": [float(x) for x in np.quantile(sims, [.025, .975])],
            "clusters": len(vals)}


def evaluate_honey(sessions, archive_count):
    out = OUT / "honey"
    out.mkdir(parents=True, exist_ok=True)
    eligible = [i for i, s in enumerate(sessions) if s["eligible"]]
    models = sorted({sessions[i]["model"] for i in eligible})
    conditions = available_conditions(sessions, models)
    cmd_docs = [" ".join(s["commands"]) if s["eligible"] else "emptytrace" for s in sessions]
    results, fold_rows = [], []
    pairs = list(itertools.permutations(models, 2))
    for fold_no, (query_model, candidate_model) in enumerate(pairs, 1):
        train_models = [m for m in models if m not in {query_model, candidate_model}]
        train_idx = [i for i in eligible if sessions[i]["model"] in train_models]
        if not train_idx:
            continue
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                     token_pattern=r"(?u)\b[\w.+-]+\b", lowercase=True)
        vectorizer.fit([cmd_docs[i] for i in train_idx])
        docs = vectorizer.transform(cmd_docs)
        vectors = {b: fit_block_vectors(sessions, train_idx, b)[0] for b in ("B", "S", "T")}
        rng = random.Random(SEED + fold_no)
        train_pairs = make_train_pairs(sessions, train_models, MAX_TRAIN_PAIRS_PER_CLASS, rng)
        if not train_pairs or len({x[2] for x in train_pairs}) < 2:
            fold_rows.append({"query_model": query_model, "candidate_model": candidate_model,
                              "status": "skipped_no_training_pairs", "training_pairs": len(train_pairs)})
            continue
        train_x = make_pair_features(train_pairs, docs, vectors)
        train_y = np.asarray([p[2] for p in train_pairs], dtype=int)
        models_fit = {
            "B": fit_pair_model(train_x, train_y, [0, 1]),
            "B_plus_S": fit_pair_model(train_x, train_y, [0, 1, 2]),
            "B_plus_S_plus_exec_duration": fit_pair_model(train_x, train_y, [0, 1, 2, 3]),
        }
        lookup = model_condition_index(sessions, eligible)
        common = sorted(set(conditions) & {c for m, c in lookup if m == query_model}
                        & {c for m, c in lookup if m == candidate_model})
        n_queries = 0
        for condition in common:
            prompt, turn_cap, environment = condition.split("/")
            query_pool = list(lookup[(query_model, condition)])
            candidate_positive_pool = list(lookup[(candidate_model, condition)])
            other_conditions = [c for c in conditions if c != condition
                                and c.split("/")[1] == turn_cap and c.split("/")[2] == environment]
            negative_pool = [i for c in other_conditions for i in lookup.get((candidate_model, c), [])]
            if len(query_pool) < QUERIES_PER_CONDITION_PAIR or len(candidate_positive_pool) < QUERIES_PER_CONDITION_PAIR or len(negative_pool) < 4:
                continue
            qids = rng.sample(query_pool, QUERIES_PER_CONDITION_PAIR)
            pids = rng.sample(candidate_positive_pool, QUERIES_PER_CONDITION_PAIR)
            for qi, pi in zip(qids, pids):
                negatives = rng.sample(negative_pool, 4)
                cand = [pi] + negatives
                records = [(qi, c, 1 if j == 0 else 0) for j, c in enumerate(cand)]
                feat = make_pair_features(records, docs, vectors)
                method_scores = {
                    "command_cosine": feat[:, 0],
                    "B_profile": models_fit["B"].predict_proba(feat[:, [0, 1]])[:, 1],
                    "B_plus_S": models_fit["B_plus_S"].predict_proba(feat[:, [0, 1, 2]])[:, 1],
                    "B_plus_S_plus_exec_duration": models_fit["B_plus_S_plus_exec_duration"].predict_proba(feat)[:, 1],
                }
                for method, score in method_scores.items():
                    hit, mrr = rank_credit(score)
                    results.append({"condition": condition, "query_model": query_model,
                                    "candidate_model": candidate_model, "method": method,
                                    "hit_at_1": hit, "mrr": mrr})
                n_queries += 1
        fold_rows.append({"query_model": query_model, "candidate_model": candidate_model,
                          "status": "complete", "training_models": len(train_models),
                          "training_pairs": len(train_pairs), "shared_conditions": len(common),
                          "test_queries": n_queries})
        print(f"Honey model-pair {fold_no}/{len(pairs)} {query_model}->{candidate_model}: {n_queries} queries", flush=True)

    methods = sorted({r["method"] for r in results})
    metrics = []
    for method in methods:
        chosen = [r for r in results if r["method"] == method]
        metrics.append({"method": method, "hit_at_1": bootstrap_macro(chosen, "hit_at_1"),
                        "mrr": bootstrap_macro(chosen, "mrr"),
                        "query_n": len(chosen), "chance_hit_at_1": .2,
                        "task_relation": "same prompt, turn-cap and environment; positive uses a different held-out model snapshot"})
    by_method_condition = collections.defaultdict(dict)
    for r in results:
        by_method_condition[r["condition"]].setdefault(r["method"], []).append(r["hit_at_1"])
    primary_delta = []
    exec_delta = []
    for condition, vals in by_method_condition.items():
        if "command_cosine" in vals and "B_plus_S" in vals:
            primary_delta.append({"condition": condition, "delta": float(np.mean(vals["B_plus_S"]) - np.mean(vals["command_cosine"]))})
        if "B_plus_S" in vals and "B_plus_S_plus_exec_duration" in vals:
            exec_delta.append({"condition": condition, "delta": float(np.mean(vals["B_plus_S_plus_exec_duration"]) - np.mean(vals["B_plus_S"]))})
    def delta_summary(rows):
        values = np.asarray([r["delta"] for r in rows], dtype=float)
        if len(values) < 2:
            return {"mean": float(values.mean()) if len(values) else None, "ci95": None, "clusters": len(values)}
        rng = np.random.default_rng(SEED + 77)
        ci = np.quantile(values[rng.integers(0, len(values), (BOOTSTRAPS, len(values)))].mean(axis=1), [.025, .975])
        return {"mean": float(values.mean()), "ci95": [float(x) for x in ci], "clusters": len(values)}
    summary = {"source": "Honey for the Agent, Zenodo version v1",
               "raw_json_sessions": archive_count, "parsed_session_files": len(sessions),
               "eligible_sessions_minimum_4_commands": len(eligible), "models": models,
               "model_count": len(models), "conditions": len(conditions),
               "prompt_count": len({s["prompt"] for s in sessions}),
               "turn_cap_count": len({s["turn_cap"] for s in sessions}),
               "environment_count": len({s["environment"] for s in sessions}),
               "command_count_quantiles": {str(q): float(np.quantile([len(s["commands"]) for s in sessions if s["eligible"]], q))
                                            for q in (0, .25, .5, .75, 1)},
               "execution_latency_ms_nonmissing": sum(x is not None for s in sessions for x in s["execution_ms"]),
               "response_latency_ms_nonmissing": sum(x is not None for s in sessions for x in s["responses_ms"]),
               "timing_interpretation": "source-reported model response and command execution/round-trip durations; no inter-event start gaps inferred",
               "methods": metrics,
               "paired_B_plus_S_minus_command_hit1": delta_summary(primary_delta),
               "paired_exec_duration_increment_over_B_plus_S": delta_summary(exec_delta),
               "held_out_model_pairs": len([x for x in fold_rows if x["status"] == "complete"]),
               "uncertainty_unit": "60 prompt x turn-cap x environment conditions; model-pair and query repeats averaged within condition"}
    write_json(out / "metrics.json", summary)
    write_json(out / "folds.json", fold_rows)
    write_json(out / "provenance.json", {"protocol": "public-source-honey-leave-two-models-out-v1",
               "seed": SEED, "archive_sha256": digest(HONEY_ZIP), "archive_bytes": HONEY_ZIP.stat().st_size,
               "feature_code_sha256": digest(ROOT / "attackagent/fingerprint_v2.py"),
               "runner_sha256": digest(pathlib.Path(__file__)), "training_pairs_per_class_cap": MAX_TRAIN_PAIRS_PER_CLASS,
               "queries_per_condition_model_pair": QUERIES_PER_CONDITION_PAIR,
               "feature_blocks": {"B": "8 local symbol/action measures", "S": "8 prefix-aware order/revisit/reuse measures",
                                  "T": "8 time measures; only measured execution durations are available in Honey"},
               "redistribution": "only aggregates; no derived session-level rows are published"})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["honey", "all"], nargs="?", default="honey")
    args = parser.parse_args()
    started = time.monotonic()
    sessions, raw_n = load_honey()
    summary = evaluate_honey(sessions, raw_n)
    source_manifest = {
        "version": "behavioral-fingerprint-v2-public-sources",
        "date": "2026-11-03", "honey": summary,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(OUT / "dataset_manifest.json", source_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
