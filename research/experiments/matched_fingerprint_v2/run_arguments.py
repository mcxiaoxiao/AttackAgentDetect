#!/usr/bin/env python3
"""Task-held-out matched CTH mixture and AIT timing association.

An exploratory experiment: one observation is an entire source trace, not a
window or a repeated gallery draw. Raw commands stay in the local inputs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
import random
import sys
import time
import warnings

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import xgboost

ROOT = pathlib.Path(__file__).resolve().parents[3]
EXP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from attackagent.fingerprint_v2 import fingerprint_matrix, summarize
from attackagent.traces import ATOM_GROUP

OUT = ROOT / "research/results/matched_fingerprint_v2/argument_profile"
SEED = 20261103
WINDOWS = (4, 8, 16)
POOLS = ("mean_std", "mean_std_trend")
FUSIONS = ("direct", "conditional")
REPRS = tuple(itertools.product(WINDOWS, POOLS, FUSIONS))
PRIOR = (8, "mean_std_trend", "conditional")
PARAMS = {"svm": (0.1, 1., 10.), "tree": (3, 5, 8), "xgb": (2, 3, 5)}
METHOD_BLOCKS = {"evidence": (), "B": ("B",), "S": ("S",), "BS": ("B", "S"),
                 "ST": ("S", "T"), "BT": ("B", "T"), "BST": ("B", "S", "T")}
FOLDS_CSV = ROOT / "research/results/overnight_fingerprint_v1/task_folds.csv"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_json(x):
    if isinstance(x, dict):
        return {str(k): finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return finite_json(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def build_cth():
    legacy = module(ROOT / "research/experiments/overnight_fingerprint_v1/run.py", "legacy")
    source = [r for r in legacy.load_corpus()
              if r["source_dataset"] == "cyber-task-horizons" and r["track"] == "ctf"]
    pools = collections.defaultdict(collections.Counter)
    for r in source:
        pools[r["task_group"]][r["mode"]] += 1
    tasks = sorted(t for t, p in pools.items() if p["human"] >= 2 and p["agent"] >= 2)
    assert len(tasks) == 17
    rows = []
    for r in source:
        if r["task_group"] in tasks:
            rows.append({"id": r["trace_id"], "task": r["task_group"],
                         "mode": r["mode"], "commands": r["commands"],
                         "actions": r["actions"], "timestamps": None,
                         "locator": r.get("record_locator", "")})
    assert len({r["id"] for r in rows}) == len(rows)
    with FOLDS_CSV.open() as f:
        folds = {r["group"]: int(r["fold"]) for r in csv.DictReader(f)
                 if r["endpoint"] == "cth_mixture" and r["group"] in tasks}
    assert len(folds) == 17
    profile_module = module(EXP / "argument_profile.py", "argument_profile")
    profiles = profile_module.cth_profiles(rows)
    for row in rows:
        profile = profiles[row["id"]]
        assert profile["recognized_heads"] == len(row["commands"])
        row["argument_profile"] = profile["features"]
    return rows, tasks, folds


def build_ait():
    pilot = module(ROOT / "research/experiments/log_fingerprint_pilot_v1/run.py", "pilot")
    raw = pilot.load_runs()
    rows = []
    for r in raw:
        ev = [e for e in r["events"] if e["login"]][:128]
        if len(ev) < 4:
            continue
        commands = [e["command"] for e in ev]
        rows.append({"id": r["id"], "task": r["task"],
                     "mode": "agent" if r["y"] else "human",
                     "commands": commands, "actions": [ATOM_GROUP.get(c) for c in commands],
                     "timestamps": [e["time"] for e in ev],
                     "raw_hash": r["raw_hash"], "source_group": r["source"]})
    assert len(rows) == 109 and len({r["id"] for r in rows}) == 109
    assert len({r["raw_hash"] for r in rows}) == len(rows), "raw duplicate audit failed"
    tasks = sorted({r["task"] for r in rows})
    assert len(tasks) == 12
    counts = collections.Counter((r["task"], r["mode"]) for r in rows)
    assert all(counts[t, mode] >= 2 for t in tasks for mode in ("human", "agent"))
    folds = {t: i % 3 for i, t in enumerate(tasks)}
    return rows, tasks, folds


def profile_cache(rows, minimum):
    cache = {}
    for window in WINDOWS:
        out = []
        for r in rows:
            blocks, audit = fingerprint_matrix(
                r["commands"], actions=r["actions"], timestamps=r["timestamps"],
                window=window, min_window_events=minimum, max_events=128)
            d = {}
            for name in ("B", "S", "T"):
                d[name] = summarize(blocks[name]) if blocks is not None else np.full(24, np.nan)
            d["B_ok"] = blocks is not None and np.isfinite(blocks["B"]).any()
            d["S_ok"] = blocks is not None and np.isfinite(blocks["S"]).any()
            d["T_ok"] = blocks is not None and np.isfinite(blocks["T"]).any()
            d["windows"] = audit["windows"]
            out.append(d)
        cache[window] = out
        print("profiles", window, "complete", sum(r["B_ok"] and r["S_ok"] for r in out),
              "/", len(out), flush=True)
    return cache


def pair_type(a, b):
    if a == b == "human":
        return "HH"
    if a == b == "agent":
        return "AA"
    return "HA"


def append_pair(out, rows, q, c, fold, rep, frac, qindex, candidate):
    a, b = rows[q], rows[c]
    same = int(a["task"] == b["task"])
    assert q != c
    if same and a.get("raw_hash") and b.get("raw_hash"):
        assert a["raw_hash"] != b["raw_hash"]
    key = f"{rep}:{frac}:{qindex}:{candidate}"
    out.append({"pair_id": key, "fold": fold, "replicate": rep, "fraction": frac,
                "query_idx": q, "candidate_idx": c, "query_id": a["id"],
                "candidate_id": b["id"], "query_task": a["task"],
                "candidate_task": b["task"], "query_mode": a["mode"],
                "candidate_mode": b["mode"], "pair_type": pair_type(a["mode"], b["mode"]),
                "same_task": same, "candidate_position": candidate})


def cth_pairs(rows, tasks, folds):
    pools = collections.defaultdict(lambda: {"human": [], "agent": []})
    for i, r in enumerate(rows):
        pools[r["task"]][r["mode"]].append(i)
    result, roles = [], []
    for rep in range(170):
        rng = random.Random(SEED + rep)
        excluded = tasks[rep % 17]
        chosen = [t for t in tasks if t != excluded]
        slots = [(t, role) for t in chosen for role in ("query", "gallery")]
        order = list(range(32))
        rng.shuffle(order)
        picked = {t: {mode: rng.sample(pools[t][mode], 2)
                      for mode in ("human", "agent")} for t in chosen}
        negatives = {t: rng.sample([x for x in chosen if folds[x] == folds[t] and x != t], 3)
                     for t in chosen}
        for count in (0, 8, 16, 24, 32):
            agents = set(order[:count])
            modes = {slot: ("agent" if k in agents else "human") for k, slot in enumerate(slots)}
            assert sum(x == "agent" for x in modes.values()) == count
            for slot in slots:
                t, role = slot
                mode = modes[slot]
                idx = picked[t][mode][0 if role == "query" else 1]
                roles.append({"replicate": rep, "agent_slots": count, "task": t,
                              "role": role, "mode": mode, "trace_id": rows[idx]["id"]})
            for qi, t in enumerate(chosen):
                q = picked[t][modes[t, "query"]][0]
                for j, u in enumerate([t] + negatives[t]):
                    c = picked[u][modes[u, "gallery"]][1]
                    append_pair(result, rows, q, c, folds[t], rep, count / 32, qi, j)
    assert len(result) == 170 * 5 * 16 * 4
    return result, roles


def gallery(rows, taskset, fold, reps, seed, fraction=-1):
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["task"] in taskset:
            by[r["task"]].append(i)
    tasks = sorted(by)
    assert len(tasks) >= 4
    out = []
    for rep in range(reps):
        rng = random.Random(seed + rep)
        qi = 0
        for task in tasks:
            for q in by[task]:
                pos = [i for i in by[task] if i != q and
                       (not rows[q].get("raw_hash") or rows[q]["raw_hash"] != rows[i].get("raw_hash"))]
                if not pos:
                    continue
                others = rng.sample([t for t in tasks if t != task], 3)
                candidates = [rng.choice(pos)] + [rng.choice(by[t]) for t in others]
                for j, c in enumerate(candidates):
                    append_pair(out, rows, q, c, fold, rep, fraction, qi, j)
                qi += 1
    return out


def train_pairs(rows, taskset, seed):
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["task"] in taskset:
            by[r["task"]].append(i)
    tasks = sorted(by)
    positives = []
    for task in tasks:
        pairs = [(a, b) for a, b in itertools.combinations(by[task], 2)
                 if not rows[a].get("raw_hash") or rows[a]["raw_hash"] != rows[b].get("raw_hash")]
        rng.shuffle(pairs)
        positives.extend((a, b, 1) for a, b in pairs[:8])
    assert positives
    negatives = set()
    attempts = 0
    while len(negatives) < len(positives) and attempts < len(positives) * 100:
        ta, tb = rng.sample(tasks, 2)
        a, b = rng.choice(by[ta]), rng.choice(by[tb])
        negatives.add((min(a, b), max(a, b), 0))
        attempts += 1
    assert len(negatives) == len(positives)
    # Sorting prevents per-process hash randomization from changing the sample.
    pairs = positives + sorted(negatives)
    rng.shuffle(pairs)
    return pairs


def as_arrays(records):
    a = np.asarray([r["query_idx"] for r in records], dtype=np.int32)
    b = np.asarray([r["candidate_idx"] for r in records], dtype=np.int32)
    y = np.asarray([r["same_task"] for r in records], dtype=np.int8)
    tasks = np.asarray([r["query_task"] for r in records], dtype=object)
    return a, b, y, tasks


def train_arrays(pairs):
    return tuple(np.asarray([p[i] for p in pairs], dtype=np.int32) for i in range(3))


def write_gz_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError(f"empty output {path}")
    with gzip.open(path, "wt", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


class Features:
    """Fit trace-level vocabulary and B/S/T transforms on training traces only."""
    def __init__(self, rows, train_idx, profiles, window, pooling):
        self.rows, self.profiles, self.window = rows, profiles[window], window
        command = [" ".join(r["commands"]) or "EMPTYTOKEN" for r in rows]
        actions = []
        for r in rows:
            pairs = [f"{a}__{b}" for a, b in zip(r["actions"], r["actions"][1:])
                     if a is not None and b is not None]
            actions.append(" ".join(pairs) or "EMPTYTOKEN")
        arguments = [" ".join(r.get("argument_profile", [])) or "EMPTYTOKEN" for r in rows]
        self.matrix = []
        for docs, ngram in ((command, (1, 2)), (actions, (1, 1)), (arguments, (1, 1))):
            vec = TfidfVectorizer(ngram_range=ngram, sublinear_tf=True,
                                  token_pattern=r"(?u)\b[\w.+-]+\b", lowercase=True)
            vec.fit([docs[i] for i in train_idx])
            self.matrix.append(vec.transform(docs))
        self.z, self.ok = {}, {}
        keep = np.asarray([i for i in range(24) if pooling == "mean_std_trend" or i % 3 != 2])
        for block in ("B", "S", "T"):
            raw = np.vstack([r[block][keep] for r in self.profiles])
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
            fit = imp.fit_transform(raw[train_idx])
            scaler = StandardScaler().fit(fit)
            self.z[block] = scaler.transform(imp.transform(raw))
            self.ok[block] = np.asarray([r[block + "_ok"] for r in self.profiles], dtype=bool)

    def evidence(self, a, b):
        return np.column_stack([np.asarray(m[a].multiply(m[b]).sum(axis=1)).ravel()
                                for m in self.matrix])

    def evidence_head_only(self, a, b):
        return self.evidence(a, b)[:, :2]

    def full(self, e, a, b, blocks, fusion):
        parts = [e]
        valid = np.ones(len(a), dtype=bool)
        for block in blocks:
            diff = np.abs(self.z[block][a] - self.z[block][b])
            if fusion == "conditional":
                diff = diff * e[:, :1]
            parts.append(diff)
            valid &= self.ok[block][a] & self.ok[block][b]
        return np.column_stack(parts), ~valid


def classifier(kind, param):
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=float(param), kernel="rbf", gamma="scale",
                                                  class_weight="balanced"))
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=int(param), min_samples_leaf=3,
                                      class_weight="balanced", random_state=SEED)
    if kind == "xgb":
        return XGBClassifier(n_estimators=100, max_depth=int(param), learning_rate=0.05,
                             min_child_weight=3, reg_lambda=1., subsample=1.,
                             colsample_bytree=1., objective="binary:logistic",
                             tree_method="hist", n_jobs=1, random_state=SEED, verbosity=0)
    raise ValueError(kind)


def fit_scores(kind, param, xtrain, ytrain, xval):
    model = classifier(kind, param)
    model.fit(xtrain, ytrain)
    if kind == "svm":
        return model.decision_function(xval)
    return model.predict_proba(xval)[:, 1]


def threshold(y, scores, fallback=None, fallback_pred=None):
    """Maximize pooled recall subject to pooled validation FMR <= .10."""
    if fallback is None:
        fallback = np.zeros(len(y), dtype=bool)
        fallback_pred = np.zeros(len(y), dtype=bool)
    neg, pos = y == 0, y == 1
    assert neg.any() and pos.any()
    candidates = np.r_[np.inf, np.sort(np.unique(scores[~fallback]))[::-1]]
    best_recall, best = -1., math.inf
    for t in candidates:
        pred = scores >= t
        pred[fallback] = fallback_pred[fallback]
        fmr = float(pred[neg].mean())
        recall = float(pred[pos].mean())
        if fmr <= .1 + 1e-12 and recall > best_recall + 1e-12:
            best_recall, best = recall, float(t)
    if best_recall < 0:
        raise ValueError("infeasible threshold due to baseline fallback predictions")
    return best


def decision(scores, t, fallback=None, base_pred=None):
    pred = scores >= t
    if fallback is not None:
        pred[fallback] = base_pred[fallback]
    return pred


def macro(y, pred, tasks):
    rates = []
    for task in sorted(set(tasks)):
        m = tasks == task
        pos, neg = m & (y == 1), m & (y == 0)
        if pos.any() and neg.any():
            fsr = float(np.mean(~pred[pos]))
            fmr = float(np.mean(pred[neg]))
            rates.append((fmr, fsr, 1 - (fmr + fsr) / 2))
    assert rates
    return np.mean(np.asarray(rates), axis=0)


def repr_rank(rep):
    window, pooling, fusion = rep
    return (int(rep == PRIOR), -abs(window - 8),
            int(pooling == "mean_std_trend"), int(fusion == "conditional"), -window)


def inner_contexts(rows, outer_train_tasks, profiles, fold, dataset):
    tasks = sorted(outer_train_tasks)
    inner_map = {task: i % 2 for i, task in enumerate(tasks)}
    result = []
    for inner in (0, 1):
        val_tasks = {t for t in tasks if inner_map[t] == inner}
        fit_tasks = set(tasks) - val_tasks
        assert len(val_tasks) >= 4 and len(fit_tasks) >= 4
        train_idx = [i for i, r in enumerate(rows) if r["task"] in fit_tasks]
        train = train_pairs(rows, fit_tasks, SEED + 10000 * fold + 257 * inner)
        val = gallery(rows, val_tasks, inner, 1, SEED + 50000 * fold + 101 * inner)
        a, b, y, tasks_val = as_arrays(val)
        aa, bb, yy = train_arrays(train)
        features = {}
        for window in WINDOWS:
            for pooling in POOLS:
                f = Features(rows, train_idx, profiles, window, pooling)
                etrain, eval_ = f.evidence(aa, bb), f.evidence(a, b)
                features[(window, pooling)] = (f, etrain, eval_)
        result.append({"inner": inner, "train_idx": train_idx, "train_a": aa, "train_b": bb,
                       "train_y": yy, "val_a": a, "val_b": b, "val_y": y,
                       "val_tasks": tasks_val, "features": features,
                       "fit_tasks": sorted(fit_tasks), "validation_tasks": sorted(val_tasks),
                       "train_pairs": len(train), "validation_pairs": len(val)})
    return result


def oof_base(contexts, kind, param):
    chunks = []
    for ctx in contexts:
        _, e_train, e_val = ctx["features"][(8, "mean_std_trend")]
        chunks.append(fit_scores(kind, param, e_train, ctx["train_y"], e_val))
    return np.concatenate(chunks)


def oof_head_only(contexts, kind, param):
    chunks = []
    for ctx in contexts:
        f, _, _ = ctx["features"][(8, "mean_std_trend")]
        train = f.evidence_head_only(ctx["train_a"], ctx["train_b"])
        val = f.evidence_head_only(ctx["val_a"], ctx["val_b"])
        chunks.append(fit_scores(kind, param, train, ctx["train_y"], val))
    return np.concatenate(chunks)


def oof_variant(contexts, kind, param, rep, blocks):
    window, pooling, fusion = rep
    vals, missing = [], []
    for ctx in contexts:
        f, etrain, eval_ = ctx["features"][(window, pooling)]
        xtrain, _ = f.full(etrain, ctx["train_a"], ctx["train_b"], blocks, fusion)
        xval, no = f.full(eval_, ctx["val_a"], ctx["val_b"], blocks, fusion)
        vals.append(fit_scores(kind, param, xtrain, ctx["train_y"], xval))
        missing.append(no)
    return np.concatenate(vals), np.concatenate(missing)


def evaluate_oof(y, tasks, base_scores, full_scores, missing):
    base_t = threshold(y, base_scores)
    bp = decision(base_scores, base_t)
    t = threshold(y, full_scores, missing, bp)
    fp = decision(full_scores, t, missing, bp)
    bm, fm = macro(y, bp, tasks), macro(y, fp, tasks)
    return {"base_threshold": base_t, "threshold": t, "base_metrics": bm,
            "full_metrics": fm, "objective": float((bm[2] + fm[2]) / 2),
            "fallback_count": int(missing.sum())}


def select_repr_xgb(contexts, blocks):
    y = np.concatenate([c["val_y"] for c in contexts])
    tasks = np.concatenate([c["val_tasks"] for c in contexts])
    bases = {p: oof_base(contexts, "xgb", p) for p in PARAMS["xgb"]}
    candidates, scores = [], {}
    for rep in REPRS:
        started = time.monotonic()
        for p in PARAMS["xgb"]:
            full, missing = oof_variant(contexts, "xgb", p, rep, blocks)
            result = evaluate_oof(y, tasks, bases[p], full, missing)
            item = {"representation": list(rep), "parameter": p, **result}
            candidates.append(item)
            scores[(rep, p)] = (full, missing)
        print("candidate", "/".join(map(str, rep)), f"{time.monotonic()-started:.1f}s",
              flush=True)
    # Earlier candidate priority determines exact ties after rounded equality.
    best = max(candidates, key=lambda r: (round(r["objective"], 12),
                                          repr_rank(tuple(r["representation"])),
                                          -float(r["parameter"])))
    rep, param = tuple(best["representation"]), int(best["parameter"])
    return rep, param, candidates, bases[param], scores[(rep, param)]


def select_param(contexts, kind, rep, blocks):
    y = np.concatenate([c["val_y"] for c in contexts])
    tasks = np.concatenate([c["val_tasks"] for c in contexts])
    candidates, cache = [], {}
    for p in PARAMS[kind]:
        base = oof_base(contexts, kind, p)
        full, missing = oof_variant(contexts, kind, p, rep, blocks)
        result = evaluate_oof(y, tasks, base, full, missing)
        candidates.append({"parameter": p, **result})
        cache[p] = (base, full, missing)
    selected = max(candidates, key=lambda r: (round(r["objective"], 12), -float(r["parameter"])))
    p = selected["parameter"]
    return p, candidates, cache[p]


def output_predictions(path, records, dataset, fold, kind, variant, rep, param, score, t,
                       base_score, base_t, fallback):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ("dataset", "fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
            "query_task", "candidate_task", "query_mode", "candidate_mode", "pair_type",
            "same_task", "candidate_position", "learner", "variant", "representation",
            "parameter", "score", "threshold", "predicted_link", "fallback")
    with gzip.open(path, "wt", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, pair in enumerate(records):
            use_base = bool(fallback[i])
            actual_score = float(base_score[i] if use_base else score[i])
            actual_t = float(base_t if use_base else t)
            item = {k: pair[k] for k in cols if k in pair}
            item.update(dataset=dataset, fold=fold, learner=kind, variant=variant,
                        representation="/".join(map(str, rep)), parameter=param,
                        score=f"{actual_score:.12g}", threshold=f"{actual_t:.12g}",
                        predicted_link=int(actual_score >= actual_t), fallback=int(use_base))
            w.writerow(item)


def run_fold(dataset, rows, tasks, folds, profiles, all_pairs, fold):
    started = time.monotonic()
    test_tasks = {t for t in tasks if folds[t] == fold}
    train_tasks = set(tasks) - test_tasks
    assert not test_tasks & train_tasks
    train_idx = [i for i, r in enumerate(rows) if r["task"] in train_tasks]
    test_pairs = [p for p in all_pairs if p["fold"] == fold]
    assert test_pairs and all(p["query_task"] in test_tasks and p["candidate_task"] in test_tasks
                              for p in test_pairs)
    print(dataset, "fold", fold, "train_tasks", len(train_tasks), "test_tasks",
          len(test_tasks), "test_pairs", len(test_pairs), flush=True)
    ctx = inner_contexts(rows, train_tasks, profiles, fold, dataset)
    y = np.concatenate([c["val_y"] for c in ctx])
    vt = np.concatenate([c["val_tasks"] for c in ctx])
    blocks = ("B", "S") if dataset == "cth" else ("B", "S", "T")
    rep, xgb_param, repr_candidates, xbase, xfull = select_repr_xgb(ctx, blocks)
    print(dataset, fold, "selected representation", rep, "XGB", xgb_param, flush=True)

    selections = {"xgb": {"parameter": xgb_param, "base": xbase, "full": xfull[0],
                          "missing": xfull[1], "candidates": repr_candidates}}
    if dataset == "cth":
        for kind in ("svm", "tree"):
            param, candidates, cache = select_param(ctx, kind, rep, blocks)
            selections[kind] = {"parameter": param, "base": cache[0], "full": cache[1],
                                "missing": cache[2], "candidates": candidates}
            print(dataset, fold, "selected", kind, param, flush=True)

    train = train_pairs(rows, train_tasks, SEED + 900000 + fold)
    ta, tb, ty = train_arrays(train)
    va, vb, vy, _ = as_arrays(test_pairs)
    f = Features(rows, train_idx, profiles, rep[0], rep[1])
    etrain, etest = f.evidence(ta, tb), f.evidence(va, vb)
    audit = {"dataset": dataset, "fold": fold, "train_tasks": sorted(train_tasks),
             "test_tasks": sorted(test_tasks), "inner_folds": [
                 {k: c[k] for k in ("inner", "fit_tasks", "validation_tasks",
                                     "train_pairs", "validation_pairs")} for c in ctx],
             "representation": rep, "train_pairs": len(train), "test_pairs": len(test_pairs),
             "selection": {}}
    for kind, sel in selections.items():
        p = sel["parameter"]
        head_oof = oof_head_only(ctx, kind, p)
        head_t = threshold(y, head_oof)
        head_train = f.evidence_head_only(ta, tb)
        head_test = f.evidence_head_only(va, vb)
        head_scores = fit_scores(kind, p, head_train, ty, head_test)
        output_predictions(OUT / dataset / f"fold-{fold}-{kind}-head_only.csv.gz",
                           test_pairs, dataset, fold, kind, "head_only", rep, p,
                           head_scores, head_t, head_scores, head_t,
                           np.zeros(len(test_pairs), dtype=bool))
        base_t = threshold(y, sel["base"])
        base_oof_pred = decision(sel["base"], base_t)
        base_test = fit_scores(kind, p, etrain, ty, etest)
        base_test_pred = decision(base_test, base_t)
        output_predictions(OUT / dataset / f"fold-{fold}-{kind}-evidence.csv.gz",
                           test_pairs, dataset, fold, kind, "evidence", rep, p, base_test,
                           base_t, base_test, base_t, np.zeros(len(test_pairs), dtype=bool))
        wanted = {"BS": ("B", "S")} if dataset == "cth" else {"BST": ("B", "S", "T")}
        if kind == "xgb":
            wanted = ({"BS": ("B", "S"), "B": ("B",), "S": ("S",)}
                      if dataset == "cth" else
                      {"BST": ("B", "S", "T"), "ST": ("S", "T"),
                       "BT": ("B", "T"), "BS": ("B", "S")})
        variant_audit = {"head_only": {"threshold": head_t},
                         "evidence": {"threshold": base_t}}
        for variant, block_names in wanted.items():
            if tuple(block_names) == blocks:
                ov, fallback_oof = sel["full"], sel["missing"]
            else:
                ov, fallback_oof = oof_variant(ctx, kind, p, rep, block_names)
            vt_ = threshold(y, ov, fallback_oof, base_oof_pred)
            xtrain, _ = f.full(etrain, ta, tb, block_names, rep[2])
            xtest, missing = f.full(etest, va, vb, block_names, rep[2])
            values = fit_scores(kind, p, xtrain, ty, xtest)
            output_predictions(OUT / dataset / f"fold-{fold}-{kind}-{variant}.csv.gz",
                               test_pairs, dataset, fold, kind, variant, rep, p, values, vt_,
                               base_test, base_t, missing)
            iv = macro(y, decision(ov, vt_, fallback_oof, base_oof_pred), vt)
            variant_audit[variant] = {"threshold": vt_, "fallback_pairs": int(missing.sum()),
                                      "fallback_fraction": float(missing.mean()),
                                      "inner_oof_macro": iv.tolist()}
            print(dataset, "fold", fold, kind, variant, "fallback",
                  int(missing.sum()), "/", len(missing), flush=True)
        audit["selection"][kind] = {"parameter": p, "candidates": sel["candidates"],
                                    "variants": variant_audit}
    audit["elapsed_seconds"] = round(time.monotonic() - started, 2)
    write_json(OUT / dataset / f"fold-{fold}-audit.json", finite_json(audit))
    print(dataset, "fold", fold, "complete", audit["elapsed_seconds"], "seconds", flush=True)


def run(dataset, only_fold=None):
    rows, tasks, folds = build_cth() if dataset == "cth" else build_ait()
    profiles = profile_cache(rows, minimum=2 if dataset == "cth" else 4)
    if dataset == "cth":
        pairs, roles = cth_pairs(rows, tasks, folds)
        write_gz_csv(OUT / dataset / "role_allocations.csv.gz", roles)
    else:
        pairs = []
        for fold in range(3):
            foldtasks = {t for t in tasks if folds[t] == fold}
            pairs.extend(gallery(rows, foldtasks, fold, 100, SEED + 100000 * fold))
    write_gz_csv(OUT / dataset / "pairs.csv.gz", pairs)
    counts = collections.Counter((r["task"], r["mode"]) for r in rows)
    audit = {"dataset": dataset, "trace_count": len(rows), "task_count": len(tasks),
             "task_mode_counts": [{"task": t, "human": counts[t, "human"],
                                   "agent": counts[t, "agent"]} for t in tasks],
             "folds": folds, "pair_count": len(pairs),
             "profile_unavailable_by_window": {
                 str(w): [rows[i]["id"] for i, p in enumerate(profiles[w])
                          if not (p["B_ok"] and p["S_ok"])] for w in WINDOWS},
             "time_block_unavailable_by_window": {
                 str(w): sum(not p["T_ok"] for p in profiles[w]) for w in WINDOWS},
             "run_ids_unique": len({r["id"] for r in rows}) == len(rows),
             "raw_hashes_unique": (len({r["raw_hash"] for r in rows}) == len(rows)
                                   if dataset == "ait" else None),
             "raw_record_locators": (len({r["locator"] for r in rows})
                                     if dataset == "cth" else None),
             "argument_profile_traces": (sum(bool(r.get("argument_profile")) for r in rows)
                                         if dataset == "cth" else 0),
             "argument_profile_feature_count": (sum(len(r.get("argument_profile", [])) for r in rows)
                                                 if dataset == "cth" else 0)}
    write_json(OUT / dataset / "input_audit.json", finite_json(audit))
    print(dataset, "traces", len(rows), "tasks", len(tasks), "pairs", len(pairs), flush=True)
    for fold in sorted(set(folds.values())):
        if only_fold is None or fold == only_fold:
            run_fold(dataset, rows, tasks, folds, profiles, pairs, fold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cth", "ait"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "environment.json", {
        "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__, "seed": SEED, "source_sha256": sha(pathlib.Path(__file__)),
        "cth_registry_sha256": sha(ROOT / "research/results/behavior_corpus_v1/trace_registry.csv"),
        "cth_evidence_sha256": sha(ROOT / "research/results/behavior_corpus_v1/evidence.jsonl"),
        "folds_sha256": sha(FOLDS_CSV),
        "ait_archive_sha256": sha(ROOT / "research/data/external/ait-shell-pilot-v1/data.zip"),
        "fingerprint_sha256": sha(ROOT / "attackagent/fingerprint_v2.py"),
        "argument_profile_sha256": sha(EXP / "argument_profile.py"),
        "python_seed": SEED, "baseline_commit": "553664fb5760858a215c12c9cd95a5add302b898"})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN.*")
        run(args.dataset, args.fold)


if __name__ == "__main__":
    main()
