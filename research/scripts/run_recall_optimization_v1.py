#!/usr/bin/env python3
"""Nested task-held-out recall optimization from audited association errors."""
from __future__ import annotations

import collections
import csv
import datetime as dt
import difflib
import hashlib
import json
import math
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_association_error_analysis as old
import run_association_operating_points as op
import run_mixed_gallery as gallery
from run_feature_ablation import BASE_SEED, grouped_folds
from attackagent.profiles import ACTION_LABELS, FEATURE_NAMES, extract_behavior_features

OUT = ROOT / "results" / "recall_optimization_v1"
METHODS = ("behavior", "behavior_align", "behavior_multiscale",
           "behavior_align_hardneg", "behavior_align_profile")
TARGET = 0.20
TARGETS = (0.10, 0.20, 0.30)
BOOTSTRAPS = 5000
GRID = 101


def lcs_features(a, b):
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    short = min(len(a), len(b))
    return [matcher.ratio(), matched / short if short else 0.0]


def phase_action(rows, bins):
    matrix = []
    labels = ACTION_LABELS
    index = {label: i for i, label in enumerate(labels)}
    for row in rows:
        values = np.zeros((bins, len(labels)), dtype=float)
        for phase in range(bins):
            start = len(row.groups) * phase // bins
            end = len(row.groups) * (phase + 1) // bins
            for group in row.groups[start:end]:
                values[phase, index[group]] += 1
            total = values[phase].sum()
            if total:
                values[phase] /= total
        matrix.append(values)
    return np.asarray(matrix)


def pair_vector(left, right, base, rows, index, idf, action_bins, profiles):
    common = gallery.pair_features(left, right, base, np.zeros(len(rows)), with_mode=False)
    a, b = rows[left], rows[right]
    command_align = lcs_features(a.names, b.names)
    action_align = lcs_features(a.groups, b.groups)
    ca, cb = collections.Counter(a.names), collections.Counter(b.names)
    overlap = sum(idf.get(token, 1.0) * min(count, cb[token]) for token, count in ca.items())
    mass_a = sum(idf.get(token, 1.0) * count for token, count in ca.items())
    mass_b = sum(idf.get(token, 1.0) * count for token, count in cb.items())
    weighted_cover = overlap / min(mass_a, mass_b) if min(mass_a, mass_b) else 0.0
    weighted_dice = 2 * overlap / (mass_a + mass_b) if mass_a + mass_b else 0.0
    align = [*command_align, *action_align, weighted_cover, weighted_dice]
    multiscale = []
    for bins, matrices in action_bins.items():
        u, v = matrices[left].reshape(-1), matrices[right].reshape(-1)
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        cosine = float(np.dot(u, v) / denom) if denom else 0.0
        multiscale.extend([cosine, float(np.exp(-np.mean(np.abs(u - v))))])
    return np.asarray(common), np.asarray(align), np.asarray(multiscale), np.abs(profiles[left] - profiles[right])


def sample_hard_pairs(rows, pair_arrays, seed):
    by_task = collections.defaultdict(list)
    by_bench = collections.defaultdict(list)
    for i, row in enumerate(rows):
        by_task[(row.benchmark, row.task)].append(i)
    for key in by_task:
        by_bench[key[0]].append(key)
    rng = np.random.default_rng(seed)
    pairs, labels, seen = [], [], set()
    for key, members in sorted(by_task.items()):
        positives = list(__import__("itertools").combinations(members, 2))
        rng.shuffle(positives)
        positives = positives[:32]
        if not positives:
            continue
        negatives = []
        for other in by_bench[key[0]]:
            if other == key:
                continue
            negatives.extend((a, b) for a in members for b in by_task[other])
        if not negatives:
            continue
        def hardness(pair):
            x, alignment, _multi, _profile = pair_arrays[(min(pair), max(pair))]
            return float(np.mean([x[0], x[1], x[2], alignment[0], alignment[4]]))
        negatives.sort(key=hardness, reverse=True)
        selected = negatives[:len(positives)]
        if len(selected) < len(positives):
            remainder = negatives[len(selected):]
            rng.shuffle(remainder)
            selected.extend(remainder[:len(positives) - len(selected)])
        for left, right in [*positives, *selected]:
            pair = (min(left, right), max(left, right))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            labels.append(int((rows[left].benchmark, rows[left].task) ==
                              (rows[right].benchmark, rows[right].task)))
    return pairs, np.asarray(labels, dtype=int)


def fit_scorer(train_rows, score_rows, seed, hardneg=False):
    rows = [*train_rows, *score_rows]
    n_train = len(train_rows)
    base = {
        "rows": rows,
        "command": gallery.cosine_matrix(train_rows, rows, gallery.text_commands, (1, 2)),
        "action": gallery.cosine_matrix(train_rows, rows, gallery.action_bigram_text, (1, 1)),
    }
    base["procgrep"], _ = gallery.procgrep_matrix(train_rows, rows)
    counts = collections.Counter(token for row in train_rows for token in set(row.names))
    idf = {token: math.log((n_train + 1) / (count + 1)) + 1 for token, count in counts.items()}
    action_bins = {bins: phase_action(rows, bins) for bins in (2, 4, 8)}
    raw = np.asarray([[np.nan if vals[name] is None else vals[name]
                       for name in FEATURE_NAMES]
                      for vals in (extract_behavior_features(r.names, r.groups) for r in rows)], dtype=float)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_prof = scaler.fit_transform(imputer.fit_transform(raw[:n_train]))
    all_prof = scaler.transform(imputer.transform(raw))
    index = {row.trace_id: i for i, row in enumerate(rows)}

    pair_arrays = {}
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            pair_arrays[(left, right)] = pair_vector(left, right, base, rows, index, idf,
                                                       action_bins, all_prof)
    train_pairs, y = (sample_hard_pairs(train_rows,
                       {(a,b): pair_arrays[(a,b)] for a,b in pair_arrays if b < n_train}, seed)
                      if hardneg else gallery.training_pairs(train_rows, seed))
    designs = {name: [] for name in METHODS if name != "behavior_multiscale"}
    designs["behavior_multiscale"] = []
    for left, right in train_pairs:
        common, align, multi, profile = pair_arrays[(min(left, right), max(left, right))]
        designs["behavior"].append(common)
        designs["behavior_align"].append(np.r_[common, align])
        designs["behavior_multiscale"].append(np.r_[common, multi])
        designs["behavior_align_hardneg"].append(np.r_[common, align])
        designs["behavior_align_profile"].append(np.r_[common, align, profile])
    models = {}
    for name, values in designs.items():
        if name == "behavior_align_hardneg" and not hardneg:
            continue
        if name != "behavior_align_hardneg" and hardneg:
            continue
        model = LogisticRegression(C=0.05 if name == "behavior_align_profile" else 0.5,
                                   class_weight="balanced", max_iter=3000, random_state=seed)
        model.fit(np.asarray(values), y)
        models[name] = model

    def score_pairs(row_pairs):
        collect = {name: [] for name in models}
        command_scores = []
        for left_row, right_row in row_pairs:
            left, right = index[left_row.trace_id], index[right_row.trace_id]
            common, align, multi, profile = pair_arrays[(min(left, right), max(left, right))]
            full = {"behavior": common, "behavior_align": np.r_[common, align],
                    "behavior_multiscale": np.r_[common, multi],
                    "behavior_align_hardneg": np.r_[common, align],
                    "behavior_align_profile": np.r_[common, align, profile]}
            for name in models:
                collect[name].append(full[name])
            command_scores.append(base["command"][left, right])
        out = {name: model.predict_proba(np.asarray(collect[name]))[:, 1]
               for name, model in models.items()}
        out["command_cosine"] = np.asarray(command_scores)
        return out
    return score_pairs, len(train_pairs)


def run():
    rows = gallery.load_rows()
    if len(rows) != 177 or len({(r.benchmark, r.task) for r in rows}) != 40:
        raise SystemExit("Pinned cohort changed; re-audit before evaluation")
    OUT.mkdir(parents=True, exist_ok=True)
    all_pairs, threshold_rows, rate_rows, rank_rows, fold_rows = [], [], [], [], []
    folds = grouped_folds(rows, BASE_SEED, n_folds=3)
    for fold, (train_keys, test_keys) in enumerate(folds, 1):
        train = [r for r in rows if (r.benchmark, r.task) in train_keys]
        test = [r for r in rows if (r.benchmark, r.task) in test_keys]
        calibration_by_method = {m: [] for m in METHODS}
        for inner, (fit_keys, valid_keys) in enumerate(grouped_folds(train, 20261730 + fold, 3), 1):
            fit = [r for r in train if (r.benchmark, r.task) in fit_keys]
            valid = [r for r in train if (r.benchmark, r.task) in valid_keys]
            scorer, _ = fit_scorer(fit, valid, BASE_SEED + fold * 100 + inner, hardneg=False)
            for row in op.make_pairs(valid, scorer, fold, "threshold_oof"):
                for method in ("behavior", "behavior_align", "behavior_multiscale", "behavior_align_profile"):
                    if method in row: calibration_by_method[method].append(row)
            scorer, _ = fit_scorer(fit, valid, BASE_SEED + fold * 100 + inner, hardneg=True)
            calibration_by_method["behavior_align_hardneg"].extend(
                op.make_pairs(valid, scorer, fold, "threshold_oof"))
        thresholds = {method: {} for method in METHODS}
        selected_by_target = {}
        for method in METHODS:
            for target in TARGETS:
                thresholds[method][target] = op.choose_threshold(
                    calibration_by_method[method], method, target, GRID)
                threshold_rows.append({"outer_fold": fold, "method": method,
                                       "target": target, **thresholds[method][target]})
        for target in TARGETS:
            selected_by_target[target] = min(
                METHODS,
                key=lambda method: (thresholds[method][target]["training_macro_false_split_rate"],
                                    method != "behavior"),
            )
        current_outer = {}
        fold_pair_counts = {}
        for hardneg, expected in ((False, ("behavior", "behavior_align", "behavior_multiscale", "behavior_align_profile")),
                                  (True, ("behavior_align_hardneg",))):
            scorer, count = fit_scorer(train, test, BASE_SEED + fold, hardneg)
            fold_pair_counts["hard_negative" if hardneg else "random"] = count
            current = op.make_pairs(test, scorer, fold, "outer_test")
            for row in current:
                for method in expected:
                    for target in TARGETS:
                        th = thresholds[method][target]["threshold"]
                        tag = int(target * 100)
                        row[f"threshold_{method}_{tag}"] = th
                        row[f"accepted_{method}_{tag}"] = int(row[method] >= th)
            for row in current:
                key = (row["query_trace_id"], row["candidate_trace_id"])
                current_outer.setdefault(key, {}).update(row)
        outer_rows = list(current_outer.values())
        for target in TARGETS:
            selected = selected_by_target[target]
            tag = int(target * 100)
            for row in outer_rows:
                row[f"selected_method_{tag}"] = selected
                row[f"selected_score_{tag}"] = row[selected]
                row[f"selected_threshold_{tag}"] = thresholds[selected][target]["threshold"]
                row[f"selected_accepted_{tag}"] = int(
                    row[selected] >= thresholds[selected][target]["threshold"])
            selected_rows = [{**row, "inner_selected": row[f"selected_score_{tag}"]} for row in outer_rows]
            for method in METHODS:
                task_rates = old._query_rates(outer_rows, method, thresholds[method][target]["threshold"])
                rate_rows.extend({"outer_fold": fold, "target": target, "method": method, **r}
                                 for r in task_rates)
            task_rates = old._query_rates(selected_rows, "inner_selected", thresholds[selected][target]["threshold"])
            rate_rows.extend({"outer_fold": fold, "target": target, "method": "inner_selected", **r}
                             for r in task_rates)
        all_pairs.extend(outer_rows)
        ranking = op.retrieval_task_metrics(test, outer_rows, METHODS)
        rank_rows.extend({"outer_fold": fold, **r} for r in ranking)
        fold_rows.append({"outer_fold": fold, "train_task_groups": len(train_keys),
                          "test_task_groups": len(test_keys), "train_records": len(train),
                          "test_records": len(test), "pair_training_count_by_sampler": fold_pair_counts,
                          "inner_selected_method_by_target": {
                              str(int(target * 100)): selected_by_target[target] for target in TARGETS}})
        selected20 = selected_by_target[TARGET]
        selected_rows = [{**row, "inner_selected": row[f"selected_score_{int(TARGET * 100)}"]}
                         for row in outer_rows]
        ranking = op.retrieval_task_metrics(test, selected_rows, ("inner_selected",))
        rank_rows.extend({"outer_fold": fold, **r} for r in ranking)
        print(f"outer fold {fold}/3 complete", flush=True)

    metrics = ("false_split_rate", "false_merge_rate", "false_split_human_agent",
               "false_merge_human_agent", "balanced_accuracy")
    rate_summary = {}
    deltas = {}
    for target in TARGETS:
        subset = [r for r in rate_rows if r["target"] == target]
        rate_summary[str(int(target * 100))] = op.summary_with_ci(
            subset, (*METHODS, "inner_selected"), metrics, BOOTSTRAPS,
            BASE_SEED + int(target * 1000))
        deltas[str(int(target * 100))] = {
            method: {metric: op.paired_delta(subset, metric, method, "behavior", BOOTSTRAPS,
                         BASE_SEED + 1000 + mi * 20 + index)
                     for mi, metric in enumerate(("false_split_rate", "false_merge_rate",
                                                   "false_split_human_agent", "false_merge_human_agent",
                                                   "balanced_accuracy"))}
            for index, method in enumerate((*METHODS, "inner_selected")) if method != "behavior"}
    rank_summary = op.summary_with_ci(rank_rows, (*METHODS, "inner_selected"), op.RANK_METRICS,
                                      BOOTSTRAPS, BASE_SEED + 800)
    for name, records in (("pair_scores.csv", all_pairs), ("thresholds.csv", threshold_rows),
                          ("task_metrics.csv", [*rate_rows, *rank_rows])):
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            fields = list(dict.fromkeys(k for r in records for k in r))
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(records)
    data_manifest = ROOT / "data/manifests/crossbenchmark_public_subset.json"
    result = {"status": "exploratory nested task-held-out evaluation",
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "protocol": "plans/14-漏关联优化实验计划.md", "dataset": {"records":len(rows),"task_groups":40,
              "revision":json.loads(data_manifest.read_text())["revision"],
              "label_scope":"same CTF task proxy; not attack campaign identity"},
              "methods": list(METHODS), "target_training_false_merge": TARGET,
              "outer_folds":fold_rows,"thresholds":threshold_rows,
              "false_merge_targets":rate_summary,"ranking":rank_summary,
              "paired_deltas_vs_behavior":deltas,
              "provenance":{"head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "runner_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
              "dataset_manifest_sha256":hashlib.sha256(data_manifest.read_bytes()).hexdigest()},
              "interpretation_guard":"Exploratory reuse of previously inspected CTF tasks; no claim of real attack attribution."}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    plot(result)
    print(json.dumps({"outputs":str(OUT.relative_to(ROOT)),
                      "fsr":{m:rate_summary["20"][m]["false_split_rate"] for m in (*METHODS,"inner_selected")},
                      "fmr":{m:rate_summary["20"][m]["false_merge_rate"] for m in (*METHODS,"inner_selected")},
                      "selected_by_fold":{str(row["outer_fold"]):row["inner_selected_method_by_target"]
                                          for row in fold_rows}},ensure_ascii=False))
    return result


def plot(result):
    rates = result["false_merge_targets"]["20"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
    methods = (*METHODS, "inner_selected")
    colors = [*colors, "#332288"]
    for color, method in zip(colors, methods):
        fmr = rates[method]["false_merge_rate"]; fsr = rates[method]["false_split_rate"]
        ax.errorbar(fmr["estimate"], fsr["estimate"],
                    xerr=[[fmr["estimate"]-fmr["ci_95"][0]],[fmr["ci_95"][1]-fmr["estimate"]]],
                    yerr=[[fsr["estimate"]-fsr["ci_95"][0]],[fsr["ci_95"][1]-fsr["estimate"]]],
                    marker="o", capsize=3, color=color, label=method)
        ax.annotate(method.replace("behavior_", ""), (fmr["estimate"], fsr["estimate"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="False merge rate", ylabel="False split rate",
           title="Recall–false-link trade-off · 20% training FMR target")
    ax.set_xlim(0, 0.45); ax.set_ylim(0, 1); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "recall_optimization_v1.png", dpi=180); fig.savefig(OUT / "recall_optimization_v1.pdf")
    plt.close(fig)


if __name__ == "__main__":
    run()
