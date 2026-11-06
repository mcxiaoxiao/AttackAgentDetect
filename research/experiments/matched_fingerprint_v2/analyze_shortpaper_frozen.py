#!/usr/bin/env python3
"""Frozen-score diagnostics for the same-challenge retrieval short paper."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "research/results"
EXP = ROOT / "research/experiments/matched_fingerprint_v2"
OUT = RESULTS / "matched_fingerprint_v2/shortpaper_frozen_20261105"
SEED = 20261105
N_BOOT = 10_000
SHARE_COUNTS = (0, 8, 16, 24, 32)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(obj)
    return obj


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def ci_summary(values: np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=float)
    ix = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    draws = values[ix].mean(axis=1)
    return {"mean": float(values.mean()),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "task_count": int(len(values))}


def gallery_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_cols = [c for c in ("fold", "replicate", "fraction", "query_id", "query_task")
                  if c in frame.columns]
    for key, group in frame.groupby(group_cols, sort=False, dropna=False):
        y = group["same_task"].to_numpy(dtype=bool)
        scores = group["score"].to_numpy(dtype=float)
        threshold = group["threshold"].to_numpy(dtype=float)
        pred = group["predicted_link"].to_numpy(dtype=bool)
        pos = np.flatnonzero(y)
        neg = np.flatnonzero(~y)
        if len(pos) != 1 or len(neg) == 0:
            continue
        ps = scores[pos[0]]
        ns = scores[neg]
        greater = int((ns > ps).sum())
        ties = int((ns == ps).sum())
        records.append({"fold": group["fold"].iloc[0] if "fold" in group else -1,
                        "replicate": group["replicate"].iloc[0] if "replicate" in group else -1,
                        "fraction": group["fraction"].iloc[0] if "fraction" in group else -1,
                        "query_id": group["query_id"].iloc[0],
                        "query_task": group["query_task"].iloc[0],
                        "fmr": float(pred[neg].mean()),
                        "fsr": float(not pred[pos[0]]),
                        "ba": float(1 - (pred[neg].mean() + float(not pred[pos[0]])) / 2),
                        "hit1": float(1 / (1 + greater + ties)),
                        "gallery_auc": float(((ns < ps).sum() + .5 * ties) / len(neg)),
                        "pair_count": int(len(group))})
    return pd.DataFrame.from_records(records)


def task_values(galleries: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return galleries.groupby(group_cols + ["query_task"], as_index=False)[
        ["fmr", "fsr", "ba", "hit1", "gallery_auc"]].mean()


def metrics_table(galleries: pd.DataFrame, arm: str, source: str,
                  task_col: str = "query_task") -> list[dict]:
    rows = []
    rng = np.random.default_rng(SEED)
    for share, values in galleries.groupby("fraction", sort=True):
        task_means = values.groupby(task_col)[["fmr", "fsr", "ba", "hit1", "gallery_auc"]].mean()
        for metric in task_means.columns:
            out = ci_summary(task_means[metric].to_numpy(), rng)
            rows.append({"source": source, "arm": arm, "agent_share": float(share),
                         "metric": metric, **out})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nyu-source", type=Path,
                        default=Path("/tmp/nyu-ctf-submissions-20261103"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    match = load(EXP.parent / "matched_fingerprint_v1/run.py", "matched_v1_shortpaper")
    legacy = load(EXP.parent / "overnight_fingerprint_v1/run.py", "legacy_shortpaper")
    rows, tasks, folds = match.build_cth()
    pairs, _ = match.cth_pairs(rows, tasks, folds)
    pair_df = pd.DataFrame(pairs)
    pair_df["fraction"] = pair_df["fraction"].astype(float)
    index = np.random.default_rng(SEED).integers(0, len(tasks), size=(N_BOOT, len(tasks)))

    exact_path = RESULTS / "association_fingerprint_v3/exact_mixture.json"
    exact = json.loads(exact_path.read_text())
    scores_by_method = {}
    source_paths = [Path(__file__), exact_path, EXP.parent / "matched_fingerprint_v1/run.py",
                    EXP.parent / "overnight_fingerprint_v1/run.py",
                    RESULTS / "overnight_fingerprint_v1/task_folds.csv"]
    for fold in sorted(set(folds.values())):
        selected = [p for p in pairs if p["fold"] == fold]
        indices = sorted({p[k] for p in selected for k in ("query_idx", "candidate_idx")})
        train = [i for i, row in enumerate(rows) if folds[row["task"]] != fold]
        for method in ("command_cosine", "action_bigram_cosine"):
            matrix, mapping = legacy.cos_matrix(rows, train, indices, method)
            a = np.asarray([mapping[p["query_idx"]] for p in selected], dtype=int)
            b = np.asarray([mapping[p["candidate_idx"]] for p in selected], dtype=int)
            scores = np.asarray(matrix[a].multiply(matrix[b]).sum(axis=1)).ravel()
            threshold = float(exact["thresholds"][str((fold, method))])
            scores_by_method.setdefault(method, []).extend(
                {"pair_id": p["pair_id"], "score": float(s), "threshold": threshold,
                 "predicted_link": int(s >= threshold)} for p, s in zip(selected, scores))
    rows_by_id = {r["id"]: r for r in rows}
    ranking_records, metric_rows = [], []
    cosine_frames = {}
    for method, scores in scores_by_method.items():
        scored = pair_df.merge(pd.DataFrame(scores), on="pair_id", validate="one_to_one")
        scored["same_task"] = scored["same_task"].astype(int)
        cosine_frames[method] = scored
        galleries = gallery_metrics(scored)
        assert len(galleries) == 170 * 16 * 5
        metric_rows.extend(metrics_table(galleries, method, "CTH exact mixture"))
        for share, group in galleries.groupby("fraction", sort=True):
            task_mean = group.groupby("query_task")[
                ["fmr", "fsr", "ba", "hit1", "gallery_auc"]].mean()
            draw = task_mean.to_numpy()[index].mean(axis=1)
            for col_idx, metric in enumerate(task_mean.columns):
                ranking_records.append({"source": "CTH exact mixture", "arm": method,
                                        "agent_share": float(share), "metric": metric,
                                        "mean": float(task_mean[metric].mean()),
                                        "ci95": [float(np.quantile(draw[:, col_idx], .025)),
                                                 float(np.quantile(draw[:, col_idx], .975))]})
        # Check the re-created threshold rates against the original task means.
        for count in SHARE_COUNTS:
            task_mean = galleries[galleries.fraction == count / 32].groupby("query_task")[
                ["fmr", "fsr", "ba"]].mean()
            expected = exact["task_means"][f"{count}:{method}"]
            for task in task_mean.index:
                got = task_mean.loc[task].to_numpy()
                want = np.asarray(expected[task], dtype=float)
                assert np.max(np.abs(got - want)) < 1e-12, (method, count, task, got, want)
        scored.to_csv(out / f"{method}_pair_scores.csv.gz", index=False,
                      compression={"method": "gzip", "mtime": 0})

    pd.DataFrame(ranking_records).to_csv(out / "mixture_metrics.csv", index=False)

    # Frozen XGBoost results: no refit, ranking and no-match stress are score diagnostics.
    frozen_parts = []
    for fold in sorted(set(folds.values())):
        for variant in ("evidence", "BS"):
            p = RESULTS / f"matched_fingerprint_v1/cth/fold-{fold}-xgb-{variant}.csv.gz"
            source_paths.append(p)
            frame = pd.read_csv(p)
            frame["arm"] = "XGB E" if variant == "evidence" else "XGB E+B/S"
            frozen_parts.append(frame)
    frozen = pd.concat(frozen_parts, ignore_index=True)
    frozen["same_task"] = frozen["same_task"].astype(int)
    frozen["predicted_link"] = frozen["predicted_link"].astype(int)
    frozen["fallback"] = frozen["fallback"].astype(int)
    frozen["score"] = frozen["score"].astype(float)
    frozen["threshold"] = frozen["threshold"].astype(float)
    xgb_galleries = {}
    no_match_rows = []
    for arm, df in frozen.groupby("arm"):
        galleries = gallery_metrics(df)
        xgb_galleries[arm] = galleries
        metric_rows.extend(metrics_table(galleries, arm, "CTH matched XGBoost"))
        for share, group in df.groupby("fraction", sort=True):
            for key, gallery in group.groupby(["replicate", "query_id", "query_task"], sort=False):
                neg = gallery[gallery.same_task == 0]
                if len(neg) != 3 or int(gallery.same_task.sum()) != 1:
                    continue
                no_match_rows.append({"arm": arm, "agent_share": float(share),
                                      "query_task": key[2],
                                      "any_negative_accepted": int(neg.predicted_link.astype(int).any()),
                                      "negative_candidate_count": 3})
    no_match = pd.DataFrame(no_match_rows)
    no_match_task = no_match.groupby(["arm", "agent_share", "query_task"], as_index=False)[
        "any_negative_accepted"].mean()
    for (arm, share), group in no_match_task.groupby(["arm", "agent_share"]):
        summary = ci_summary(group.any_negative_accepted.to_numpy(), np.random.default_rng(SEED))
        metric_rows.append({"source": "CTH no-match stress", "arm": arm,
                            "agent_share": float(share), "metric": "any_negative_accepted",
                            **summary})

    # Retrospective operating-point reading, with whole tied-score blocks per held-out fold/share.
    operating = []
    for arm, df in frozen.groupby("arm"):
        fold_task_rows = []
        for (fold, share), part in df.groupby(["fold", "fraction"], sort=True):
            y = part.same_task.to_numpy(dtype=bool)
            s = part.score.to_numpy(dtype=float)
            candidates = np.r_[np.inf, np.sort(np.unique(s))[::-1]]
            threshold, best_recall, achieved_fmr = np.inf, -1.0, 0.0
            for t in candidates:
                pred = s >= t
                fmr = float(pred[~y].mean())
                recall = float(pred[y].mean())
                if fmr <= .100000000001 and recall > best_recall + 1e-12:
                    threshold, best_recall, achieved_fmr = float(t), recall, fmr
            pred = s >= threshold
            q = part.copy()
            q["predicted_link"] = pred.astype(int)
            tm = gallery_metrics(q)
            per_task = tm.groupby("query_task")[["fmr", "fsr", "ba"]].mean().reset_index()
            for _, r in per_task.iterrows():
                fold_task_rows.append({"fold": int(fold), "share": float(share),
                                       "query_task": r.query_task, "fmr": r.fmr,
                                       "fsr": r.fsr, "recall": 1-r.fsr,
                                       "threshold": threshold,
                                       "fold_pooled_fmr": achieved_fmr})
        tmp = pd.DataFrame(fold_task_rows)
        for share, part in tmp.groupby("share"):
            for metric in ("fmr", "fsr", "recall"):
                task_values_ = part.groupby("query_task")[metric].mean()
                stats = ci_summary(task_values_.to_numpy(), np.random.default_rng(SEED))
                operating.append({"arm": arm, "agent_share": share, "metric": metric,
                                  "retrospective_task_macro": stats,
                                  "mean_fold_pooled_fmr_at_selected_cutoff": float(
                                      part.groupby("fold").fold_pooled_fmr.first().mean()),
                                  "note": "test scores selected per outer fold/share; not independent threshold validation"})
    dump(out / "retrospective_fmr10.json", operating)
    no_match.to_csv(out / "no_match_gallery_rows.csv.gz", index=False,
                    compression={"method": "gzip", "mtime": 0})

    # Task removal sensitivity: drop the task as query and discard any complete gallery
    # containing it as a candidate, so remaining galleries retain the original 1:3 design.
    loo = []
    action = pd.DataFrame(scores_by_method["action_bigram_cosine"]).merge(
        pair_df, on="pair_id", validate="one_to_one")
    action["same_task"] = action["same_task"].astype(int)
    for omitted in tasks:
        chosen = action[(action.query_task != omitted) & (action.candidate_task != omitted)]
        gal = gallery_metrics(chosen)
        for share, part in gal.groupby("fraction"):
            task_means = part.groupby("query_task")[
                ["fmr", "fsr", "ba", "hit1", "gallery_auc"]].mean()
            for metric in task_means.columns:
                loo.append({"omitted_task": omitted, "agent_share": float(share),
                            "metric": metric, "estimate": float(task_means[metric].mean()),
                            "retained_query_tasks": int(len(task_means)),
                            "complete_galleries": int(len(part))})
    loo_df = pd.DataFrame(loo)
    loo_summary = []
    for metric in ("fmr", "fsr", "ba", "hit1", "gallery_auc"):
        sub = loo_df[loo_df.metric == metric]
        pivot = sub.pivot(index="omitted_task", columns="agent_share", values="estimate")
        delta = pivot[1.0] - pivot[0.0]
        loo_summary.append({"metric": metric, "endpoint_difference": {
            "mean": float(delta.mean()), "minimum": float(delta.min()),
            "maximum": float(delta.max()), "positive_omissions": int((delta > 0).sum()),
            "negative_omissions": int((delta < 0).sum()), "omissions": int(len(delta))},
            "per_task": {str(k): float(v) for k, v in delta.items()}})
    loo_df.to_csv(out / "leave_one_task_out.csv", index=False)
    dump(out / "leave_one_task_summary.json", loo_summary)

    # Sequence-level duplicates across held-out folds and the NYU external confirmation.
    seq_by_id = {r["id"]: tuple(r["commands"]) for r in rows}
    cth_fold_overlap = []
    for fold in sorted(set(folds.values())):
        train_ids = [r["id"] for r in rows if folds[r["task"]] != fold]
        test_ids = [r["id"] for r in rows if folds[r["task"]] == fold]
        train_seq = defaultdict(list)
        for trace_id in train_ids:
            train_seq[seq_by_id[trace_id]].append(trace_id)
        overlap = [trace_id for trace_id in test_ids if seq_by_id[trace_id] in train_seq]
        cth_fold_overlap.append({"fold": fold, "training_traces": len(train_ids),
                                 "test_traces": len(test_ids), "duplicate_test_traces": len(overlap),
                                 "test_trace_ids": overlap,
                                 "tasks": sorted({next(r["task"] for r in rows if r["id"] == x)
                                                  for x in overlap})})

    nyu_rows, nyu_manifest = [], None
    nyu_overlap_tasks = []
    if args.nyu_source.exists():
        nyt = load(EXP / "run_independent_nyu_agent.py", "nyu_runner_audit")
        nyu_rows, nyu_manifest = nyt.load_external(args.nyu_source.resolve())
        train_seq = defaultdict(list)
        for row in rows:
            train_seq[tuple(row["commands"])].append(row["id"])
        overlap = [r for r in nyu_rows if tuple(r["commands"]) in train_seq]
        nyu_overlap_tasks = sorted({r["task"] for r in overlap})
    nyu_task_slugs = set(nyu_manifest["eligible_tasks"][i]["task"]
                         for i in range(len(nyu_manifest["eligible_tasks"]))) if nyu_manifest else set()
    training_slugs = {r["task"].rsplit("/", 1)[-1] for r in rows}
    alias_overlap = sorted(training_slugs & nyu_task_slugs)
    if nyu_manifest:
        source_paths.extend([RESULTS / "matched_fingerprint_v2/independent_nyu_task_manifest.json",
                             EXP / "run_independent_nyu_agent.py"])
    audit = {"cth_training_tasks": len(tasks), "cth_training_traces": len(rows),
             "cth_test_fold_sequence_overlap": cth_fold_overlap,
             "nyu_exact_normalized_sequence_overlap_count": (sum(
                 tuple(r["commands"]) in {tuple(x["commands"]) for x in rows} for r in nyu_rows)
                 if nyu_rows else None),
             "nyu_exact_overlap_task_count": len(nyu_overlap_tasks),
             "nyu_exact_overlap_tasks": nyu_overlap_tasks,
             "nyu_eligible_task_slug_overlap_with_cth_training": alias_overlap,
             "nyu_manifest_preexcluded_task_slug_overlap_with_cth_training": sorted(
                 training_slugs & set(nyu_manifest.get("preexcluded_task_slugs", []))) if nyu_manifest else None,
             "nyu_manifest_preexcluded_count": nyu_manifest.get("preexcluded_task_count") if nyu_manifest else None,
             "interpretation": "Exact sequence and slug matches are audit flags, not proof of identical source tasks.",
             "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths if p.exists()}}
    dump(out / "sequence_and_alias_audit.json", audit)

    # Remove every complete gallery containing a test trace whose normalized sequence
    # also occurs in that outer fold's training tasks. This is a deliberately strict
    # train/test sequence-overlap sensitivity check.
    overlap_summaries = []
    overlap_task_values = {}
    overlap_counts = {}
    flagged_by_fold = {}
    for fold in sorted(set(folds.values())):
        train_sequences = {tuple(r["commands"]) for r in rows if folds[r["task"]] != fold}
        test_ids = {r["id"] for r in rows if folds[r["task"]] == fold}
        flagged_by_fold[fold] = {trace_id for trace_id in test_ids
                                 if seq_by_id[trace_id] in train_sequences}

    def without_flagged_galleries(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        key = ["fold", "replicate", "fraction", "query_id", "query_task"]
        flags = frame.apply(lambda r: r.query_id in flagged_by_fold[int(r.fold)] or
                            r.candidate_id in flagged_by_fold[int(r.fold)], axis=1)
        dropped = frame.loc[flags, key].drop_duplicates()
        marked = frame.merge(dropped.assign(_drop=1), on=key, how="left")
        clean = marked[marked._drop.isna()].drop(columns="_drop")
        overlap_counts[name] = {"all_candidate_pairs": int(len(frame)),
                                "removed_galleries": int(len(dropped)),
                                "removed_pair_rows": int(len(dropped) * 4),
                                "retained_complete_galleries": int(len(clean) // 4),
                                "retained_query_tasks": int(clean.query_task.nunique())}
        return clean

    overlap_sources = {**cosine_frames}
    overlap_sources.update({arm: df.copy() for arm, df in frozen.groupby("arm")})
    for arm, frame in overlap_sources.items():
        frame = frame.copy()
        frame["fold"] = frame.fold.astype(int)
        frame["same_task"] = frame.same_task.astype(int)
        full_galleries = gallery_metrics(frame)
        clean_frame = without_flagged_galleries(frame, arm)
        clean_galleries = gallery_metrics(clean_frame)
        overlap_counts[arm]["all_galleries"] = int(len(full_galleries))
        for subset, galleries in (("all", full_galleries),
                                  ("drop_train_sequence_overlap", clean_galleries)):
            for share, part in galleries.groupby("fraction", sort=True):
                task_mean = part.groupby("query_task")[
                    ["fmr", "fsr", "ba", "hit1", "gallery_auc"]].mean()
                overlap_task_values[(arm, subset, float(share))] = task_mean
                for metric in task_mean:
                    rng = np.random.default_rng(SEED + int(share * 1000))
                    stats = ci_summary(task_mean[metric].to_numpy(), rng)
                    overlap_summaries.append({"arm": arm, "subset": subset,
                                              "agent_share": float(share),
                                              "metric": metric, **stats,
                                              "complete_galleries": int(len(part))})
    overlap_endpoint = []
    for arm in overlap_sources:
        for subset in ("all", "drop_train_sequence_overlap"):
            a = overlap_task_values.get((arm, subset, 0.0))
            b = overlap_task_values.get((arm, subset, 1.0))
            if a is None or b is None:
                continue
            common = sorted(set(a.index) & set(b.index))
            for metric in a.columns:
                delta = (b.loc[common, metric] - a.loc[common, metric]).to_numpy(dtype=float)
                rng = np.random.default_rng(SEED + len(common))
                ix = rng.integers(0, len(delta), size=(N_BOOT, len(delta)))
                draws = delta[ix].mean(axis=1)
                overlap_endpoint.append({"arm": arm, "subset": subset, "metric": metric,
                                         "mean_endpoint_difference": float(delta.mean()),
                                         "ci95": [float(np.quantile(draws, .025)),
                                                  float(np.quantile(draws, .975))],
                                         "tasks_paired": len(common)})
    pd.DataFrame(overlap_summaries).to_csv(out / "cth_exact_sequence_sensitivity.csv", index=False)
    dump(out / "cth_exact_sequence_sensitivity.json", {
        "rule": "drop complete 4-candidate galleries if query or any candidate sequence occurs in that outer fold training tasks",
        "flagged_test_traces_by_fold": {str(k): sorted(v) for k, v in flagged_by_fold.items()},
        "counts": overlap_counts, "endpoint_differences": overlap_endpoint})

    # If exact normalized command sequences cross from CTH training into NYU,
    # remove each affected task as query and candidate and report frozen score sensitivity.
    if nyu_rows and nyu_overlap_tasks:
        pred_path = RESULTS / "matched_fingerprint_v2/independent_nyu_agent/predictions.csv.gz"
        source_paths.append(pred_path)
        pred = pd.read_csv(pred_path)
        pred["same_task"] = pred.same_task.astype(int)
        pred["predicted_link"] = pred.predicted_link.astype(int)
        pred["score"] = pred.score.astype(float)
        pred["threshold"] = pred.threshold.astype(float)
        sensitivity = []
        for learner, frame in pred.groupby("learner"):
            for variant, arm in frame.groupby("variant"):
                base = gallery_metrics(arm)
                remain = arm[(~arm.query_task.isin(nyu_overlap_tasks)) &
                             (~arm.candidate_task.isin(nyu_overlap_tasks))]
                after = gallery_metrics(remain)
                for name, data in (("all", base), ("drop_overlap_tasks", after)):
                    task_means = data.groupby("query_task")[
                        ["fmr", "fsr", "ba", "hit1", "gallery_auc"]].mean()
                    for metric in task_means:
                        sensitivity.append({"learner": learner, "variant": variant,
                                            "subset": name, "metric": metric,
                                            "mean": float(task_means[metric].mean()),
                                            "task_count": int(len(task_means)),
                                            "complete_galleries": int(len(data))})
        pd.DataFrame(sensitivity).to_csv(out / "nyu_duplicate_task_sensitivity.csv", index=False)

    metric_rows = [{**r, "percent_mean": 100*r["mean"],
                    "percent_ci95": [100*x for x in r["ci95"]]} for r in metric_rows]
    pd.DataFrame(metric_rows).to_csv(out / "frozen_metric_summaries.csv", index=False)
    dump(out / "run_manifest.json", {
        "protocol": "shortpaper_review_protocol.md", "seed": SEED, "bootstrap_draws": N_BOOT,
        "frozen_scores_only": True, "cth_pairs": len(pair_df), "cth_tasks": len(tasks),
        "environment": {"python": __import__("sys").version,
                        "numpy": np.__version__, "pandas": pd.__version__},
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths if p.exists()},
        "outputs": ["mixture_metrics.csv", "frozen_metric_summaries.csv",
                    "retrospective_fmr10.json", "no_match_gallery_rows.csv.gz",
                    "leave_one_task_summary.json", "sequence_and_alias_audit.json",
                    "cth_exact_sequence_sensitivity.json"]})
    make_figure(pd.DataFrame(ranking_records), out)
    print(json.dumps({"out": str(out), "mixture_rows": len(ranking_records),
                      "metrics": len(metric_rows), "cth_fold_exact_sequence_duplicates":
                      [x["duplicate_test_traces"] for x in cth_fold_overlap],
                      "nyu_cross_source_exact_sequence_duplicates": audit[
                          "nyu_exact_normalized_sequence_overlap_count"],
                      "figure": str(out / "mixture_ranking.pdf")}, indent=2))


def make_figure(metrics: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    shares = np.asarray([0, 25, 50, 75, 100], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    left = axes[0]
    for method, metric, label, color, marker in (
        ("action_bigram_cosine", "fmr", "False merge", "#c05746", "o"),
        ("action_bigram_cosine", "fsr", "Missed link", "#286b8e", "s")):
        vals = metrics[(metrics.arm == method) & (metrics.metric == metric)].sort_values("agent_share")
        y = vals["mean"].to_numpy() * 100
        ci = np.asarray(vals["ci95"].tolist(), dtype=float).T * 100
        left.errorbar(shares, y, yerr=np.vstack([y-ci[0], ci[1]-y]), label=label,
                      color=color, marker=marker, linewidth=1.7, capsize=2.5, markersize=4)
    left.set(title="(a) Fixed-threshold errors", xlabel="Agent-role share (%)",
             ylabel="Task-macro rate (%)", xticks=shares, ylim=(0, 105))
    left.grid(axis="y", alpha=.22)
    left.legend(frameon=False, loc="center right")

    right = axes[1]
    for method, metric, label, color, marker in (
        ("action_bigram_cosine", "hit1", "Action bigram · Hit@1", "#238b8e", "o"),
        ("action_bigram_cosine", "gallery_auc", "Action bigram · Gallery AUC", "#5964a6", "s")):
        vals = metrics[(metrics.arm == method) & (metrics.metric == metric)].sort_values("agent_share")
        y = vals["mean"].to_numpy() * 100
        ci = np.asarray(vals["ci95"].tolist(), dtype=float).T * 100
        right.errorbar(shares, y, yerr=np.vstack([y-ci[0], ci[1]-y]), label=label,
                       color=color, marker=marker, linewidth=1.7, capsize=2.5, markersize=4)
    right.set(title="(b) Threshold-free ranking", xlabel="Agent-role share (%)",
              ylabel="Task-macro score (%)", xticks=shares, ylim=(0, 105))
    right.grid(axis="y", alpha=.22)
    right.legend(frameon=False, loc="best", fontsize=8)
    fig.savefig(out / "mixture_ranking.pdf", bbox_inches="tight")
    fig.savefig(out / "mixture_ranking.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
