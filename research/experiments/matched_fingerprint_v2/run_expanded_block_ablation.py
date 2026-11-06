#!/usr/bin/env python3
"""Complete the predeclared CTH/AIT block ablation without model retuning."""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "research/results/matched_fingerprint_v2/expanded_ablation_20261106"
V1 = ROOT / "research/results/matched_fingerprint_v1"
NYU = ROOT / "research/results/matched_fingerprint_v2/independent_nyu_agent/predictions.csv.gz"
PROTOCOL = ROOT / "research/experiments/matched_fingerprint_v2/expanded_block_ablation_20261106_protocol.md"
N_BOOT = 10_000
BOOT_SEED = 20261106
PREDICTION_ID = ["fold", "pair_id", "replicate", "fraction", "query_id", "candidate_id",
                 "query_task", "candidate_task", "same_task", "candidate_position"]
GALLERY_ID = ["fold", "fraction", "replicate", "query_id", "query_task"]
METRICS = ["ba", "fmr", "fsr", "hit1", "gallery_auc", "pair_ap"]


def load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def safe(item):
        if isinstance(item, dict):
            return {str(k): safe(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(v) for v in item]
        if isinstance(item, np.generic):
            return safe(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return "Infinity" if item > 0 else "-Infinity" if item < 0 else "NaN"
        return item
    path.write_text(json.dumps(safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_records(path: pathlib.Path) -> list[dict]:
    with gzip.open(path, "rt", newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        out = list(reader)
    integer_fields = {"fold", "replicate", "query_idx", "candidate_idx", "same_task",
                      "candidate_position"}
    for row in out:
        for key in integer_fields:
            row[key] = int(row[key])
        row["fraction"] = float(row["fraction"])
    return out


def identity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame[PREDICTION_ID].sort_values(PREDICTION_ID, kind="stable")
            .reset_index(drop=True))


def verify_same_pairs(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    if len(left) != len(right):
        raise AssertionError(f"{label}: candidate decision counts differ")
    pd.testing.assert_frame_equal(identity_frame(left), identity_frame(right), check_dtype=False)


def frozen_prediction(dataset: str, fold: int, learner: str, variant: str) -> pd.DataFrame:
    return pd.read_csv(V1 / dataset / f"fold-{fold}-{learner}-{variant}.csv.gz",
                       compression="gzip")


def additions_for(dataset: str) -> dict[str, tuple[str, ...]]:
    if dataset == "cth":
        return {"svm": ("B", "S"), "tree": ("B", "S")}
    return {"xgb": ("B", "S", "T")}


def run_dataset(runner, dataset: str, sources: list[pathlib.Path]) -> tuple[list[pd.DataFrame], dict]:
    rows, tasks, folds = runner.build_cth() if dataset == "cth" else runner.build_ait()
    profile_path = ROOT / "attackagent/fingerprint_v2.py"
    pairs_path = V1 / dataset / "pairs.csv.gz"
    sources.extend([pairs_path, profile_path, runner.FOLDS_CSV])
    profiles = runner.profile_cache(rows, minimum=2 if dataset == "cth" else 4)
    pairs = read_records(pairs_path)
    expected_pairs = 170 * 5 * 16 * 4 if dataset == "cth" else 43_600
    if len(pairs) != expected_pairs:
        raise AssertionError(f"{dataset}: expected {expected_pairs} frozen pairs, found {len(pairs)}")
    pairs_df = pd.DataFrame(pairs)
    if set(pairs_df.fold.unique()) != {0, 1, 2}:
        raise AssertionError(f"{dataset}: outer fold set changed")
    written: list[pd.DataFrame] = []
    audit = {"dataset": dataset, "trace_count": len(rows), "task_count": len(tasks),
             "pair_decisions_per_arm": expected_pairs, "folds": {}, "added_arms": {}}

    for fold in sorted(set(folds.values())):
        outer_test = {task for task in tasks if folds[task] == fold}
        outer_train = set(tasks) - outer_test
        if outer_train & outer_test:
            raise AssertionError(f"{dataset} fold {fold}: task leakage")
        test_pairs = [row for row in pairs if row["fold"] == fold]
        test_df = pd.DataFrame(test_pairs)
        if set(test_df.query_task) != outer_test or set(test_df.candidate_task) != outer_test:
            raise AssertionError(f"{dataset} fold {fold}: evaluation tasks differ from frozen split")
        source_audit_path = V1 / dataset / f"fold-{fold}-audit.json"
        sources.append(source_audit_path)
        source_audit = json.loads(source_audit_path.read_text())
        rep = tuple(source_audit["representation"])
        train_idx = [i for i, row in enumerate(rows) if row["task"] in outer_train]
        contexts = runner.inner_contexts(rows, outer_train, profiles, fold, dataset)
        y = np.concatenate([chunk["val_y"] for chunk in contexts])
        val_tasks = np.concatenate([chunk["val_tasks"] for chunk in contexts])
        train = runner.train_pairs(rows, outer_train, runner.SEED + 900_000 + fold)
        ta, tb, ty = runner.train_arrays(train)
        va = np.asarray([row["query_idx"] for row in test_pairs], dtype=np.int32)
        vb = np.asarray([row["candidate_idx"] for row in test_pairs], dtype=np.int32)

        frozen_models = {}
        for frozen_learner, selection in source_audit["selection"].items():
            frozen_models[frozen_learner] = {
                "parameter": selection["parameter"],
                "variants": selection.get("variants", {}),
            }
        fold_audit = {"outer_train_tasks": sorted(outer_train), "outer_test_tasks": sorted(outer_test),
                      "representation": list(rep), "train_pair_count": len(train),
                      "test_pair_count": len(test_pairs), "frozen_model_settings": frozen_models,
                      "learners": {}}
        for learner, variants in additions_for(dataset).items():
            chosen = source_audit["selection"][learner]
            param = chosen["parameter"]
            base_oof = runner.oof_base(contexts, learner, param)
            base_threshold = runner.threshold(y, base_oof)
            base_oof_pred = runner.decision(base_oof, base_threshold)
            frozen_e = frozen_prediction(dataset, fold, learner, "evidence")
            verify_same_pairs(test_df, frozen_e, f"{dataset}/fold-{fold}/{learner}/E")
            base_train_features = runner.Features(rows, train_idx, profiles, rep[0], rep[1])
            e_train = base_train_features.evidence(ta, tb)
            e_test = base_train_features.evidence(va, vb)
            base_test_score = runner.fit_scores(learner, param, e_train, ty, e_test)
            frozen_e = frozen_e.set_index("pair_id").loc[[r["pair_id"] for r in test_pairs]]
            if not np.allclose(base_test_score, frozen_e.score.astype(float).to_numpy(), rtol=1e-9, atol=1e-10):
                raise AssertionError(f"{dataset} fold {fold} {learner}: re-created E scores differ")
            frozen_thresholds = frozen_e.threshold.astype(float).unique()
            if len(frozen_thresholds) != 1 or not np.isclose(base_threshold, frozen_thresholds[0], rtol=1e-9, atol=1e-10):
                raise AssertionError(f"{dataset} fold {fold} {learner}: E threshold differs")

            learner_audit = {"parameter": param, "baseline_threshold": float(base_threshold),
                             "base_oof_fmr": float(np.mean(base_oof_pred[y == 0])),
                             "added_arms": {}}
            for variant in variants:
                blocks = (variant,)
                oof_score, missing_oof = runner.oof_variant(contexts, learner, param, rep, blocks)
                arm_threshold = runner.threshold(y, oof_score, missing_oof, base_oof_pred)
                oof_pred = runner.decision(oof_score, arm_threshold, missing_oof, base_oof_pred)
                oof_fmr = float(np.mean(oof_pred[y == 0]))
                if oof_fmr > .1 + 1e-12:
                    raise AssertionError(f"{dataset} fold {fold} {learner}/{variant}: OOF FMR > 10%")

                arm_features = runner.Features(rows, train_idx, profiles, rep[0], rep[1])
                x_train, _ = arm_features.full(e_train, ta, tb, blocks, rep[2])
                x_test, missing_test = arm_features.full(e_test, va, vb, blocks, rep[2])
                raw_score = runner.fit_scores(learner, param, x_train, ty, x_test)
                out_path = OUT / "added_predictions" / f"{dataset}-fold-{fold}-{learner}-{variant}.csv.gz"
                runner.output_predictions(out_path, test_pairs, dataset, fold, learner, variant,
                                          rep, param, raw_score, arm_threshold, base_test_score,
                                          base_threshold, missing_test)
                saved = pd.read_csv(out_path, compression="gzip")
                verify_same_pairs(test_df, saved, f"{dataset}/fold-{fold}/{learner}/{variant}")
                # An arm's stored E fallback must equal the frozen E decision exactly.
                fallback = saved.fallback.astype(int).to_numpy().astype(bool)
                base_rows = frozen_e.reset_index()
                if fallback.any():
                    saved_by_id = saved.set_index("pair_id").loc[base_rows.pair_id]
                    base_pred = base_rows.predicted_link.astype(int).to_numpy()
                    if not np.array_equal(saved_by_id.loc[fallback, "predicted_link"].astype(int).to_numpy(),
                                          base_pred[fallback]):
                        raise AssertionError("fallback decisions differ from corresponding frozen E rows")
                macro_oof = runner.macro(y, oof_pred, val_tasks)
                arm_record = {"variant": variant, "blocks": list(blocks),
                              "threshold": float(arm_threshold),
                              "oof_fmr": oof_fmr,
                              "oof_macro_fmr_fsr_ba": [float(v) for v in macro_oof],
                              "oof_fallback_pairs": int(missing_oof.sum()),
                              "test_fallback_pairs": int(fallback.sum()),
                              "test_pair_count": len(saved),
                              "test_threshold_rejects_all": bool(saved.predicted_link.astype(int).sum() == 0),
                              "prediction_file": str(out_path.relative_to(ROOT)),
                              "prediction_sha256": sha(out_path)}
                learner_audit["added_arms"][variant] = arm_record
                audit["added_arms"][f"{learner}:{variant}"] = True
                written.append(saved)
            fold_audit["learners"][learner] = learner_audit
        audit["folds"][str(fold)] = fold_audit
    return written, audit


def canonical_arm(source: str, variant: str) -> str:
    mapping = {
        "cth": {"evidence": "E", "B": "E+B", "S": "E+S", "BS": "E+B+S"},
        "ait": {"evidence": "E", "B": "E+B", "S": "E+S", "T": "E+T",
                "BS": "E+B+S", "BT": "E+B+T", "ST": "E+S+T", "BST": "E+B+S+T1--T6"},
        "nyu": {"E": "E", "E_BS": "E+B+S", "E+B+S": "E+B+S"},
    }
    return mapping[source][variant]


def pairwise_to_ranking_scores(frame: pd.DataFrame, base: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    current = frame.copy()
    current["_source_index"] = current.index
    baseline = base[["fold", "pair_id", "score"]].rename(columns={"score": "e_score"})
    current = current.merge(baseline, on=["fold", "pair_id"], how="left", validate="one_to_one")
    if current.e_score.isna().any():
        raise AssertionError("missing matched E score for a fallback/ranking candidate")
    current["gallery_fallback"] = current.groupby(GALLERY_ID, sort=False).fallback.transform("max").astype(int)
    current["ranking_score"] = np.where(current.gallery_fallback.astype(bool),
                                        current.e_score.astype(float), current.score.astype(float))
    if (current.loc[current.gallery_fallback.astype(bool), "ranking_score"].to_numpy() !=
            current.loc[current.gallery_fallback.astype(bool), "e_score"].to_numpy()).any():
        raise AssertionError("gallery fallback did not restore all four E scores")
    detail = {"pairwise_fallback_pairs": int(current.fallback.astype(int).sum()),
              "gallery_count": int(current.groupby(GALLERY_ID, observed=True).ngroups),
              "gallery_fallback_count": int(current.drop_duplicates(GALLERY_ID).gallery_fallback.sum()),
              "gallery_level_rule": "if any of four pair rows has missing required block, replace all four ranking scores with matched E scores"}
    ordered = current.sort_values(GALLERY_ID + ["candidate_position"], kind="stable")
    labels = ordered.same_task.to_numpy(dtype=int).reshape(-1, 4)
    scores = ordered.ranking_score.to_numpy(dtype=float).reshape(-1, 4)
    if not np.all(labels.sum(axis=1) == 1):
        raise AssertionError("ranking requires one positive and three negatives")
    positive = scores[np.arange(len(scores)), labels.argmax(axis=1)]
    negatives = scores[labels == 0].reshape(-1, 3)
    detail["positive_tie_galleries"] = int(np.any(negatives == positive[:, None], axis=1).sum())
    detail["any_score_tie_galleries"] = int((np.diff(np.sort(scores, axis=1), axis=1) == 0).any(axis=1).sum())
    return current, detail


def fast_task_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Vectorized equivalent of the frozen gallery metrics for large arm grids."""
    frame = frame.copy()
    frame["fraction"] = frame.fraction.astype(str)
    ordered = frame.sort_values(GALLERY_ID + ["candidate_position"], kind="stable")
    groups = ordered[GALLERY_ID].drop_duplicates().reset_index(drop=True)
    if len(ordered) != len(groups) * 4:
        raise AssertionError("each query gallery must contain four candidate rows")
    labels = ordered.same_task.to_numpy(dtype=int).reshape(-1, 4)
    scores = ordered.score.to_numpy(dtype=float).reshape(-1, 4)
    predictions = ordered.predicted_link.to_numpy(dtype=int).reshape(-1, 4)
    if not np.all(labels.sum(axis=1) == 1):
        raise AssertionError("each gallery must have exactly one positive")
    positive = scores[np.arange(len(scores)), labels.argmax(axis=1)]
    negatives = scores[labels == 0].reshape(-1, 3)
    greater = (negatives > positive[:, None]).sum(axis=1)
    tied = (negatives == positive[:, None]).sum(axis=1)
    query = groups[["query_task", "fraction"]].copy()
    query["hit1"] = np.where(greater == 0, 1 / (1 + tied), 0.0)
    query["gallery_auc"] = ((negatives < positive[:, None]) +
                            .5 * (negatives == positive[:, None])).mean(axis=1)
    query["fmr"] = predictions[labels == 0].reshape(-1, 3).mean(axis=1)
    query["fsr"] = 1 - predictions[labels == 1]
    task_share = query.groupby(["query_task", "fraction"], observed=True).agg(
        hit1=("hit1", "mean"), gallery_auc=("gallery_auc", "mean"),
        fmr=("fmr", "mean"), fsr=("fsr", "mean")).reset_index()
    ap_rows = []
    for (task, fraction), group in frame.groupby(["query_task", "fraction"], sort=False, observed=True):
        ap_rows.append({"query_task": task, "fraction": str(fraction),
                        "pair_ap": float(average_precision_score(
                            group.same_task.astype(int), group.ranking_score.astype(float)))})
    task_share = task_share.merge(pd.DataFrame(ap_rows), on=["query_task", "fraction"],
                                  validate="one_to_one")
    task = task_share.groupby("query_task", observed=True)[
        ["fmr", "fsr", "hit1", "gallery_auc", "pair_ap"]].mean()
    task["ba"] = 1 - (task.fmr + task.fsr) / 2
    return task[METRICS].sort_index()


def estimate(values: pd.DataFrame, indexes: np.ndarray, metrics) -> dict:
    point = values[metrics].mean(axis=0)
    draws = values[metrics].to_numpy()[indexes].mean(axis=1)
    bounds = np.quantile(draws, [0.025, 0.975], axis=0).T
    return {metric: {"mean": float(point[metric]),
                     "ci95": [float(bounds[i, 0]), float(bounds[i, 1])]}
            for i, metric in enumerate(metrics)}


def input_frame(predictions: pd.DataFrame, source: str, variant_column: str,
                learner_filter=None) -> pd.DataFrame:
    frame = predictions.copy()
    frame["source"] = source
    frame["arm"] = [canonical_arm(source, v) for v in frame[variant_column].astype(str)]
    for column in ("fold", "replicate", "same_task", "candidate_position", "fallback", "predicted_link"):
        frame[column] = frame[column].astype(int)
    frame["fraction"] = frame.fraction.astype(str)
    frame["score"] = frame.score.astype(float)
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runner_path = ROOT / "research/experiments/matched_fingerprint_v1/run.py"
    metrics_path = ROOT / "research/experiments/matched_fingerprint_v2/analyze_frozen_ranking_metrics.py"
    runner = load(runner_path, "matched_fingerprint_v1_runner_expanded")
    sources = [runner_path, metrics_path, PROTOCOL, runner.FOLDS_CSV,
               ROOT / "attackagent/fingerprint_v2.py",
               ROOT / "research/results/behavior_corpus_v1/trace_registry.csv",
               ROOT / "research/results/behavior_corpus_v1/evidence.jsonl",
               ROOT / "research/data/external/ait-shell-pilot-v1/data.zip",
               V1 / "pair_predictions.csv.gz", NYU]
    added_parts, source_audits = [], {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN.*")
        for dataset in ("cth", "ait"):
            parts, audit = run_dataset(runner, dataset, sources)
            added_parts.extend(parts)
            source_audits[dataset] = audit

    added = pd.concat(added_parts, ignore_index=True)
    v1 = pd.read_csv(V1 / "pair_predictions.csv.gz", compression="gzip")
    v1 = v1[((v1.dataset == "cth") & v1.learner.isin(["svm", "tree", "xgb"]) &
             v1.variant.isin(["evidence", "BS", "B", "S"])) |
            ((v1.dataset == "ait") & (v1.learner == "xgb") &
             v1.variant.isin(["evidence", "BS", "BT", "ST", "BST"]))].copy()
    v1["source"] = v1.dataset
    v1["arm"] = [canonical_arm(source, variant) for source, variant in zip(v1.dataset, v1.variant)]
    for col in ("fold", "replicate", "same_task", "candidate_position", "fallback", "predicted_link"):
        v1[col] = v1[col].astype(int)
    v1["fraction"] = v1.fraction.astype(str)
    v1["score"] = v1.score.astype(float)
    added["source"] = added.dataset
    added["arm"] = [canonical_arm(source, variant) for source, variant in zip(added.dataset, added.variant)]
    for col in ("fold", "replicate", "same_task", "candidate_position", "fallback", "predicted_link"):
        added[col] = added[col].astype(int)
    added["fraction"] = added.fraction.astype(str)
    added["score"] = added.score.astype(float)
    combined = pd.concat([v1, added], ignore_index=True, sort=False)

    # Check the fixed candidate set for every feature arm, including the added fits.
    for (source, learner), group in combined.groupby(["source", "learner"], observed=True):
        base = group[group.arm == "E"]
        if base.empty:
            raise AssertionError(f"missing E baseline for {source}/{learner}")
        base = base[base.fold == base.fold.min()]
        full_group = combined[(combined.source == source) & (combined.learner == learner)]
        for arm, arm_frame in full_group.groupby("arm", observed=True):
            if arm_frame.arm.iloc[0] == "E":
                continue
            verify_same_pairs(group[group.arm == "E"], arm_frame,
                              f"candidate identity {source}/{learner}/{arm}")

    nyu = pd.read_csv(NYU, compression="gzip")
    nyu["source"] = "nyu"
    nyu["arm"] = [canonical_arm("nyu", variant) for variant in nyu.variant.astype(str)]
    for col in ("fold", "replicate", "same_task", "candidate_position", "fallback", "predicted_link"):
        nyu[col] = nyu[col].astype(int)
    nyu["fraction"] = nyu.fraction.astype(str)
    nyu["score"] = nyu.score.astype(float)
    for learner, group in nyu.groupby("learner", observed=True):
        verify_same_pairs(group[group.variant == "E"], group[group.variant == "E_BS"],
                          f"NYU frozen pair identity {learner}")
    # A joined analysis frame includes NYU but does not refit or reselect any NYU model.
    all_rows = pd.concat([combined, nyu], ignore_index=True, sort=False)
    all_rows["ranking_score"] = np.nan
    all_rows["gallery_fallback"] = 0
    fallback_audit = {}
    task_frames = {}
    for (source, learner, arm), frame in all_rows.groupby(["source", "learner", "arm"], observed=True):
        e = all_rows[(all_rows.source == source) & (all_rows.learner == learner) & (all_rows.arm == "E")]
        if e.empty:
            raise AssertionError(f"missing source/learner E score {source}/{learner}")
        ranked, detail = pairwise_to_ranking_scores(frame, e)
        ix = ranked._source_index.to_numpy(dtype=int)
        all_rows.loc[ix, "ranking_score"] = ranked.ranking_score.to_numpy()
        all_rows.loc[ix, "gallery_fallback"] = ranked.gallery_fallback.to_numpy()
        fallback_audit[f"{source}:{learner}:{arm}"] = detail
        metric_frame = ranked.copy()
        metric_frame["score"] = metric_frame.ranking_score.astype(float)
        task_frames[(source, learner, arm)] = fast_task_values(metric_frame)

    rng = np.random.default_rng(BOOT_SEED)
    task_orders = {}
    results = {}
    for key, task_values in task_frames.items():
        source, learner, arm = key
        task_list = sorted(task_values.index)
        if source not in task_orders:
            task_orders[source] = {"tasks": task_list,
                                   "indexes": rng.integers(0, len(task_list), size=(N_BOOT, len(task_list)))}
        elif task_orders[source]["tasks"] != task_list:
            raise AssertionError(f"task support differs within {source}/{learner}/{arm}")
        indexes = task_orders[source]["indexes"]
        values = task_values[METRICS]
        absol = estimate(values, indexes, METRICS)
        frame = all_rows[(all_rows.source == source) & (all_rows.learner == learner) & (all_rows.arm == arm)]
        tie = fallback_audit[f"{source}:{learner}:{arm}"]
        results[f"{source}:{learner}:{arm}"] = {
            "source": source, "learner": learner, "arm": arm,
            "task_count": len(task_list), "pair_decisions_per_arm": int(len(frame)),
            "positive_pair_prevalence": float(frame.same_task.mean()),
            "decision_and_gallery_rank_metrics": absol,
            "fallback_and_ties": tie,
            "threshold_rejects_all": bool(frame.predicted_link.sum() == 0),
        }

    contrasts = {}
    for key, task_values in task_frames.items():
        source, learner, arm = key
        baseline_arm = "E"
        if arm == baseline_arm:
            continue
        base_values = task_frames[(source, learner, baseline_arm)][METRICS]
        if task_values.index.tolist() != task_frames[(source, learner, baseline_arm)].index.tolist():
            raise AssertionError(f"task support differs for {source}/{learner}/{arm}")
        delta = task_values[METRICS] - base_values
        indexes = task_orders[source]["indexes"]
        contrasts[f"{source}:{learner}:{baseline_arm}->{arm}"] = {
            "baseline": baseline_arm, "candidate": arm,
            "task_count": len(task_values), "pair_decisions_per_arm": int(
                all_rows[(all_rows.source == source) & (all_rows.learner == learner) &
                         (all_rows.arm == arm)].shape[0]),
            "paired_delta": estimate(delta, indexes, METRICS),
        }
    for learner in ["xgb"]:
        base = task_frames[("ait", learner, "E+B+S")][METRICS]
        full = task_frames[("ait", learner, "E+B+S+T1--T6")][METRICS]
        if base.index.tolist() != full.index.tolist():
            raise AssertionError("AIT T increment task support differs")
        contrasts["ait:xgb:E+B+S->E+B+S+T1--T6"] = {
            "baseline": "E+B+S", "candidate": "E+B+S+T1--T6",
            "task_count": len(base), "pair_decisions_per_arm": int(
                all_rows[(all_rows.source == "ait") & (all_rows.learner == learner) &
                         (all_rows.arm == "E+B+S")].shape[0]),
            "paired_delta": estimate(full - base, task_orders["ait"]["indexes"], METRICS),
        }

    pred_cols = [c for c in all_rows.columns if c not in ("e_score", "_source_index")]
    final_predictions = all_rows[pred_cols].copy()
    out_pred = OUT / "expanded_predictions.csv.gz"
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    final_predictions.to_csv(out_pred, index=False, compression="gzip")
    metadata = {
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha(PROTOCOL),
        "analysis_script": str(pathlib.Path(__file__).relative_to(ROOT)),
        "analysis_script_sha256": sha(pathlib.Path(__file__)),
        "runner": str(runner_path.relative_to(ROOT)), "runner_sha256": sha(runner_path),
        "metric_script_sha256": sha(metrics_path), "fingerprint_script_sha256": sha(ROOT / "attackagent/fingerprint_v2.py"),
        "scope": "one bounded supplementary block-ablation batch; no representation, parameter, or feature selection rerun",
        "nyu_used_for_selection": False,
        "bootstrap": {"unit": "query task", "draws": N_BOOT, "seed": BOOT_SEED,
                      "paired_arm_indices": True},
        "model_and_pair_seed": runner.SEED,
        "threshold": "per-arm inner OOF maximize recall subject to pooled FMR <= 10%; missing pair falls back to same-fold E score, threshold, and decision",
        "ranking_fallback": "if any row in a four-candidate gallery falls back, all four ranking scores use matched E scores",
        "rank_metric_ties": "uniform fractional tie credit for Hit@1 and gallery AUC; pair AP computed from ranking_score within task and CTH share",
        "block_sets": {
            "cth": ["E", "E+B", "E+S", "E+B+S"],
            "ait": ["E", "E+B", "E+S", "E+T1--T6", "E+B+S", "E+B+T1--T6", "E+S+T1--T6", "E+B+S+T1--T6"],
            "nyu": ["E", "E+B+S (frozen external)"],
        },
        "cohorts": source_audits,
        "inputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(sources))},
        "added_prediction_sha256": {r["prediction_file"]: r["prediction_sha256"]
                                     for dataset in source_audits.values()
                                     for fold in dataset["folds"].values()
                                     for learner in fold["learners"].values()
                                     for r in learner["added_arms"].values()},
        "merged_predictions": str(out_pred.relative_to(ROOT)),
        "merged_predictions_sha256": sha(out_pred),
        "results": results,
        "contrasts_vs_E": contrasts,
    }
    write_json(OUT / "analysis.json", metadata)
    write_json(OUT / "run_manifest.json", {
        "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit_learn": runner.sklearn.__version__,
                        "xgboost": runner.xgboost.__version__},
        "bootstrap_seed": BOOT_SEED, "bootstrap_draws": N_BOOT,
        "analysis_sha256": sha(OUT / "analysis.json"),
        "merged_predictions_sha256": sha(out_pred),
        "added_arms": metadata["added_prediction_sha256"],
        "nyu_predictions_sha256": sha(NYU),
    })
    print(f"wrote {OUT / 'analysis.json'}")
    print(f"merged predictions: {len(final_predictions):,} rows")
    for name, item in contrasts.items():
        d = item["paired_delta"]
        print(name, " ".join(f"{metric}={100*d[metric]['mean']:+.2f}pp"
                             for metric in ("ba", "fmr", "fsr", "hit1", "gallery_auc", "pair_ap")))


if __name__ == "__main__":
    main()
