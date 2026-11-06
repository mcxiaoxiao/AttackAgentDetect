#!/usr/bin/env python3
"""Write the human-readable audit record for the expanded block ablation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "research/results/matched_fingerprint_v2/expanded_ablation_20261106/analysis.json"
OUT = ROOT / "research/reports/expanded-block-ablation-2026-11-06.md"
METRIC_LABELS = [("ba", "BA"), ("fmr", "FMR"), ("fsr", "FSR"),
                 ("hit1", "Hit@1"), ("gallery_auc", "Gallery AUC"), ("pair_ap", "Pair AP")]


def value(x: float) -> str:
    return f"{x * 100:.2f}"


def ci(item: dict) -> str:
    return f"{value(item['mean'])} [{value(item['ci95'][0])}, {value(item['ci95'][1])}]"


def delta_ci(item: dict) -> str:
    lo, hi = item["ci95"]
    return f"{100*item['mean']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]"


def clean(value_):
    if value_ is None:
        return "not recorded"
    if isinstance(value_, str) and value_ in ("Infinity", "-Infinity"):
        return "+∞" if value_ == "Infinity" else "−∞"
    if isinstance(value_, (float, int)):
        return f"{value_:.12g}"
    return str(value_)


def source_name(source: str) -> str:
    return {"cth": "CTH", "ait": "AIT", "nyu": "NYU"}.get(source, source)


def threshold(value_):
    # The original finite_json audit stores an infinite all-reject cutoff as null.
    if value_ is None:
        return "+∞"
    return clean(value_)


def main() -> None:
    d = json.loads(RESULT.read_text())
    out = [
        "# Expanded feature-block ablation (2026-11-06)",
        "",
        "## Question and frozen design",
        "",
        "This fixed batch completes the CTH single-block comparisons for all three learners and the AIT XGBoost factorial. Feature arms use the original task-disjoint folds, selected representations, learner parameters, candidate pairs, inner out-of-fold threshold rule, and pairwise E fallback. No representation, parameter, threshold rule, feature subset, or NYU result was used to choose an arm.",
        "",
        "| Cohort | Task clusters | Trace/run count | Pair decisions per arm | Arm grid |",
        "|---|---:|---:|---:|---|",
        "| CTH mixed roles | 17 | 85 | 54,400 | E, E+B, E+S, E+B+S; RBF-SVM, tree, XGBoost |",
        "| AIT audit logs | 12 | 109 | 43,600 | E plus all seven B/S/T1--T6 subsets; XGBoost |",
        "| NYU Agent-only | 160 | 585 | 234,000 | Frozen E and E+B+S; three learners |",
        "",
        "B denotes local behavior, S sequence, and T denotes only the six observed AIT launch-gap summaries T1--T6. T7--T8 remain unobserved. NYU predictions are unchanged from the frozen external analysis.",
        "",
        "## Absolute task-macro outcomes",
        "",
        "Cells are point estimate [paired 95% task-cluster interval], in percent. CTH first averages the five role shares within each task. AIT and NYU average within task. The same task clusters and frozen pair identities are used across arms.",
        "",
        "| Source | Learner | Arm | BA | FMR | FSR | Hit@1 | Gallery AUC | Pair AP |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, result in d["results"].items():
        source, learner, arm = key.split(":", 2)
        metrics = result["decision_and_gallery_rank_metrics"]
        out.append("| " + " | ".join([source_name(source), learner, arm,
                                      *[ci(metrics[name]) for name, _ in METRIC_LABELS]]) + " |")

    out.extend(["", "## Paired differences", "",
                "Changes are percentage points; values are mean [paired 95% task-cluster interval]. Each feature arm is paired against the same learner/source E baseline. The separate AIT timing row compares E+B+S+T1--T6 against E+B+S.",
                "",
                "| Source | Learner | Contrast | ΔBA | ΔFMR | ΔFSR | ΔHit@1 | ΔGallery AUC | ΔPair AP |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for key, result in d["contrasts_vs_E"].items():
        source, learner, contrast = key.split(":", 2)
        changes = result["paired_delta"]
        out.append("| " + " | ".join([source_name(source), learner, contrast,
                                      *[delta_ci(changes[name]) for name, _ in METRIC_LABELS]]) + " |")

    out.extend(["", "## Fold configurations and thresholds", "",
                "Representations and parameters below are the original frozen selections. Added arms use those settings; only their inner OOF thresholds are fit separately under the same pooled FMR ≤10% rule. `∞` means the maximum-recall rule selected the all-reject threshold. OOF FMR includes the corresponding E fallback.",
                "",
                "| Source | Fold | Train/test tasks | Representation | Learner | Parameter | Arm thresholds (OOF FMR; test fallback pairs; all reject) |",
                "|---|---:|---:|---|---|---:|---|"])
    for source, cohort in d["cohorts"].items():
        for fold, fold_data in cohort["folds"].items():
            config = fold_data["frozen_model_settings"]
            for learner, learner_config in config.items():
                added = fold_data["learners"].get(learner, {}).get("added_arms", {})
                arm_bits = []
                # Include every frozen arm's threshold/fallback as a provenance check.
                for arm, info in learner_config.get("variants", {}).items():
                    arm_bits.append(f"{arm}: t={threshold(info.get('threshold'))}, fallback={info.get('fallback_pairs', 0)}")
                # The newly fitted arms add the OOF rate and exact outer fallback count.
                for arm, info in added.items():
                    arm_bits.append(f"{arm}: t={threshold(info['threshold'])}, OOF FMR={100*info['oof_fmr']:.2f}%, fallback={info['test_fallback_pairs']}, all-reject={info['test_threshold_rejects_all']}")
                task_count = f"{len(fold_data['outer_train_tasks'])}/{len(fold_data['outer_test_tasks'])}"
                representation = "/".join(str(v) for v in fold_data["representation"])
                out.append("| " + " | ".join([source_name(source), fold, task_count, representation,
                                              learner, clean(learner_config["parameter"]),
                                              "; ".join(arm_bits)]) + " |")

    out.extend(["", "## Fallback, ties, and all-reject checks", "",
                "Decision metrics use pairwise E fallback as in the frozen classifier. Ranking uses a separate gallery-level rule: if one of the four candidate pairs lacks a required block, all four ranking scores are replaced by matched E scores so a gallery is never ranked on mixed score scales. Hit@1 awards fractional credit for ties; gallery AUC gives half credit to equal positive/negative scores. Pair AP is calculated within query task and, for CTH, role share.",
                "",
                "| Source | Learner | Arm | Pair fallback rows | Gallery fallbacks / galleries | Positive tied galleries | Any-score-tie galleries | Threshold rejects all? |",
                "|---|---|---|---:|---:|---:|---:|---|"])
    for key, result in d["results"].items():
        source, learner, arm = key.split(":", 2)
        item = result["fallback_and_ties"]
        out.append("| " + " | ".join([source_name(source), learner, arm,
                                      str(item["pairwise_fallback_pairs"]),
                                      f"{item['gallery_fallback_count']} / {item['gallery_count']}",
                                      str(item["positive_tie_galleries"]),
                                      str(item["any_score_tie_galleries"]),
                                      str(result["threshold_rejects_all"])]) + " |")

    out.extend(["", "## Integrity checks and reproducibility", "",
                "- Every arm was checked against the same pair identity set, including fold, pair ID, query/candidate IDs and tasks, label, and candidate position.",
                "- Outer training and test task lists were checked disjoint; held-out candidate tasks equal that fold's frozen test tasks.",
                "- Recreated E scores and thresholds were checked against the frozen per-fold E files before added predictions were accepted.",
                "- Added-arm OOF pooled FMR was checked against the 10% constraint, with the original E fallback applied.",
                "- Vectorized aggregation was checked against the prior aggregation implementation on one task each from CTH, AIT, and NYU; all six outcomes matched exactly.",
                "- NYU is evaluation only: no new fitting, feature choice, or threshold selection was performed on it.",
                "",
                f"Bootstrap: {d['bootstrap']['draws']:,} paired task-cluster resamples, seed `{d['bootstrap']['seed']}`. Model/pair seed: `{d['model_and_pair_seed']}`. Repeated decisions are not independent samples.",
                "",
                "### Provenance hashes",
                ""])
    for name, digest in d["inputs_sha256"].items():
        out.append(f"- `{name}`: `{digest}`")
    out.extend(["", f"- Analysis script: `{d['analysis_script']}` — `{d['analysis_script_sha256']}`",
                f"- Frozen runner: `{d['runner']}` — `{d['runner_sha256']}`",
                f"- Protocol: `{d['protocol']}` — `{d['protocol_sha256']}`",
                f"- Merged per-arm predictions ({d['merged_predictions']}): `{d['merged_predictions_sha256']}`",
                "- Per-fold added prediction files, their hashes, all thresholds, and task-level estimates are recorded in `research/results/matched_fingerprint_v2/expanded_ablation_20261106/analysis.json` and `run_manifest.json`.",
                "",
                "## Interpretation",
                "",
                "The expanded comparison sharpens rather than overturns the existing result. CTH B/S effects vary by learner; thresholded improvements exchange FMR for FSR in different ways, and the decision tree's full arm rejects all candidates. AIT's T1--T6 increment over B/S is small and its paired intervals include zero; all factorial AIT arms have lower ranking point estimates than E. The frozen NYU XGBoost comparison improves ranking and BA while exceeding the CTH 10% calibration target on FMR. These are task-retrieval observations under the named collection and candidate protocols, not evidence of actor or campaign identity.",
                ""])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out))
    print(OUT)


if __name__ == "__main__":
    main()
