#!/usr/bin/env python3
"""Render the reviewed short-paper ablation and ranking tables from frozen JSON."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "research/results/matched_fingerprint_v2/expanded_ablation_20261106/analysis.json"
TABLE3 = ROOT / "research/paper/matched-v2-main-tables.tex"
TABLE4 = ROOT / "research/paper/matched-v2-ranking-table.tex"
FULL_RANK = ROOT / "research/paper/matched-v2-full-ranking-ablation.tex"

GOOD = {"light": "AblGoodLight", "mid": "AblGoodMid", "deep": "AblGoodDeep"}
BAD = {"light": "AblBadLight", "mid": "AblBadMid", "deep": "AblBadDeep"}


def pct(item: dict, metric: str) -> float:
    return 100 * item["decision_and_gallery_rank_metrics"][metric]["mean"]


def delta_pp(data: dict, source: str, learner: str, arm: str, metric: str,
             baseline: str = "E") -> float:
    name = f"{source}:{learner}:{baseline}->{arm}"
    return 100 * data["contrasts_vs_E"][name]["paired_delta"][metric]["mean"]


def band(change: float) -> str | None:
    amount = abs(change)
    if amount < 1:
        return None
    if amount < 3:
        return "light"
    if amount < 8:
        return "mid"
    return "deep"


def cell(absolute: float, change: float | None, higher_is_better: bool,
         bold: bool = False) -> str:
    value = f"{absolute:.2f}"
    if bold:
        value = f"\\textbf{{{value}}}"
    if change is None:
        return value
    level = band(change)
    if level is None:
        colorized = value
    else:
        favorable = change > 0 if higher_is_better else change < 0
        color = (GOOD if favorable else BAD)[level]
        colorized = f"\\cellcolor{{{color}}}{value}"
    return f"{colorized}{{\\scriptsize\\,({change:+.2f})}}"


def arm_result(data: dict, source: str, learner: str, arm: str) -> dict:
    return data["results"][f"{source}:{learner}:{arm}"]


def feature_table(data: dict) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\caption{Full feature-block ablation for same-challenge association.}",
        r"\label{tab:feature-block-ablation}",
        r"\begin{tabularx}{\linewidth}{@{}p{.88in}Yrrr@{}}",
        r"\toprule",
        r"Learner & Feature arm & BA (\%) & FMR (\%) & FSR (\%) \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textbf{CTH mixed roles (17 tasks; 54,400 pair decisions per arm)}} \\",
    ]
    cth_arms = ["E", "E+B", "E+S", "E+B+S"]
    for learner, display in (("svm", "RBF-SVM"), ("tree", "Decision tree"), ("xgb", "XGBoost")):
        for i, arm in enumerate(cth_arms):
            r = arm_result(data, "cth", learner, arm)
            values = []
            for metric in ("ba", "fmr", "fsr"):
                change = None if arm == "E" else delta_pp(data, "cth", learner, arm, metric)
                higher = metric == "ba"
                values.append(cell(pct(r, metric), change, higher, bold=arm == "E"))
            name = rf"\textbf{{{display}}}" if i == 0 else ""
            feature = r"\textbf{E (baseline)}" if arm == "E" else rf"\quad {arm}"
            lines.append(" & ".join([name, feature, *values]) + r" \\")
        if learner != "xgb":
            lines.append(r"\addlinespace[.18em]")
    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textbf{AIT audit logs (12 tasks; XGBoost; 43,600 pair decisions per arm)}} \\",
    ])
    ait_arms = ["E", "E+B", "E+S", "E+T", "E+B+S", "E+B+T", "E+S+T", "E+B+S+T1--T6"]
    labels = {"E+T": r"\quad E+T1--T6", "E+B+T": r"\quad E+B+T1--T6",
              "E+S+T": r"\quad E+S+T1--T6", "E+B+S+T1--T6": r"\quad E+B+S+T1--T6$^{\dagger}$"}
    for i, arm in enumerate(ait_arms):
        r = arm_result(data, "ait", "xgb", arm)
        values = []
        for metric in ("ba", "fmr", "fsr"):
            change = None if arm == "E" else delta_pp(data, "ait", "xgb", arm, metric)
            values.append(cell(pct(r, metric), change, metric == "ba", bold=arm == "E"))
        learner = r"\textbf{XGBoost}" if i == 0 else ""
        feature = r"\textbf{E (baseline)}" if arm == "E" else labels.get(arm, rf"\quad {arm}")
        lines.append(" & ".join([learner, feature, *values]) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\par\vspace{.16em}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"Each nonbaseline cell gives the absolute task-macro percentage followed by its signed percentage-point change from E. BA is higher-is-better; lower FMR/FSR is favorable. Changes smaller than 1 point remain unshaded; teal marks favorable and red unfavorable changes, with darker fills for larger changes. Shading is descriptive and does not encode significance. AIT measures only launch-gap T1--T6; $\dagger$ marks the full arm whose incremental contrast is E+B+S+T1--T6 versus E+B+S (BA +0.98, FMR +0.09, FSR $-2.05$ points; paired task intervals are in the experiment record). CTH and AIT retain their original fold-specific representations, learners, thresholds, and candidate pairs. Full intervals, denominators, thresholds, fallback counts, and fold settings are in the experiment record.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def rank_table(data: dict) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\caption{Candidate-ranking performance for the predeclared feature comparisons.}",
        r"\label{tab:ranking-diagnostics}",
        r"\begin{tabularx}{\linewidth}{@{}p{1.02in}Yrrr@{}}",
        r"\toprule",
        r"Learner & Scored evidence & Hit@1 (\%) & Gallery AUC (\%) & Pair AP (\%) \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textbf{CTH mixed roles: E+B+S compared with E}} \\",
    ]
    for learner, display in (("svm", "RBF-SVM"), ("tree", "Decision tree"), ("xgb", "XGBoost")):
        for i, arm in enumerate(("E", "E+B+S")):
            r = arm_result(data, "cth", learner, arm)
            changes = [None if arm == "E" else delta_pp(data, "cth", learner, arm, metric)
                       for metric in ("hit1", "gallery_auc", "pair_ap")]
            values = [cell(pct(r, metric), d, True, bold=arm == "E")
                      for metric, d in zip(("hit1", "gallery_auc", "pair_ap"), changes)]
            label = rf"\textbf{{{display}}}" if i == 0 else ""
            feature = r"\textbf{E (baseline)}" if arm == "E" else r"\quad E+B+S"
            lines.append(" & ".join([label, feature, *values]) + r" \\")
        if learner != "xgb":
            lines.append(r"\addlinespace[.15em]")
    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textbf{AIT XGBoost: T increment is compared with E+B+S}} \\",
    ])
    for i, arm in enumerate(("E", "E+B+S", "E+B+S+T1--T6")):
        r = arm_result(data, "ait", "xgb", arm)
        baseline = "E+B+S" if arm == "E+B+S+T1--T6" else "E"
        comparison = (f"ait:xgb:{baseline}->{arm}" if arm != "E" else None)
        changes = ([None, None, None] if arm == "E" else
                   [100 * data["contrasts_vs_E"][comparison]["paired_delta"][metric]["mean"]
                    for metric in ("hit1", "gallery_auc", "pair_ap")])
        values = [cell(pct(r, metric), d, True, bold=arm == "E")
                  for metric, d in zip(("hit1", "gallery_auc", "pair_ap"), changes)]
        display = {"E": r"\textbf{E (baseline)}", "E+B+S": r"\quad E+B+S",
                   "E+B+S+T1--T6": r"\quad E+B+S+T1--T6$^{\dagger}$"}[arm]
        lines.append(" & ".join([r"\textbf{XGBoost}" if i == 0 else "", display, *values]) + r" \\")
    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textbf{NYU Agent-only external evaluation: frozen E+B+S compared with E}} \\",
    ])
    for learner, display in (("svm", "RBF-SVM"), ("tree", "Decision tree"), ("xgb", "XGBoost")):
        for i, arm in enumerate(("E", "E+B+S")):
            r = arm_result(data, "nyu", learner, arm)
            changes = [None if arm == "E" else delta_pp(data, "nyu", learner, arm, metric)
                       for metric in ("hit1", "gallery_auc", "pair_ap")]
            values = [cell(pct(r, metric), d, True, bold=arm == "E")
                      for metric, d in zip(("hit1", "gallery_auc", "pair_ap"), changes)]
            label = rf"\textbf{{{display}}}" if i == 0 else ""
            feature = r"\textbf{E (baseline)}" if arm == "E" else r"\quad E+B+S"
            lines.append(" & ".join([label, feature, *values]) + r" \\")
        if learner != "xgb":
            lines.append(r"\addlinespace[.15em]")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\par\vspace{.16em}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"Cells give absolute task-macro percentages and signed paired changes in parentheses. CTH and NYU changes compare E+B+S with E; the AIT $\dagger$ row gives the incremental change over E+B+S. Higher ranking values are favorable. Pale-to-deeper teal/red backgrounds encode fixed absolute-change bands of 1--$<3$, 3--$<8$, and $\geq8$ points; changes below 1 point are unshaded. Colors do not encode statistical significance. If any pair in a four-candidate gallery lacks a required block, all four ranking scores use the matched E score. Complete factorial ranking estimates are in the appendix; task-cluster intervals are in the experiment record.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def full_ranking_table(data: dict) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.07}",
        r"\caption{Complete factorial ranking ablation. Values are absolute percentages followed by changes from E.}",
        r"\label{tab:full-ranking-ablation}",
        r"\begin{tabularx}{\linewidth}{@{}p{.82in}p{1.15in}Yrrr@{}}",
        r"\toprule Source & Learner & Feature arm & Hit@1 & Gallery AUC & Pair AP \\",
        r"\midrule",
    ]
    groups = [("cth", [("svm", "RBF-SVM"), ("tree", "Decision tree"), ("xgb", "XGBoost")],
               ["E", "E+B", "E+S", "E+B+S"]),
              ("ait", [("xgb", "XGBoost")],
               ["E", "E+B", "E+S", "E+T", "E+B+S", "E+B+T", "E+S+T", "E+B+S+T1--T6"])]
    for source, learners, arms in groups:
        label_source = "CTH" if source == "cth" else "AIT"
        for learner, display in learners:
            for i, arm in enumerate(arms):
                r = arm_result(data, source, learner, arm)
                changes = [None if arm == "E" else delta_pp(data, source, learner, arm, metric)
                           for metric in ("hit1", "gallery_auc", "pair_ap")]
                values = [cell(pct(r, metric), d, True, bold=arm == "E")
                          for metric, d in zip(("hit1", "gallery_auc", "pair_ap"), changes)]
                lines.append(" & ".join([label_source if (learner == learners[0][0] and i == 0) else "",
                                         rf"\textbf{{{display}}}" if i == 0 else "",
                                         r"\textbf{E (baseline)}" if arm == "E" else
                                         rf"\quad {arm if 'T1--T6' in arm else arm.replace('+T', '+T1--T6')}",
                                         *values]) + r" \\")
            lines.append(r"\addlinespace[.12em]")
        if source == "cth":
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\par\vspace{.12em}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"Each change is a paired task-level percentage-point difference from the same learner/source E baseline; the AIT $+$T increment is separately shown in the main ranking table. Teal/red shading uses the fixed magnitude bands described in Table~\ref{tab:ranking-diagnostics}; colors do not encode significance. CTH includes 17 task clusters and 54,400 pair decisions per arm; AIT includes 12 clusters and 43,600 decisions per arm. Gallery ranking uses whole-gallery E fallback when any candidate lacks a required block. Intervals, tie counts, and fallback details are recorded in the accompanying experiment Markdown and JSON.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    data = json.loads(RESULT.read_text())
    TABLE3.write_text(feature_table(data))
    TABLE4.write_text(rank_table(data))
    FULL_RANK.write_text(full_ranking_table(data))
    print(TABLE3)
    print(TABLE4)
    print(FULL_RANK)


if __name__ == "__main__":
    main()
