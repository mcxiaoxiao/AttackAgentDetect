#!/usr/bin/env python3
"""Create supplementary matched-result tables from aggregate predictions."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "research/results/matched_fingerprint_v1"
r = json.loads((OUT / "metrics.json").read_text())


def entry(dataset, learner, variant, frac, typ="ALL"):
    return next(x for x in r[dataset]["methods"] if x["learner"] == learner
                and x["variant"] == variant and x["fraction"] == frac and x["pair_type"] == typ)


def pct(x):
    return "--" if x is None else f"{100*x:.1f}"


def pair(dataset, learner, variant, frac, typ, metric):
    a = entry(dataset, learner, "evidence", frac, typ)["metrics"][metric]["point"]
    b = entry(dataset, learner, variant, frac, typ)["metrics"][metric]["point"]
    return "--" if a is None or b is None else pct(a) + r"$\rightarrow$" + pct(b)


def delta(dataset, learner, variant, frac, typ="ALL"):
    x = entry(dataset, learner, variant, frac, typ)["vs_evidence"]["ba"]
    if x["point"] is None or x["ci95"] is None:
        return "--"
    return f"{100*x['point']:+.1f} [{100*x['ci95'][0]:+.1f},{100*x['ci95'][1]:+.1f}]"


def latex_table(headers, rows, caption, label):
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return (r"\begin{table}[ht]" + "\n" + r"\centering\compacttable" + "\n"
            + r"\caption{" + caption + "}" + "\n" + r"\label{" + label + "}" + "\n"
            + r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrr@{}}" + "\n"
            + r"\toprule " + " & ".join(headers) + r" \\" + "\n"
            + r"\midrule" + "\n" + body + "\n" + r"\bottomrule" + "\n"
            + r"\end{tabular*}" + "\n" + r"\end{table}" + "\n")


def pair_rows(dataset, learner, variant, fractions, typ="ALL"):
    rows = []
    for frac in fractions:
        a = entry(dataset, learner, "evidence", frac, typ)
        b = entry(dataset, learner, variant, frac, typ)
        share = f"{100*float(frac):.0f}" if dataset == "cth" else "mixed"
        rows.append([learner.upper(), share, pair(dataset, learner, variant, frac, typ, "ba"),
                     pair(dataset, learner, variant, frac, typ, "fmr"),
                     pair(dataset, learner, variant, frac, typ, "fsr"),
                     delta(dataset, learner, variant, frac, typ)])
    return rows


fractions = ["0.0", "0.25", "0.5", "0.75", "1.0"]
parts = [r"\section{Matched experiment: complete outcomes and checks}\label{app:matched}",
         r"Table~\ref{tab:matched-shares} reports every role-share outcome, while Table~\ref{tab:matched-types} separates Human--Human (HH), Human--Agent (HA), and Agent--Agent (AA). Rates are task-macro percentages. Paired intervals resample query tasks; repeated draws and reused candidate tasks are not independent observations."]
rows = []
for learner in ("svm", "tree", "xgb"):
    rows.extend(pair_rows("cth", learner, "BS", fractions))
parts.append(latex_table(["Learner", r"Share (\%)", "BA", "FMR", "FSR", r"$\Delta$BA [95\% CI]"],
                         rows,
                         r"All CTH role shares. Each metric shows evidence-only $\rightarrow$ evidence plus B/S. $\Delta$BA is the paired difference in percentage points; intervals are exploratory.",
                         "tab:matched-shares"))

# Longtable keeps all 45 Human/Agent strata in the paper without shrinking type.
head = r"""
\begin{longtable}{@{}llcrrrr@{}}
\caption{CTH pair types under the identical galleries. E and B/S are evidence-only and augmented predictions; FMR and FSR show E$\rightarrow$B/S, in percent. Empty strata are not estimable. BA differences have paired task-cluster intervals.}\label{tab:matched-types}\\
\toprule Learner & Share & Type & Pos./neg. & FMR & FSR & $\Delta$BA [95\% CI] \\
\midrule\endfirsthead
\toprule Learner & Share & Type & Pos./neg. & FMR & FSR & $\Delta$BA [95\% CI] \\
\midrule\endhead
"""
longrows = []
for learner in ("svm", "tree", "xgb"):
    for frac in fractions:
        for typ in ("HH", "HA", "AA"):
            x = entry("cth", learner, "BS", frac, typ)
            counts = f"{x['positive_records']}/{x['negative_records']}"
            longrows.append(" & ".join([learner.upper(), f"{100*float(frac):.0f}\\%", typ, counts,
                                        pair("cth", learner, "BS", frac, typ, "fmr"),
                                        pair("cth", learner, "BS", frac, typ, "fsr"),
                                        delta("cth", learner, "BS", frac, typ)]) + r" \\")
parts.append(head + "\n".join(longrows) + "\n" + r"\bottomrule\end{longtable}")

parts.append(r"The AIT association panel has no exact role-share manipulation. The E and B/S/T comparison below uses the same mixed-mode pairs and thresholds fixed on training tasks.")
aitrows = []
for typ in ("HH", "HA", "AA"):
    x = entry("ait", "xgb", "BST", "-1", typ)
    aitrows.append(["XGB", typ, pair("ait", "xgb", "BST", "-1", typ, "ba"),
                    pair("ait", "xgb", "BST", "-1", typ, "fmr"),
                    pair("ait", "xgb", "BST", "-1", typ, "fsr"),
                    delta("ait", "xgb", "BST", "-1", typ)])
parts.append(latex_table(["Learner", "Type", "BA", "FMR", "FSR", r"$\Delta$BA [95\% CI]"],
                         aitrows,
                         r"AIT pair types. Each metric is evidence-only $\rightarrow$ evidence plus B/S/T; comparisons are within this troubleshooting cohort.",
                         "tab:matched-ait-types"))
for dataset in ("cth", "ait"):
    audit = [json.loads((OUT / dataset / f"fold-{fold}-audit.json").read_text()) for fold in range(3)]
    desc = []
    for a in audit:
        w, pool, fusion = a["representation"]
        p = a["selection"]["xgb"]["parameter"]
        desc.append(f"fold {a['fold']}: {w}-event, {pool.replace('_', ' ')}, {fusion}, depth {p}")
    parts.append(r"\noindent\textbf{" + dataset.upper() + r" selected training-side configurations.} " +
                 "; ".join(desc) + ".")
    ds = [100*x["delta_ba"] for x in r[dataset]["leave_one_task_out"]]
    parts.append(r"Deleting each task from both query and candidate positions gives a full-versus-evidence BA change range of " +
                 f"[{min(ds):+.2f},{max(ds):+.2f}]" + r" percentage points. These are sensitivity checks, not independent new test sets.")
parts.append(r"Fold-level thresholds, all optimization candidates, FMR/FSR paired intervals, exact role assignments and record-level predictions are stored under \texttt{research/results/matched\_fingerprint\_v1/}. A null threshold in the JSON selection audit denotes positive infinity, an all-reject decision.")
(OUT / "matched-appendix.tex").write_text("\n\n".join(parts) + "\n")
print("Built supplementary matched tables")
