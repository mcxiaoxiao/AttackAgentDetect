#!/usr/bin/env python3
"""Generate a concise draft while preserving each existing TeX table."""
import csv
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[3]
PAPER = ROOT / "research/paper"
OUT = ROOT / "research/results/matched_fingerprint_v1"
original = (PAPER / "human-agent-fingerprint-overnight.tex").read_text()
results = json.loads((OUT / "metrics.json").read_text())


def item(dataset, learner, variant, fraction, pair_type="ALL"):
    return next(x for x in results[dataset]["methods"]
                if x["learner"] == learner and x["variant"] == variant
                and x["fraction"] == fraction and x["pair_type"] == pair_type)


def pc(v, signed=False):
    return f"{100 * v:+.2f}" if signed else f"{100 * v:.2f}"


def interval(v, signed=False):
    return "[" + ",".join(pc(x, signed) for x in v) + "]"


def delta(m, ref="vs_evidence"):
    x = m[ref]["ba"]
    return pc(x["point"], True) + " pp " + interval(x["ci95"], True)


cth = {v: item("cth", "xgb", v, "average_5_fractions")
       for v in ("BS", "S", "B", "evidence")}
ait = {v: item("ait", "xgb", v, "-1")
       for v in ("BST", "ST", "BT", "BS", "evidence")}
svm = item("cth", "svm", "BS", "average_5_fractions")
tree = item("cth", "tree", "BS", "average_5_fractions")
full = cth["BS"]
t_row = ait["BS"]["vs_xgb_full"]["ba"]
t_delta = -t_row["point"]
t_ci = [-t_row["ci95"][1], -t_row["ci95"][0]]
ha0 = item("cth", "xgb", "evidence", "0.5", "HA")
ha1 = item("cth", "xgb", "BS", "0.5", "HA")
with (OUT / "paired_score_cases.csv").open() as f:
    cases = list(csv.DictReader(f))
rescued = next(x for x in cases if x["dataset"] == "cth" and x["error_transition"] == "missed_link_corrected")
new_merge = next(x for x in cases if x["dataset"] == "cth" and x["error_transition"] == "new_false_merge")


def table_row(dataset, title, m):
    ba = m["metrics"]["ba"]
    d = m["vs_xgb_full"]["ba"]
    reference = abs(d["point"]) < 1e-12 and d["ci95"] == [0.0, 0.0]
    dcell = "Reference" if reference else pc(d["point"], True) + " " + interval(d["ci95"], True)
    return (" & ".join([dataset, title, pc(ba["point"]) + " " + interval(ba["ci95"]),
                        pc(m["metrics"]["fmr"]["point"]), pc(m["metrics"]["fsr"]["point"]),
                        dcell]) + r" \\")


table_rows = []
for data, lookup, variants in (
        ("CTH", cth, (("BS", "E+B+S (full)"), ("S", "Remove B"), ("B", "Remove S"),
                      ("evidence", "Evidence only"))),
        ("AIT", ait, (("BST", "E+B+S+T (full)"), ("ST", "Remove B"),
                      ("BT", "Remove S"), ("BS", "Remove T"), ("evidence", "Evidence only")))):
    if table_rows:
        table_rows.append(r"\midrule")
    for i, (variant, title) in enumerate(variants):
        table_rows.append(table_row(data if i == 0 else "", title, lookup[variant]))

table = r"""
\begin{table}[ht]
\centering\compacttable
\caption{Matched XGBoost association ablation. E denotes command and action-bigram evidence. CTH entries average five Agent-role shares; AIT uses a fixed mixed-mode gallery. Rates are percentages; $\Delta$BA is relative to the same dataset's full row in percentage points. BA and $\Delta$ show 95\% task-cluster intervals. The datasets use different tasks and are interpreted separately.}
\label{tab:matched-fingerprint-ablation}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrr@{}}
\toprule
Dataset & Configuration & BA [95\% CI] & FMR & FSR & $\Delta$BA [95\% CI] \\
\midrule
@ROWS@
\bottomrule
\end{tabular*}
\end{table}
""".replace("@ROWS@", "\n".join(table_rows))

header = original[:original.index(r"\begin{document}")]
intro_start = original.index(r"\section{Introduction:")
appendix_start = original.index(r"\appendix")
bib_start = original.index(r"\begin{thebibliography}")
bib_end = original.index(r"\end{thebibliography}", bib_start) + len(r"\end{thebibliography}")
oldbody = original[intro_start:appendix_start]
oldappendix = original[appendix_start + len(r"\appendix"):bib_start]
bib = original[bib_start:bib_end]
fig_match = re.search(r"\\begin\{figure\}(?:\[[^]]+\])?.*?\\end\{figure\}", oldbody, re.S)
assert fig_match and r"\label{fig:architecture}" in fig_match.group()
architecture = fig_match.group()
oldbody = oldbody[:fig_match.start()] + oldbody[fig_match.end():]
oldbody = oldbody.replace(r"\subsection{", r"\subsubsection{")
oldbody = oldbody.replace(r"\section{", r"\subsection{")
oldbody = oldbody.replace(r"\subsection{Data and observation contracts}",
                          r"\subsection{Data and observation contracts}\label{app:legacy-data}")
oldappendix = oldappendix.replace(r"Section~3.", r"Appendix~\ref{app:legacy-data}.")
oldappendix = oldappendix.replace(r"\section{Reproduction, versions, and data handling}",
                                  r"\clearpage" + "\n" + r"\section{Reproduction, versions, and data handling}")
oldappendix = oldappendix.replace(
    r"The fingerprint implementation is \texttt{attackagent/fingerprint\_v2.py}. Experiment scripts, fixed splits, predictions, and figure sources are organized under \texttt{research/experiments/} and \texttt{research/results/}, in the \texttt{behavioral\_fingerprint\_v2} and \texttt{association\_fingerprint\_v3} directories.",
    r"The fingerprint implementation is \texttt{attackagent/fingerprint\_v2.py}. Experiment scripts, fixed splits, predictions, and figure sources are grouped by protocol under \texttt{research/experiments/} and \texttt{research/results/}.")
if r"\bibitem{ait}" not in bib:
    bib = bib.replace(r"\end{thebibliography}",
                      r"\bibitem{ait} AIT/AECID. \textit{AI-Agent vs. Human System Administrators in Web Application Troubleshooting}. \url{https://github.com/ait-aecid/llm-user-shell-sim}" + "\n" + r"\end{thebibliography}")

subs = {
    "@CTHBASE@": pc(cth["evidence"]["metrics"]["ba"]["point"]),
    "@CTHFULL@": pc(full["metrics"]["ba"]["point"]),
    "@CTHDELTA@": delta(full),
    "@CTHFMR0@": pc(cth["evidence"]["metrics"]["fmr"]["point"]),
    "@CTHFMR1@": pc(full["metrics"]["fmr"]["point"]),
    "@CTHFSR0@": pc(cth["evidence"]["metrics"]["fsr"]["point"]),
    "@CTHFSR1@": pc(full["metrics"]["fsr"]["point"]),
    "@SVMDELTA@": delta(svm),
    "@TREEDELTA@": delta(tree),
    "@AITBASE@": pc(ait["evidence"]["metrics"]["ba"]["point"]),
    "@AITFULL@": pc(ait["BST"]["metrics"]["ba"]["point"]),
    "@AITDELTA@": delta(ait["BST"]),
    "@AITFMR0@": pc(ait["evidence"]["metrics"]["fmr"]["point"]),
    "@AITFMR1@": pc(ait["BST"]["metrics"]["fmr"]["point"]),
    "@AITFSR0@": pc(ait["evidence"]["metrics"]["fsr"]["point"]),
    "@AITFSR1@": pc(ait["BST"]["metrics"]["fsr"]["point"]),
    "@TDELTA@": pc(t_delta, True) + " pp " + interval(t_ci, True),
    "@FALLBACK@": pc(full["fallback_records"] / full["records"]),
    "@HA_FMR0@": pc(ha0["metrics"]["fmr"]["point"]),
    "@HA_FMR1@": pc(ha1["metrics"]["fmr"]["point"]),
    "@HA_FSR0@": pc(ha0["metrics"]["fsr"]["point"]),
    "@HA_FSR1@": pc(ha1["metrics"]["fsr"]["point"]),
    "@RESCUE_BS@": f"{float(rescued['evidence_score']):.3f}",
    "@RESCUE_BT@": f"{float(rescued['evidence_threshold']):.3f}",
    "@RESCUE_FS@": f"{float(rescued['full_score']):.3f}",
    "@RESCUE_FT@": f"{float(rescued['full_threshold']):.3f}",
    "@MERGE_BS@": f"{float(new_merge['evidence_score']):.3f}",
    "@MERGE_BT@": f"{float(new_merge['evidence_threshold']):.3f}",
    "@MERGE_FS@": f"{float(new_merge['full_score']):.3f}",
    "@MERGE_FT@": f"{float(new_merge['full_threshold']):.3f}",
    "@ARCHITECTURE@": architecture,
    "@TABLE@": table,
}

main = r"""
\maketitle

\begin{abstract}
AI agents can alter the behavior used to associate system traces. We assess whether a log-derived fingerprint improves same-task association when Human and Agent traces are mixed. On 17 CTH tasks, we compare original evidence against B/S fingerprint features on identical galleries at five exact Agent-role shares, using SVM, decision trees, and XGBoost. Across shares, XGBoost changes from @CTHBASE@\% to @CTHFULL@\% task-macro balanced accuracy (BA), a paired @CTHDELTA@, 95\% task-cluster interval. The corresponding SVM and tree changes are @SVMDELTA@ and @TREEDELTA@. On 109 AIT troubleshooting runs, XGBoost with B/S/T changes BA from @AITBASE@\% to @AITFULL@\% (@AITDELTA@); adding T over B/S changes BA by @TDELTA@. We observe trade-offs between missed links and false merges rather than a consistent cross-model gain. These exploratory same-task results motivate observation-aware association rather than operational attribution claims.
\end{abstract}

\section{Introduction}
Behavioral association links logs that merit joint examination. Command and action similarity can provide useful evidence, but its meaning may change when Human and AI Agent traces enter the same candidate set. Different tasks can use similar tools, while the same task can be executed through different sequences. We study two errors: accepting traces from different benchmark tasks as a link (false merge, FMR), and rejecting traces from the same task (false split, FSR).

In a fixed 17-task Cyber Task Horizons (CTH) mixture, two prior similarity baselines lose balanced accuracy as the fraction of Agent roles increases. At the original thresholds they accept fewer links: FMR decreases while FSR increases. This is an operating-point observation for a task-proxy benchmark, not evidence that every association algorithm fails with agents.

We ask whether complementary execution evidence changes this trade-off. Our contributions are: (1) an interpretable windowed fingerprint with local behavior, sequence evolution and observed timing; (2) a matched five-share comparison of three association classifiers before and after fingerprint input; and (3) an aligned CTH/AIT ablation that tests the incremental value of timing when genuine timestamps exist. We retain negative and inconclusive comparisons.

\section{Related work}
Agent identification can exploit terminal traces, model-specific interaction patterns, or active probes~\cite{trace,lidar,known,procgrep}. Those identity questions differ from predicting a relation between two traces. Cross-Agent studies linkage with synthetic campaign labels~\cite{crossagent}; provenance approaches such as ATLAS, HOLMES and OSTINATO use richer system evidence to reconstruct or associate attack behavior~\cite{atlas,holmes,ostinato}. Our labels encode benchmark task or troubleshooting scenario, not a verified shared campaign or operator. AIT's Human/Agent troubleshooting study provides audit logs for an additional timed analysis~\cite{ait}; its execution-mode setting is not an attack benchmark.

\section{Fingerprint and association}
For each ordered trace, successive windows retain up to 128 events. Block B summarizes local command diversity, action organization and repetition; S captures sequence novelty, revisits and reuse; T records available launch gaps or durations. A feature that cannot be observed remains missing. Window columns are summarized by means and standard deviations, optionally including trends. Figure~\ref{fig:architecture} illustrates the measurement and separate association readout.

@ARCHITECTURE@

The association input E comprises command TF--IDF cosine and action-bigram TF--IDF cosine. Fingerprint summaries are standardized using training traces only; absolute pair differences are either concatenated with E or multiplied by command similarity. SVM, decision-tree and XGBoost classifiers score same-task versus different-task pairs without receiving Human/Agent labels, source identifiers or task IDs. Profiles are not a sequence-model input, and classifier scores are not assumed calibrated.

\section{Experimental setup}
\textbf{CTH mixture.} The frozen core contains 17 tasks. In each of 170 coupled draws, 16 tasks supply 32 query/gallery role slots with exactly 0, 8, 16, 24 or 32 Agent slots. Each query has one distinct same-task and three different-task candidates. All methods use identical pairs within each share; proportions refer to role slots, not a resampled full dataset~\cite{cth}. Original task folds keep test tasks out of training. A one-command trace cannot produce the required B/S profile: @FALLBACK@\% of pair records use the paired original-evidence decision as fallback.

\textbf{AIT ablation.} We use 109 Human/Agent troubleshooting runs across 12 tasks~\cite{ait}. Three four-task outer folds each use 100 fixed gallery draws per query. Positive pairs are distinct same-task runs; negative candidates are from the other three test tasks. T uses observed launch gaps, not fabricated execution durations. AIT and CTH have different tasks and observation contracts, so absolute metrics are not directly ranked across datasets.

\textbf{Selection and inference.} Within each outer training partition, two task-disjoint folds select one of 12 window, summary, and fusion combinations and small fixed hyperparameter grids; the objective averages original-evidence and full-fingerprint task-macro BA. SVM and tree reuse the representation selected for XGBoost. Each version's threshold maximizes inner-fold recall subject to FMR at most 10\%. The threshold is then fixed across test shares and pair types; test FMR need not meet this constraint. Ablations use the full version's configuration. Held-out metrics average per query task, then CTH's five shares equally. Paired 95\% intervals resample tasks 10,000 times with draws and methods coupled; candidate sharing makes intervals approximate.

\section{Results}
\subsection{Matched mixture}
Figure~\ref{fig:matched-mixture} compares all three learners on identical galleries. The five-share XGBoost mean changes from @CTHBASE@\% to @CTHFULL@\% BA (@CTHDELTA@). FMR changes from @CTHFMR0@\% to @CTHFMR1@\%, and FSR from @CTHFSR0@\% to @CTHFSR1@\%. The corresponding SVM and decision-tree BA changes are @SVMDELTA@ and @TREEDELTA@. None of these paired intervals excludes zero. Under this constrained operating point, the full decision tree rejects all evaluated links; its zero FMR accompanies 100\% FSR. A 10\% training-side FMR target is not a guarantee of a useful test-side working point.

\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{matched-mixture-v1.pdf}
\caption{Matched evidence-only versus B/S-augmented association across five exact role shares. Rows show BA, false merges and missed links; columns use the same gallery for each classifier's pair of variants. Bands are 95\% task-cluster intervals. BA and FSR axes start near 45\%, and the FMR axis displays 0--30\%; each row shares one scale. The full decision tree rejects every test pair at its training-selected thresholds.}
\label{fig:matched-mixture}
\end{figure}

\subsection{What B, S and T contribute}
Table~\ref{tab:matched-fingerprint-ablation} places the CTH B/S and AIT B/S/T experiments in one view. On AIT, full B/S/T changes BA from @AITBASE@\% to @AITFULL@\% relative to E (@AITDELTA@). FMR changes from @AITFMR0@\% to @AITFMR1@\%, while FSR changes from @AITFSR0@\% to @AITFSR1@\%; the latter remains high. Adding T to B/S changes BA by @TDELTA@, so the independent timing contribution remains unresolved. CTH removes no T because its retained command view lacks event timestamps.

@TABLE@

At 50\% Agent roles, the XGBoost Human--Agent subgroup's FSR changes from @HA_FSR0@\% to @HA_FSR1@\%, while FMR changes from @HA_FMR0@\% to @HA_FMR1@\%. These descriptive rates reuse the same sessions and do not isolate execution mode from collection. For one same-task Human--Agent pair, the evidence score @RESCUE_BS@ is below its @RESCUE_BT@ threshold; the full score @RESCUE_FS@ exceeds its @RESCUE_FT@ threshold. Conversely, one different-task pair changes from @MERGE_BS@ below @MERGE_BT@ to @MERGE_FS@ above @MERGE_FT@. These scores come from separately fitted models, not calibrated probabilities or causal contributions. Full stratified rates, error corrections and newly introduced errors are retained with the predictions and summarized in Appendix~\ref{app:matched}. Earlier 40-task CTH ablation and other source-held-out analyses use different protocols; their tables remain intact rather than being replaced by the present 17-task results.

\section{Discussion and conclusion}
These experiments do not establish a universal fingerprint benefit. The matched CTH comparison shows model-dependent changes and a substantial false-merge/missed-link trade-off; AIT provides a separate, noisy test of observed timing. CTH terminal recording and Agent tool-call collection differ, and AIT mode, task and collection workflow remain confounded. Repeated draws reuse a small number of traces; task-cluster intervals do not eliminate candidate sharing. The datasets informed prior development, so the present results are exploratory and are not an untouched external confirmation.

A stronger study needs independently collected Human and Agent runs under aligned telemetry, repeated tasks and validated activity labels. Until then, the fingerprint is an additional source of evidence whose benefit and decision threshold must be established for each observation contract.
"""
for k, v in subs.items():
    main = main.replace(k, v)
assert not re.search(r"@[A-Z_]+@", main)
draft = (header + r"\begin{document}" + "\n" + main + "\n" + bib
         + "\n" + r"\clearpage" + "\n" + r"\appendix" + "\n"
         + (OUT / "matched-appendix.tex").read_text() + "\n"
         + r"\section{Previous exploratory protocols and results}" + "\n"
         + r"The following preserved material documents previous protocols. Its data, models, and candidate sets differ from the matched study in the main text." + "\n"
         + oldbody + "\n" + oldappendix + "\n" + r"\end{document}" + "\n")
(PAPER / "human-agent-fingerprint-draft.tex").write_text(draft)
(OUT / "matched-ablation-table.tex").write_text(table)
print("Wrote draft TeX and matched table")
