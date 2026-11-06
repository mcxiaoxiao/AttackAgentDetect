# Fingerprint dimension and ranking follow-up

Date: 2026-11-05
Status: exploratory follow-up on the existing CTH/AIT experiments; the frozen NYU evaluation was used only for post hoc ranking metrics.

## What the 24 dimensions mean

The current fingerprint is a per-window schema with three blocks of eight measurements (B/S/T), not a statistically selected 24-feature optimum. B1–B8 reuse the earlier common-log K1–K8 measurements for local command/action diversity, entropy, switching, run length, and repetition. S1–S8 extend this with prefix-relative novelty, revisits, action returns, reused command fragments, and action-distribution shift. T1–T8 summarize observed launch gaps, gap variability/order, event duration, and idle time. The blocks are an interpretable design taxonomy; their membership was not chosen by optimizing the association labels.

This is distinct from the earlier “24-value K1–K8 readout”: that readout took eight per-window measures and summarized each by mean, standard deviation, and slope (8 × 3). The present B/S/T matrix has 24 columns per window; its pooled readout has up to 72 values (24 × 3), or 48 for B/S.

At eight-event windows, the B/S blocks were eligible for 84/85 CTH traces and all 109 AIT and 585 NYU traces. Availability varies by individual measurement: all B measures are finite in 84 CTH traces; S2/S3 in 42, S4 in 58, S5 in 55, S7 in 64, and S8 in 26. In AIT, S4 is finite in 82/109 traces, S5 in 107, S8 in 101, T1–T6 in 109, and T7/T8 in zero because matched event durations are absent. The unlisted B/S measures are finite in all CTH/AIT traces eligible for their block; the frozen NYU manifest records 585 eligible B/S traces. These counts mean “at least one finite value across a trace’s windows”; unavailable values are median-imputed from the training fold. Low coverage alone does not establish that a feature is irrelevant.

## Experiments completed

### Ranking metrics on frozen pairs

Using the exact frozen pair predictions, I calculated tie-aware Hit@1/Hit@2, MRR, within-query gallery AUC, candidate-pair average precision (AP), and the stored-threshold BA. No model, feature, or threshold was refit. Hit@1/AUC/AP intervals resample task clusters (10,000 draws, seed 20261104). The table keeps the canonical BA intervals from the existing matched analyses so its BA values match the manuscript's original result tables; the new JSON also records BA with the same bootstrap draws as the ranking metrics. CTH averages the five role-share cells equally within task; AIT and NYU average query outcomes within task. AP pools candidate-pair scores within task and CTH role share at the fixed 1:3 positive/negative ratio. The repeated pairs and galleries are not independent units.

| Comparison | Tasks; pair decisions/arm | ΔHit@1 pp [95% interval] | Δgallery AUC pp [95% interval] | Δpair AP pp [95% interval] | ΔBA pp [95% interval] |
|---|---:|---:|---:|---:|---:|
| CTH SVM, E → E+B/S | 17; 54,400 | +4.77 [−3.03, +13.18] | +3.60 [−2.29, +10.85] | +3.08 [−2.36, +8.89] | +1.38 [−3.83, +6.53] |
| CTH tree, E → E+B/S | 17; 54,400 | +1.21 [−6.26, +9.02] | −0.78 [−6.71, +5.28] | +1.26 [−2.61, +5.14] | −2.80 [−6.61, +0.57] |
| CTH XGBoost, E → E+B/S | 17; 54,400 | +2.37 [−5.62, +10.83] | +2.40 [−4.32, +9.54] | +0.27 [−4.95, +5.70] | −2.38 [−7.13, +2.25] |
| AIT XGBoost, E+B/S → E+B/S+T | 12; 43,600 | +0.68 [−1.46, +2.75] | +0.93 [−0.80, +2.66] | +1.50 [−0.61, +3.45] | +0.98 [−0.21, +2.10] |
| NYU tree, E → E+B/S | 160; 234,000 | −5.59 [−8.11, −3.04] | −7.13 [−9.28, −4.94] | −3.00 [−4.61, −1.37] | +3.64 [+1.97, +5.37] |
| NYU XGBoost, E → E+B/S | 160; 234,000 | +3.20 [+0.51, +5.92] | +1.82 [−0.38, +3.97] | +2.07 [+0.04, +4.08] | +5.71 [+3.63, +7.73] |
| NYU SVM, E → E+B/S | 160; 234,000 | −1.06 [−3.71, +1.47] | −0.18 [−2.08, +1.60] | −1.41 [−3.40, +0.45] | −8.23 [−10.22, −6.28] |

The ranking metrics separate candidate ordering from a thresholded decision. CTH ranking changes are uncertain for all three learners. On the frozen NYU panel, XGBoost improves Hit@1 and AP, with AUC still uncertain; the tree’s BA gain accompanies worse Hit@1, AUC, and AP. The BA gain therefore does not mean that every learner ranks candidates better. For the isolated AIT timing increment, every ranking and BA interval spans zero. The whole E→E+B/S+T AIT comparison is not a substitute for this isolated T contrast: it combines B/S and T.

### Fold-internal single-measure reduction

On the 17-task CTH development cohort, each outer fold compared the full B/S arm with 16 candidates, one for each possible B/S measure removal. Candidate selection maximized inner task-macro BA under the same pooled validation FMR≤10% threshold rule. The selected removals were B6, S7, and B2 across the three folds, so there was no stable single measurement to remove.

| Outer comparison: selected compact minus full B/S | Change, pp [95% task-cluster interval] |
|---|---:|
| BA | −0.59 [−2.02, +0.71] |
| Hit@1 | −0.24 [−3.13, +2.15] |
| Gallery AUC | −0.32 [−1.92, +1.03] |
| Pair AP | −0.14 [−1.75, +1.38] |

Both ranking and decision metrics now use the E score and threshold on the 1,089 fallback pairs per B/S arm. The compact arm's intervals all include zero. On the same outer predictions, full B/S versus command-only changed BA by +0.27 pp [−3.79, +4.22], Hit@1 by +3.52 pp [−5.16, +13.13], gallery AUC by +4.88 pp [−2.64, +13.02], and AP by +1.90 pp [−3.56, +7.85]. Full B/S versus the original two-channel evidence E changed BA by −2.38 pp [−7.20, +2.31].

This is nested with respect to the feature-removal choice, but it reuses a CTH cohort examined in earlier iterations and reuses the representation and XGBoost parameter selected by the original run in each outer training fold. It is exploratory, not independent confirmation. It tests one-measure-at-a-time deletion, not arbitrary combinations or interaction-aware feature search.

## Assessment of the proposed follow-ups

| Proposal | Assessment from the available data |
|---|---|
| Add AUC, PR-AUC, and Hit@k | Completed as gallery AUC, candidate-pair AP, and Hit@1/Hit@2. This materially improves interpretation because threshold BA and ranking quality can move in opposite directions. Keep BA plus FMR/FSR for operating-point decisions, and ranking measures for candidate review. |
| Use NYU model snapshot as a same-source label | The separate [snapshot retrieval experiment](fingerprint-target-and-ranker-2026-11-05.md) holds out challenges and keeps all four candidates on one other challenge. It measures retrieval of four known submission configurations in D-CIPHER. |
| Add Human/Agent identity or pair-type conditioning | Existing CTH HH/HA/AA breakdowns and pair-type/threshold analyses already probe part of this suggestion; their support overlaps and does not yield a stable aggregate gain. The labels mark collection mode, not verified human identity or common Agent identity. Use them as descriptive strata; a predicted mode gate or manually downweighted “style” block would add an unvalidated modeling choice. |
| Remove capture-process artifacts and emphasize cross-source holdout | This is the largest validity issue and cannot be solved with the current trace fields: CTH Human terminal input and Agent tool calls differ in collection path, and NYU shares a framework. Existing source-held-out task-association results are sensitivity evidence, not proof that capture artifacts were removed. A clean test needs matched tasks and a common logging contract across execution modes. |
| Select fewer dimensions inside training folds; compare compact/full/command-only | Completed for one-at-a-time B/S deletion, alongside all 16 features and command-only comparisons. Results do not select a stable subset. Keep the 24-column schema as the preregistered/design representation for now; do not claim every feature is necessary. Revisit pruning with more independent task groups and a selection rule fixed before evaluation. |

## Paper and reproducibility updates

The draft now explains the design origin and the two meanings of “24,” reports trace-level coverage and median imputation, defines the ranking metrics, uses point estimates in the main ranking table, and includes the exploratory fold-internal reduction result with its limits. The local 14-page working PDF was rebuilt and visually checked. The site-hosted PDF and source ZIP remain the previous packaged snapshot. The report and machine-readable outputs preserve deltas, intervals, task counts, seeds, and input hashes.

## CTH block ablation behind Table 3

The separate B-only and S-only entries in Table 3 are a post-hoc recomputation from the frozen v1 XGBoost pair predictions, not a new model run. All arms use the same 54,400 pair decisions, the selected per-fold XGBoost settings, equal weighting over the five Agent shares within each task, and 10,000 paired task-cluster bootstrap draws (seed 20261104). The 95% intervals below are nominal and are not adjusted for examining multiple ablation arms.

| CTH feature arm | BA % | FMR % | FSR % | $\Delta$BA vs E+B+S, pp [95% interval] | $\Delta$FMR, pp | $\Delta$FSR, pp |
|---|---:|---:|---:|---:|---:|---:|
| E | 56.06 | 5.97 | 81.91 | +2.38 [−2.28, +7.29] | −9.46 | +4.70 |
| E+B | 52.75 | 13.13 | 81.37 | −0.93 [−2.52, +0.59] | −2.30 | +4.15 |
| E+S | 56.10 | 14.47 | 73.34 | +2.42 [−3.09, +8.05] | −0.96 | −3.88 |
| E+B+S | 53.68 | 15.43 | 77.21 | Reference | — | — |

For the AIT timing increment, the same 43,600 pairs give E+B+S BA/FMR/FSR of 50.94/5.28/92.84%; E+B+S+T1--T6 gives 51.92/5.37/90.79%. The paired changes are +0.98 pp in BA (95% interval [−0.21, +2.10]), +0.09 pp in FMR, and −2.05 pp in FSR. T1--T6 use observed launch gaps; duration-dependent T7--T8 are unavailable. The machine-readable source is `research/results/matched_fingerprint_v2/unified_ablation/unified_bst_ablation.json`.

Reproduce the post hoc ranking analysis and audits from the repository root with:

```sh
.venv/bin/python research/experiments/matched_fingerprint_v2/run_fold_internal_dimension_selection.py
.venv/bin/python research/experiments/matched_fingerprint_v2/analyze_frozen_ranking_metrics.py
.venv/bin/python research/experiments/matched_fingerprint_v2/audit_window_feature_coverage.py
```

Outputs are in `research/results/matched_fingerprint_v2/ranking_diagnostics_20261104/` and `research/results/matched_fingerprint_v2/dimension_selection_20261104/`. The ranking script hashes the original frozen prediction files; the dimension-selection record hashes the runner and frozen input predictions. No NYU label or result entered feature selection.
