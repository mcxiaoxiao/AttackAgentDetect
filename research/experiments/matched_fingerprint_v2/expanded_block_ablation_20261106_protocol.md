# Expanded block ablation (2026-11-06)

## Question and stop rule

This single fixed batch completes the already specified feature-block comparisons. It asks whether the observed B/S and measured-timing increments persist when each block is added alone and in combination, with the original held-out candidate pairs and fold-selected model settings. No further parameter, representation, threshold, feature-selection, or seed search is planned after this batch. NYU remains a frozen external E versus E+B+S comparison and is not used to select any setting.

## Frozen evaluation units and candidates

| Cohort | Rows | Tasks | Candidate decisions | Arms to report |
|---|---:|---:|---:|---|
| CTH mixed-role | 85 traces | 17 | 54,400 per arm | E, E+B, E+S, E+B+S for RBF-SVM, tree, XGBoost |
| AIT audit logs | 109 runs | 12 | 43,600 per arm | E, E+B, E+S, E+T, E+B+S, E+B+T, E+S+T, E+B+S+T for XGBoost |
| NYU D-CIPHER Agent-only | 585 runs | 160 | 234,000 per arm | Frozen E and E+B+S for three learners; no refit |

CTH and AIT use their original task-held-out outer folds, inner OOF thresholds, fixed candidate lists, selected representation, and per-fold model parameter from `matched_fingerprint_v1`. Added CTH arms are B and S for SVM and tree. Added AIT arms are B, S, and T. Existing frozen predictions supply all other CTH and AIT arms. NYU is included only to contextualize the already frozen external comparison.

## Fit and fallback rules

For each added arm, the per-fold learner parameter and representation are read from the existing outer-fold audit. They are not reselected. The feature vectorizer, median imputation, standardization, and decision threshold are fit using only that fold's training tasks and its original two inner task splits. The threshold maximizes inner OOF positive recall subject to pooled false-merge rate at most 10%; feature-missing pairs reuse that outer fold's E score, threshold, and decision. This reproduces the original pairwise fallback policy.

Decision metrics use the saved pairwise fallback decisions. Ranking is computed inside each original four-candidate gallery. If any candidate in a gallery uses a missing-feature fallback, every score in that gallery is replaced by its corresponding E score before Hit@1, gallery AUC, and pair AP are calculated. This keeps scores within a ranking comparison on one scoring scale. We record both pairwise fallback counts and affected gallery counts. T denotes only observed AIT launch-gap measurements T1--T6; duration-dependent T7--T8 remain unavailable.

## Outcomes and uncertainty

Report task-macro BA, FMR, and FSR, plus Hit@1, gallery AUC, and pair AP. CTH first averages the five fixed role shares within each task; AIT and NYU average within each task. Point estimates and paired differences use the same task support. Ninety-five-percent percentile intervals use 10,000 paired bootstrap resamples of task clusters, seed `20261106`. Repeated pair decisions are not independent samples. Hit@1 and gallery AUC award uniform fractional credit for score ties; pair AP is computed within task and CTH role share at the frozen 1:3 candidate prevalence.

All cells in the main decision table show an absolute percentage followed by the signed percentage-point difference from E in parentheses. The AIT T increment is separately contrasted as E+B+S+T1--T6 versus E+B+S. The ranking table reports absolute percentages and paired changes for predeclared core comparisons; the complete factorial ranking contrasts are retained in the machine-readable results and experiment record. Fixed shading bands are `<1`, `1--<3`, `3--<8`, and `>=8` percentage points; direction follows each metric (higher is favorable for BA and ranking, lower for FMR/FSR). Color conveys magnitude and direction only, never statistical significance.

## Integrity checks

The runner must verify that all arms share the original pair identities within each cohort, that outer train/test tasks remain disjoint, and that each added fold threshold satisfies its OOF FMR constraint with the recorded E fallback. It must record frozen parameter/representation, thresholds, fallback counts, score-tie counts, task denominators, input and code hashes, environment, and every added prediction. Final table values and deltas are recomputed from those predictions. A result remains exploratory if the paired task interval spans zero or if its direction depends on the learner/source.
