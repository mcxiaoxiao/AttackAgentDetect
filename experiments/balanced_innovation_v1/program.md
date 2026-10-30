# Balanced trace association: exploratory follow-up

Date: 2026-10-31. Scope: development tasks only. The 14-task locked external set is not read, tuned, or rerun.

## Question

Can rare-command local alignment and phase-consistent action evidence produce a more balanced operating point than the existing command and profile scores? The prior 0% false-split point was too permissive: it accepted nearly every different-task pair. This experiment gives false merges and missed links equal weight through balanced accuracy, then checks whether ranking and query coverage hold up.

## Data and split

Use the already-inspected 97 Cyber Task Horizons traces across 40 benchmark-qualified task groups (27 InterCode-CTF, 6 CyBench, and 7 NYUCTF). The frozen cohort audit is the source of truth; an initial run exposed that the earlier draft's 177-trace count was wrong, so that run stopped before fitting any model. The runner now reads only these development sources and the selected Sonnet traces, excluding external tasks and unrelated model logs before parsing. Run one three-fold outer task-group split. Within each outer training partition, form three-fold out-of-fold predictions to choose thresholds and methods. Each outer test fold is evaluated once. This remains exploratory reuse, not independent confirmation.

## Fixed candidates

Keep the six existing methods and add two training-only features:

1. `idf_lcs_coverage`: IDF-weighted command coverage from sequence-alignment blocks, combined with IDF command overlap and penalized for large trace-length mismatch.
2. `phase_gated_lcs`: geometric mean of `idf_lcs_coverage` and same-phase action-profile similarity across four relative trajectory phases.

IDF is calculated from the training traces of each fold. Human/Agent labels, task IDs, prompts, outputs, model names, and paths are excluded from scores.

## Balanced selection rule

For each candidate, choose its threshold from inner out-of-fold training scores to maximize the unweighted mean of (a) task-macro balanced accuracy, (b) mean of Hit@1, MRR, Recall@5, and NDCG@5, and (c) query coverage. Balanced accuracy gives false merges and missed links equal cost. Ties prefer higher balanced accuracy, then lower FMR, then lower FSR, then simpler method. Select the method with the highest inner utility and compare its outer-fold results with a command TF–IDF threshold selected by the same rule.

Report each error rate, subgroup rate, balanced accuracy, coverage, all four ranking metrics, task-cluster paired intervals, selected method per fold, and the complete Pareto set. Do not claim a winner if the outer results do not improve multiple metrics without a material regression elsewhere.

## Interpretation and stop condition

All 40 task groups have informed earlier experiments, so favorable results are hypothesis-generating only. Do not tune on the locked external tasks. If the balanced objective still yields weak discrimination or unstable method selection, preserve the result and prioritize a new, same-interface Human/Agent/Script cohort before any confirmatory claim.
