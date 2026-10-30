# Balanced innovation v12: hard-negative verification under a recall gate

## Question

Can a regularized hard-negative pair verifier lower false merges substantially while preserving the command baseline's missed-link rate and all other prespecified metrics?

## Motivation and one primary change

The v11 recall-first artifact score produced no outer missed links, but its 75.9% task-macro FMR remains unusable. The v11 audit also showed that the top-16 candidate union contained every observed positive pair in the outer predictions, while adding many negative candidates. This round fixes the union width at 16 and selects a regularized verifier trained on same-task positives, top-ranked wrong-task hard negatives, and a small random background sample. The verifier uses command, distinctive artifact, sequence, behavioral-profile, phase-tensor, and reciprocal-neighborhood features; Human/Agent type interactions are excluded in this primary run.

Threshold choice is made only from inner task-grouped out-of-fold predictions. It must strictly reduce FMR and keep overall and every estimable subgroup FSR, other subgroup errors, balanced accuracy, query coverage, and all ranking metrics no worse than the same-fold command baseline. Among feasible thresholds, choose the highest verifier cutoff to reduce false merges within the recall gate. If none passes, retain the baseline.

## Frozen data and claim boundary

- Reuse the 97-trace, 40-task development cohort, the v11 corrected Human command recovery, seed `20261031`, and the same nested task-grouped folds.
- The top-16 candidate width is fixed in this round; no outer-fold labels choose candidate width or threshold.
- Every task has only one Agent trace, so Agent-to-Agent missed-link performance is not estimable.
- Labels are same-task retrieval proxies, not actor, campaign, organization, or identity labels.
- The cohort has been reused in earlier development rounds. These are exploratory estimates, not independent confirmation.
- Any outer-fold missed-link increase means failure. Report absolute FMR even when relative improvement is large; a high absolute error rate is not an operational success.

## Outputs

`results/balanced_innovation_v12/results.json` is the canonical summary. Pair scores, task metrics, frozen fold thresholds, directional outcomes, candidate recall, and English figures are emitted beside it. Keep failed runs and their error cases in the repository. A new task cohort is required for independent confirmation.
