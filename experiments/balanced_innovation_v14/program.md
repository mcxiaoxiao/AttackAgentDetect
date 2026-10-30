# Balanced innovation v14: query-local rank cap

## Question

Can a query-local top-k decision reduce false merges while preserving missed-link, subgroup-error, coverage, balanced-accuracy, and ranking gates?

## Motivation and one primary change

In v11, 97.12% of positive pairs were already in the top five by the selected artifact-containment score, while a single global low threshold accepted 75.87% of different-task pairs. v12–v13 showed that a globally calibrated verifier can reduce FMR, but no tested threshold passed the inner missed-link gate. This round changes only the decision rule: rank candidates independently within each query, then accept the first k, with k selected only from inner task-grouped out-of-fold predictions.

The candidate generator, artifact-containment ranking score, corrected Human command reconstruction, 97-trace/40-task cohort, seed, folds, and metrics stay fixed. Candidate ties use average rank; all candidates tied at the cutoff receive the same decision. The k grid is fixed at 1–60, the maximum union size implied by three evidence families capped at 20 each. Among k values meeting every inner hard gate, choose the smallest feasible k (highest reciprocal-rank cutoff) to minimize false merges. If none qualifies, use the same-fold command baseline.

## Hard gates and failure rule

An alternative must strictly reduce overall FMR, keep overall and every estimable subgroup FSR/FMR no higher than the inner command baseline, and keep balanced accuracy, query coverage, and all four ranking metrics no lower. Any outer-fold FSR increase over the same-fold baseline fails the round. Report the selected k, absolute FMR/FSR, subgroup errors, ranking metrics, and paired task-cluster intervals.

## Claim boundary

This is exploratory reuse of the previously inspected development cohort, not independent confirmation. Same-task labels are retrieval proxies, not common-actor or campaign labels. One Agent trace per task means Agent-to-Agent FSR remains unestimable. Do not call the method useful for attribution if absolute FMR remains high, even if all relative gates pass.

## Outputs

`results/balanced_innovation_v14/results.json` is canonical. The run records fold thresholds, pair-level scores and ranks, task metrics, directional errors, and English figures. Keep a failed result as a method-development record; reserve new tasks for one final confirmation.
