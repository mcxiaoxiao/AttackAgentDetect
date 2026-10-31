# Balanced innovation v13: exact verifier threshold boundaries

## Question

Does an exhaustive threshold scan over observed inner-fold verifier scores reveal a joint operating point that the 101-quantile scan in v12 missed?

## Motivation and one primary change

In v12, the verifier had low FMR but no threshold on the 101-point quantile grid passed every inner hard gate. This round keeps the verifier, candidate width, feature set, data, folds, and recall/coverage/ranking gates fixed. It evaluates every distinct observed verifier score as a possible threshold, so every distinct pair-decision boundary is represented. The selected threshold is still chosen only from inner grouped out-of-fold predictions and must strictly lower FMR while keeping FSR, every estimable subgroup error, balanced accuracy, coverage, and ranking no worse than the same-fold command baseline.

Among feasible thresholds, choose the highest cutoff to minimize false merges subject to the recall gate. If none passes, retain the command baseline. No outer labels determine threshold choice.

## Frozen data and claim boundary

The cohort remains the reused 97-trace/40-task development set with corrected Human command recovery and seed `20261031`. This remains exploratory development, not independent confirmation. One Agent trace per task means Agent-to-Agent FSR is not estimable. Same-task labels are retrieval proxies, not identity or campaign labels. A fold-level or aggregate outer FSR increase fails the round, and the absolute FMR must remain visible.

## Outputs

`results/balanced_innovation_v13/results.json` is the canonical summary. The run writes pair scores, task metrics, thresholds, directional outcomes, candidate recall, and English figures. Preserve failed rounds and error audits; final confirmation requires new task groups.
