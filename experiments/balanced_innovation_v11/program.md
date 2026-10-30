# Balanced innovation v11: recall-first cutoff under joint hard gates

## Question

Can a recall-first cutoff avoid the outer-fold missed-link increase seen in v10 while retaining lower false merges and non-decreasing coverage, balanced accuracy, and ranking?

## Motivation and change

In v10, the pooled outer estimate improved, but one held-out fold had a higher FSR than its same-fold command baseline. The selector had maximized an equal-weight utility inside the feasible region. This round changes only the selection objective: among thresholds that pass every inner hard gate, choose the lowest cutoff. A lower cutoff prioritizes accepting borderline candidates; it is allowed only while inner FMR remains strictly lower and every estimable error, balanced accuracy, query coverage, and ranking metric stays no worse than the same-fold baseline.

The feature score, candidate generation, Human command recovery, 97-trace/40-task cohort, seed (`20261031`), outer folds, and inner grouped out-of-fold procedure are unchanged from v10. Artifact containment is training-IDF weighted and excludes generic tool/path terms.

## Claim boundary

This is another exploratory iteration on the same reused development cohort. A fold-level or aggregate outer FSR increase fails the round. Report absolute rates and paired task-cluster intervals. Even if all point estimates pass, do not call the approach operationally useful if its absolute FMR remains high. Same-task labels are retrieval proxies, not actor, campaign, organization, or identity labels. One Agent trace per task means Agent-to-Agent FSR is not estimable.

## Outputs

`results/balanced_innovation_v11/results.json` is the canonical summary. The script writes fold thresholds, pair scores, task metrics, directional errors, and English figures. Keep failed runs and error audits as development history; independent confirmation requires new task groups.
