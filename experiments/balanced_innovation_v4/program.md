# Balanced association iteration 4: Pareto-constrained selection

Date: 2026-10-31. Scope: the reused 97-trace / 40-task development cohort only. The 14-task locked external set is not read, tuned, or rerun.

## Why this iteration

Normalized operand features produced the strongest ranking result so far. The inner selector chose operand Jaccard in all three outer folds. Across outer tasks, its macro FMR was 76.8% versus 78.9% for the command baseline; FSR was 9.0% versus 6.8%, while balanced accuracy was essentially unchanged (57.1% versus 57.1%). Hit@1, MRR, Recall@5, and NDCG@5 rose by 23.4, 16.2, 3.0, and 11.7 points. Human–Agent FMR also fell, but Human–Agent FSR and Agent–Agent FMR worsened. This is useful ranking evidence, not an all-metric win.

## Testable change

Keep the same candidate scores and frozen task-group splits. Replace the weighted-sum method selector with an inner-OOF Pareto rule: a method/threshold is eligible only if it is no worse than the command baseline on every estimable error rate (overall and Human–Human, Human–Agent, Agent–Agent), balanced accuracy, query coverage, and each of the four ranking metrics. Among eligible policies, select the highest equal-weight utility of balanced accuracy, mean rank quality, and coverage. The command baseline remains eligible, so the rule cannot knowingly select an inner-fold regression on a measured metric.

Rival: inner Pareto dominance may be too strict to find any alternative, and an inner-dominant policy may still regress on held-out tasks. If so, the reused cohort does not support a reliable all-metric operating point.

## Evaluation and stop condition

Run three-fold outer task-group evaluation and three-fold inner out-of-fold selection. Report the same task-cluster 95% intervals and all subgroup, coverage, and ranking metrics. The outer folds are used once and never to choose a threshold. All task groups were already used in earlier exploratory work, so this is still hypothesis-generating.

If no candidate dominates the baseline on outer estimates, or any material subgroup regression remains, stop score tuning on this proxy. The next experiment must use newly collected Human/Agent/Script traces under a common event schema with explicit actor-level labels; same-task relevance cannot establish shared identity.
