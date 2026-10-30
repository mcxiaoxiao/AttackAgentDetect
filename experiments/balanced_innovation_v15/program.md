# Balanced innovation v15: discriminative pair classification with open-set rejection

## Question

Can a fundamental restructuring — from similarity-based retrieval to discriminative pair classification with task-aware features and open-set rejection — break the FSR/FMR deadlock that v10–v14 could not resolve?

## Root-cause diagnosis from v10–v14

| Observed failure | Root cause | v15 response |
|---|---|---|
| FMR 75–95% across all methods | Generic shell vocabulary (`ls`, `cat`, `file`) collides across tasks; similarity-based scoring cannot distinguish same-task from different-task pairs that share common commands | **Chi-squared task-discriminative weighting**: weight each command token by its chi-squared statistic against task labels, so tokens that appear in many tasks get penalized more aggressively than IDF alone |
| FSR concentrates in short/long cross-mode pairs (median length ratio 0.40 vs 0.75 for found pairs) | Symmetric similarity (Jaccard, cosine) penalizes length mismatch; short traces have insufficient vocabulary to establish overlap | **Length-aware asymmetric containment**: use `overlap / min(left_mass, right_mass)` with an explicit `sqrt(min_len / max_len)` correction factor |
| 12 groups of cross-task pairs with identical command sequences (largest: `ls, cat` across 13 tasks) | Command-name features are identical for these pairs; no amount of threshold tuning can separate them | **Operand and artifact discrimination**: extract normalized flags, file extensions, artifact names, and URL tokens; use chi-squared weights on these finer-grained features |
| Verifier overfits inner folds (fold 3 BA collapses from 0.97 to 0.82) | Hard-negative verifier trains on top-ranked wrong-task pairs, which are fold-specific; model memorizes fold artifacts | **Cross-fold regularized pair classifier**: train logistic regression on all pair features with L2 regularization (C=0.1); use cross-fold predicted probabilities, not fold-specific hard negatives |
| H-A error asymmetry (H-A FMR and FSR differ from H-H and A-A) | Human terminal and Agent tool-call interfaces produce different feature distributions; a single threshold treats all pair types equally | **Source-pair conditioned scoring**: compute separate thresholds for H-H, H-A, A-A pairs; use predicted source-mode probabilities to condition the pair score |
| Forced linking inflates FMR | Every query must link to something; when no candidate is truly similar, the top-scoring wrong-task pair is accepted | **Open-set rejection**: if the best candidate score falls below a learned rejection threshold, the query is left unlinked; this trades coverage for precision |

## One primary change

Replace the similarity-retrieval paradigm with a **discriminative pair classifier** that:

1. Computes task-discriminative features (chi-squared weighted command TF-IDF, length-aware operand containment, action-sequence edit distance, source-pair type indicators).
2. Feeds these features into a regularized logistic regression trained on same-task positives vs. different-task negatives.
3. Applies source-pair conditioned thresholds (separate for H-H, H-A, A-A).
4. Uses open-set rejection: queries whose best candidate score falls below a learned threshold are left unlinked.

All transforms (chi-squared weights, IDF, pair classifier, rejection threshold) are fitted on inner training folds only. Outer folds are scored with frozen transforms.

## Competing explanations

- If chi-squared weights overfit to training tasks, test-fold performance will degrade. The nested CV design prevents this: chi-squared is computed on inner training tasks only.
- If the pair classifier memorizes fold-specific patterns, cross-fold performance will be unstable. L2 regularization (C=0.1) and the small feature set (12 dimensions) limit capacity.
- If open-set rejection sacrifices too much coverage, the method is not useful. The rejection threshold is selected to maintain coverage >= 90% on inner folds.

## Acceptance criteria

Inner-fold selection must simultaneously satisfy:
- Overall FSR no higher than same-fold command baseline
- Overall FMR strictly lower than same-fold command baseline
- All estimable subgroup error rates (H-H, H-A, A-A) no worse
- Balanced accuracy, query coverage, and all four rank metrics no worse

If no threshold passes all gates, the round fails and falls back to command baseline. Report absolute rates and paired task-cluster intervals.

## Claim boundary

This is another exploratory iteration on the same reused 97-trace / 40-task development cohort. Same-task labels are retrieval proxies, not actor, campaign, organization, or identity labels. One Agent trace per task means Agent-to-Agent FSR is not estimable. A fold-level or aggregate outer FSR increase fails the round. Even if all point estimates pass, do not call the approach operationally useful if its absolute FMR remains high.

## Outputs

`results/balanced_innovation_v15/results.json` is the canonical summary. The script writes fold thresholds, pair scores, task metrics, directional errors, and English figures. Keep failed rounds and error audits as development history.
