# Balanced association iteration 5: reciprocal retrieval with graph verification

Date: 2026-10-31. This is an exploratory continuation on the reused 97-trace / 40-task development cohort. The 14-task external set remains locked and is not read or tuned.

## Research question and objective

The earlier operand-signature candidate improved ranking and modestly reduced overall false merges, but increased missed links and several subgroup errors. A scalar feature blend may be the wrong architecture. Test whether a two-stage process can preserve same-task candidates while suppressing generic cross-task matches:

1. **Candidate retrieval:** rank each trace against other traces in the same benchmark using three evidence families: content (command TF–IDF and normalized operand signatures), sequence (partial command sequence and IDF-LCS), and behavior (24D profile, 4×8×3 tensor, and phase-gated score).
2. **Reciprocal verification:** compute query-local percentile ranks in each view and require symmetric evidence from both trace directions. `reciprocal_consensus` averages the strongest two of three view scores, so correlated signals inside one family cannot contribute multiple votes.
3. **Neighborhood verification:** calculate top-3 neighbor sets per evidence family. `neighborhood_verified` scales the reciprocal score by shared-neighbor support; graph support can confirm a direct candidate but cannot create a link independently.

The candidate methods, top-3 neighborhood size, and aggregation rule are fixed before the nested run. Only thresholds and method choice are selected inside the training folds.

## Data and evaluation

Use the same frozen cohort and exact task-grouped nested split as iteration 4: three outer task-group folds, with three inner out-of-fold folds per outer training partition. All traces from a task stay in one fold. This cohort has already informed prior experiments; no result is independent confirmation. The label is same-task retrieval relevance only, not actor, person, agent, or organization identity.

Compare against `command_tfidf` and retain the prior six methods plus iteration 1–3 candidates. Choose the operating method only if its inner out-of-fold policy is no worse than the command baseline for every estimable overall and Human–Human / Human–Agent / Agent–Agent error rate, balanced accuracy, query coverage, and ranking metric. Otherwise fall back to the baseline. The outer test fold is evaluated once after method and threshold are fixed.

Report FMR, FSR, balanced accuracy, query coverage, all estimable subgroup errors, Hit@1, MRR, Recall@5, NDCG@5, task-cluster 95% intervals, selected policies, and pair-level outer predictions. Use English-only labels in figures. Do not read, retune, or rerun the locked external set.

## Why these features

The prior ranking result suggests normalized operands contain useful task evidence, while the error audit showed that individual strong rankings can still connect unrelated tasks. Reciprocal ranks address directionality: a pair must look plausible from both traces. Grouping evidence into content, order, and behavior avoids treating correlated command features as independent corroboration. Shared-neighbor evidence asks whether two traces occupy a compatible local neighborhood, a graph-level consistency signal that can rescue links supported indirectly while down-weighting isolated generic similarities.

## Limits and stop condition

The corpus is small, task labels are only a same-task retrieval proxy, and the same groups were examined earlier. The neighborhood graph is formed within each held-out benchmark fold using unlabeled pair scores; it does not use task IDs, actor labels, prompts, outputs, or evaluation labels. Thresholds are nested and no locked external data are used. A favorable point estimate remains exploratory. If the Pareto selector returns the baseline or outer folds show material subgroup regressions, report the candidate as inconclusive rather than presenting it as a usable identity-linkage system. Confirmatory identity claims require newly collected, actor-level ground truth under a common event schema.
