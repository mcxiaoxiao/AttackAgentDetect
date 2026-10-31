# Balanced association iteration 2: mutual rank fusion

Date: 2026-10-31. Scope: the reused 97-trace / 40-task development cohort only. The 14-task locked external set is not read, tuned, or rerun.

## Why this iteration

The first balanced nested run selected the command baseline, 4×8×3 tensor, and partial sequence once each; neither new sequence candidate was selected. The fold-wise selected policy regressed on false merges, missed links, balanced accuracy, and Human–Agent false merges, while only the ranking point estimates moved slightly upward. Its task-macro development estimates were FMR 86.1%, FSR 7.8%, balanced accuracy 53.0%, and Human–Agent FMR 81.9%. This is not a useful operating point.

Error review found false links concentrated among Human–Human and Human–Agent traces for different InterCode reverse-engineering tasks. Multiple command-head and action-pattern similarities appear generic across those tasks. The current source representation discards command arguments and other execution context, so adding another head-sequence score is unlikely to fix the ambiguity.

## Testable candidate and rival explanation

Candidate: fuse query-relative rankings from command TF–IDF, the 24D profile, the 4×8×3 stage tensor, and IDF local sequence using reciprocal-rank fusion with fixed `k=5`; then take the geometric mean of each directed pair's fused score in both directions. This is a scale-free, multi-view consensus score. It may suppress a pair that looks similar only because of one generic modality while retaining pairs ranked highly across both traces.

Rival: the component ranks may make similar false links across all views, and mutual ranking may punish true pairs when a trace is short or has few candidates. In that case, the bottleneck is missing discriminative evidence in the traces or weak same-task labels, not score calibration.

## Data and evaluation

Use the same 97 observations from 27 InterCode-CTF, 6 CyBench, and 7 NYUCTF task groups. Run three-fold outer task-group evaluation; choose thresholds and methods using three-fold inner out-of-fold predictions only. The four-metric rank mean, task-macro balanced accuracy, and query coverage retain equal weight in the predeclared selection utility. Ties prefer balanced accuracy, lower FMR, lower FSR, then the simpler method.

Compare all original methods and both iteration-1 candidates plus the single new mutual-rank candidate. Report FMR, FSR, Human–Agent subgroup errors, balanced accuracy, query coverage, Hit@1, MRR, Recall@5, NDCG@5, task-cluster intervals, selected method per fold, and Pareto dominance. Do not claim success if any estimable primary error or ranking metric materially regresses. All 40 task groups informed earlier exploratory work, so this remains hypothesis-generating, not independent confirmation.

## Stop condition

If mutual rank fusion does not produce a balanced point with lower FMR and no material FSR, coverage, or ranking regression, preserve the negative result and prioritize collecting same-interface Human/Agent/Script traces with explicit actor-level ground truth and richer execution events (arguments, tool outcomes, and timestamps) before further tuning. Same-task relevance remains a retrieval proxy, not actor identity.
