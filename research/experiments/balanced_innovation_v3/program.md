# Balanced association iteration 3: normalized execution operands

Date: 2026-10-31. Scope: the reused 97-trace / 40-task development cohort only. The 14-task locked external set is not read, tuned, or rerun.

## Why this iteration

Iteration 1's rare-sequence and phase-gating scores were not selected in any outer fold; the fold-wise policy had task-macro FMR 86.1%, FSR 7.8%, and balanced accuracy 53.0%, all worse than its command-baseline comparison. Iteration 2's mutual rank fusion improved ranking point estimates (Hit@1 42.8%, MRR 58.0%, Recall@5 74.8%, NDCG@5 59.7%) but its thresholded point had FMR 83.4%, FSR 16.0%, balanced accuracy 50.3%, and 88.9% query coverage. It was not selected.

Outer error pairs repeatedly linked different InterCode reverse-engineering tasks with similar generic command heads and action patterns. The present parser discards shell operands, filenames, flags, and URL context. This iteration tests whether those available event details add task-discriminative evidence.

## Candidate features and rival explanation

Extract only normalized features from observed command lines: option flags, file extensions, artifact-name tokens, and URL host/path tokens. Exclude raw command arguments, prompt/output text, model names, task identifiers, and paths from the score. Fit token IDF on outer-training traces only; unseen test tokens receive the maximum training IDF.

Compare two predeclared scores:

1. `operand_jaccard`: training-IDF weighted Jaccard similarity over normalized operands.
2. `operand_aware_command`: 0.45 command TF–IDF plus 0.55 operand similarity.

Rival: Human and Agent may solve the same task with different artifacts, while different tasks may share common flags and filenames. If so, the extra context will not improve pair discrimination and may overfit the already-reused cohort.

## Evaluation

Use the same frozen 97 traces: 67 InterCode-CTF, 14 CyBench, and 16 NYUCTF, spanning 27, 6, and 7 task groups respectively. Run three-fold outer task-group evaluation and three-fold inner out-of-fold threshold/method selection. Compare all earlier methods and candidates under the same equal-weight utility: task-macro balanced accuracy, mean of Hit@1/MRR/Recall@5/NDCG@5, and query coverage. Report task-cluster 95% intervals for FMR, FSR, Human–Agent subgroup errors, balanced accuracy, coverage, and all rank metrics. No external-set tuning is allowed.

Treat this as a final bounded development iteration. If neither candidate produces lower FMR without a material FSR, coverage, subgroup, or ranking regression, stop optimizing this same-task proxy. The next valid step is a newly collected cohort with common Human/Agent/Script instrumentation and explicit actor-level labels; same-task relevance cannot establish shared identity.
