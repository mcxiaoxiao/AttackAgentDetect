# Balanced association iteration 7: inner-selected candidate recall

Date: 2026-10-31. This iteration changes one primary component from iteration 6: candidate-pool width. Iteration 6's top-12 union admitted only about 81.8% of the positive pairs in the outer folds, so a verifier could not recover positives excluded before scoring. The current 97-trace / 40-task cohort remains an exploratory same-task retrieval proxy; it does not label actor or organization identity.

## Hypothesis

Increasing candidate recall from a union of top-12 results per evidence family may reduce missed links while a hard-negative verifier rejects lookalike traces from other tasks. A wider candidate pool is useful only if it preserves or improves the false-merge rate and downstream retrieval quality.

## Fixed protocol

1. **Candidate-width sweep:** within each benchmark and query, form the union of top-12, top-16, or top-20 results from three evidence families: content, sequence, and behavior. The width is selected only with inner task-grouped out-of-fold predictions. The outer fold is never used to select the width.
2. **Hard-negative verifier:** for each training fold, use all same-task positives, up to 12 highest-ranked wrong-task candidates per query from the top-20 training candidate union, and up to 4 sampled background negatives. Fit a regularized logistic verifier (C=0.1). Score candidates from the same three evidence families; candidates outside a selected top-k union cannot be accepted.
3. **Human/Agent profile:** train the source-mode predictor on the 24D behavior profile only. Make training probabilities by task-grouped cross-fitting and held-out probabilities from the outer-training fit. The verifier receives probabilities and their profile/tensor/phase interactions, not true mode labels, task IDs, model names, or capture paths.
4. **Predeclared ablations:** generic verifier; Human/Agent-conditioned verifier; remove Human-side profile interactions; remove Agent-side profile interactions; remove phase/tensor evidence; remove neighborhood evidence. The split, candidate width, hard-negative sample, and regularization remain matched.
5. **Baselines:** retain command TF–IDF, operand signature, and neighborhood-only methods as comparisons. The locked external set is not read or retuned.

## Selection and acceptance

Keep the same three-fold outer task-group split and three-fold inner grouped selection. An alternative can be selected only if, on inner out-of-fold data, its FSR is no higher than the same-fold command baseline, its FMR is strictly lower, and every estimable Human/Human, Human/Agent, Agent/Human, and Agent/Agent error rate, balanced accuracy, query coverage, and four ranking metrics is no worse. If no candidate passes, use the command baseline and record that no alternative passed.

Outer evaluation is one-time for this protocol. If outer FSR exceeds the same-fold baseline, the iteration fails its primary gate regardless of FMR improvement; do not present it as a winning method or tune against that outer result. Report absolute rates, paired deltas, 95% task-cluster intervals, and per-fold results. A claim of significant improvement requires the interval and absolute effect to support it; the reused development cohort is not independent confirmation.

## Saved evidence

Save all outer pair scores and candidate flags, inner-selected thresholds, fold-level and directional metrics, candidate recall for each top-k width, source hashes, and English figures. Keep failures and their error cases in the research record. Separate true execution behavior from capture-format artifacts when reviewing cases; do not infer a cause without trace-level evidence.

## Interpretation boundary

The cohort has 57 Human traces and 40 Agent traces, with one Agent trace per task group and 1–2 Human traces. Human→Agent and Agent→Human retrieval can be measured; positive Agent→Agent pairs are absent, so Agent→Agent missed-link performance is not estimable. Actor identity requires stable subject IDs and repeated runs, which this dataset does not provide.
