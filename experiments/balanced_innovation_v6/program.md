# Balanced association iteration 6: recall-first verifier and source-profile ablations

Date: 2026-10-31. The target is same-activity retrieval within a benchmark. The current 97-trace / 40-task cohort supplies a same-task proxy and is reused for method development; it contains 57 Human and 40 Agent traces, exactly one Agent trace in each task group, and 1–2 Human traces per group. It can compare Human→Agent and Agent→Human retrieval, but has no positive Agent–Agent pairs and no actor-level match labels.

## Why this iteration

Correcting the Pareto direction in iteration 4/5 changed the selected method but did not satisfy the target: the selected outer-fold policy lowered FMR by 2.70 percentage points while raising FSR by 1.23 points. The neighborhood-only candidate lowered FMR to 73.91% but raised FSR to 8.85%. These are trade-offs, so neither is accepted.

## Fixed architecture

1. **Candidate generation:** within each benchmark, retain the union of the top 12 candidates from content (command and operands), sequence, and behavior (profile, tensor, phase). This protects recall when one evidence family is weak.
2. **Hard-negative verifier:** train a regularized logistic model on all available same-task positive pairs, up to 12 highest-ranked different-task candidates per query, and up to 4 sampled background negatives. Feature values use the existing command, 24D profile, 4×8×3 tensor, sequence, operand, reciprocal-rank, and neighborhood scores. Exclude the label-trained fusion score to avoid reusing fitted training labels as a feature.
3. **Mode-conditioned profile:** train a Human/Agent source predictor on the 24D behavioral profile only. Generate its training probabilities by task-grouped cross-fitting; apply the model fitted on training tasks to held-out traces. Use those probabilities to form query-side and candidate-side Human/Agent interactions with profile, tensor, and phase scores. The verifier never receives true Human/Agent labels, task IDs, model names, or capture paths.
4. **Predeclared ablations:** generic verifier; mode-conditioned verifier; remove Human-side profile interactions; remove Agent-side interactions; remove phase/tensor evidence; remove neighborhood evidence. Use the same data split, candidate pool, hard negatives, and regularization (C=0.1) for each variant.

## Evaluation and acceptance

Keep the same three-fold outer task-group split and three-fold inner grouped out-of-fold selection. Tune method and threshold only within each outer training set. The selected policy must have inner FSR no higher than the same-fold command baseline and a lower FMR; coverage and all four ranking metrics must also be no worse. Compare outer results once, with task-cluster paired 95% intervals and separate Human–Human, Human→Agent, Agent→Human, and Agent–Agent strata where estimable. A strategy that raises outer FSR is recorded as a failed iteration and is not shown as the improved method.

Save the candidate lists, hard negatives, fold assignments, mode probabilities, pair predictions, thresholds, all metric rows, figures, input hashes, and model coefficients. Figures use English labels. The locked external set is not read or retuned.

## Interpretation boundary

The public cohort is useful for exploratory same-task retrieval. Human capture uses terminal input and Agent capture uses model tool calls, so mode prediction can learn source-interface differences. The current filter selects a single Claude Sonnet 4.6 Agent trace per task. Identity and Agent–Agent recall require repeated traces with stable pseudonymous actor or agent-instance IDs; any identity claim awaits that data.
