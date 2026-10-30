# Balanced association iteration 9: stabilize the repaired command-plus-operand method

Date: 2026-10-31. Iteration 8 repaired Human command extraction and found a promising outer-fold point for `operand_aware_command`, but fold-wise method switching produced a selected policy whose aggregate FSR rose. This iteration keeps the repaired terminal-output parser and candidate scoring fixed, then limits operational selection to two choices: command TF–IDF or command plus normalized operands. Profile, tensor, neighborhood, and verifier methods remain reported as diagnostics and ablations.

## Hypothesis

The command-plus-operand method may reduce false merges and improve retrieval while retaining same-task positives once Human command lines are reconstructed correctly. Restricting selection to one proposed alternative and the command fallback may reduce instability from choosing among many small-sample candidates.

## Fixed protocol

1. Recover Human shell commands from terminal output prompt echoes, applying cursor and editing controls; use the keystroke parser only when no prompt command can be recovered. Agent shell commands continue to come from tool-call arguments. Task answers, target flags, score results, and task labels are excluded from features.
2. Keep iteration 8's normalized operand signatures, top-12/16/20 candidate sweep, hard-negative verifier, profile branches, ablations, dataset revision, and grouped splits unchanged.
3. Compare only `command_tfidf` and `operand_aware_command` for the operational selected policy. Choose method and threshold within inner grouped out-of-fold predictions. Retain all other methods in saved outer diagnostics.

## Acceptance

An alternative is eligible only when inner out-of-fold evaluation has strictly lower overall FMR, no higher overall or estimable subgroup FSR, and no worse other subgroup errors, balanced accuracy, query coverage, or four ranking metrics than the same-fold command baseline. If it fails, use the command baseline for that fold. Outer evaluation remains one-time for this protocol; an overall outer FSR increase marks the iteration as failed, regardless of FMR or ranking gains. Report absolute error rates, paired changes, task-cluster 95% intervals, and fold results. The reused development cohort is not independent confirmation.

## Data boundary

The frozen cohort contains 97 traces in 40 task groups: 57 Human and 40 Agent, with one Agent trace per task group. It supports exploratory same-task retrieval and Human↔Agent direction analysis. It has no positive Agent→Agent pairs and no stable actor IDs, so it cannot establish Agent identity or organizational attribution.
