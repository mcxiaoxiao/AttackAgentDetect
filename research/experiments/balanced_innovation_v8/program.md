# Balanced association iteration 8: reconstruct Human shell commands

Date: 2026-10-31. This iteration corrects one input-measurement issue found in the iteration 7 error audit. The Human source stores interactive terminal keystrokes, while the Agent source stores tool-call arguments. The previous adapter read Human keystrokes as if each carriage return were a complete command and ignored line-editor history, cursor, and completion behavior. That produced fragments such as `cat chall` where the executed command was `cat chall_1.S`.

## Primary change

Extract Human commands only from shell-prompt echoes in the saved terminal output. Reconstruct the edited line using backspace, cursor movement, and terminal insert/delete controls. Keep the task's score and submit commands out of the command vocabulary. Do not read the answer, target, score result, or task label into any feature. Agent commands continue to come from recorded bash tool calls.

This change corrects the captured behavior before changing the association method. The v7 candidate sweep (top-12/16/20 evidence-family union), hard-negative verifier, Human/Agent profile branches, ablations, split, and threshold selection remain fixed so the effect of repairing Human command extraction can be measured.

## Evaluation and acceptance

Use the same frozen 97-trace / 40-task development cohort, three outer task-group folds, three inner grouped folds, and 95% task-cluster intervals. The locked external set is not read or retuned. Select an alternative only if inner out-of-fold evaluation shows strictly lower overall FMR, no increase in overall or Human/Human, Human/Agent, Agent/Agent FSR, and no degradation in the other estimable subgroup errors, balanced accuracy, query coverage, or four ranking metrics. If none passes, keep the same-fold command baseline.

Outer evaluation remains one-time for this protocol. Any outer FSR increase is a failed iteration regardless of an FMR reduction. Keep absolute rates, paired changes, per-fold results, and missed-link examples in the research record; do not present a failed method as a win. This reused cohort supports method development, not independent confirmation or actor/organization identity claims.

## Saved evidence and interpretation

Save input hashes, the Human command reconstruction audit, outer pair scores and candidate flags, thresholds, directional metrics, and English figures. The cohort has one Agent trace per task group, so positive Agent→Agent missed-link performance is unavailable. Same-task labels do not prove that two traces came from the same actor or organization.
