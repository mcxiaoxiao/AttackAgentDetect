# Iteration 7 error audit

Iteration 7 did not pass the acceptance gate. Its selected policy lowered task-macro FMR from 78.94% to 76.24%, but raised FSR from 6.82% to 8.05% (+1.23 percentage points); query coverage also fell from 94.83% to 94.00%. The neighborhood-only point had FMR 73.91% and FSR 8.85%. Neither is an accepted improvement.

## Selected-policy missed links

The selected policy introduced six directed false-split rows, representing three unordered Human–Agent pairs in two InterCode tasks (`task_77` and `task_81`). They were scored by `operand_aware_command`, not filtered by the candidate pool. Their command-TF–IDF similarities were 0.070–0.101, operand Jaccard was 0.003–0.014, and the selected scores (0.035–0.047) fell below a 0.050 threshold.

The raw terminal output shows why the operand feature was misleading. Human prompt echoes contain `cat chall_1.S` for `task_77` and `cat chall_3.S` for `task_81`; the Agent tool calls read those same filenames. The old Human input-keystroke parser instead emitted fragments such as `cat chall` and `cat cha`, because it discarded cursor edits and completion. This was a recording-parser error, not evidence that the paired traces used different files. Iteration 8 reads executed shell commands from output prompt echoes and records a fallback when no prompt command can be recovered.

The corrected parser changed command heads or normalized operands for 41 of the 57 Human traces in the selected cohort. Recognized Human command heads increased from 534 to 619. For 49 selected Human traces, the output transcript supplied prompt echoes; 8 used the legacy keystroke fallback. The trace-level differences are in `human_command_extraction_audit.csv` under the iteration 8 results.

## Neighborhood-only missed links

Compared with the command baseline, neighborhood verification rejected 20 directed positive rows (10 unordered Human–Agent pairs) spanning seven task groups. All were cross-mode. Fourteen rows had no shared top-3 neighbor; four had support 0.13 or 0.20; two had support 0.67 but weak reciprocal consensus (0.083). The neighborhood score reduced valid cross-mode pairs; the added false splits were not caused by its candidate generator.

Candidate filtering was a separate verifier issue. The three-family union had 81.8% micro / 93.2% task-macro positive-pair recall at top-12, 94.6% / 98.4% at top-16, and 100% at top-20. The hard-negative verifier still had 96.4% FMR at top-16, so widening the candidate pool alone did not make it useful.

## Interpretation

The v7 result is retained as a failed iteration. Its FSR increase is not acceptable, and the profile-conditioned verifier did not establish useful discrimination. The most actionable root cause was faulty Human command extraction; evaluate the repaired features before drawing conclusions about profile utility. This dataset remains a same-task proxy with no positive Agent–Agent pairs, so it cannot establish actor identity or Agent–Agent missed-link performance.
