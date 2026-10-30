# v12–v13 hard-negative verifier audit

## Decision

The top-16 candidate union retained all observed positive pairs in the 40 outer task groups, but the regularized hard-negative verifier did not provide a threshold that passed every inner hard gate. In v13, the threshold search exhaustively tested every distinct observed score, not just quantiles. All three outer-fold selectors found zero feasible verifier thresholds and correctly fell back to the command baseline. The verifier was not accepted.

## Error tradeoff

At its independently inner-selected equal-weight utility threshold, the generic verifier had 19.07% task-macro FMR and 16.93% FSR, versus 78.07% and 7.23% for the command baseline. It substantially reduced false merges while missing too many same-task candidates. The Human/Agent-conditioned verifier had 23.21% FMR and 18.17% FSR. These are diagnostic operating points, not hard-gate-selected winners.

The mode-conditioned ablations were mixed:

- Removing the Human-profile branch: 24.48% FMR, 17.96% FSR.
- Removing the Agent-profile branch: 21.60% FMR, 16.93% FSR.
- Removing neighborhood evidence: 28.23% FMR, 16.31% FSR.
- Removing phase/tensor evidence: 19.80% FMR, 22.08% FSR.

The pattern suggests the Human and Agent branches, neighborhood context, and phase/tensor evidence affect errors differently. It does not establish a robust Human/Agent-specific profile gain: mode conditioning performed worse than the generic verifier on both overall error rates, and every verifier diagnostic exceeded the 7.23% baseline FSR.

## Interpretation

The top-16 retrieval union was not the source of the misses: its observed candidate recall was 100% in this reused cohort. The verifier's score threshold rejected positives while deciding among a large pool of wrong-task pairs. Exact scanning confirmed that no inner threshold met the joint constraints of strictly lower FMR, no higher overall/subgroup errors, and no loss in balanced accuracy, query coverage, or ranking.

Same-task labels are retrieval proxies, not actor or campaign identity labels. Because the same task groups were repeatedly used during development, the results remain exploratory and require a new task cohort for confirmation.
