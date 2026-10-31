# Balanced innovation v10: distinctive artifact containment

## Question

Can a short trace and a longer trace still be linked when they share a distinctive file or artifact, even if the longer trace contains many unrelated commands? Can this recover those links without adding false merges or hurting coverage and ranking?

## Motivation from error review

Earlier iterations exposed positive same-task pairs where a Human trace and an Agent trace used the same artifact, but set-based Jaccard similarity diluted the shared evidence because the Agent trace was much longer. Examples included `chall_3.S` and `rock`. Those observations motivate this change; they are not used to set the outer-fold threshold.

## Frozen cohort and claim boundary

- Reuse the existing 97-trace, 40-task development cohort and the same task-grouped three-fold outer split seed (`20261031`).
- Each task currently has one Agent trace, so this cohort cannot estimate Agent-to-Agent missed-link performance.
- The label is a same-task retrieval proxy. It does not establish common actor, campaign, organization, or identity.
- Earlier iterations inspected this cohort. This is development evidence, not independent confirmation. Keep external results as a separate reference.

## One primary change

Keep the v8 Human terminal-output command recovery and the v9 command baseline. Add a training-only IDF-weighted containment score over distinctive artifact-name tokens:

`shared artifact mass / artifact mass of the shorter trace`

Generic tool, shell, and path words are excluded. An unseen test token receives the maximum IDF weight fitted on the training fold. The candidate score is `0.45 × command TF–IDF + 0.55 × artifact containment`. No task ID, label, target, answer, or test-fold statistic enters features or weights.

Candidate generation remains the union of content, sequence, and behavior evidence. Other methods and profile/tensor ablations remain diagnostics; only the command baseline and the new artifact-containment method may be selected.

## Nested selection and acceptance

For each outer task-held-out fold, fit features on the outer training set, obtain inner grouped out-of-fold predictions, and select the threshold there. Select the new method only when one inner threshold has strictly lower overall false-merge rate than the same-fold command baseline while satisfying all of the following:

- Overall and every estimable Human/Human, Human/Agent, and Agent/Agent false-split rate is no higher.
- Every other estimable subgroup error rate is no higher.
- Balanced accuracy, query coverage, Hit@1, MRR, Recall@5, and NDCG@5 are no lower.
- Otherwise, retain the command baseline for that outer fold.

Evaluate the frozen choice once on the outer test tasks. Any outer-fold or aggregate missed-link increase means the method fails this round, even if false merges or ranking improve. Report absolute rates, paired deltas, subgroup results, and task-cluster 95% intervals. Report absent or non-estimable strata as such; do not impute them as zero.

## Outputs

`results/balanced_innovation_v10/results.json` is the canonical summary. Pair-level scores, task metrics, fold thresholds, directional outcomes, and English figures are emitted beside it. Retain failed experiments and their error audits as development history. Do not characterize a development-set result as an external or deployment result.
