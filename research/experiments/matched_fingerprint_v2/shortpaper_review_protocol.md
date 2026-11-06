# Same-challenge retrieval short-paper reanalysis

## Frozen question and limits

This bounded reanalysis asks (1) how the recorded Human/Agent role composition in fixed CTH query galleries changes same-task retrieval errors and score ranking, and (2) whether the B/S fingerprint adds evidence beyond the existing command/action-bigram similarities under a matched candidate design. “Same task” and “same challenge” remain retrieval proxies. Neither establishes the same operator, Agent instance, campaign, or real-world activity. CTH role contrasts do not isolate execution behavior from collection interface: Human terminal input and Agent tool calls are recorded differently, and the source does not establish that Human participants never used AI.

## Analysis batch and stop rule

The batch is fixed at the existing 17-task CTH folds, frozen 54,400-pair five-share gallery, original fold-specific XGBoost depth/representation, and five B/S sequence-shuffle seeds `20261105`–`20261109`. No NYU outcome, test task, test threshold, or model selection is used to alter the representation or configuration. A shuffle permutes each trace's normalized command-head and mapped-action sequence with the same permutation; command evidence, sequence-length controls, raw chunk counts, labels, pair list, and thresholds remain otherwise fixed. Vocabulary, imputation, feature scaling, and thresholds are fitted inside the training portion of each fold.

Primary outcomes are query-task macro FMR/FSR/BA at training-fold thresholds. Secondary outcomes are gallery Hit@1 and gallery AUC. The primary mechanism contrasts are `E+records+LCS` versus `E+records`, and real B/S versus the same record-count control; shuffled B/S is a negative control for order-sensitive information. The five shuffled-seed results are reported individually. Intervals resample the 17 query-task clusters and do not treat repeated galleries or pairs as independent. This is one diagnostic batch; no follow-up tuning is planned if results are inconclusive.

## Frozen operating-point diagnostics

Cosine ranking metrics, task-cluster leave-one-task-out sensitivity, the retrospective maximum-recall reading at observed FMR ≤10%, and the three-negative no-match pressure check use existing scores and pair lists only. The 10% reading is descriptive and test-score-selected within each outer fold and share; it is not an independent threshold validation. The no-match check removes the positive from each original gallery and asks whether any of its three original negatives crosses its arm's frozen threshold.

For exact normalized-command-sequence overlap, identify held-out traces whose full command-head sequence appears among the corresponding outer fold's training tasks. Remove the entire four-candidate gallery if the query or any candidate is flagged; do not replace candidates or refit the model. Report the retained gallery/task counts and both full and filtered estimates. Apply the same task-complete-gallery rule to any exact CTH-training/NYU sequence overlap. A task-slug match is audited separately and is not treated as proof that two source-specific tasks are identical.

## Inputs and provenance

The CTH mixed cohort and fold assignments come from `matched_fingerprint_v1/run.py` and `overnight_fingerprint_v1/task_folds.csv`; exact-mixture cosine thresholds and task means come from `association_fingerprint_v3/exact_mixture.json`. Frozen matched CTH XGBoost scores come from the v1 per-fold prediction files. CTH Agent-only and NYU Agent-only confirmations are summarized from their already-frozen manifests, analyses, and pair predictions. AIT remains an appendix result. The runner writes source hashes, protocol hash, environment, fold configuration, pair counts, thresholds, and all arm predictions to `results/matched_fingerprint_v2/shortpaper_mechanism_20261105/`.

## Predeclared interpretation

Do not call the B/S block a robust sequence contribution if its point estimate depends on learner, source, sequence length, or uncontrolled trace volume. A reduction in one error rate accompanied by an increase in the other is an error trade-off. A positive ranking change is described separately from a thresholded decision change. If deletion of a task changes the mixture direction, identify the influential task and narrow the statement. If real B/S does not exceed the record-count or shuffled-order control with an interval excluding zero, report that the current data do not isolate an order-specific increment.
