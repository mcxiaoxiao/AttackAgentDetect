# Independent CTH Agent–Agent task confirmation

## Purpose and freeze point

This is a task-held-out confirmation of whether the B/S fingerprint adds association evidence for Agent–Agent pairs. It follows diagnostics on the development cohort and does not reuse their task groups. The independent unit is the source-qualified CTF task group; the label remains a same-challenge retrieval proxy, not an attack, actor, campaign, or organization identity label.

The task manifest is frozen before fitting the confirmation models. Candidate tasks come from the pinned Cyber Task Horizons CTF records and must have at least two distinct Agent runs with complete B/S features under a four-event window, two-event minimum, and 128-event cap. We exclude every task group with any Human run, every task in previous CTH mixture/cross-mode or Agent-only retrieval predictions, every challenge slug used in the prior Antieval CyBench gallery, and every test sequence exactly duplicated in a development trace. The resulting cohort is AA-only and does not support HH or HA estimates.

The audit and complete frozen task/trace list are in `research/results/matched_fingerprint_v2/independent_cth_task_manifest.json`. The audit records the exact historical prediction files checked, input hashes, excluded groups, and usable trace IDs.

## Fixed models and comparison

Training uses only the existing 17-task CTH mixed-mode development cohort. There are 85 development traces and the original three task folds. Model settings are selected without confirmation outcomes by taking the majority setting in the three frozen v2 outer-fold audits:

- Fingerprint representation: four-event B/S windows, mean and standard-deviation pooling, command-conditioned fusion.
- SVM: RBF, C=1; decision tree: maximum depth 3; XGBoost: depth 3, with all other settings held to the v2 runner defaults.
- Each classifier is trained on the same development pair list and the same pair labels across feature versions.

Four input versions are compared for each classifier: `E` (command and action-bigram similarities), `E+Args` (E plus value-redacted command syntax), `E+B/S`, and `E+Args+B/S`. This makes the original-evidence contrast explicit while checking whether recovered arguments change the fingerprint increment. For a trace pair lacking a required B/S block, the fingerprint version uses its corresponding E or E+Args decision.

Each version's threshold is calibrated on one-gallery-per-query, three-fold out-of-fold predictions from the 17 development tasks. The threshold maximizes recall under validation FMR ≤10%, and is then frozen. The external outcome is not used for threshold, representation, feature, or parameter selection.

## Confirmation candidate panel

The 26 frozen task groups are assigned to three candidate partitions by a deterministic SHA-256 rule. For each eligible Agent trace used as a query, each of 100 seeded gallery draws contains one different Agent trace from the same task and one Agent trace from each of three different tasks in the same partition. All classifiers and feature versions receive the identical pair manifest. The dataset is therefore 100% Agent by role, and every evaluated pair is AA.

Primary rate outcomes are task-macro FMR, FSR, and BA at the frozen thresholds. Pairwise score ranking (AUC, MRR, and Hit@1) is a threshold-independent diagnostic. Confidence intervals resample the 26 task groups in 10,000 paired draws; repeated galleries are averaged within task and do not count as new independent tasks.

## Interpretation limits

This is an independent task-group confirmation within the CTH source, not a new-source replication. It evaluates only Agent–Agent CTF same-task association. It cannot establish performance for Human–Human or Human–Agent pairs, and CTF task identity cannot establish actor, activity, campaign, or attribution identity. Any positive AA result would be a bounded signal; a mixed-role or cross-source claim would still require new data.
