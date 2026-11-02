# Overnight fingerprint v1 results

Status: **complete for the frozen, eligible tracks**. This directory contains actual predictions and measured outcomes, not only preflight counts. The exact input hashes and split/feature protocol are recorded in `recognition/provenance.json`, `mixture/provenance.json`, and `retrieval/provenance.json`.

## What ran

- AIT Human/Agent label recognition: 89 eligible audit-log runs (45 Human, 44 Agent), 12 task groups; fixed out-of-fold predictions for task, source, application, and crossed task/source holdouts.
- CTH mixture retrieval: 17 task groups, 85 records; 100 repeated role allocations at five Agent concentrations; command cosine and action-bigram cosine; pair types HH/HA/AA.
- Agent-only task retrieval: CTH (121 traces / 48 tasks, five candidates) and Antieval Cybench (31 / 12, three candidates), each with a paired command-only and matrix-fusion comparison.
- Exploratory two-way source transfer: train on one Agent-only source and evaluate source-native task ranking in the other. This was added after the primary tracks were observed, is explicitly exploratory, and does not link task IDs across sources.

## Main measured outcomes

- AIT task-held-out BA: command TF--IDF 96.8%; whole-session K1--K8 55.2%; window-summary matrix 56.3%; matrix DTW 3-NN 53.3%. Window summary minus whole-session K1--K8 is +1.1 percentage points (95% task-bootstrap interval --14.3 to +15.4; 12 tasks). Under source holdout, window summary falls to 39.9% while command TF--IDF is 82.6%.
- CTH composite mixture outcome: using target Agent-role shares 0/25/50/75/100% (the 25% and 75% targets realize as 23.5% and 76.5% across 34 integer role slots), command-cosine task-macro BA falls 53.8% to 46.0% (paired change -7.73 pp; 95% paired task-cluster interval -16.02 to -0.42; 17 tasks). Action-bigram BA falls 57.3% to 46.0% (-11.34 pp; interval -20.84 to -3.26). FMR falls while FSR rises, shifting the errors toward false splits rather than showing more false merges.
- Agent-only retrieval: matrix fusion changes CTH Hit@1 by +0.35 pp (95% task-bootstrap interval --3.82 to +4.17; 48 tasks) and Antieval by +2.08 pp (--6.25 to +12.50; 12 tasks). Neither interval excludes zero.

These outcomes are task/source-qualified benchmark proxies. They do not establish same-actor, same-attack, or same-campaign relationships. The CTH Human and Agent capture interfaces differ. The mixed Human/Agent matrix-increment track was **not run**: only one eligible 16-command task group was found, against a frozen minimum of 15.

## Key outputs

- `recognition/metrics.json`, `recognition/predictions.csv`, `recognition/folds.csv`
- `mixture/metrics.json`, `mixture/composite-balance.json`, `mixture/pair_type_task_metrics.json`, `mixture/pair_predictions.csv`
- `retrieval/metrics.json`, `retrieval/cross_source_metrics.json`, `retrieval/predictions.csv`
- `figures/`: publication PDF and PNG charts built from the metrics above
- `run_ledger.csv`: completed stages and early implementation failures. The latter are bookkeeping failures, not performance observations.
- `execution_status.json`: final runner status.

Reproduce from the repository root with the commands in `research/experiments/overnight_fingerprint_v1/START.md`.
