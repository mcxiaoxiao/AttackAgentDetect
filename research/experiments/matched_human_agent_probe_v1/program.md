# Same-challenge Human query / Agent gallery probe

Date: 2026-11-02. Exploratory follow-up after the frozen overnight study. Test whether matrix fusion improves same-challenge retrieval when a CTH Human trace queries an Antieval Agent gallery on a normalized CyBench challenge ID.

Crosswalk: exact benchmark slug after removing the Antieval cybench: prefix and (hard) suffix. Keep all five shared IDs; use common-command traces with at least four commands. The same challenge ID is a task proxy only; fixture/version equivalence and capture parity remain unverified.

Design: leave one of five challenge IDs out as the query task. Fit score normalization and logistic fusion on Agent-only Antieval pairs from the other four task IDs. Each query ranks one same-task Agent candidate and one Agent negative from each of the other four tasks. Twenty repeated gallery draws per Human query; same candidates for all methods. No parameter tuning on the query challenge.

Primary metric: task-macro Hit@1. Secondary: MRR and pairwise AUC. Average within task before paired task-cluster bootstrap (10,000 draws). Report all tasks and methods. Fixed scope: five shared challenges, seven Human traces with >=4 commands, no new downloads, CPU-only.
