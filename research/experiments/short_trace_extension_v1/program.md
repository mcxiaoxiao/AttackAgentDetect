# Short-trace retrieval extension, 2026-11-02

Exploratory follow-up after the overnight results were observed. Objective: expand eligible CTF task coverage and test whether retaining separate matrix dimensions improves retrieval over scalar fusion. No claim of an independent confirmatory test.

Inputs: existing pinned behavior_corpus_v1; no new downloads in this extension. Sources remain separate. Use Agent-only CTF records with >=8 retained command heads, maximum 128, four-command windows (tail >=4). Exact command sequences are deduplicated within each source/task; test traces identical to a training trace are purged. Tasks are assigned deterministically to three folds before pairs/candidates; all source metadata are excluded from features.

Prespecified comparisons: command cosine; command + length; command + whole-session K1–K8 differences; command + 24 window-summary differences; command + shuffled-window-summary differences. Session scaling and pair classifiers are fit on training tasks only. Logistic regression C=1, balanced classes, maximum 500 positive and 500 negative training pairs per fold; no test-based tuning or winner selection.

Each query ranks one same-task candidate and four different-task candidates; 20 common repeated candidate draws per query. Primary outcome: task-macro Hit@1, secondary MRR and query pairwise AUC. Ties receive expected rank credit. Average all draws/queries within task before paired task-cluster bootstrap (10,000 resamples). Report every comparison, source separately, coverage and exclusions. Repeated draws do not add independent tasks. Stop after this fixed batch (local CPU budget 15 minutes; no paid API calls). Preserve original study unchanged.
