# CTH cross-task model-snapshot retrieval

## Question and status

This frozen exploratory follow-up tests whether B/S adds retrieval value when the label is closer to an execution-style fingerprint: whether two Agent traces came from the same recorded model snapshot, despite coming from different CTF tasks. This is a trace-source retrieval proxy. It does not identify a human participant, a persistent Agent instance, an operator, an activity, or an attack campaign.

The CTH catalog contains 107 CTF task groups with exactly one trace from each of three model snapshots (`claude-opus-4.6`, `claude-sonnet-4.6`, and `gpt-5.5`). All three snapshots use the same recorded Inspect-ReAct collection interface. Participant IDs are blank in the catalog. The primary cohort is the subset of task groups in which all three model traces contain at least eight normalized command heads; this yields 47 task groups and permits an equal-length prefix test. The same 47 groups are used for the full-trace sensitivity.

## Frozen protocol

Assign task groups to three folds by sorting `SHA256("cth-snapshot-v1:20261106:" + task_group)` and assigning in round-robin order. In each outer fold, train only on task groups outside that fold. Training examples are all unordered pairs of traces from different training tasks; the target is whether the recorded model snapshot is the same. No within-task pairs enter training or testing.

For each held-out query task and each different held-out gallery task in the same fold, evaluate all three query snapshots against the gallery task's three traces. Each gallery therefore has exactly one same-snapshot candidate and two different-snapshot candidates. Query and gallery tasks are both absent from that fold's training data. The gallery list and labels are identical across feature arms and classifiers.

The primary view uses exactly the first eight command heads and their mapped actions for every trace. The full-trace view uses all normalized command heads, with log trace count as an explicit control. Compare four fixed feature arms: (1) command-head and action-bigram cosine evidence plus trace-count features (`E+length`); (2) the same plus normalized command LCS; (3) the same plus real B/S summaries; and (4) the same plus B/S summaries after shuffling each trace's commands and actions in lockstep. The five fixed shuffle seeds are 20261106--20261110. B/S uses four-event windows, mean/standard-deviation pooling, conditional fusion, and training-fold-only imputation/scaling. No source model name, task name, file path, or raw argument enters the features.

Use the existing fixed learner families and parameters: RBF-SVM C=1, decision-tree depth 3, and XGBoost depth 3. The primary outcomes are gallery Hit@1 and gallery AUC; MRR is also reported. There is no decision threshold. Report all three learners, both views, every shuffle seed, and all declared arms. The primary comparison is real B/S versus `E+length` in the equal-eight-command view. A common gain requires positive paired changes for all three learners; any uncertainty remains visible.

Uncertainty uses 10,000 paired two-way task-cluster bootstrap draws. Within each fold, query-task and gallery-task IDs are resampled independently; a task-pair gallery receives the product of its query- and gallery-task multiplicities. The diagonal is excluded. Fold means are weighted by the number of held-out query tasks. This preserves task reuse in both gallery roles while keeping all arms paired. This analysis is exploratory and does not replace the main same-task retrieval estimand.

## Interpretation rule

If B/S improves same-snapshot retrieval under the fixed-length view, report it as evidence of source-snapshot information in these three CTH Agent configurations, not as same-activity association or actor identification. If the gain appears only on full traces, treat trace volume as a plausible explanation. If real B/S does not outperform the shuffled-order control, do not attribute any gain to sequence order. Do not select a new window, learner, threshold, or task subset after viewing the results.

## Post-hoc fusion sensitivity

The initial run uses conditional fusion, matching the majority setting in the earlier CTH fold selections. After inspecting it, a separate direct-fusion sensitivity is run because conditional fusion gates B/S differences by command-head cosine, which can suppress them on deliberately cross-task galleries. The direct run keeps the same 47 tasks, folds, pairs, arms, learners, parameters, bootstrap, and shuffle seeds. It is reported alongside the initial run as a post-hoc sensitivity and cannot replace the primary result or be described as confirmatory.
