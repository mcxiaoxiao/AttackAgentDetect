# Exact CTH mixture: original baseline protocol

This exploratory rerun uses the frozen 17-task CTH cohort and the original task-held-out FMR ≤ 10% training threshold. Each of 170 coupled draws excludes one task, rotating the exclusion evenly, and assigns the 32 query/gallery roles across 16 tasks to exactly 0%, 25%, 50%, 75%, or 100% Agent roles. Candidate tasks and role order are shared across concentrations. Each query uses one same-task and three different-task candidates from its held-out fold. The original report used five candidates and realized 23.5% and 76.5% at its intermediate targets, so this result is a new controlled rerun rather than a relabeling of the old scores.

| Agent share | Command BA | Action-bigram BA |
| --- | ---: | ---: |
| 0% | 53.59% | 57.40% |
| 25% | 51.72% | 54.34% |
| 50% | 49.70% | 51.31% |
| 75% | 48.23% | 48.62% |
| 100% | 46.50% | 46.48% |

For 100% minus 0%, the task-paired BA change is −7.10 percentage points (95% task-cluster interval −15.45 to +0.06) for commands and −10.92 points (−20.61 to −3.06) for action bigrams. False-split rates increase while false-merge rates decrease; these results do not establish that every type of error increases. All labels are same-benchmark-task proxies, and Human and Agent capture paths differ. The fingerprint fusion increment has not yet been evaluated under this protocol.

Reproduce with `.venv/bin/python research/experiments/association_fingerprint_v3/run_exact_mixture.py`. The JSON contains the task-level averages and paired endpoint intervals; repeated draws are averaged before task-cluster resampling.
