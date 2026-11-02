# Short trace and matched challenge follow-up

Date: 2026-11-02. These exploratory follow-ups were added after the initial overnight results. They use pinned local records and make no claim of independent external confirmation.

## Why change the trace gate

The initial fingerprint used eight-command windows and required at least 16 retained commands so each trace had two windows. That was a design choice for observing change across windows, not a universal evidence requirement. It excluded most of the Antieval CyBench sessions. This follow-up uses four-command windows and an eight-command minimum, keeps the 8-dimensional feature definition unchanged, and tests all sessions under the same candidate and task-holdout protocol. Exact command sequences are deduplicated within a task, and test sequences duplicated in training are purged.

| Source | Raw Agent CTF records | Eligible traces after length and deduplication | Tasks | Command Hit@1 | Command + 24-value matrix Hit@1 | Paired change (95% task-cluster CI) |
|---|---:|---:|---:|---:|---:|---:|
| CTH | 415 | 219 | 86 | 57.1% | 53.5% | -3.57 pp (-6.82, -0.43) |
| Antieval CyBench | 189 | 120 | 27 | 49.7% | 46.8% | -2.89 pp (-7.50, +1.79) |

Each query ranks one same-task trace against four different-task traces. Candidate draws are repeated 20 times, but intervals resample task groups after first averaging within task. In CTH, MRR changes by -3.09 pp (95% CI -5.44 to -0.84) and pairwise AUC by -3.95 pp (-6.66 to -1.42). In Antieval, MRR changes by -3.23 pp (-6.68 to +0.16) and pairwise AUC by -4.23 pp (-7.62 to -1.02). The matrix does not improve these Agent-only results under the broader short-trace gate; its ranking signal is slightly worse in CTH, while Antieval Hit@1 remains uncertain.

The extension uses 86 and 27 task clusters, improving independent-task coverage from the previous 48 and 12 task panels. A four-command window measures only local behavior and is not directly interchangeable with the original two-window 8-command fingerprint. This is a planned sensitivity panel, not a replacement score selected for a favorable result.

## Direct same-challenge Human query / Agent gallery

An exact normalized challenge-slug join between CTH Human CyBench and Antieval Agent CyBench gives five candidate challenges. At the four-command gate, the panel has seven Human queries and 30 Agent gallery traces. We leave one challenge out at a time; the matrix score is trained on Agent-only pairs from the other four challenges. Each Human query ranks one same-challenge Agent trace and one trace from each of the four other challenges, with 20 repeated galleries.

| Method | Task-macro Hit@1 | Paired change vs. command (95% task-cluster CI) |
|---|---:|---:|
| Command cosine | 44.0% | reference |
| Command + matrix DTW | 42.5% | -1.5 pp (-4.5, 0.0) |

The paired MRR change is -1.0 pp (95% CI -2.67 to 0.0); pairwise AUC changes by -0.75 pp (-1.75 to 0.0). This small panel gives no indication that the matrix helps Human-query-to-Agent-gallery retrieval. Its five task clusters make interval estimates coarse, and matching by the challenge slug does not verify fixture/version or capture equivalence. Treat these as exploratory same-challenge task-retrieval results, not real activity attribution or a controlled execution-mode comparison.

## What would improve the experiment

The data now support two honest directions. First, keep the 16-command/two-window analysis as the longer-trace panel and report four-command windows as a prespecified local-behavior sensitivity panel; do not search more window lengths until one produces a positive matrix result. Second, collect repeated Human and Agent runs on the same CyBench challenge version, environment, permissions, and logger. The current five-challenge cross-source panel is useful for debugging candidate retrieval and feature stability, but it cannot separate execution mode from source and capture format. An evaluation with at least 15 independent matched tasks, repeated Human and Agent sessions per task, and participant/model holdouts is needed before claiming a reliable correction benefit.

Reproduce with:

```sh
.venv/bin/python research/experiments/short_trace_extension_v1/run.py
.venv/bin/python research/experiments/matched_human_agent_probe_v1/run.py
```
