# Human task trace extension v1

Date: 2026-11-02. This extends the public Human-data inventory and adds a small exploratory cross-source same-challenge panel. It does not change the frozen overnight results.

## New Human source

Pinned Zenodo release 10.5281/zenodo.10298187 (CC BY 4.0) contains two KYPO Locust 3302 training cohorts with 52 and 48 trainees. The verified archives total 856,272 bytes. Command and task events were exported from process_mining.csv; each record is grouped by source variant, anonymized sandbox participant, and author-labeled training level. Only explicit bash-command names are adapted to the repository's lossy known-command view; Metasploit and game events are retained in counts but excluded from that common view. Participant IDs are hashed before export.

| Gate | Nonempty task segments | Distinct levels | Levels with >=2 Human participants | Participants |
|---:|---:|---:|---:|---:|
| Any command | 265 | 11 | 10 | 100 |
| >=4 commands | 197 | 10 | 10 | 97 |
| >=8 commands | 146 | 10 | 9 | 82 |
| >=16 commands | 68 | 8 | 7 | 55 |
| >=32 commands | 19 | 6 | 4 | 17 |

These are 265 participant-by-level segments, not 265 independent people or tasks. Every split must group by participant; data1 and data2 task labels remain namespaced separately. The source contains no Agent records, and none of its levels has a matched Agent run. It can support Human-only within-source behavior descriptions, task retrieval, and short-trace sensitivity checks; it cannot increase matched Human/Agent CTF tasks.

## Candidate matched challenge panel

CTH contains Human traces for six CyBench challenge IDs. The pinned Antieval CyBench release contains five matching IDs after removing its explicit `cybench:` prefix and `(hard)` suffix: `back_to_the_past`, `flag_command`, `locktalk`, `lootstash`, and `packedaway`. At the >=8 gate, four challenges provide five eligible Human queries. At the >=4 gate, all five provide seven Human queries and 30 Agent gallery traces. The canonical challenge IDs support a source-qualified task proxy, but exact fixture/version equivalence and capture parity are not established. Any result must be labeled exploratory cross-source same-challenge retrieval and report each task, query count, and source protocol.

This is a concrete path toward an Agent gallery / Human query test. It does not establish a balanced, same-environment Human/Agent collection. The stronger follow-up remains repeated Human and Agent runs on the same pinned challenge version, environment, permissions, and common logger.

## Reproduction and files

`.venv/bin/python research/scripts/build_human_task_supplement_v1.py` validates pinned archive SHA-256 values and rebuilds the task-level sidecar, command-name evidence, length coverage, and public site registry. The source archives are under `research/data/external/kypo-human-v1/`; the pinned source manifest is `research/data/manifests/human_task_kypo_v1.json`.

Source: [Zenodo KYPO behavior dataset](https://zenodo.org/records/10298187); [paper and data description](https://doi.org/10.1016/j.dib.2023.109956).
