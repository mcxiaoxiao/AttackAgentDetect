# Execution Behavior Trace Dataset v2

A source-qualified command-view dataset with current analysis outputs.

## Contents

- `traces/trace-catalog-v2.json`: normalized records and ordered command/action sequences where source terms allow.
- `traces/trace-catalog-v2.csv`: the same records; sequence and count fields are JSON-encoded cells.
- `relations/human-agent-cybench-crosswalk.csv`: candidate shared challenge slugs, not verified fixture equivalence.
- `analysis/analysis-v2.json`: recognition, CTF mixture, short-trace, matched challenge, and pair-evaluation summaries.
- `analysis/ctf-behavior-trace-pairs.csv` and `ctf-behavior-trace-summary.json`: the 97-record CTF pair-evaluation panel.
- `protocols/`: experiment reports documenting methods and limits.

## Dataset composition

The catalog has 1,407 source-qualified records: 1,142 from Cyber Task Horizons, Antieval CyBench, and AgentTrace, plus 265 Human-only KYPO task segments. It contains 344 Human and 1,063 Agent records across four source families.
Sequences contain normalized command names, not raw terminal logs. CTH, AgentTrace, and KYPO include the complete normalized sequence. Antieval CyBench does not state a source license; those records retain only the already-published command prefix (up to 24 entries), aggregates, and upstream locator. Review the pinned source terms before reuse or redistribution.

## How to use

Use `source_dataset` plus `task_group` as the grouping key; do not merge same-named tasks across sources by default. Keep every KYPO record from a participant in one split. The five-row CyBench crosswalk is exploratory: challenge version, fixture, and capture parity have not been verified.
Suitable analyses include within-source same-task retrieval, Human-only task-level variation, command-view sensitivity, and source-held-out pilots. The records do not label real campaigns, operators, or person identities. The AIT recognition experiment is a separate analysis panel and is not part of the 1,407 trace catalog.

## License and attribution

No single license applies to the whole bundle. Use each row's `license`, `source_url`, `source_revision`, and `record_locator`. `not-stated-by-source` is not permission to reuse or redistribute. KYPO and CTH are CC BY 4.0; AgentTrace includes Apache 2.0 and upstream task terms. Preserve upstream attribution.

## Citation

AttackAgent Detect. *Execution Behavior Trace Dataset v2: Source-qualified command-view records and analysis*. 2026. Cite the upstream source for each subset used.
