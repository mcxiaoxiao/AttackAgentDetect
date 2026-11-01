# Common-log fingerprint pilot v1

Exploratory troubleshooting data; not attack attribution evidence.

- `provenance.json`: frozen protocol/code/input hashes, measurement checks, grouping, admission, fold sizes, and runtime versions.
- `metrics.json` / `metrics.csv`: all 48 prespecified method/view/panel outcomes; JSON also contains task-conditional intervals and application breakdowns.
- `predictions.csv`: one out-of-fold prediction per eligible run, method, view, and panel. Source and task columns are evaluator sidecars, never model features.
- `features.csv`: measured K1--K8 plus evaluator labels and quality/control columns. The fitted model receives only explicitly selected feature columns.
- `run_registry.csv`: 109 original run intervals and inclusion-relevant counts; no raw command streams.
- `parser_audit.json`: event joining and extraction quality counts.
- `source_review.json`: posthoc command-presence diagnostics and reviewed upstream source hashes, not a tuned feature set.

Main interpretation: `research/reports/log-fingerprint-pilot-v1.md`. Raw archive and supporting external source remain in ignored local storage. No simulated measurements, repeated test-set optimization, or same-task-as-campaign labels are used.
