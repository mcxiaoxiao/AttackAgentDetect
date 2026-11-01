# Overnight study: preparation outputs

Only eligibility and split preparation have run. No new classification, matrix, mixture or retrieval performance results exist in this directory yet.

- `preflight.json`: frozen-input check, actual cohort counts and execution gates.
- `ait_matrix_eligibility.csv`: 89 admitted run IDs and quality counts.
- `ait_matrix_exclusions.csv`: 6 records from the previous 95-run pilot that fail the new minimum-evidence rule.
- `task_folds.csv`: fixed task folds for CTF mixture, source-separated Agent retrieval and AIT leave-one-task-out.

The input sequences and source metadata remain separate; these evaluator-side IDs and labels must never become model features. Repeated draws, windows and session pairs are not independent data points. The execution phase must add a real run ledger, predictions and metrics rather than reinterpret the preparation counts as performance.
