# Overnight fingerprint study v1

Status: **executed**. The frozen protocol in `protocol.json` was used for three completed tracks: AIT mode-label recognition, CTH CTF mixture retrieval, and source-local Agent-only retrieval (CTH and Antieval Cybench). A post-observation two-way source-transfer analysis is reported separately as exploratory.

- [`program.md`](program.md): fixed design, gates, outputs, and interpretation boundaries.
- [`protocol.json`](protocol.json): pre-run protocol and input hashes. Its `prepared_not_launched` status is the original preregistration snapshot; execution status is recorded separately below.
- [`START.md`](START.md): reproduction commands and the final-run scope.
- [`../../results/overnight_fingerprint_v1/README.md`](../../results/overnight_fingerprint_v1/README.md): eligibility, completed results, and interpretation.
- [`../../paper/human-agent-fingerprint-overnight.tex`](../../paper/human-agent-fingerprint-overnight.tex): concise method/data/results manuscript source.

Run from the repository root:

```sh
.venv/bin/python research/experiments/overnight_fingerprint_v1/run.py all
.venv/bin/python research/experiments/overnight_fingerprint_v1/build_figures.py
```

`run.py all` verifies the frozen input hashes before writing outputs. Existing predictions and metrics are reproducible under the frozen seed; the run ledger preserves earlier implementation failures as well as completed attempts. Those bookkeeping failures are not model results. No Git action is part of this workflow.
