# Overnight fingerprint study v1

Status: prepared, not launched. This is an eight-hour research and delivery runbook, not an already implemented full experiment runner.

- [Complete overnight plan](program.md): stages, fixed comparisons, failure rules, new manuscript and page acceptance criteria.
- [Frozen machine-readable protocol](protocol.json): input hashes, seeds, eligibility and compute limits.
- [Actual eligibility audit](../../results/overnight_fingerprint_v1/preflight.json): primary AIT recognition and CTF baseline mixture gates pass; the Human/Agent matrix-mixture increment gate fails.
- [Entry prompt for the execution turn](START.md).

Reproduce the eligibility audit from the repository root:

```sh
.venv/bin/python research/experiments/overnight_fingerprint_v1/preflight.py
```

This parses the pinned local AIT archive and verifies the normalized CTF catalog. It fits no model, overwrites no historical experiment output, and starts no background process. The first implementation stage must build and check `run.py` before any full batch is started. Do not confuse a passed eligibility check with a successful fingerprint method.
