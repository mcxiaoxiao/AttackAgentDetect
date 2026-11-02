# Final run reproduction entry

The planned run is complete. This entry is now a reproducibility note, not a request to launch unattended work.

From the repository root:

```sh
.venv/bin/python research/experiments/overnight_fingerprint_v1/run.py all
.venv/bin/python research/experiments/overnight_fingerprint_v1/build_figures.py
```

The runner checks the frozen input hashes in `protocol.json` and writes to `research/results/overnight_fingerprint_v1/`. No new data downloads or paid API calls are used. The output retains the exact folds, predictions, eligibility records, metrics, figure sources, and run ledger.

The mixed Human/Agent matrix increment remains unavailable under the frozen gate: only one CTH task has the required 16-command evidence, while 15 are required. Do not replace this result with Agent-only retrieval or describe the matrix as having corrected mixture errors.
