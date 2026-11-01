# Fixed common-log fingerprint pilot

Run from the repository root with the existing environment:

```sh
.venv/bin/python research/experiments/log_fingerprint_pilot_v1/run.py
```

Input: the public AIT/AECID archive stored locally at `research/data/external/ait-shell-pilot-v1/data.zip`, with the content hash fixed in `program.md` and `run.py`. Raw logs are not redistributed. No upstream code is executed. Dependencies are the repository's NumPy/SciPy/scikit-learn environment.

Read `program.md` before interpreting results. This is a completed exploratory batch, without hyperparameter search or an independent final-test claim. The eight scalar features did not discriminate reliably in this pilot. A lexical baseline scored higher, with posthoc evidence of possible shell/collection confounding. No association-benefit experiment was run.

Findings: [`../../reports/log-fingerprint-pilot-v1.md`](../../reports/log-fingerprint-pilot-v1.md). Complete aggregate results: `research/results/log_fingerprint_pilot_v1/`.
