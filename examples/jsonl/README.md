# JSONL interface smoke example

These short records are hand-written placeholders for checking the CLI schema and output shape. They are **not** experimental data, malware traces, or performance evidence. Replace them only with traces you are authorized to process and labels that were assigned independently of the observed features.

From the repository root:

```bash
PYTHONPATH=. .venv/bin/python -m attackagent.jsonl_cli \
  --train examples/jsonl/train.jsonl \
  --queries examples/jsonl/queries.jsonl \
  --candidates examples/jsonl/candidates.jsonl
```

Training accepts only labeled `human` and `agent` rows. A `script` query is explicitly abstained because this release has no script training data. Scores are uncalibrated; candidate ranks are within-benchmark research rankings and do not estimate APT or activity identity.
