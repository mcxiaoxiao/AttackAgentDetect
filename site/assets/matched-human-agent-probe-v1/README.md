# Matched Human-query / Agent-gallery probe

Five normalized CyBench challenge IDs link CTH Human queries to Antieval Agent candidates. At the four-command gate, the fixed cross-source panel contains seven Human queries and 30 Agent traces. Leave-one-challenge-out command cosine Hit@1 is 44.0%; command + matrix DTW is 42.5%, paired delta -1.5 pp (95% task-cluster interval -4.5 to 0.0; five task clusters). This is an exploratory same-challenge proxy; fixture-version equivalence and capture parity are unverified.

Reproduce with `.venv/bin/python research/experiments/matched_human_agent_probe_v1/run.py`.
