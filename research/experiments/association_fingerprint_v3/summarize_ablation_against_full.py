#!/usr/bin/env python3
"""Paired task-bootstrap comparison against the fixed-window full B+S model."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
result = ROOT / 'research/results/association_fingerprint_v3/ba_tree_ablation.json'
out = ROOT / 'research/results/association_fingerprint_v3/ba_tree_ablation_against_full.json'
data = json.loads(result.read_text())
by = {}
for row in data['task_predictions']:
    by.setdefault(row['method'], {})[row['task']] = row['ba']
methods = ('evidence', 'length_control', 'gated_S', 'gated_B', 'gated_components')
tasks = sorted(by['gated_components'])
assert len(tasks) == 40 and all(set(by[m]) == set(tasks) for m in methods)
rng = np.random.default_rng(20261103)
indices = rng.integers(0, len(tasks), (10000, len(tasks)))
full = np.asarray([by['gated_components'][t] for t in tasks])
summary = []
for method in methods:
    vals = np.asarray([by[method][t] for t in tasks])
    diff = vals - full
    ci = np.quantile(diff[indices].mean(axis=1), [.025, .975])
    summary.append({'method': method, 'ba': float(vals.mean()),
                    'delta_vs_full': float(diff.mean()), 'paired_task_ci95': ci.tolist()})
out.write_text(json.dumps({'protocol': 'CTH task-held-out, fixed 8-event windows; paired task-cluster bootstrap against full B+S; exploratory and unadjusted',
                           'source': result.name, 'seed': 20261103, 'task_count': len(tasks),
                           'bootstrap_samples': 10000, 'summary': summary}, indent=2) + '\n')
print(json.dumps(summary, indent=2))
