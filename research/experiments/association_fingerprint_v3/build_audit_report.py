#!/usr/bin/env python3
"""Summarize all recorded BA pilots without selecting on held-out scores."""
import hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[3];D=R/'research/results/association_fingerprint_v3';E=R/'research/experiments/association_fingerprint_v3'
files=['ba_pilot.json','ba_matched_negatives.json','ba_blend_25.json','ba_blend_50.json','ba_local_alignment.json','ba_action_profiles.json','ba_trees.json','ba_tree_window4.json','ba_tree_window16.json','ba_tree_ablation.json','ba_tree_window_shuffle.json']
out=['# Exploratory BA study audit','','All experiments follow inspection of earlier results. Intervals are unadjusted; variants and held-out scores must not be treated as a fresh confirmatory test. CTH labels are same benchmark task; execution mode and capture path are confounded. The task-held-out CTH galleries are identical across these pilots.','','| Experiment | Method | BA (%) | Difference from its matched evidence baseline (pp; 95% task interval) |','| --- | --- | ---: | --- |']
for name in files:
 d=json.loads((D/name).read_text())
 for r in d['summary']:
  lo,hi=r['paired_task_ci95'];out.append(f"| {name.removesuffix('.json')} | {r['method']} | {r['ba']*100:.2f} | {r['delta_ba']*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}] |")
out+=['','The matched-hard-negative-only baseline drops to 46.80% BA, below the original evidence baseline (63.36%). Its apparent +7.51 pp fingerprint increase is therefore not evidence of a net improvement over the stronger prior baseline. Mixed-negative, action-profile, and local-alignment attempts are retained above.','','## Frozen-family holdouts','','The gated-component shallow-tree family is evaluated with identical candidates and train-side thresholds in each comparison. Within-source windows are chosen on inner folds; source-held-out comparisons use an eight-event window. Known normalized CyBench task-name overlap is purged from outer and inner source-held-out training.','','| Source | Holdout | Evidence BA | Fingerprint BA | Paired delta [95% interval], pp |','| --- | --- | ---: | ---: | --- |']
w=json.loads((D/'ba_tree_nested_multisource.json').read_text())['results'];h=json.loads((D/'ba_leave_source_out_purged.json').read_text())['results']
for source in ('cth','kypo','antieval'):
 for kind,d in [('task or participant',w[source]),('entire source',h[source])]:
  base=d.get('baseline_ba',d.get('evidence_ba'));value=d.get('ba',d.get('fingerprint_ba'));lo,hi=d['paired_cluster_ci95'];out.append(f"| {source.upper()} | {kind} | {base*100:.2f}% | {value*100:.2f}% | {d['delta_ba']*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}] |")
out+=['','CTH source holdout has 175 queries and 40 task units: FSR 85.75% → 72.88%, FMR 10.29% → 13.75%. Its BA increase coexists with substantial remaining missed links. KYPO intervals resample query participants while candidate galleries remain fixed. Matching challenge names are a conservative proxy for overlap; unrecognized task/environment overlap may remain.','', 'Each source-held-out candidate score, threshold, and grouped tree-path contribution is recorded in `ba_leave_source_out_purged_pairs.csv`. Tree-path contributions sum to the model probability and are descriptive model decompositions, not causal attribution.']
(D/'BA-PILOTS.md').write_text('\n'.join(out)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'status':'post-observation exploratory','seed':20261103,'data_sha256':digest(R/'site/data/behavior-trace-dataset-v2.json'),'script_sha256':{str(p.relative_to(R)):digest(p) for p in sorted(E.glob('*.py'))},'result_sha256':{p.name:digest(p) for p in sorted(D.glob('*.json'))}}
(D/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Wrote full BA pilot audit and hashes.')
