#!/usr/bin/env python3
"""Task-cluster error contrasts in the fixed 50% mixture; exploratory diagnostics."""
import collections,csv,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];OUT=ROOT/'research/results/behavioral_fingerprint_v2/refinement';OUT.mkdir(parents=True,exist_ok=True)
rows=[r for r in csv.DictReader(open(ROOT/'research/results/behavioral_fingerprint_v2/cth_mixture/pair_predictions.csv')) if float(r['fraction'])==.5 and r['method'] in ('command_cosine','command_plus_BS')]
d=collections.defaultdict(list)
for r in rows:d[(r['method'],r['query_task'],r['pair_type'],r['same_task']=='True')].append(int(r['pred_link']))
tasks=sorted({r['query_task'] for r in rows})
def ci(x):
 a=np.array(x);return np.quantile(np.random.default_rng(20261103).choice(a,(10000,len(a)),replace=True).mean(1),[.025,.975]).tolist()
summary=[];deltas=[]
for typ in ('HH','HA','AA'):
 for pos,metric in ((True,'FSR'),(False,'FMR')):
  for method in ('command_cosine','command_plus_BS'):
   vals=[(1-np.mean(d[(method,t,typ,pos)])) if pos else np.mean(d[(method,t,typ,pos)]) for t in tasks]
   summary.append({'method':method,'type':typ,'metric':metric,'value':float(np.mean(vals)),'ci95':ci(vals)})
  vals=[]
  for t in tasks:
   a,b=[np.mean(d[(m,t,typ,pos)]) for m in ('command_cosine','command_plus_BS')];vals.append(a-b if pos else b-a)
  deltas.append({'type':typ,'metric':metric,'tasks':len(tasks),'delta':float(np.mean(vals)),'ci95':ci(vals)})
vals=[np.mean(d[('command_cosine',t,'HH',True)])-np.mean(d[('command_cosine',t,'HA',True)]) for t in tasks]
gap={'comparison':'HA_minus_HH_FSR_command','tasks':len(tasks),'delta':float(np.mean(vals)),'ci95':ci(vals)}
(OUT/'error_strata.json').write_text(json.dumps({'summary':summary,'paired_deltas':deltas,'mode_gap':gap,'protocol':'50% role mixture, fixed original predictions; average allocations within query task before 10000 task-cluster resamples; all HH/HA/AA FSR and FMR contrasts retained; exploratory intervals unadjusted for multiple comparisons'},indent=2)+'\n')
