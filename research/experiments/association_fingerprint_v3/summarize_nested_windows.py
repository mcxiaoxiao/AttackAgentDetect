#!/usr/bin/env python3
"""Select existing window/depth candidates using each outer fold's inner scores."""
import collections,json,pathlib,warnings
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];D=ROOT/'research/results/association_fingerprint_v3';output={}
for source in ('antieval','kypo'):
 runs={4:json.loads((D/f'ba_tree4_external_{source}.json').read_text()),8:json.loads((D/f'ba_tree_external_{source}.json').read_text()),16:json.loads((D/f'ba_tree16_external_{source}.json').read_text())};nfold=len(runs[8]['folds']); selected=[];test=[];base=[]
 for fold in range(nfold):
  candidates=[]
  for w,run in runs.items():
   for row in run['folds'][fold]['inner_results']:
    if row['method']=='gated_components':candidates.append((row['inner_ba'],-w,-float(row['C']),w,row['C']))
  best=max(candidates);w=best[-2];selected.append({'fold':fold,'window':w,'depth_selector':best[-1],'inner_ba':best[0]})
  test.extend(r for r in runs[w]['query_predictions'] if r['fold']==fold and r['method']=='gated_components')
  base.extend(r for r in runs[8]['query_predictions'] if r['fold']==fold and r['method']=='evidence')
 assert {(r['fold'],r['query']) for r in test}=={(r['fold'],r['query']) for r in base}
 by=collections.defaultdict(lambda:collections.defaultdict(list))
 for method,items in [('fingerprint',test),('evidence',base)]:
  for r in items:by[method][(r['task'],r['participant'] if source=='kypo' else r['task'])].append(r['ba'])
 units=sorted({u for t,u in by['evidence']});tasks=sorted({t for t,u in by['evidence']});arr={}
 for method in ('fingerprint','evidence'):
  mat=np.full((len(units),len(tasks)),np.nan)
  for i,u in enumerate(units):
   for j,t in enumerate(tasks):
    if (t,u) in by[method]:mat[i,j]=np.mean(by[method][t,u])
  arr[method]=mat
 rng=np.random.default_rng(20261103);ix=rng.integers(0,len(units),(5000,len(units)))
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  val=float(np.nanmean(np.nanmean(arr['fingerprint'],axis=0)));b=float(np.nanmean(np.nanmean(arr['evidence'],axis=0)));vboot=np.nanmean(np.nanmean(arr['fingerprint'][ix],axis=1),axis=1);bboot=np.nanmean(np.nanmean(arr['evidence'][ix],axis=1),axis=1)
 output[source]={'ba':val,'baseline_ba':b,'delta_ba':val-b,'paired_cluster_ci95':np.nanquantile(vboot-bboot,[.025,.975]).tolist(),'queries':len(test),'independent_units':len(units),'selected':selected}
cth_runs={4:json.loads((D/'ba_tree_window4.json').read_text()),8:json.loads((D/'ba_trees.json').read_text()),16:json.loads((D/'ba_tree_window16.json').read_text())}
cth_test=[];cth_base=[];cth_selected=[]
for fold in range(5):
 choices=[]
 for w,run in cth_runs.items():
  for row in run['fold_choices'][fold]['inner_results']:
   if row['method']=='gated_components':choices.append((row['inner_ba'],-w,-float(row['C']),w,row['C']))
 best=max(choices);w=best[-2];cth_selected.append({'fold':fold,'window':w,'depth_selector':best[-1],'inner_ba':best[0]})
 cth_test.extend(r for r in cth_runs[w]['task_predictions'] if r['fold']==fold and r['method']=='gated_components')
 cth_base.extend(r for r in cth_runs[8]['task_predictions'] if r['fold']==fold and r['method']=='evidence')
a={r['task']:r['ba'] for r in cth_test};b={r['task']:r['ba'] for r in cth_base};assert set(a)==set(b) and len(a)==40
diff=np.array([a[t]-b[t] for t in sorted(a)]);rng=np.random.default_rng(20261103);boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
output['cth']={'ba':float(np.mean(list(a.values()))),'baseline_ba':float(np.mean(list(b.values()))),'delta_ba':float(diff.mean()),'paired_cluster_ci95':np.quantile(boot,[.025,.975]).tolist(),'independent_units':40,'selected':cth_selected}
(D/'ba_tree_nested_multisource.json').write_text(json.dumps({'status':'post-observation exploratory; fixed gated-components family, select window and depth from inner folds','results':output},indent=2)+'\n');print(json.dumps(output,indent=2))
