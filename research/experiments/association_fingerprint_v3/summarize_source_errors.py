#!/usr/bin/env python3
"""Audit held-out-source errors and exact tree-path score decomposition."""
import collections,csv,hashlib,json,pathlib,warnings
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];D=ROOT/'research/results/association_fingerprint_v3'
result=json.loads((D/'ba_leave_source_out_purged.json').read_text())['results']
with (D/'ba_leave_source_out_purged_pairs.csv').open(newline='') as f:pairs=list(csv.DictReader(f))
out={}
for source,source_result in result.items():
 rows=[r for r in pairs if r['target']==source]; unit_by_query={r['query']:(r['task'],r['unit']) for r in source_result['query_predictions']};by=collections.defaultdict(list)
 for r in rows:by[r['method'],r['query']].append(r)
 byunit=collections.defaultdict(lambda:collections.defaultdict(list));scores={}
 for (method,q),items in by.items():
  pos=[r for r in items if int(r['same_task'])==1];neg=[r for r in items if int(r['same_task'])==0];assert len(pos)==1 and len(neg)==7
  fmr=np.mean([int(r['predicted_link']) for r in neg]);fsr=1-int(pos[0]['predicted_link']);task,unit=unit_by_query[q];byunit[method][task,unit].append((fmr,fsr,1-(fmr+fsr)/2))
  for r in items:
   key=(r['query'],r['candidate']);scores.setdefault(key,{})[method]=r
   score=float(r['root_probability'])+float(r['evidence_contribution'])+float(r['B_contribution'])+float(r['S_contribution']);assert abs(score-float(r['score']))<1e-9
 units=sorted({u for t,u in byunit['evidence']});tasks=sorted({t for t,u in byunit['evidence']});mats={}
 for method in ('evidence','gated_components'):
  mat=np.full((len(units),len(tasks),3),np.nan)
  for i,u in enumerate(units):
   for j,t in enumerate(tasks):
    if (t,u) in byunit[method]:mat[i,j,:]=np.mean(byunit[method][t,u],axis=0)
  mats[method]=mat
 rng=np.random.default_rng(20261103);ix=rng.integers(0,len(units),(5000,len(units)))
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);baseline=np.nanmean(np.nanmean(mats['evidence'],axis=0),axis=0);finger=np.nanmean(np.nanmean(mats['gated_components'],axis=0),axis=0);bboot=np.nanmean(np.nanmean(mats['evidence'][ix],axis=1),axis=1);fboot=np.nanmean(np.nanmean(mats['gated_components'][ix],axis=1),axis=1)
 assert abs(finger[2]-source_result['fingerprint_ba'])<1e-12
 changes={};examples={}
 for name,label,before,after in [('recovered_positive',1,0,1),('new_missed_positive',1,1,0),('avoided_false_merge',0,1,0),('new_false_merge',0,0,1)]:
  selected=[]
  for key,models in scores.items():
   b=models['evidence'];a=models['gated_components']
   if int(b['same_task'])==label and int(b['predicted_link'])==before and int(a['predicted_link'])==after:selected.append((key,b,a))
  changes[name]=len(selected)
  if selected:
   key,b,a=min(selected,key=lambda z:hashlib.sha256(('|'.join(z[0])).encode()).hexdigest());examples[name]={'query':key[0],'candidate':key[1],'baseline_score':float(b['score']),'baseline_threshold':float(b['threshold']),'fingerprint_score':float(a['score']),'fingerprint_threshold':float(a['threshold']),'root_probability':float(a['root_probability']),'evidence_contribution':float(a['evidence_contribution']),'B_contribution':float(a['B_contribution']),'S_contribution':float(a['S_contribution'])}
 metrics={}
 for i,name in enumerate(('fmr','fsr','ba')):metrics[name]={'baseline':float(baseline[i]),'fingerprint':float(finger[i]),'delta':float(finger[i]-baseline[i]),'paired_cluster_ci95':np.nanquantile(fboot[:,i]-bboot[:,i],[.025,.975]).tolist()}
 out[source]={'metrics':metrics,'changed_pair_counts':changes,'examples':examples,'bootstrap_unit':'query participant' if source=='kypo' else 'task','note':'Candidate galleries are fixed during query-unit resampling; repeated candidate participants are not independently resampled.'}
(D/'ba_leave_source_out_purged_errors.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:{'metrics':v['metrics'],'changed_pair_counts':v['changed_pair_counts']} for k,v in out.items()},indent=2))
