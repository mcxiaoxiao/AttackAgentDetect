#!/usr/bin/env python3
"""Human-only known-task retrieval on unseen KYPO participants."""
import collections,hashlib,importlib.util,json,pathlib,random
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('cross',HERE/'run_cth_cross_mode.py'); c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
m=c.mix;ROOT=c.ROOT;OUT=ROOT/'research/results/behavioral_fingerprint_v2/kypo_human';OUT.mkdir(parents=True,exist_ok=True)

def main():
 data=json.loads((ROOT/'site/data/behavior-trace-dataset-v2.json').read_text())['records'];rows=[]
 for x in data:
  if x['source_dataset']!='kypo-locust-human-training':continue
  blocks,audit=m.fingerprint_matrix(x['commands'],window=8,min_window_events=4,max_events=128)
  if blocks is None:continue
  rows.append({**x,'task':x['task_group'],'profile_eligible':True,'B':blocks['B'],'S':blocks['S'],'B_summary':m.summarize(blocks['B']),'S_summary':m.summarize(blocks['S']),'actions':[m.ATOM_GROUP.get(z) for z in x['commands']]})
 people=sorted({x['participant_id'] for x in rows},key=lambda x:hashlib.sha256(x.encode()).hexdigest());folds={p:i%5 for i,p in enumerate(people)};pred=[];audit=[]
 for fold in range(5):
  train=[i for i,x in enumerate(rows) if folds[x['participant_id']]!=fold];test=[i for i,x in enumerate(rows) if folds[x['participant_id']]==fold];rng=random.Random(c.SEED+fold)
  pairs=[p for p in m.sample_training_pairs(rows,train,rng) if rows[p[0]]['participant_id']!=rows[p[1]]['participant_id']];feat,_=m.prepare_features(rows,train,list(range(len(rows))),c.SEED);fit=m.fit_models(feat(pairs),np.array([p[2] for p in pairs]))
  by=collections.defaultdict(list)
  for i in test:by[rows[i]['task']].append(i)
  by={t:ids for t,ids in by.items() if len(ids)>=2};nq=0
  if len(by)<5:raise RuntimeError('Insufficient held-out task pool')
  for task,qs in sorted(by.items()):
   others=sorted(set(by)-{task})
   for q in qs:
    nq+=1
    for rep in range(20):
     positive=rng.choice([z for z in qs if rows[z]['participant_id']!=rows[q]['participant_id']]);neg_tasks=rng.sample(others,4)
     candidates=[positive]+[rng.choice([z for z in by[t] if rows[z]['participant_id']!=rows[q]['participant_id']]) for t in neg_tasks]
     x=feat([(q,z,int(j==0)) for j,z in enumerate(candidates)])
     scores={'command_cosine':x[:,0],'command_plus_BS':fit['command_plus_BS'].predict_proba(x[:,m.METHOD_COLUMNS['command_plus_BS']])[:,1]}
     for method,sc in scores.items():
      h,mrr=c.rank_credit(sc,0);pred.append({'fold':fold,'participant':rows[q]['participant_id'],'task':task,'query_trace_id':rows[q]['trace_id'],'replicate':rep,'method':method,'hit_at_1':h,'mrr':mrr})
  audit.append({'fold':fold,'train_participants':len({rows[i]['participant_id'] for i in train}),'test_participants':len({rows[i]['participant_id'] for i in test}),'eligible_gallery_tasks':len(by),'queries':nq})
 # Average repeated draws per participant/task first, then task macro.
 D=collections.defaultdict(list)
 for x in pred:D[(x['method'],x['participant'],x['task'])].append(x)
 subjects=sorted({x['participant'] for x in pred});tasks=sorted({x['task'] for x in pred}); metrics=[]
 for metric in ('hit_at_1','mrr'):
  mats={}
  for method in ('command_cosine','command_plus_BS'):
   arr=np.full((len(subjects),len(tasks)),np.nan)
   for i,p in enumerate(subjects):
    for j,t in enumerate(tasks):
     if (method,p,t) in D:arr[i,j]=np.mean([x[metric] for x in D[(method,p,t)]])
   mats[method]=arr
  rng=np.random.default_rng(c.SEED);ix=rng.integers(0,len(subjects),(5000,len(subjects)))
  base=np.nanmean(np.nanmean(mats['command_cosine'][ix],axis=1),axis=1)
  for method,arr in mats.items():
   val=float(np.nanmean(np.nanmean(arr,axis=0)));bs=np.nanmean(np.nanmean(arr[ix],axis=1),axis=1)
   metrics.append({'method':method,'metric':metric,'value':val,'participant_cluster_ci95':np.quantile(bs,[.025,.975]).tolist(),'paired_delta':val-float(np.nanmean(np.nanmean(mats['command_cosine'],axis=0))),'paired_delta_ci95':np.quantile(bs-base,[.025,.975]).tolist()})
 c.write_csv(OUT/'query_predictions.csv',pred);c.write_json(OUT/'metrics.json',{'cohort_records':len(rows),'cohort_participants':len(people),'evaluated_tasks':len(tasks),'evaluated_query_traces':len({x['query_trace_id'] for x in pred}),'evaluated_participants':len(subjects),'metrics':metrics,'folds':audit,'protocol':'fixed C=1; five participant-disjoint folds; known task labels; 20 galleries per query; one same-task and four other-task candidates; different query/candidate participants; training-only transformations','uncertainty':'5000 participant-cluster resamples, recompute task macro after averaging repeated draws','labels':'same source variant and training level; Human-only known-task retrieval','runner_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()});print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
