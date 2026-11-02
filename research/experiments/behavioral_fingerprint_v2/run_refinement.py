#!/usr/bin/env python3
"""Bounded nested refinement on the existing CTH cross-mode protocol."""
import collections, csv, hashlib, importlib.util, json, pathlib, random, sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
ROOT=pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from attackagent.fingerprint_v2 import masked_dtw, fit_scale, apply_scale
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cross', HERE/'run_cth_cross_mode.py'); cross=importlib.util.module_from_spec(spec); spec.loader.exec_module(cross)
mix=cross.mix
OUT=ROOT/'research/results/behavioral_fingerprint_v2/refinement'; OUT.mkdir(parents=True,exist_ok=True)
SEED=20261103; FAMILIES=('scalar','conditional','components','local_dtw'); CS=(.1,1.,10.)
def model(c): return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=SEED))
def build(rows, train):
 base,_=mix.prepare_features(rows,train,list(range(len(rows))),SEED)
 raw=np.vstack([np.r_[r['B_summary'],r['S_summary']] for r in rows])
 scale=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()).fit(raw[train]); profiles=scale.transform(raw)
 matrices=[np.hstack([r['B'],r['S']]) for r in rows]
 center,sd=fit_scale([matrices[i] for i in train]); mats=[apply_scale(x,center,sd) for x in matrices]; cache={}
 def features(pairs):
  x=base(pairs); cos=x[:,0:1]; dist=x[:,2:4]
  rich=np.asarray([np.abs(profiles[a]-profiles[b]) for a,b,_ in pairs])
  dt=[]
  for a,b,_ in pairs:
   key=tuple(sorted((a,b)))
   if key not in cache: cache[key]=masked_dtw(mats[a],mats[b],min_shared_columns=3)
   dt.append(cache[key])
  dt=np.asarray(dt)[:,None]
  return {'scalar':np.c_[cos,dist], 'conditional':np.c_[cos,dist,cos*dist],
          'components':np.c_[cos,rich], 'local_dtw':np.c_[cos,dist,-dt,cos*(-dt)]}
 return features

def inner_gallery(rows, val, seed):
 rng=random.Random(seed); by=collections.defaultdict(list)
 for i in val: by[(rows[i]['task'],rows[i]['mode'])].append(i)
 tasks=sorted({rows[i]['task'] for i in val}); pairs=[]; groups=[]
 for q in val:
  mode='agent' if rows[q]['mode']=='human' else 'human'
  candidates=[rng.choice(by[(t,mode)]) for t in tasks]
  start=len(pairs); pairs.extend((q,c,int(rows[q]['task']==rows[c]['task'])) for c in candidates)
  groups.append((rows[q]['task'],start,len(pairs),tasks.index(rows[q]['task'])))
 return pairs,groups

def rank_metrics(scores, groups):
 by=collections.defaultdict(list)
 for task,a,b,pos in groups: by[task].append(cross.rank_credit(scores[a:b],pos))
 return np.mean([np.mean(v,axis=0) for v in by.values()],axis=0)

def main():
 allrows=mix.load_rows(); counts=collections.defaultdict(set)
 for r in allrows:
  if r['profile_eligible']: counts[r['task']].add(r['mode'])
 rows=[r for r in allrows if r['profile_eligible'] and len(counts[r['task']])==2]
 lookup={r['trace_id']:i for i,r in enumerate(rows)}
 original=list(csv.DictReader(open(ROOT/'research/results/behavioral_fingerprint_v2/cth_cross_mode/candidate_scores.csv')))
 original=[r for r in original if r['method']=='command_cosine']
 predictions=[]; choices=[]
 for fold in range(5):
  testrows=[r for r in original if int(r['fold'])==fold]; testtasks={r['query_task'] for r in testrows}; train=[i for i,r in enumerate(rows) if r['task'] not in testtasks]
  tasks=sorted({rows[i]['task'] for i in train}); innerfold={t:j%3 for j,t in enumerate(tasks)}
  inner_values=collections.defaultdict(list)
  for inner in range(3):
   fit=[i for i in train if innerfold[rows[i]['task']]!=inner]; val=[i for i in train if innerfold[rows[i]['task']]==inner]
   pairs=mix.sample_training_pairs(rows,fit,random.Random(SEED+fold*1009+inner)); labels=np.array([p[2] for p in pairs]); vp,groups=inner_gallery(rows,val,SEED+fold*100+inner)
   feat=build(rows,fit); fx,vx=feat(pairs),feat(vp)
   for family in FAMILIES:
    for c in CS:
     fitted=model(c).fit(fx[family],labels); score=fitted.predict_proba(vx[family])[:,1]
     inner_values[(family,c)].append(rank_metrics(score,groups))
  average={k:np.mean(v,axis=0) for k,v in inner_values.items()}
  selected={f:max(CS,key=lambda c:tuple(average[(f,c)])) for f in FAMILIES}
  best=max(FAMILIES,key=lambda f:tuple(average[(f,selected[f])]))
  choices.append({'fold':fold,'selected_family':best,'selected_C':selected[best],'all_inner_scores':[{'family':f,'C':c,'hit_at_1':float(v[0]),'mrr':float(v[1])} for (f,c),v in average.items()]})
  pairs=mix.sample_training_pairs(rows,train,random.Random(SEED+fold*1009)); labels=np.array([p[2] for p in pairs]); evalpairs=[(lookup[r['query_trace_id']],lookup[r['candidate_trace_id']],int(r['same_task_proxy'])) for r in testrows]
  feat=build(rows,train); fx,ex=feat(pairs),feat(evalpairs)
  scores={f:model(selected[f]).fit(fx[f],labels).predict_proba(ex[f])[:,1] for f in FAMILIES}; scores['nested_selected']=scores[best]; scores['command_cosine']=np.array([float(r['score']) for r in testrows])
  groups=collections.defaultdict(list)
  for i,r in enumerate(testrows): groups[r['query_trace_id']].append(i)
  for method,s in scores.items():
   for q,ids in groups.items():
    pos=next(j for j,k in enumerate(ids) if int(testrows[k]['same_task_proxy'])==1); hit,mrr=cross.rank_credit(s[ids],pos)
    predictions.append({'fold':fold,'query_trace_id':q,'task':testrows[ids[0]]['query_task'],'method':method,'hit_at_1':hit,'mrr':mrr})
  print('completed outer fold',fold,'selected',best,flush=True)
 by=collections.defaultdict(lambda:collections.defaultdict(list))
 for p in predictions: by[p['method']][p['task']].append(p)
 metrics=[]
 for method,tasks in by.items():
  rec={'method':method,'tasks':len(tasks),'queries':sum(map(len,tasks.values()))}
  for metric in ('hit_at_1','mrr'):
   v=np.array([np.mean([p[metric] for p in tasks[t]]) for t in sorted(tasks)])
   b=np.array([np.mean([p[metric] for p in by['command_cosine'][t]]) for t in sorted(tasks)])
   diff=v-b; boot=np.random.default_rng(SEED).choice(diff,(10000,len(diff)),replace=True).mean(axis=1)
   rec[metric]=float(v.mean()); rec['delta_'+metric]=float(diff.mean()); rec['delta_'+metric+'_ci95']=np.quantile(boot,[.025,.975]).tolist()
  metrics.append(rec)
 cross.write_csv(OUT/'query_predictions.csv',predictions); cross.write_json(OUT/'inner_choices.json',choices); cross.write_json(OUT/'metrics.json',metrics)
 cross.write_json(OUT/'provenance.json',{'status':'post_observation_exploratory','seed':SEED,'input_candidate_scores_sha256':hashlib.sha256((ROOT/'research/results/behavioral_fingerprint_v2/cth_cross_mode/candidate_scores.csv').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((HERE/'refinement-protocol.md').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'outer_test_selection':False,'family_and_C_selection':'inner task-held-out Hit@1; MRR breaks ties','candidate_pool':'identical to original CTH cross-mode panel'})
 print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
