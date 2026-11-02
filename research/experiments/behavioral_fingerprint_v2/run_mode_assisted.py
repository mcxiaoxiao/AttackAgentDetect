#!/usr/bin/env python3
"""Cross-fit the fingerprint's mode readout before adding it to association."""
import collections,csv,hashlib,importlib.util,json,pathlib,random,sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
ROOT=pathlib.Path(__file__).resolve().parents[3]; HERE=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('refine',HERE/'run_refinement.py'); r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
mix=r.mix; cross=r.cross;OUT=r.OUT

def main():
 raw=mix.load_rows();groups=collections.defaultdict(set)
 for x in raw:
  if x['profile_eligible']: groups[x['task']].add(x['mode'])
 rows=[x for x in raw if x['profile_eligible'] and len(groups[x['task']])==2]; ids={x['trace_id']:i for i,x in enumerate(rows)}
 X=np.vstack([np.r_[x['B_summary'],x['S_summary']] for x in rows]); y=np.array([int(x['mode']=='agent') for x in rows])
 base=[x for x in csv.DictReader(open(ROOT/'research/results/behavioral_fingerprint_v2/cth_cross_mode/candidate_scores.csv')) if x['method']=='command_cosine']
 predictions=[];mode_predictions=[]
 for fold in range(5):
  testrows=[x for x in base if int(x['fold'])==fold];tasks={x['query_task'] for x in testrows};train=[i for i,x in enumerate(rows) if x['task'] not in tasks];test=[i for i,x in enumerate(rows) if x['task'] in tasks]
  train_tasks=sorted({rows[i]['task'] for i in train});fm={t:j%3 for j,t in enumerate(train_tasks)};p=np.full(len(rows),np.nan)
  for inner in range(3):
   fit=[i for i in train if fm[rows[i]['task']]!=inner];val=[i for i in train if fm[rows[i]['task']]==inner]
   m=r.model(1).fit(X[fit],y[fit]);p[val]=m.predict_proba(X[val])[:,1]
  m=r.model(1).fit(X[train],y[train]);p[test]=m.predict_proba(X[test])[:,1]
  for i in test:mode_predictions.append({'task':rows[i]['task'],'trace_id':rows[i]['trace_id'],'truth':int(y[i]),'score':float(p[i])})
  pairfeatures,_=mix.prepare_features(rows,train,list(range(len(rows))),r.SEED)
  trainpairs=mix.sample_training_pairs(rows,train,random.Random(r.SEED+fold*1009));ev=[(ids[x['query_trace_id']],ids[x['candidate_trace_id']],int(x['same_task_proxy'])) for x in testrows]
  def features(pairs):
   v=pairfeatures(pairs); z=np.array([[p[a]*p[b],(1-p[a])*(1-p[b]),p[a]*(1-p[b])+(1-p[a])*p[b]] for a,b,_ in pairs]);return np.c_[v[:,0],z,v[:,0:1]*z],np.c_[v[:,[0,2,3]],z,v[:,0:1]*z]
  tr,te=features(trainpairs),features(ev);lab=[x[2] for x in trainpairs]
  for name,xx,vv in zip(('command_plus_mode','command_BS_plus_mode'),tr,te):
   score=r.model(1).fit(xx,lab).predict_proba(vv)[:,1];qs=collections.defaultdict(list)
   for k,x in enumerate(testrows):qs[x['query_trace_id']].append(k)
   for q,js in qs.items():
    pos=next(j for j,k in enumerate(js) if int(testrows[k]['same_task_proxy']));hit,mrr=cross.rank_credit(score[js],pos)
    predictions.append({'method':name,'task':testrows[js[0]]['query_task'],'query_trace_id':q,'hit_at_1':hit,'mrr':mrr})
 orig=list(csv.DictReader(open(ROOT/'research/results/behavioral_fingerprint_v2/cth_cross_mode/query_predictions.csv')));orig=[x for x in orig if x['method']=='command_cosine']
 B=collections.defaultdict(list)
 for x in orig:B[x['query_task']].append(x)
 metrics=[]
 for method in ('command_plus_mode','command_BS_plus_mode'):
  D=collections.defaultdict(list)
  for x in predictions:
   if x['method']==method:D[x['task']].append(x)
  out={'method':method,'tasks':len(D)}
  for metric in ('hit_at_1','mrr'):
   v=np.array([np.mean([x[metric] for x in D[t]]) for t in sorted(D)]);b=np.array([np.mean([float(x[metric]) for x in B[t]]) for t in sorted(D)]);d=v-b
   out[metric]=float(v.mean());out['delta_'+metric]=float(d.mean());out['delta_'+metric+'_ci95']=np.quantile(np.random.default_rng(r.SEED).choice(d,(10000,len(d)),replace=True).mean(1),[.025,.975]).tolist()
  metrics.append(out)
 cross.write_json(OUT/'mode_assisted_metrics.json',metrics);cross.write_csv(OUT/'mode_assisted_predictions.csv',predictions);cross.write_csv(OUT/'mode_readout_predictions.csv',mode_predictions)
 cross.write_json(OUT/'mode_assisted_provenance.json',{'status':'post_observation_exploratory','mode_features':'B/S summaries only; three task-fold crossfit for training-pair mode scores; outer-training fit for test mode scores','fusion':'fixed C=1; cosine, probabilities of HH/HA/AA, and cosine interactions; optional scalar B/S distances','mode_label_usage':'supervised training only; true test mode never enters pair score','runner_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()})
 print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
