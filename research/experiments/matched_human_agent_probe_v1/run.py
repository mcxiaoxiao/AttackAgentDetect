"""Exploratory same-challenge Human-query / Agent-gallery retrieval probe."""
import collections, csv, hashlib, importlib.util, itertools, json, pathlib, re, sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from attackagent.fingerprints import fit_column_scale, matrix, transform_matrix, dtw_distance
spec=importlib.util.spec_from_file_location('overnight',ROOT/'research/experiments/overnight_fingerprint_v1/run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
SEED=20261102;OUT=ROOT/'research/results/matched_human_agent_probe_v1';OUT.mkdir(parents=True,exist_ok=True)

def boot(values):
 a=np.asarray(values,float);g=np.random.default_rng(SEED)
 return np.quantile(a[g.integers(0,len(a),(10000,len(a)))].mean(axis=1),[.025,.975]).tolist()

def main():
 rows=r.load_corpus();human=collections.defaultdict(list);agent=collections.defaultdict(list)
 for x in rows:
  if x['benchmark']!='cybench' or x['track']!='ctf' or len(x['commands'])<4:continue
  if x['source_dataset']=='cyber-task-horizons' and x['mode']=='human': key=x['task_id'];human[key].append(x)
  elif x['source_dataset']=='antieval/cybench-trajectories' and x['mode']=='agent':
   key=re.sub(r'^cybench:|\s*\(hard\)$','',x['task_id']);agent[key].append(x)
 tasks=sorted(set(human)&set(agent));assert len(tasks)==5,tasks
 for t in tasks:human[t].sort(key=lambda x:x['trace_id']);agent[t].sort(key=lambda x:x['trace_id'])
 sessions={x['trace_id']:x for t in tasks for x in human[t]+agent[t]};ids=list(sessions);pos={k:i for i,k in enumerate(ids)}
 commands={k:x['commands'][:128] for k,x in sessions.items()}
 mats={k:matrix(v,window=4,max_commands=128,min_commands=4,min_tail=4)[0] for k,v in commands.items()}
 preds=[];folds=[]
 for fold,target in enumerate(tasks):
  train_tasks=[t for t in tasks if t!=target];train=[x['trace_id'] for t in train_tasks for x in agent[t]]
  vec=r.vectorizer().fit([r.command_text(commands[k]) for k in train]);docs=vec.transform([r.command_text(commands[k]) for k in ids])
  center,scale=fit_column_scale([mats[k] for k in train]);scaled={k:transform_matrix(mats[k],center,scale) for k in ids}
  pairs=[]
  for t in train_tasks:
   pairs.extend((a['trace_id'],b['trace_id'],1) for a,b in itertools.combinations(agent[t],2))
  neg=[]
  for i,t in enumerate(train_tasks):
   for u in train_tasks[i+1:]:neg.extend((a['trace_id'],b['trace_id'],0) for a in agent[t] for b in agent[u])
  rng=np.random.default_rng(SEED+fold)
  if len(neg)>len(pairs):neg=[neg[i] for i in rng.choice(len(neg),len(pairs),replace=False)]
  pairs+=neg;xtrain_matrix=[];xtrain_length=[];y=[]
  for a,b,label in pairs:
   cs=r.cosine(docs[pos[a]],docs[pos[b]]);d=dtw_distance(scaled[a],scaled[b]);ld=abs(np.log1p(len(commands[a]))-np.log1p(len(commands[b])))
   xtrain_matrix.append([cs,-d if np.isfinite(d) else np.nan]);xtrain_length.append([cs,ld]);y.append(label)
  def fit_model(x):
   return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
       LogisticRegression(C=1,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=SEED)).fit(x,y)
  model_matrix=fit_model(xtrain_matrix);model_length=fit_model(xtrain_length)
  other=[t for t in tasks if t!=target];folds.append({'query_task':target,'training_agent_tasks':len(train_tasks),'training_pairs':len(pairs),'human_queries':len(human[target])})
  for q in human[target]:
   qid=q['trace_id']
   for rep in range(20):
    candidates=[str(rng.choice([x['trace_id'] for x in agent[target]]))]+[str(rng.choice([x['trace_id'] for x in agent[t]])) for t in other]
    cs=[r.cosine(docs[pos[qid]],docs[pos[c]]) for c in candidates]
    ds=[dtw_distance(scaled[qid],scaled[c]) for c in candidates]
    ls=[abs(np.log1p(len(commands[qid]))-np.log1p(len(commands[c]))) for c in candidates]
    fused=model_matrix.predict_proba([[s,-d if np.isfinite(d) else np.nan] for s,d in zip(cs,ds)])[:,1]
    length=model_length.predict_proba([[s,l] for s,l in zip(cs,ls)])[:,1]
    for method,scores in [('command',cs),('command_matrix',fused),('command_length',length)]:
     equal=[j for j,v in enumerate(scores) if np.isclose(v,scores[0],rtol=1e-10,atol=1e-12)]
     above=sum(v>scores[0] and not np.isclose(v,scores[0],rtol=1e-10,atol=1e-12) for v in scores)
     preds.append({'task':target,'query':qid,'replicate':rep,'method':method,
       'hit1':(1/len(equal) if not above else 0.),'mrr':float(np.mean([1/(above+j) for j in range(1,len(equal)+1)])),
       'auc':float(np.mean([float(scores[0]>v) if not np.isclose(scores[0],v,rtol=1e-10,atol=1e-12) else .5 for v in scores[1:]]))})
 by=collections.defaultdict(lambda:collections.defaultdict(list))
 for x in preds:by[x['method']][x['task']].append(x)
 metrics=[]
 for method,taskdata in by.items():
  for metric in ('hit1','mrr','auc'):
   vals={t:float(np.mean([x[metric] for x in ps])) for t,ps in taskdata.items()}
   base={t:float(np.mean([x[metric] for x in by['command'][t]])) for t in vals};delta=[vals[t]-base[t] for t in sorted(vals)]
   metrics.append({'method':method,'metric':metric,'value':float(np.mean(list(vals.values()))),'tasks':len(vals),'paired_delta':float(np.mean(delta)),'paired_95ci':boot(delta)})
 result={'status':'exploratory_cross_source_same_challenge_proxy','task_join':'normalized CyBench slug only; fixture equivalence unverified',
   'human_traces_ge4':sum(len(human[t]) for t in tasks),'agent_traces_ge4':sum(len(agent[t]) for t in tasks),
   'tasks':tasks,'human_queries':sum(len(human[t]) for t in tasks),'draws_per_query':20,'candidates_per_query':5,'folds':folds,'metrics':metrics}
 (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
 with (OUT/'predictions.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=preds[0]);w.writeheader();w.writerows(preds)
 (OUT/'provenance.json').write_text(json.dumps({'seed':SEED,'inputs':['research/results/behavior_corpus_v1/evidence.jsonl','research/results/behavior_corpus_v1/trace_registry.csv'],
   'runner_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'program_sha256':hashlib.sha256(pathlib.Path(__file__).with_name('program.md').read_bytes()).hexdigest()},indent=2)+'\n')
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
