#!/usr/bin/env python3
"""Nested task-held-out BA pilot for evidence-conditioned fingerprint fusion."""
import collections,csv,importlib.util,json,pathlib,random,sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
p=ROOT/'research/experiments/behavioral_fingerprint_v2/run_cth_cross_mode.py'
spec=importlib.util.spec_from_file_location('cross',p);cross=importlib.util.module_from_spec(spec);spec.loader.exec_module(cross);mix=cross.mix
OUT=ROOT/'research/results/association_fingerprint_v3';OUT.mkdir(parents=True,exist_ok=True)
SEED=20261103; CS=(.1,1.,10.); METHODS=('evidence','lexical_local','fingerprint_local')
def fit(x,y,c):
 model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=SEED));return model.fit(x,y)
def build(rows,train):
 from attackagent.fingerprint_v2 import fit_scale,apply_scale
 base,_=mix.prepare_features(rows,train,list(range(len(rows))),SEED)
 mats=[np.hstack([r['B'],r['S']]) for r in rows]; center,scale=fit_scale([mats[i] for i in train]); mats=[apply_scale(x,center,scale) for x in mats]
 windows=[[set(r['commands'][i:i+8]) for i in range(0,min(len(r['commands']),128),8) if len(r['commands'][i:i+8])>=2] for r in rows]
 cache={}
 def align(a,b,weight):
  key=(a,b,weight)
  if key in cache:return cache[key]
  left,right=mats[a],mats[b];n,m=len(left),len(right);cost=np.full((n+1,m+1),np.inf);cost[0,0]=0;back={}; local={}
  for i in range(1,n+1):cost[i,0]=i*.25;back[i,0]=(i-1,0,False)
  for j in range(1,m+1):cost[0,j]=j*.25;back[0,j]=(0,j-1,False)
  for i in range(1,n+1):
   for j in range(1,m+1):
    x,y=windows[a][i-1],windows[b][j-1];anchor=len(x & y)/max(1,len(x | y));shared=np.isfinite(left[i-1]) & np.isfinite(right[j-1]);dist=float(np.mean(np.abs(left[i-1,shared]-right[j-1,shared]))) if shared.sum()>=3 else 0.
    match=1-anchor+weight*dist if (weight==0 or shared.sum()>=3) else float('inf');local[i,j]=(anchor,dist)
    options=[(cost[i-1,j-1]+match,i-1,j-1,True),(cost[i-1,j]+.25,i-1,j,False),(cost[i,j-1]+.25,i,j-1,False)];best=min(options,key=lambda z:z[0]);cost[i,j]=best[0];back[i,j]=best[1:]
  i,j=n,m;matched=[]
  while i or j:
   pi,pj,match=back[i,j]
   if match:matched.append(local[i,j])
   i,j=pi,pj
  coverage=len(matched)/max(1,n,m);meananchor=float(np.mean([x[0] for x in matched])) if matched else 0.;meandist=float(np.mean([x[1] for x in matched])) if matched else 0.
  ans=(float(cost[n,m]/max(1,n+m)),coverage,meananchor,meandist);cache[key]=ans;return ans
 def features(pairs):
  x=base(pairs);e=x[:,:2];lex=np.asarray([align(a,b,0.)[:3] for a,b,_ in pairs]);fpr=np.asarray([align(a,b,.5) for a,b,_ in pairs]);gate=x[:,0:1]
  return {'evidence':e,'lexical_local':np.c_[e,lex],'fingerprint_local':np.c_[e,lex,gate*fpr]}
 return features
def gallery(rows,idx,seed):
 rng=random.Random(seed); by=collections.defaultdict(list)
 for i in idx:by[rows[i]['task'],rows[i]['mode']].append(i)
 tasks=sorted({rows[i]['task'] for i in idx}); pairs=[]
 for q in idx:
  mode='agent' if rows[q]['mode']=='human' else 'human'
  for t in tasks:pairs.append((q,rng.choice(by[t,mode]),int(rows[q]['task']==t)))
 return pairs
def ba_by_task(scores,labels,tasks,threshold):
 result={}
 for t in sorted(set(tasks)):
  mask=np.array([x==t for x in tasks]); y=labels[mask];pred=scores[mask]>=threshold
  if np.any(y==1) and np.any(y==0):result[t]=1-(np.mean(pred[y==0])+np.mean(~pred[y==1]))/2
 return result
def choose_threshold(scores,labels,tasks):
 candidates=np.unique(np.quantile(scores,np.linspace(0,1,101))); best=(-1.,float('inf'))
 for th in candidates:
  vals=ba_by_task(scores,labels,tasks,th); v=float(np.mean(list(vals.values())))
  if v>best[0]+1e-12 or (abs(v-best[0])<1e-12 and th>best[1]):best=(v,float(th))
 return best
allrows=mix.load_rows(); modes=collections.defaultdict(set)
for r in allrows:
 if r['profile_eligible']:modes[r['task']].add(r['mode'])
rows=[r for r in allrows if r['profile_eligible'] and len(modes[r['task']])==2]; lookup={r['trace_id']:i for i,r in enumerate(rows)}
assert len(rows)==175 and len({r['task'] for r in rows})==40
with (ROOT/'research/results/behavioral_fingerprint_v2/cth_cross_mode/candidate_scores.csv').open() as f:
 original=[r for r in csv.DictReader(f) if r['method']=='command_cosine']
predictions=[];choices=[]
for fold in range(5):
 testrows=[r for r in original if int(r['fold'])==fold]; held={r['query_task'] for r in testrows}; train=[i for i,r in enumerate(rows) if r['task'] not in held]
 tasks=sorted({rows[i]['task'] for i in train}); innerfold={t:i%3 for i,t in enumerate(tasks)}; oof=collections.defaultdict(list);ys=[];ts=[]
 for inner in range(3):
  tr=[i for i in train if innerfold[rows[i]['task']]!=inner];val=[i for i in train if innerfold[rows[i]['task']]==inner]
  trainpairs=mix.sample_training_pairs(rows,tr,random.Random(SEED+fold*1009+inner));validpairs=gallery(rows,val,SEED+fold*100+inner)
  f=build(rows,tr); X,V=f(trainpairs),f(validpairs); y=np.array([p[2] for p in trainpairs]);ys.extend(p[2] for p in validpairs);ts.extend(rows[p[0]]['task'] for p in validpairs)
  for method in METHODS:
   for c in CS:oof[method,c].extend(fit(X[method],y,c).predict_proba(V[method])[:,1])
 yv=np.array(ys);selected={}; audit=[]
 for method in METHODS:
  options=[]
  for c in CS:
   score,th=choose_threshold(np.asarray(oof[method,c]),yv,ts); options.append((score,-c,th,c));audit.append({'method':method,'C':c,'inner_ba':score,'threshold':th})
  score,_,th,c=max(options);selected[method]=(c,th)
 trainpairs=mix.sample_training_pairs(rows,train,random.Random(SEED+fold*1009));testpairs=[(lookup[r['query_trace_id']],lookup[r['candidate_trace_id']],int(r['same_task_proxy'])) for r in testrows]
 f=build(rows,train);X,V=f(trainpairs),f(testpairs);y=np.array([p[2] for p in trainpairs]);yt=np.array([p[2] for p in testpairs]);tasknames=[r['query_task'] for r in testrows]
 for method,(c,th) in selected.items():
  scores=fit(X[method],y,c).predict_proba(V[method])[:,1]
  for t,ba in ba_by_task(scores,yt,tasknames,th).items():predictions.append({'fold':fold,'task':t,'method':method,'ba':ba})
 choices.append({'fold':fold,'selected':selected,'inner_results':audit});print('fold',fold,'done',flush=True)
by=collections.defaultdict(dict)
for r in predictions:by[r['method']][r['task']]=r['ba']
summary=[]
for method in METHODS:
 keys=sorted(by[method]); v=np.asarray([by[method][t] for t in keys]); diff=v-np.asarray([by['evidence'][t] for t in keys]); rng=np.random.default_rng(SEED); boot=diff[rng.integers(0,len(keys),(10000,len(keys)))].mean(axis=1)
 summary.append({'method':method,'tasks':len(keys),'ba':float(v.mean()),'delta_ba':float(diff.mean()),'paired_task_ci95':np.quantile(boot,[.025,.975]).tolist()})
(OUT/'ba_local_alignment.json').write_text(json.dumps({'protocol':'post-observation exploratory; same fixed CTH galleries; five outer task folds; nested three task folds choose C and BA threshold; local alignment is exploratory; matched lexical-only alignment and fingerprint alignment share training pairs and logistic learner','summary':summary,'fold_choices':choices,'task_predictions':predictions},indent=2)+'\n')
print(json.dumps(summary,indent=2))
