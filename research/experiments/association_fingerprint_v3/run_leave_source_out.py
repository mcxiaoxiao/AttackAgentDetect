#!/usr/bin/env python3
"""Leave-one-source-out association evaluation on common command views."""
import collections,csv,importlib.util,itertools,json,pathlib,random,sys,warnings
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('cross',ROOT/'research/experiments/behavioral_fingerprint_v2/run_cth_cross_mode.py');cross=importlib.util.module_from_spec(spec);spec.loader.exec_module(cross);mix=cross.mix
SEED=20261103; CS=(.1,1.,10.);METHODS=('evidence','gated_components'); data=json.loads((ROOT/'site/data/behavior-trace-dataset-v2.json').read_text())['records'];rows=[];seen=set()
source_map={'cyber-task-horizons':'cth','antieval/cybench-trajectories':'antieval','kypo-locust-human-training':'kypo'}
for x in data:
 source=source_map.get(x['source_dataset']);
 if not source or (source=='cth' and x['track']!='ctf'):continue
 cmds=x['commands'];key=(source,x['task_group'],tuple(cmds));minimum=2 if source=='cth' else 4
 if len(cmds)<minimum or (source=='antieval' and key in seen):continue
 seen.add(key);blocks,_=mix.fingerprint_matrix(cmds,window=8,min_window_events=minimum,max_events=128)
 if blocks is None:continue
 rows.append({**x,'source':source,'task':x['task_group'],'profile_eligible':True,'B':blocks['B'],'S':blocks['S'],'B_summary':mix.summarize(blocks['B']),'S_summary':mix.summarize(blocks['S']),'actions':[mix.ATOM_GROUP.get(z) for z in cmds]})
cth_modes=collections.defaultdict(set);ant_counts=collections.Counter()
for r in rows:
 if r['source']=='cth':cth_modes[r['task']].add(r['mode'])
 if r['source']=='antieval':ant_counts[r['task']]+=1
rows=[r for r in rows if (r['source']!='cth' or len(cth_modes[r['task']])==2) and (r['source']!='antieval' or ant_counts[r['task']]>=2)]
def valid(a,b):return a!=b and not (rows[a]['source']=='kypo' and rows[a]['participant_id']==rows[b]['participant_id'])
def sample(idx,seed):
 rng=random.Random(seed);bysource=collections.defaultdict(lambda:collections.defaultdict(list))
 for i in idx:bysource[rows[i]['source']][rows[i]['task']].append(i)
 source_pairs={}
 for src,by in sorted(bysource.items()):
  positives=[]
  for t,ids in sorted(by.items()):
   ps=[(a,b,1) for a,b in itertools.combinations(ids,2) if valid(a,b)];rng.shuffle(ps);positives.extend(ps[:8])
  neg=[]
  for a,b,_ in positives:
   options=[i for t,ids in by.items() if t!=rows[a]['task'] for i in ids if rows[i]['mode']==rows[b]['mode'] and valid(a,i)]
   if options:neg.append((a,rng.choice(options),0))
  n=min(len(positives),len(neg));rng.shuffle(positives);rng.shuffle(neg);source_pairs[src]=(positives[:n],neg[:n])
 n=min(min(len(p),len(q)) for p,q in source_pairs.values());assert n>0
 pairs=[]
 for p,q in source_pairs.values():pairs.extend(p[:n]+q[:n])
 rng.shuffle(pairs);return pairs
def gallery(idx,seed):
 rng=random.Random(seed);by=collections.defaultdict(list)
 for i in idx:by[rows[i]['task'],rows[i]['mode']].append(i)
 tasks=sorted({rows[i]['task'] for i in idx});pairs=[];spans=[]
 for q in idx:
  mode=('agent' if rows[q]['mode']=='human' else 'human') if rows[q]['source']=='cth' else rows[q]['mode']
  pos=[i for i in by[rows[q]['task'],mode] if valid(q,i)];neg=[t for t in tasks if t!=rows[q]['task'] and any(valid(q,i) for i in by[t,mode])]
  if not pos or len(neg)<7:continue
  chosen=rng.sample(neg,7);start=len(pairs);pairs.append((q,rng.choice(pos),1))
  for t in chosen:pairs.append((q,rng.choice([i for i in by[t,mode] if valid(q,i)]),0))
  spans.append((q,start,len(pairs)))
 return pairs,spans
def features(train,pairs):
 base,_=mix.prepare_features(rows,train,list(range(len(rows))),SEED);x=base(pairs);raw=np.vstack([np.r_[r['B_summary'],r['S_summary']] for r in rows]);sc=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()).fit(raw[train]);v=sc.transform(raw);delta=np.asarray([np.abs(v[a]-v[b]) for a,b,_ in pairs]);return {'evidence':x[:,:2],'gated_components':np.c_[x[:,:2],x[:,0:1]*delta]}
def fit(x,y,c):return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),ExtraTreesClassifier(n_estimators=64,max_depth={.1:3,1.:5,10.:8}[c],min_samples_leaf=3,class_weight='balanced',max_features=1.0,n_jobs=1,random_state=SEED)).fit(x,y)
def task_ba(scores,labels,tasks,th):
 out={}
 for t in sorted(set(tasks)):
  mask=np.array([z==t for z in tasks]);y=labels[mask];pred=scores[mask]>=th
  if np.any(y==0) and np.any(y==1):out[t]=1-(np.mean(pred[y==0])+np.mean(~pred[y==1]))/2
 return out
def threshold(scores,labels,tasks):
 best=(-1.,0.)
 for t in np.unique(np.quantile(scores,np.linspace(0,1,101))):
  vals=task_ba(scores,labels,tasks,t);v=np.mean(list(vals.values()))
  if v>best[0]+1e-12:best=(float(v),float(t))
 return best
out={}
for target in ('cth','antieval','kypo'):
 train_sources=[s for s in ('cth','antieval','kypo') if s!=target];train=[i for i,r in enumerate(rows) if r['source'] in train_sources];test=[i for i,r in enumerate(rows) if r['source']==target];oof=collections.defaultdict(list);ys=[];ts=[]
 for k,valid_source in enumerate(train_sources):
  tr=[i for i in train if rows[i]['source']!=valid_source];val=[i for i in train if rows[i]['source']==valid_source];tp=sample(tr,SEED+k);vp,_=gallery(val,SEED+k);assert vp
  X,V=features(tr,tp),features(tr,vp);y=np.array([p[2] for p in tp]);ys.extend(p[2] for p in vp);ts.extend(rows[p[0]]['source']+'/'+rows[p[0]]['task'] for p in vp)
  for method in METHODS:
   for c in CS:oof[method,c].extend(fit(X[method],y,c).predict_proba(V[method])[:,1])
 selected={};audit=[];yv=np.array(ys)
 for method in METHODS:
  opts=[]
  for c in CS:
   score,th=threshold(np.asarray(oof[method,c]),yv,ts);opts.append((score,-c,th,c));audit.append({'method':method,'depth_selector':c,'inner_ba':score,'threshold':th})
  score,_,th,c=max(opts);selected[method]=(c,th)
 tp=sample(train,SEED+100);vp,spans=gallery(test,SEED+200);assert spans;X,V=features(train,tp),features(train,vp);y=np.array([p[2] for p in tp]);pred=[]
 for method,(c,th) in selected.items():
  scores=fit(X[method],y,c).predict_proba(V[method])[:,1]
  for q,start,end in spans:
   v=scores[start:end]>=th;ba=1-((1-int(v[0]))+np.mean(v[1:]))/2;pred.append({'task':rows[q]['task'],'unit':rows[q]['participant_id'] if target=='kypo' else rows[q]['task'],'query':rows[q]['trace_id'],'method':method,'ba':float(ba)})
 by=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in pred:by[r['method']][r['task'],r['unit']].append(r['ba'])
 units=sorted({u for t,u in by['evidence']});tasks=sorted({t for t,u in by['evidence']});arr={}
 for method in METHODS:
  mat=np.full((len(units),len(tasks)),np.nan)
  for i,u in enumerate(units):
   for j,t in enumerate(tasks):
    if (t,u) in by[method]:mat[i,j]=np.mean(by[method][t,u])
  arr[method]=mat
 rng=np.random.default_rng(SEED);ix=rng.integers(0,len(units),(5000,len(units)))
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);a=float(np.nanmean(np.nanmean(arr['gated_components'],axis=0)));b=float(np.nanmean(np.nanmean(arr['evidence'],axis=0)));ab=np.nanmean(np.nanmean(arr['gated_components'][ix],axis=1),axis=1);bb=np.nanmean(np.nanmean(arr['evidence'][ix],axis=1),axis=1)
 out[target]={'test_records':len(test),'queries':len(spans),'units':len(units),'fingerprint_ba':a,'evidence_ba':b,'delta_ba':a-b,'paired_cluster_ci95':np.nanquantile(ab-bb,[.025,.975]).tolist(),'selected':selected,'inner_results':audit,'query_predictions':pred};print(target,'done',out[target]['delta_ba'],flush=True)
p=ROOT/'research/results/association_fingerprint_v3/ba_leave_source_out.json';p.write_text(json.dumps({'protocol':'post-observation exploratory; entire target source held out; 8 candidates; inner leave-one-training-source-out; source-balanced same-source negative sampling; 8-event fingerprint; no source identifiers in feature vectors','results':out},indent=2)+'\n');print(json.dumps({k:{x:v for x,v in z.items() if x not in ('query_predictions','inner_results')} for k,z in out.items()},indent=2))
