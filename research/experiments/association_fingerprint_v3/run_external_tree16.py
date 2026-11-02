#!/usr/bin/env python3
"""Frozen BA fusion pilot on held-out Antieval tasks or KYPO participants."""
import collections,csv,hashlib,importlib.util,itertools,json,pathlib,random,sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('cross',ROOT/'research/experiments/behavioral_fingerprint_v2/run_cth_cross_mode.py');cross=importlib.util.module_from_spec(spec);spec.loader.exec_module(cross);mix=cross.mix
SEED=20261103;CS=(.1,1.,10.);METHODS=('evidence','additive','gated','gated_components');source=sys.argv[1];assert source in ('antieval','kypo')
raw=json.loads((ROOT/'site/data/behavior-trace-dataset-v2.json').read_text())['records'];wanted='antieval/cybench-trajectories' if source=='antieval' else 'kypo-locust-human-training';rows=[];seen=set()
for x in raw:
 if x['source_dataset']!=wanted:continue
 cmds=x['commands']; key=(x['task_group'],tuple(cmds))
 if len(cmds)<4 or (source=='antieval' and key in seen):continue
 seen.add(key);blocks,audit=mix.fingerprint_matrix(cmds,window=16,min_window_events=4,max_events=128)
 if blocks is None:continue
 rows.append({**x,'task':x['task_group'],'profile_eligible':True,'B':blocks['B'],'S':blocks['S'],'B_summary':mix.summarize(blocks['B']),'S_summary':mix.summarize(blocks['S']),'actions':[mix.ATOM_GROUP.get(z) for z in cmds]})
if source=='antieval':
 counts=collections.Counter(r['task'] for r in rows);rows=[r for r in rows if counts[r['task']]>=2]
groupkey='task' if source=='antieval' else 'participant_id';groups=sorted({r[groupkey] for r in rows},key=lambda x:hashlib.sha256(x.encode()).hexdigest());nf=3 if source=='antieval' else 5;folds={x:i%nf for i,x in enumerate(groups)}
def valid_pair(i,j):return i!=j and (source!='kypo' or rows[i]['participant_id']!=rows[j]['participant_id'])
def train_pairs(idx,seed):
 rng=random.Random(seed);by=collections.defaultdict(list)
 for i in idx:by[rows[i]['task']].append(i)
 positives=[]
 for t,ids in sorted(by.items()):
  p=[(a,b,1) for a,b in itertools.combinations(ids,2) if valid_pair(a,b)];rng.shuffle(p);positives.extend(p[:8])
 negatives=[];tasks=sorted(by)
 for a,b,_ in positives:
  options=[j for t in tasks if t!=rows[a]['task'] for j in by[t] if valid_pair(a,j)]
  if options:negatives.append((a,rng.choice(options),0))
 assert positives and negatives
 out=positives+negatives;rng.shuffle(out);return out
def gallery(idx,seed):
 rng=random.Random(seed);by=collections.defaultdict(list)
 for i in idx:by[rows[i]['task']].append(i)
 pairs=[];spans=[]
 for q in idx:
  pos=[j for j in by[rows[q]['task']] if valid_pair(q,j)]
  negatives=[t for t,ids in by.items() if t!=rows[q]['task'] and any(valid_pair(q,j) for j in ids)]
  if not pos or len(negatives)<2:continue
  chosen=sorted(negatives);start=len(pairs);pairs.append((q,rng.choice(pos),1))
  for t in chosen:pairs.append((q,rng.choice([j for j in by[t] if valid_pair(q,j)]),0))
  spans.append((q,start,len(pairs)))
 return pairs,spans
def feat_builder(train):
 base,_=mix.prepare_features(rows,train,list(range(len(rows))),SEED)
 raw=np.vstack([np.r_[r['B_summary'],r['S_summary']] for r in rows]);sc=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()).fit(raw[train]);v=sc.transform(raw)
 def f(pairs):
  x=base(pairs);e=x[:,:2];gate=x[:,0:1];delta=np.asarray([np.abs(v[a]-v[b]) for a,b,_ in pairs]);return {'evidence':e,'additive':np.c_[e,x[:,2:4]],'gated':np.c_[e,gate*x[:,2:4]],'gated_components':np.c_[e,gate*delta]}
 return f
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
pred=[];audits=[]
for fold in range(nf):
 train=[i for i,r in enumerate(rows) if folds[r[groupkey]]!=fold];test=[i for i,r in enumerate(rows) if folds[r[groupkey]]==fold]
 if source=='antieval':
  seqs={tuple(rows[i]['commands']) for i in train};test=[i for i in test if tuple(rows[i]['commands']) not in seqs]
 innergroups=sorted({rows[i][groupkey] for i in train},key=lambda x:hashlib.sha256(x.encode()).hexdigest());innerfold={t:i%3 for i,t in enumerate(innergroups)};oof=collections.defaultdict(list);yall=[];tall=[]
 for inner in range(3):
  tr=[i for i in train if innerfold[rows[i][groupkey]]!=inner];val=[i for i in train if innerfold[rows[i][groupkey]]==inner];tp=train_pairs(tr,SEED+fold*100+inner);vp,_=gallery(val,SEED+fold*200+inner);assert vp
  f=feat_builder(tr);X,V=f(tp),f(vp);y=np.array([z[2] for z in tp]);yall.extend(z[2] for z in vp);tall.extend(rows[z[0]]['task'] for z in vp)
  for method in METHODS:
   for c in CS:oof[method,c].extend(fit(X[method],y,c).predict_proba(V[method])[:,1])
 selected={};audit=[];yv=np.array(yall)
 for method in METHODS:
  opts=[]
  for c in CS:
   score,th=threshold(np.asarray(oof[method,c]),yv,tall);opts.append((score,-c,th,c));audit.append({'method':method,'C':c,'inner_ba':score,'threshold':th})
  score,_,th,c=max(opts);selected[method]=(c,th)
 tp=train_pairs(train,SEED+fold*1000);vp,spans=gallery(test,SEED+fold*2000);assert spans
 f=feat_builder(train);X,V=f(tp),f(vp);y=np.array([z[2] for z in tp])
 for method,(c,th) in selected.items():
  scores=fit(X[method],y,c).predict_proba(V[method])[:,1]
  for q,start,end in spans:
   x=scores[start:end]>=th;ba=1-((1-int(x[0]))+np.mean(x[1:]))/2
   pred.append({'fold':fold,'task':rows[q]['task'],'participant':rows[q].get('participant_id',''),'query':rows[q]['trace_id'],'method':method,'ba':float(ba)})
 audits.append({'fold':fold,'test_groups':len({rows[i][groupkey] for i in test}),'queries':len(spans),'selected':selected,'inner_results':audit});print(source,'fold',fold,'done',flush=True)
by=collections.defaultdict(lambda:collections.defaultdict(list))
for r in pred:by[r['method']][(r['task'],r['participant'] if source=='kypo' else r['task'])].append(r['ba'])
units=sorted(set(k[1] for k in by['evidence']));tasks=sorted(set(k[0] for k in by['evidence']));arrays={}
for method in METHODS:
 mat=np.full((len(units),len(tasks)),np.nan)
 for i,u in enumerate(units):
  for j,t in enumerate(tasks):
   if (t,u) in by[method]:mat[i,j]=np.mean(by[method][t,u])
 arrays[method]=mat
rng=np.random.default_rng(SEED);ix=rng.integers(0,len(units),(5000,len(units)));summary=[]
for method in METHODS:
 vals=arrays[method];base=arrays['evidence'];mean=float(np.nanmean(np.nanmean(vals,axis=0)));basemean=float(np.nanmean(np.nanmean(base,axis=0)));bs=np.nanmean(np.nanmean(vals[ix],axis=1),axis=1);bbase=np.nanmean(np.nanmean(base[ix],axis=1),axis=1)
 summary.append({'method':method,'ba':mean,'delta_ba':mean-basemean,'paired_cluster_ci95':np.nanquantile(bs-bbase,[.025,.975]).tolist()})
out=ROOT/'research/results/association_fingerprint_v3';out.mkdir(parents=True,exist_ok=True)
(out/f'ba_tree16_external_{source}.json').write_text(json.dumps({'source':source,'records':len(rows),'task_groups':len({r['task'] for r in rows}),'holdout':groupkey,'independent_units':len(units),'protocol':'16-event window shallow tree feature families and nested depth/threshold selection; same candidate pairs; retrospective exploratory evaluation','summary':summary,'folds':audits,'query_predictions':pred},indent=2)+'\n');print(json.dumps(summary,indent=2))
