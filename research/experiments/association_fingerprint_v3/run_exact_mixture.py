#!/usr/bin/env python3
"""Paired exact-role mixture on the original 17 CTH tasks."""
import collections, csv, importlib.util, json, pathlib, random
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('legacy',ROOT/'research/experiments/overnight_fingerprint_v1/run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OUT=ROOT/'research/results/association_fingerprint_v3'; OUT.mkdir(parents=True,exist_ok=True)
rows=[r for r in m.load_corpus() if r['source_dataset']=='cyber-task-horizons' and r['track']=='ctf']
pools=collections.defaultdict(lambda: {'human':[],'agent':[]})
for i,r in enumerate(rows): pools[r['task_group']][r['mode']].append(i)
tasks=sorted(t for t,p in pools.items() if len(p['human'])>=2 and len(p['agent'])>=2)
assert len(tasks)==17
with (ROOT/'research/results/overnight_fingerprint_v1/task_folds.csv').open() as f:
    folds={r['group']:int(r['fold']) for r in csv.DictReader(f) if r['endpoint']=='cth_mixture'}
methods=('command_cosine','action_bigram_cosine'); cache={}; thresholds={}
for fold in sorted(set(folds[t] for t in tasks)):
    train=[i for i,r in enumerate(rows) if r['task_group'] in tasks and folds[r['task_group']]!=fold]
    test=[i for i,r in enumerate(rows) if r['task_group'] in tasks and folds[r['task_group']]==fold]
    train_tasks={rows[i]['task_group'] for i in train}
    for method in methods:
        cache[fold,method]=m.cos_matrix(rows,train,test,method)
        thresholds[fold,method]=m.inner_threshold(rows,train,train_tasks,method,'cyber-task-horizons',fold)[0]
results=collections.defaultdict(lambda: collections.defaultdict(list))
for rep in range(170):
    rng=random.Random(20261103+rep); excluded=tasks[rep%17]; chosen=[t for t in tasks if t!=excluded]
    slots=[(t,role) for t in chosen for role in ('query','gallery')]; order=list(range(32)); rng.shuffle(order)
    picked={t:{mode:rng.sample(pools[t][mode],2) for mode in ('human','agent')} for t in chosen}
    negatives={t:rng.sample([x for x in chosen if folds[x]==folds[t] and x!=t],3) for t in chosen}
    for count in (0,8,16,24,32):
        agent=set(order[:count]); modes={slot:('agent' if k in agent else 'human') for k,slot in enumerate(slots)}
        assert sum(v=='agent' for v in modes.values())==count
        for t in chosen:
            fold=folds[t]; qi=picked[t][modes[t,'query']][0]; candidates=[t]+negatives[t]
            ids=[picked[u][modes[u,'gallery']][1] for u in candidates]
            for method in methods:
                mat,mapping=cache[fold,method]; scores=[m.cosine(mat[mapping[qi]],mat[mapping[c]]) for c in ids]
                pred=[s>=thresholds[fold,method] for s in scores]
                fmr=sum(pred[1:])/3; fsr=1-int(pred[0]); ba=1-(fmr+fsr)/2
                results[count,method][t].append((fmr,fsr,ba))
summary=[]
for (count,method), taskvals in sorted(results.items()):
    values=np.array([np.mean(taskvals[t],axis=0) for t in tasks]); rng=np.random.default_rng(20261103)
    boots=values[rng.integers(0,len(tasks),(10000,len(tasks)))].mean(axis=1)
    summary.append({'agent_fraction':count/32,'method':method,'task_count':len(tasks),'replicates':170,'fmr':float(values[:,0].mean()),'fsr':float(values[:,1].mean()),'ba':float(values[:,2].mean()),'ba_task_cluster_95ci':np.quantile(boots[:,2],[.025,.975]).tolist()})
task_means={f'{count}:{method}':{t:np.mean(vals,axis=0).tolist() for t,vals in taskvals.items()} for (count,method),taskvals in results.items()}
endpoint={}
for method in methods:
    delta=np.asarray([task_means[f'32:{method}'][t][2]-task_means[f'0:{method}'][t][2] for t in tasks])
    rng=np.random.default_rng(20261103); boot=delta[rng.integers(0,len(tasks),(10000,len(tasks)))].mean(axis=1)
    endpoint[method]={'delta_ba':float(delta.mean()),'task_cluster_95ci':np.quantile(boot,[.025,.975]).tolist()}
output={'task_means':task_means,'paired_endpoint':endpoint,'protocol':'original 17-task cohort; 16 query/gallery pairs per draw, 32 unique roles; 170 coupled draws; 1 same-task and 3 other-task candidates; original training-side FMR<=10% threshold','seed':20261103,'results':summary,'thresholds':{str(k):v for k,v in thresholds.items()}}
(OUT/'exact_mixture.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2))
