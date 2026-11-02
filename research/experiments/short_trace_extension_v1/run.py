"""Fixed exploratory short-trace extension; all comparisons are reported."""
import collections
import csv
import hashlib
import importlib.util
import itertools
import json
import pathlib
import sys

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from attackagent.fingerprints import core8, matrix, summary_features

spec = importlib.util.spec_from_file_location('overnight', ROOT / 'research/experiments/overnight_fingerprint_v1/run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
OUT = ROOT / 'research/results/short_trace_extension_v1'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261102
METHODS = ('length', 'static8', 'matrix24', 'shuffle24')


def bootstrap(values):
    a = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    return np.quantile(a[rng.integers(0, len(a), (10000, len(a)))].mean(axis=1), [.025, .975]).tolist()


def source_run(source):
    raw = [x for x in r.load_corpus() if x['source_dataset'] == source and x['track'] == 'ctf' and x['mode'] == 'agent']
    rows, seen = [], set()
    for x in raw:
        commands = x['commands'][:128]
        key = (x['task_group'], tuple(commands))
        if len(commands) < 8 or key in seen:
            continue
        seen.add(key)
        mat, _ = matrix(commands, window=4, min_commands=8, min_tail=4)
        rng = np.random.default_rng(SEED + int(x['trace_id'][:8], 16))
        shuffled = mat[rng.permutation(len(mat))]
        rows.append({**x, 'commands': commands, 'features': {
            'length': np.array([np.log1p(len(commands))]),
            'static8': core8(commands)[0],
            'matrix24': summary_features(mat),
            'shuffle24': summary_features(shuffled)}})
    counts = collections.Counter(x['task_group'] for x in rows)
    rows = [x for x in rows if counts[x['task_group']] >= 2]
    tasks = sorted({x['task_group'] for x in rows}, key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    fold_map = {t: i % 3 for i, t in enumerate(tasks)}
    predictions, fold_audit = [], []
    for fold in range(3):
        train = [i for i,x in enumerate(rows) if fold_map[x['task_group']] != fold]
        train_seqs = {tuple(rows[i]['commands']) for i in train}
        test = [i for i,x in enumerate(rows) if fold_map[x['task_group']] == fold and tuple(x['commands']) not in train_seqs]
        by_task = collections.defaultdict(list)
        for i in test:
            by_task[rows[i]['task_group']].append(i)
        by_task = {t: ids for t,ids in by_task.items() if len(ids) >= 2}
        if len(by_task) < 5:
            fold_audit.append({'fold':fold, 'status':'insufficient_test_tasks', 'tasks':len(by_task)})
            continue
        vec = r.vectorizer()
        vec.fit([r.command_text(rows[i]['commands']) for i in train])
        docs = vec.transform([r.command_text(x['commands']) for x in rows])
        cosine = (docs @ docs.T).toarray()
        rng = np.random.default_rng(SEED + fold)
        groups = collections.defaultdict(list)
        for i in train:
            groups[rows[i]['task_group']].append(i)
        positives = [(a,b,1) for ids in groups.values() for a,b in itertools.combinations(ids,2)]
        if len(positives) > 500:
            positives = [positives[i] for i in rng.choice(len(positives),500,replace=False)]
        negatives = set()
        while len(negatives) < min(500,len(positives)):
            a,b = rng.choice(train,2,replace=False)
            if rows[a]['task_group'] != rows[b]['task_group']:
                negatives.add((min(a,b),max(a,b),0))
        pairs = positives + sorted(negatives)
        labels = [p[2] for p in pairs]
        models, features = {}, {}
        for method in METHODS:
            scale = make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler())
            scale.fit(np.vstack([rows[i]['features'][method] for i in train]))
            features[method] = scale.transform(np.vstack([x['features'][method] for x in rows]))
            xp = np.array([[cosine[a,b], *np.abs(features[method][a]-features[method][b])] for a,b,_ in pairs])
            models[method] = make_pipeline(StandardScaler(), LogisticRegression(C=1,class_weight='balanced',solver='liblinear',random_state=SEED,max_iter=2000)).fit(xp,labels)
        fold_audit.append({'fold':fold,'train_sessions':len(train),'eligible_test_tasks':len(by_task),'test_sessions':sum(map(len,by_task.values())),'training_pairs':len(pairs)})
        task_names = sorted(by_task)
        for task in task_names:
            others = [t for t in task_names if t != task]
            for qi in by_task[task]:
                for rep in range(20):
                    positive = int(rng.choice([i for i in by_task[task] if i != qi]))
                    candidates = [positive] + [int(rng.choice(by_task[t])) for t in rng.choice(others,4,replace=False)]
                    scores = {'command':cosine[qi,candidates]}
                    for method in METHODS:
                        xp = np.array([[cosine[qi,ci], *np.abs(features[method][qi]-features[method][ci])] for ci in candidates])
                        scores[method] = models[method].predict_proba(xp)[:,1]
                    for method, s in scores.items():
                        equal = np.isclose(s,s[0],rtol=1e-10,atol=1e-12)
                        above = int(((s > s[0]) & ~equal).sum())
                        ties = int(equal.sum())
                        predictions.append({'source':source,'task':task,'query':rows[qi]['trace_id'],'replicate':rep,'method':method,
                            'hit1':1/ties if above==0 else 0., 'mrr':float(np.mean([1/(above+j) for j in range(1,ties+1)])),
                            'auc':float(np.mean([(float(s[0]>v) if not np.isclose(s[0],v,rtol=1e-10,atol=1e-12) else .5) for v in s[1:]]))})
    metrics=[]
    by=collections.defaultdict(lambda:collections.defaultdict(list))
    for p in predictions:
        by[p['method']][p['task']].append(p)
    for method, task_rows in by.items():
        for metric in ('hit1','mrr','auc'):
            task_values={t:float(np.mean([p[metric] for p in ps])) for t,ps in task_rows.items()}
            base={t:float(np.mean([p[metric] for p in by['command'][t]])) for t in task_values}
            delta=[task_values[t]-base[t] for t in sorted(task_values)]
            metrics.append({'source':source,'method':method,'metric':metric,'value':float(np.mean(list(task_values.values()))),
                'tasks':len(task_values),'paired_delta':float(np.mean(delta)),'paired_95ci':bootstrap(delta)})
    return predictions, {'source':source,'raw_agent_ctf':len(raw),'eligible_sessions_after_dedup':len(rows),'eligible_tasks':len(tasks),'folds':fold_audit,'metrics':metrics}


if __name__ == '__main__':
    results, predictions = [], []
    for source in ('cyber-task-horizons','antieval/cybench-trajectories'):
        pred, result = source_run(source)
        predictions.extend(pred); results.append(result)
        print(json.dumps(result),flush=True)
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    with (OUT/'predictions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=predictions[0]);w.writeheader();w.writerows(predictions)
    (OUT/'provenance.json').write_text(json.dumps({'seed':SEED,'status':'exploratory_followup','runner_sha256':r.sha(pathlib.Path(__file__).read_bytes()),'program_sha256':r.sha(pathlib.Path(__file__).with_name('program.md').read_bytes()),'input_sha256':r.sha((r.CORPUS/'evidence.jsonl').read_bytes())},indent=2)+'\n')
