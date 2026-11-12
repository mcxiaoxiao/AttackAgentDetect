#!/usr/bin/env python3
"""Task-held-out mixed-role retrieval using shared BST blocks."""
import argparse
import collections
import csv
import gzip
import hashlib
import itertools
import json
import pathlib
import random
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.experiments.matched_fingerprint_v1 import run as base
from attackagent.fingerprint_v2 import fingerprint_matrix, summarize, category_features, command_composition_profile

SEED = 20261111


def pairs(rows, indices, label):
    out = []
    for a, b in itertools.combinations(indices, 2):
        same = rows[a]['task'] == rows[b]['task']
        role = rows[a]['mode'] == rows[b]['mode']
        out.append((a, b, int(same and (role or label == 'task'))))
    return out


def galleries(rows, indices, fold, repetitions):
    rng = random.Random(SEED + fold)
    out = []
    for rep in range(repetitions):
        for q in indices:
            row = rows[q]
            groups = [[], [], [], []]
            for c in indices:
                if c == q:
                    continue
                same = rows[c]['task'] == row['task']
                role = rows[c]['mode'] == row['mode']
                groups[0 if same and role else 1 if same else 2 if role else 3].append(c)
            if not all(groups):
                continue
            selected = [rng.choice(g) for g in groups]
            for pos, c in enumerate(selected):
                out.append((q, c, rep, pos))
            alternatives = [c for c in groups[2] if c != selected[2]]
            if alternatives:
                selected[0] = rng.choice(alternatives)
                for pos, c in enumerate(selected):
                    out.append((q, c, rep, pos + 4))
    return out


def make_features(rows, train, minimum):
    docs = [' '.join(r['commands'][:128]) or 'EMPTYTOKEN' for r in rows]
    acts = [base.action_bigram_text(r['actions'][:128]) if hasattr(base, 'action_bigram_text') else ' '.join(str(a)+'__'+str(b) for a,b in zip(r['actions'],r['actions'][1:]) if a is not None and b is not None) or 'EMPTYTOKEN' for r in rows]
    matrices = []
    for corpus in (docs, acts):
        vec = TfidfVectorizer(ngram_range=(1, 2) if corpus is docs else (1, 1), token_pattern=r'(?u)\b[\w.+-]+\b')
        vec.fit([corpus[i] for i in train])
        matrices.append(vec.transform(corpus))
    raw = {k: [] for k in 'BST'}
    core_valid = {k: [] for k in 'BST'}
    presence = []
    for row in rows:
        commands = row['commands'][:128]
        blocks, _ = fingerprint_matrix(commands, actions=row['actions'][:128], timestamps=row['timestamps'], window=8, min_window_events=minimum, max_events=128)
        cats = category_features(commands, row['timestamps'])
        for k in 'BST':
            scalar = summarize(blocks[k]) if blocks is not None else np.full(24, np.nan)
            core_valid[k].append(bool(np.isfinite(scalar).any()))
            raw[k].append(np.r_[scalar, cats[k + '_cat']])
        presence.append(command_composition_profile(commands))
    transformed = {}
    valid = {}
    for k in 'BST':
        values = np.array(raw[k])
        valid[k] = np.asarray(core_valid[k])
        imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(values[train])
        x = imp.transform(values)
        scale = StandardScaler().fit(x[train])
        transformed[k] = scale.transform(x)
    presence = np.asarray(presence, dtype=bool)
    def features(pairlist):
        a = np.array([p[0] for p in pairlist]); b = np.array([p[1] for p in pairlist])
        evidence = np.column_stack([np.asarray(m[a].multiply(m[b]).sum(axis=1)).ravel() for m in matrices])
        blocks = {k: np.abs(v[a] - v[b]) for k,v in transformed.items()}
        blocks['W'] = blocks['B'].copy()
        union = np.logical_or(presence[a], presence[b]).sum(axis=1)
        overlap = np.divide(np.logical_and(presence[a], presence[b]).sum(axis=1), union, out=np.zeros(len(a)), where=union != 0)
        blocks['B'] = np.column_stack([blocks['B'], overlap])
        missing = {k: ~(valid[k][a] & valid[k][b]) for k in 'BST'}
        missing['W'] = missing['B']
        lengths = np.log1p([len(r['commands'][:128]) for r in rows])
        control = np.column_stack([np.abs(lengths[a]-lengths[b]), (lengths[a]+lengths[b])/2, overlap])
        return evidence, blocks, missing, control
    return features


def model_scores(kind, train, labels, test, weights):
    model = base.classifier(kind, 1 if kind == 'svm' else 3)
    if kind == 'svm':
        model.fit(train, labels, svc__sample_weight=weights)
    else:
        model.fit(train, labels, sample_weight=weights)
    return model.decision_function(test) if kind == 'svm' else model.predict_proba(test)[:, 1]


def weights_for(rows, pairlist):
    cells = [(p[2], base.pair_type(rows[p[0]]['mode'], rows[p[1]]['mode'])) for p in pairlist]
    counts = collections.Counter(cells)
    per_label = collections.Counter(label for label, _ in counts)
    return np.asarray([len(pairlist) / (2 * per_label[label] * counts[label, kind]) for label, kind in cells])


def run(dataset, out, repetitions):
    rows, tasks, folds = base.build_cth() if dataset == 'cth' else base.build_ait()
    minimum = 2 if dataset == 'cth' else 4
    out.mkdir(parents=True, exist_ok=True)
    coverage = collections.Counter()
    for row in rows:
        blocks, _ = fingerprint_matrix(row['commands'][:128], actions=row['actions'][:128], timestamps=row['timestamps'], window=8, min_window_events=minimum, max_events=128)
        for k in 'BST':
            coverage[k] += int(blocks is not None and np.isfinite(blocks[k]).any())
    manifest = {'dataset': dataset, 'seed': SEED, 'repetitions': repetitions, 'tasks': len(tasks), 'traces': len(rows), 'coverage': dict(coverage), 'folds': folds, 'runs': [{'id': r['id'], 'task': r['task'], 'mode': r['mode'], 'events': len(r['commands'][:128])} for r in rows], 'runner_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), 'fingerprint_sha256': hashlib.sha256((ROOT/'attackagent/fingerprint_v2.py').read_bytes()).hexdigest()}
    (out/f'{dataset}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    arms = ['E','B','S','BS'] if dataset == 'cth' else ['E','B','S','T','BS','BT','ST','BST']
    arms += ['C', 'CBS' if dataset == 'cth' else 'CBST', 'WS' if dataset == 'cth' else 'WST']
    learners = ['xgb','svm','tree'] if dataset == 'cth' else ['xgb']
    records = []
    for fold in range(3):
        train = [i for i,r in enumerate(rows) if folds[r['task']] != fold]
        test = [i for i,r in enumerate(rows) if folds[r['task']] == fold]
        gallery = galleries(rows, test, fold, repetitions)
        candidate_pairs = [(a,b,0) for a,b,_,_ in gallery]
        all_train = pairs(rows, train, 'mode')
        labels = ['mode', 'task']
        for target in labels:
            trainpairs = pairs(rows, train, target)
            y = np.array([p[2] for p in trainpairs])
            weights = weights_for(rows, trainpairs)
            fmap = make_features(rows, train, minimum)
            e, blocks, missing, control = fmap(trainpairs)
            te, tb, tm, tc = fmap(candidate_pairs)
            active_arms = arms if target == 'mode' else ['E', 'BS' if dataset == 'cth' else 'BST']
            for learner in learners:
                inner = []
                for inner_fold in sorted({folds[rows[i]['task']] for i in train}):
                    fit = [i for i in train if folds[rows[i]['task']] != inner_fold]
                    val = [i for i in train if folds[rows[i]['task']] == inner_fold]
                    ip = pairs(rows, fit, target)
                    vp = pairs(rows, val, target)
                    fm = make_features(rows, fit, minimum)
                    ie, ib, _, ic = fm(ip)
                    ve, vb, vm, vc = fm(vp)
                    iy = np.asarray([p[2] for p in ip])
                    vy = np.asarray([p[2] for p in vp])
                    iw = weights_for(rows, ip)
                    es = model_scores(learner, ie, iy, ve, iw)
                    inner.append((ie, ib, ve, vb, vm, iy, vy, iw, es, ic, vc))
                baseline = model_scores(learner, e, y, te, weights)
                ey = np.concatenate([item[6] for item in inner])
                eoof = np.concatenate([item[8] for item in inner])
                et = base.threshold(ey, eoof)
                for arm in active_arms:
                    cols = '' if arm == 'E' else arm.replace('C', '')
                    controls = [control] if 'C' in arm else []
                    test_controls = [tc] if 'C' in arm else []
                    x = np.column_stack([e] + controls + [blocks[k] for k in cols])
                    xt = np.column_stack([te] + test_controls + [tb[k] for k in cols])
                    score = baseline if arm == 'E' else model_scores(learner, x, y, xt, weights)
                    fallback = np.logical_or.reduce([tm[k] for k in cols]) if cols else np.zeros(len(te),dtype=bool)
                    score = np.where(fallback, baseline, score)
                    if arm != 'E':
                        oof_parts, missing_parts = [], []
                        for ie, ib, ve, vb, vm, iy, vy, iw, es, ic, vc in inner:
                            ix = np.column_stack([ie] + ([ic] if 'C' in arm else []) + [ib[k] for k in cols])
                            vx = np.column_stack([ve] + ([vc] if 'C' in arm else []) + [vb[k] for k in cols])
                            miss = np.logical_or.reduce([vm[k] for k in cols]) if cols else np.zeros(len(ve), dtype=bool)
                            oof_parts.append(model_scores(learner, ix, iy, vx, iw))
                            missing_parts.append(miss)
                        oof = np.concatenate(oof_parts)
                        oof_missing = np.concatenate(missing_parts)
                        try:
                            threshold = base.threshold(ey, oof, oof_missing, eoof >= et)
                        except ValueError:
                            threshold = float('inf')
                    else:
                        threshold = et
                    for j,(a,b,rep,pos) in enumerate(gallery):
                        cutoff = et if fallback[j] else threshold
                        records.append({'dataset':dataset,'fold':fold,'target':target,'learner':learner,'arm':arm,'query_id':rows[a]['id'],'candidate_id':rows[b]['id'],'query_task':rows[a]['task'],'candidate_task':rows[b]['task'],'query_mode':rows[a]['mode'],'candidate_mode':rows[b]['mode'],'replicate':rep,'position':pos,'same_task':int(rows[a]['task']==rows[b]['task']),'label':int(rows[a]['task']==rows[b]['task'] and (target=='task' or rows[a]['mode']==rows[b]['mode'])),'score':float(score[j]),'baseline_score':float(baseline[j]),'threshold':float(cutoff),'predicted_link':int(score[j]>=cutoff),'fallback':int(fallback[j])})
        print(dataset, 'fold', fold, 'galleries', len(gallery)//4, flush=True)
    out.mkdir(parents=True,exist_ok=True)
    with gzip.open(out/f'{dataset}_predictions.csv.gz','wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    print('saved',len(records),'predictions',flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['cth','ait'],required=True); p.add_argument('--out',type=pathlib.Path,required=True); p.add_argument('--repetitions',type=int,default=100)
    args=p.parse_args(); run(args.dataset,args.out,args.repetitions)
