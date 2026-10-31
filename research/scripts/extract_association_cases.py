#!/usr/bin/env python3
"""Audit worked examples from frozen association scores; never tune the models."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_mixed_gallery as gallery
import run_association_operating_points as operating
from run_feature_ablation import grouped_folds

PAIR_FILE = ROOT / 'results/association_operating_points_pairs_v1.csv'
RESULT_FILE = ROOT / 'results/association_operating_points_v1.json'
OUT = ROOT / 'results/association_cases_v1.json'
CATALOG = ROOT / 'results/association_case_catalog_v1.csv'
SELECTION = ('recovered_same_task', 'introduced_cross_task_link', 'persistent_same_task_miss')
ORDER = ('benchmark', 'query_task', 'candidate_task', 'query_trace_id', 'candidate_trace_id')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category(row):
    same = int(row['same_task_proxy_for_evaluation_only'])
    before = int(row['associated_behavior_20'])
    after = int(row['associated_mode_aware_20'])
    return {
        (1, 0, 1): 'recovered_same_task', (1, 0, 0): 'persistent_same_task_miss',
        (1, 1, 0): 'lost_same_task_link', (1, 1, 1): 'retained_same_task_link',
        (0, 0, 1): 'introduced_cross_task_link', (0, 1, 0): 'removed_cross_task_link',
        (0, 1, 1): 'persistent_cross_task_link', (0, 0, 0): 'retained_cross_task_rejection',
    }[(same, before, after)]


def evidence(row, probability):
    return {'trace_id': row.trace_id, 'benchmark': row.benchmark, 'task_for_evaluation_only': row.task,
            'mode_for_evaluation_only': row.label, 'model_source_for_evaluation_only': row.source,
            'command_count': len(row.names), 'command_names': list(row.names),
            'action_categories': list(row.groups), 'predicted_agent_score_raw': float(probability),
            'prediction_at_0_5': 'agent' if probability >= .5 else 'human'}


def select_ranking_flip(human_agent):
    """Find a deterministic Human query whose first result changes to a same-task Agent."""
    grouped = collections.defaultdict(list)
    for row in human_agent:
        grouped[row['query_trace_id']].append(row)
    qualifying = []
    for query_id, gallery_rows in grouped.items():
        relevance = [int(r['same_task_proxy_for_evaluation_only']) for r in gallery_rows]
        before = gallery.expected_rank_metrics([float(r['behavior']) for r in gallery_rows], relevance)
        after = gallery.expected_rank_metrics([float(r['mode_aware']) for r in gallery_rows], relevance)
        base_top = max(float(r['behavior']) for r in gallery_rows)
        mode_top = max(float(r['mode_aware']) for r in gallery_rows)
        base_ties = [r for r in gallery_rows if abs(float(r['behavior']) - base_top) <= 1e-12]
        mode_ties = [r for r in gallery_rows if abs(float(r['mode_aware']) - mode_top) <= 1e-12]
        if (before['hit_at_1'] == 0 and after['hit_at_1'] == 1
                and all(int(r['same_task_proxy_for_evaluation_only']) == 0 for r in base_ties)):
            correct = sorted((r for r in mode_ties
                              if int(r['same_task_proxy_for_evaluation_only']) == 1),
                             key=lambda r: (r['candidate_task'], r['candidate_trace_id']))
            if correct and any(r['candidate_mode_for_evaluation_only'] == 'agent' for r in correct):
                old_top = sorted(base_ties, key=lambda r: (r['candidate_task'], r['candidate_trace_id']))[0]
                right = next(r for r in correct if r['candidate_mode_for_evaluation_only'] == 'agent')
                qualifying.append((gallery_rows[0]['benchmark'], gallery_rows[0]['query_task'], query_id,
                                   right, old_top, len(gallery_rows)))
    qualifying.sort(key=lambda x: (x[0], x[1], x[2], x[3]['candidate_trace_id']))
    return qualifying


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refit', action='store_true', help='Recompute selected outer-fold scores and verify frozen CSV.')
    args = parser.parse_args()
    if not args.refit:
        parser.error('Use --refit to audit the selected scores before generating cases.')
    original = json.loads(RESULT_FILE.read_text())
    expected_hashes = original['provenance']['code_sha256']
    files = {'runner': 'scripts/run_association_operating_points.py',
             'threshold_helpers': 'scripts/run_association_error_analysis.py',
             'feature_source': 'scripts/run_feature_ablation.py',
             'baseline_source': 'scripts/run_mixed_gallery.py', 'trace_parser': 'attackagent/traces.py'}
    for key, name in files.items():
        assert digest(ROOT / name) == expected_hashes[key], f'Original method changed: {name}'
    input_digest = digest(PAIR_FILE)
    with PAIR_FILE.open(newline='') as handle:
        pairs = list(csv.DictReader(handle))
    for line, row in enumerate(pairs, 2):
        row['source_csv_line'] = line
        for method in ('behavior', 'mode_aware'):
            decision = float(row[method]) >= float(row[f'threshold_{method}_20'])
            assert int(decision) == int(row[f'associated_{method}_20'])
            fold = original['folds'][int(row['outer_fold']) - 1]
            assert float(row[f'threshold_{method}_20']) == fold['thresholds'][method]['0.2']['threshold']
    human_agent = [r for r in pairs if r['query_mode_for_evaluation_only'] == 'human'
                   and r['candidate_mode_for_evaluation_only'] == 'agent']
    human_queries = [r for r in pairs if r['query_mode_for_evaluation_only'] == 'human']
    selected = []
    for kind in SELECTION:
        options = sorted((r for r in human_agent if category(r) == kind), key=lambda r: tuple(r[k] for k in ORDER))
        assert options, kind
        selected.append((kind, options[0]))
    ranking_flips = select_ranking_flip(human_queries)
    assert ranking_flips, 'No Human-query Hit@1 correction found.'
    _, _, _, ranking_positive, ranking_old_top, ranking_gallery_size = ranking_flips[0]
    selected.extend([('ranking_correct_agent', ranking_positive), ('ranking_previous_top', ranking_old_top)])
    print('Frozen selection:', [(kind, r['query_trace_id'], r['candidate_trace_id']) for kind, r in selected], flush=True)
    rows = gallery.load_rows()
    by_id = {r.trace_id: r for r in rows}
    folds = grouped_folds(rows, 20261030)
    probabilities, audits = {}, []
    for fold_index in sorted({int(r['outer_fold']) for _, r in selected}):
        train_keys, test_keys = folds[fold_index - 1]
        assert not train_keys & test_keys
        train = [r for r in rows if (r.benchmark, r.task) in train_keys]
        test = [r for r in rows if (r.benchmark, r.task) in test_keys]
        current = [(by_id[r['query_trace_id']], by_id[r['candidate_trace_id']])
                   for _, r in selected if int(r['outer_fold']) == fold_index]
        assert all((q.benchmark, q.task) in test_keys and (c.benchmark, c.task) in test_keys for q, c in current)
        seed = 20261030 + fold_index
        scorer, _, _ = operating.fit_score_model(train, test, seed)
        recomputed = scorer(current)
        all_rows = train + test
        mode, _, _ = gallery.fit_mode_scores(train, all_rows, seed)
        probabilities.update({r.trace_id: float(p) for r, p in zip(all_rows, mode) if r.trace_id in {x.trace_id for pair in current for x in pair}})
        source = [r for _, r in selected if int(r['outer_fold']) == fold_index]
        error = max(abs(float(r[method]) - float(recomputed[method][i]))
                    for i, r in enumerate(source) for method in operating.METHODS)
        assert error < 1e-10, (fold_index, error)
        audits.append({'outer_fold': fold_index, 'seed': seed, 'selected_pairs': len(source),
                       'train_test_task_overlap': 0, 'max_absolute_score_error': error,
                       'frozen_thresholds_match': True})
        print('Verified fold', fold_index, 'maximum score error', error, flush=True)
    cases = []
    for index, (kind, row) in enumerate(selected, 1):
        q, c = by_id[row['query_trace_id']], by_id[row['candidate_trace_id']]
        assert q.label == 'human'
        if kind != 'ranking_previous_top':
            assert c.label == 'agent'
        assert int(q.task == c.task and q.benchmark == c.benchmark) == int(row['same_task_proxy_for_evaluation_only'])
        decision = {}
        gallery_rows = [r for r in pairs if r['query_trace_id'] == q.trace_id]
        for method in ('behavior', 'mode_aware'):
            score = float(row[method]); threshold = float(row[f'threshold_{method}_20'])
            scores = [float(r[method]) for r in gallery_rows]
            decision[method] = {'score': score, 'threshold': threshold, 'score_minus_threshold': score - threshold,
                                'linked': bool(score >= threshold),
                                'candidate_rank_low': 1 + sum(s > score + 1e-12 for s in scores),
                                'candidate_rank_high': sum(s >= score - 1e-12 for s in scores),
                                'query_hit_at_1': gallery.expected_rank_metrics(scores, [int(r['same_task_proxy_for_evaluation_only']) for r in gallery_rows])['hit_at_1']}
        cases.append({'case_id': f'C{index}', 'category': kind, 'outer_fold': int(row['outer_fold']),
                      'source_csv_line': row['source_csv_line'], 'query': evidence(q, probabilities[q.trace_id]),
                      'candidate': evidence(c, probabilities[c.trace_id]), 'decisions': decision})
    counts = dict(collections.Counter(category(r) for r in human_agent))
    output = {'protocol': 'association-cases-v1', 'analysis_type': 'post-hoc worked examples, no new performance estimate',
              'selection': {'threshold_cases_scope': 'human-query to Agent-candidate pairs only; reverse directions excluded',
                            'threshold_cases_rule': 'lexicographically first pair in each decision category', 'order': list(ORDER),
                            'categories': list(SELECTION), 'threshold_target': 0.2,
                            'ranking_case_scope': 'all candidates in the same held-out benchmark/fold gallery for Human queries',
                            'ranking_flip': {'rule': 'first Human query whose behavior top tie is entirely different-task and whose mode-aware Hit@1 is 1.0 with a same-task Agent at the top',
                                             'qualifying_query_count': len(ranking_flips),
                                             'selected_query_id': ranking_flips[0][2],
                                             'candidate_gallery_size': ranking_gallery_size,
                                             'cases': ['ranking_correct_agent', 'ranking_previous_top']},
                            'representativeness': 'Illustrative post-hoc examples; not random samples or evidence of prevalence.'},
              'pair_counts': {'human_query_agent_candidate': len(human_agent), 'same_task': sum(int(r['same_task_proxy_for_evaluation_only']) for r in human_agent), 'by_category': counts,
                              'interpretation': 'Unweighted directed pair counts have different class denominators; they are not the benchmark/task-macro rates.'},
              'cases': cases, 'refit_audit': audits,
              'provenance': {'dataset_revision': original['dataset']['revision'], 'data_license': original['dataset']['license'],
                             'input_pair_csv_sha256': input_digest, 'input_summary_sha256': digest(RESULT_FILE),
                             'script_sha256': digest(Path(__file__)), 'git_head_at_run': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()},
              'interpretation': 'The before/after contrast changes the representation and refits the ranker. It does not identify a causal effect of Agent execution or reproduce a published APT correlation system. Mode scores are raw detector outputs, not calibrated probabilities of actor identity.'}
    assert digest(PAIR_FILE) == input_digest
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    fields = [*ORDER, 'outer_fold', 'source_csv_line', 'same_task_proxy_for_evaluation_only', 'category',
              'behavior', 'threshold_behavior_20', 'mode_aware', 'threshold_mode_aware_20']
    with CATALOG.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in sorted(human_agent, key=lambda r: tuple(r[k] for k in ORDER)):
            writer.writerow({**{k: row[k] for k in fields if k != 'category'}, 'category': category(row)})
    print(json.dumps({'counts': counts, 'ranking_flip_queries': len(ranking_flips),
                      'cases': [{'case': c['case_id'], 'category': c['category'],
                                 'query_task': c['query']['task_for_evaluation_only'],
                                 'candidate_task': c['candidate']['task_for_evaluation_only'],
                                 'query_hit_at_1_before_after': [c['decisions']['behavior']['query_hit_at_1'], c['decisions']['mode_aware']['query_hit_at_1']]}
                                for c in cases]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
