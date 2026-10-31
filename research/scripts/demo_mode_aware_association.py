#!/usr/bin/env python3
"""Run a no-raw-log held-out example with one human query trace."""

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from attackagent.prototype import ModeAwareAssociationPrototype, from_observation  # noqa: E402
from attackagent.traces import load_crossbenchmark_observations, load_observations  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-task", help="held-out benchmark/task ID; default selects the first eligible CyBench task")
    args = parser.parse_args()

    rows = [row for row in load_observations() + load_crossbenchmark_observations() if row.names]
    labels = collections.defaultdict(set)
    for row in rows:
        labels[row.task].add(row.label)
    eligible = sorted(task for task, modes in labels.items() if modes == {"human", "agent"})
    rows = [row for row in rows if row.task in set(eligible)]
    heldout = args.query_task or next(task for task in eligible if task.startswith("cybench/"))
    test_rows = [row for row in rows if row.task == heldout]
    if not test_rows or not {row.label for row in test_rows} >= {"human", "agent"}:
        raise SystemExit(f"Task must have non-empty human and Agent records: {heldout}")

    train_rows = [row for row in rows if row.task != heldout]
    train = [from_observation(row) for row in train_rows]
    query_row = next(row for row in test_rows if row.label == "human")
    test_records = [from_observation(row) for row in test_rows]
    model = ModeAwareAssociationPrototype().fit(train)
    mode_result = model.predict_mode([from_observation(query_row)])[0]

    all_records = train + test_records
    all_rows = train_rows + test_rows
    train_probs = model.training_mode_oof_
    test_probs = [x["agent_like_score"] for x in model.predict_mode(test_records)]
    probs = list(train_probs) + test_probs
    query_index = len(train_rows) + next(i for i, row in enumerate(test_rows) if row.trace_id == query_row.trace_id)
    benchmark = query_row.benchmark
    pairs = []
    candidate_task_for_pair = []
    for j, row in enumerate(all_rows):
        if row.benchmark == benchmark and row.label == "agent" and row.trace_id != query_row.trace_id:
            pairs.append((query_index, j))
            candidate_task_for_pair.append(row.task)
    scores = model.predict_pair_scores(all_records, pairs, agent_probabilities=probs)

    task_scores = collections.defaultdict(list)
    for task, score in zip(candidate_task_for_pair, scores["mode_aware"]):
        task_scores[task].append(float(score))
    ranked = sorted(((task, sum(values) / len(values)) for task, values in task_scores.items()),
                    key=lambda item: (-item[1], item[0]))
    output = {
        "scope": "held-out same-CTF-task retrieval; task ID is a proxy label, not an intrusion/campaign identity",
        "training_task_count": len({row.task for row in train_rows}),
        "heldout_task": heldout,
        "query_trace_id": query_row.trace_id,
        "query_mode_score": mode_result,
        "candidate_ranking": [
            {"rank": rank, "task_id": task, "mean_match_score": round(score, 4), "same_task_proxy": task == heldout}
            for rank, (task, score) in enumerate(ranked, start=1)
        ],
        "score_warning": "Match scores rank candidates under this balanced public benchmark training distribution; they are not calibrated probabilities.",
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
