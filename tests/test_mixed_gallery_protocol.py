import dataclasses
import unittest

import numpy as np

from attackagent.traces import Observation
from scripts.run_mixed_gallery import expected_rank_metrics, split_tasks, training_pairs


class MixedGalleryProtocolTest(unittest.TestCase):
    def test_multi_positive_metrics_use_expected_tie_breaking(self):
        result = expected_rank_metrics([1.0, 1.0, 1.0], [1, 0, 1])
        self.assertAlmostEqual(result["hit_at_1"], 2 / 3)
        self.assertAlmostEqual(result["mrr"], 5 / 6)
        self.assertEqual(result["recall_at_5"], 1.0)
        expected_dcg = (2 / 3) * (1 + 1 / np.log2(3) + 1 / 2)
        ideal_dcg = 1 + 1 / np.log2(3)
        self.assertAlmostEqual(result["ndcg_at_5"], expected_dcg / ideal_dcg)

    def test_all_tie_mrr_matches_uniform_random_position(self):
        result = expected_rank_metrics([0.0, 0.0, 0.0, 0.0], [1, 0, 0, 0])
        self.assertAlmostEqual(result["hit_at_1"], 0.25)
        self.assertAlmostEqual(result["mrr"], (1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)

    def test_task_splits_are_disjoint_and_benchmark_stratified(self):
        rows = []
        for benchmark, count in (("small", 6), ("large", 27), ("mid", 7)):
            for task_no in range(count):
                rows.append(Observation(str(task_no), f"{benchmark}/{task_no}", "human", "h", ("ls",), ("discover",), benchmark=benchmark))
        folds = split_tasks(rows, seed=20261030, n_folds=3)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertFalse(set(fold["train_tasks"]) & set(fold["test_tasks"]))
        for benchmark, count in (("small", 6), ("large", 27), ("mid", 7)):
            counts = [sum(task.startswith(f"{benchmark}/") for task in fold["test_tasks"]) for fold in folds]
            self.assertTrue(all(value >= 2 for value in counts))
            self.assertEqual(sum(counts), count)

    def test_training_pair_sampling_does_not_depend_on_mode_or_source(self):
        rows = []
        for task_no in range(4):
            for mode, source in (("human", "expert"), ("agent", "model-a"), ("agent", "model-b")):
                rows.append(Observation(
                    trace_id=f"{task_no}-{mode}-{source}", task=f"task-{task_no}", label=mode,
                    source=source, names=("ls",), groups=("discover",), benchmark="demo",
                ))
        first_pairs, first_labels = training_pairs(rows, 9)
        changed = [dataclasses.replace(row, label="agent" if row.label == "human" else "human", source="renamed") for row in rows]
        second_pairs, second_labels = training_pairs(changed, 9)
        self.assertEqual(first_pairs, second_pairs)
        np.testing.assert_array_equal(first_labels, second_labels)


if __name__ == "__main__":
    unittest.main()
