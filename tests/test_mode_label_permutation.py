import collections
import unittest

import numpy as np

from attackagent.prototype import TraceRecord
from scripts.run_mode_label_permutation import (
    assert_group_counts_preserved,
    permute_modes_within_task,
)


def record(trace_id, benchmark, activity_id, mode):
    return TraceRecord(
        trace_id=trace_id,
        benchmark=benchmark,
        activity_id=activity_id,
        mode=mode,
        commands=("whoami",),
        actions=("exec",),
    )


class ModeLabelPermutationTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("a1", "suite-a", "task-1", "human"),
            record("a2", "suite-a", "task-1", "agent"),
            record("a3", "suite-a", "task-1", "human"),
            record("a4", "suite-a", "task-1", "agent"),
            record("b1", "suite-a", "task-2", "human"),
            record("b2", "suite-a", "task-2", "agent"),
            record("c1", "suite-b", "task-1", "human"),
            record("c2", "suite-b", "task-1", "agent"),
        ]

    def test_permutation_preserves_each_benchmark_task_label_counts(self):
        train_indices = [0, 1, 2, 3, 6, 7]
        labels = permute_modes_within_task(self.records, train_indices, np.random.default_rng(7))
        assert_group_counts_preserved(self.records, labels, train_indices)
        groups = collections.defaultdict(list)
        for row, label in zip(self.records, labels):
            groups[f"{row.benchmark}/{row.activity_id}"].append(str(label))
        for group, observed in groups.items():
            expected = [row.mode for row in self.records if f"{row.benchmark}/{row.activity_id}" == group]
            self.assertEqual(collections.Counter(observed), collections.Counter(expected))

    def test_permutation_is_reproducible_for_a_fixed_seed(self):
        train_indices = [0, 1, 2, 3, 6, 7]
        a = permute_modes_within_task(self.records, train_indices, np.random.default_rng(11))
        b = permute_modes_within_task(self.records, train_indices, np.random.default_rng(11))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(str(a[4]), self.records[4].mode)
        self.assertEqual(str(a[5]), self.records[5].mode)

    def test_groups_are_benchmark_qualified(self):
        train_indices = list(range(len(self.records)))
        labels = permute_modes_within_task(self.records, train_indices, np.random.default_rng(13))
        assert_group_counts_preserved(self.records, labels, train_indices)
        # Same task ID in different benchmark suites remains an independent block.
        first = [str(labels[i]) for i, row in enumerate(self.records) if row.benchmark == "suite-a" and row.activity_id == "task-1"]
        second = [str(labels[i]) for i, row in enumerate(self.records) if row.benchmark == "suite-b" and row.activity_id == "task-1"]
        self.assertEqual(collections.Counter(first), collections.Counter(["human", "agent", "human", "agent"]))
        self.assertEqual(collections.Counter(second), collections.Counter(["human", "agent"]))


if __name__ == "__main__":
    unittest.main()
