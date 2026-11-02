import math

import numpy as np

from attackagent.fingerprint_v2 import (
    apply_scale,
    combine_summary,
    fingerprint_matrix,
    fit_scale,
    local8,
    masked_dtw,
    summarize,
)
from attackagent.fingerprints import core8
from attackagent.traces import ATOM_GROUP


def test_b_block_matches_frozen_k8_on_known_terminal_actions():
    commands = ["ls", "cat", "cat", "chmod", "ls", "cat"]
    actions = [ATOM_GROUP.get(x) for x in commands]
    expected, _ = core8(commands)
    actual = local8(commands, actions)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_short_trace_gets_one_window_without_old_minimum_length_gate():
    blocks, audit = fingerprint_matrix(["ls", "cat", "pwd", "ls"], window=8)
    assert audit["windows"] == 1
    assert blocks["B"].shape == (1, 8)
    assert blocks["S"].shape == (1, 8)
    assert blocks["T"].shape == (1, 8)
    assert np.isnan(blocks["T"]).all()


def test_less_than_minimum_window_events_is_explicitly_unavailable():
    blocks, audit = fingerprint_matrix(["ls", "cat", "pwd"], window=8)
    assert blocks is None
    assert audit["reason"] == "no_valid_window"


def test_sequence_block_counts_revisits_and_reused_fragments():
    blocks, _ = fingerprint_matrix(["a", "b", "a", "b", "a", "b", "a", "b"],
                                   actions=["x"] * 8, window=4)
    assert blocks is not None
    # The second window revisits all symbols and repeats command fragments.
    assert blocks["S"][1, 0] == 0
    assert blocks["S"][1, 1] == 1
    assert blocks["S"][1, 5] == 1
    assert blocks["S"][1, 6] == 1


def test_timing_features_use_real_aligned_start_times_and_durations():
    blocks, audit = fingerprint_matrix(
        ["a", "b", "c", "d"], actions=["x"] * 4,
        timestamps=[0.0, 0.5, 10.5, 11.0], durations=[0.2, 0.2, 0.2, 0.2],
        window=8)
    assert blocks is not None
    assert math.isclose(blocks["T"][0, 3], 2 / 3)
    assert math.isclose(blocks["T"][0, 4], 1 / 3)
    assert math.isclose(blocks["T"][0, 6], math.log1p(0.2))
    assert audit["time"]["valid_gaps"] == 3


def test_nonmonotone_times_are_not_silently_used_as_positive_gaps():
    blocks, audit = fingerprint_matrix(["a", "b", "c", "d"], actions=["x"] * 4,
                                       timestamps=[1, 0, 2, 3], window=8)
    assert audit["time"]["valid_gaps"] == 2
    assert audit["time"]["nonmonotone_or_overlapping"] == 1
    # Only the two valid positive gaps (2s and 1s) are used; one would be <= 1s.
    assert math.isclose(blocks["T"][0, 3], 0.5)


def test_summaries_keep_block_sizes_masks_and_train_only_scaling():
    matrix = np.asarray([[1., np.nan], [3., 5.]])
    result = summarize(matrix)
    assert result.shape == (6,)
    assert result[0] == 2
    assert result[1] == 1
    assert result[2] == 2
    assert np.isnan(result[5])
    blocks, _ = fingerprint_matrix(["ls", "cat", "pwd", "ls"], window=8)
    assert combine_summary(blocks, ("B", "S")).shape == (48,)
    center, scale = fit_scale([blocks["B"]])
    transformed = apply_scale(blocks["B"], center, scale)
    assert transformed.shape == blocks["B"].shape


def test_masked_dtw_supports_arbitrary_block_width_and_missing_cells():
    left = np.asarray([[0., 1., np.nan], [1., 0., 2.]])
    right = np.asarray([[0., 1., 4.], [1., 0., 2.]])
    assert math.isfinite(masked_dtw(left, right, min_shared_columns=2))
    assert math.isnan(masked_dtw(left[:, :1], right[:, :1], min_shared_columns=2))
