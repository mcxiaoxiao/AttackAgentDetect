"""Windowed, interpretable execution fingerprints and masked DTW distance."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .traces import ATOM_GROUP

FEATURES = tuple(f"K{i}" for i in range(1, 9))


def core8(commands: Sequence[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Return the frozen common-log 8-feature definition for one ordered window."""
    names = [str(command) for command in commands]
    n = len(names)
    if n == 0:
        return np.full(8, np.nan), {"commands": 0, "mapped_actions": 0, "eligible_pairs": 0}
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    actions = [ATOM_GROUP.get(command) for command in names]
    pairs = [(a, b) for a, b in zip(actions, actions[1:]) if a is not None and b is not None]
    n_pairs = len(pairs)
    pair_counts: dict[tuple[str, str], int] = {}
    outgoing: dict[str, int] = {}
    for a, b in pairs:
        pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
        outgoing[a] = outgoing.get(a, 0) + 1

    probs = [count / n for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    runs = longest = 0
    previous = None
    for action in actions:
        if action is None:
            runs = 0
        else:
            runs = runs + 1 if action == previous else 1
            longest = max(longest, runs)
        previous = action
    cond_entropy = (
        -sum((count / n_pairs) * math.log2(count / outgoing[a])
             for (a, _), count in pair_counts.items())
        if n_pairs else np.nan
    )
    values = np.asarray([
        len(counts) / n,
        entropy,
        max(counts.values()) / n,
        sum(a != b for a, b in pairs) / n_pairs if n_pairs else np.nan,
        cond_entropy,
        longest / n if longest else np.nan,
        1.0 - len(pair_counts) / n_pairs if n_pairs else np.nan,
        sum(a == b for a, b in zip(names, names[1:])) / (n - 1) if n > 1 else np.nan,
    ], dtype=float)
    if not np.isfinite(values[~np.isnan(values)]).all():
        raise ValueError("Non-finite core fingerprint value")
    return values, {"commands": n, "mapped_actions": sum(a is not None for a in actions), "eligible_pairs": n_pairs}


def matrix(commands: Sequence[str], window: int = 8, max_commands: int = 128,
           min_commands: int = 16, min_tail: int = 4, min_pairs: int = 5):
    """Build a variable-row matrix without assigning fixed attack phases."""
    prefix = list(commands[:max_commands])
    if len(prefix) < min_commands:
        return None, {"reason": "short_prefix", "commands": len(prefix), "windows": 0}
    rows, quality = [], []
    for start in range(0, len(prefix), window):
        chunk = prefix[start:start + window]
        if len(chunk) < min_tail:
            break
        row, audit = core8(chunk)
        rows.append(row)
        quality.append(audit)
    if not rows:
        return None, {"reason": "no_eligible_window", "commands": len(prefix), "windows": 0}
    return np.vstack(rows), {"reason": None, "commands": len(prefix), "windows": len(rows),
                             "eligible_pairs": sum(item["eligible_pairs"] for item in quality),
                             "mapped_actions": sum(item["mapped_actions"] for item in quality)}


def summary_features(values: np.ndarray) -> np.ndarray:
    """Fixed 24-value readout: per-column mean, population SD, and OLS slope."""
    if values.ndim != 2 or values.shape[1] != 8 or values.shape[0] == 0:
        raise ValueError("Expected a non-empty J x 8 fingerprint matrix")
    x = np.linspace(0.0, 1.0, values.shape[0]) if values.shape[0] > 1 else np.zeros(1)
    output = []
    for column in range(8):
        y = values[:, column]
        valid = np.isfinite(y)
        if not valid.any():
            output.extend([np.nan, np.nan, np.nan])
            continue
        observed = y[valid]
        output.append(float(np.mean(observed)))
        output.append(float(np.std(observed, ddof=0)))
        if valid.sum() < 2:
            output.append(np.nan)
        else:
            xv = x[valid]
            centered = xv - xv.mean()
            denom = float(np.dot(centered, centered))
            output.append(float(np.dot(centered, observed - observed.mean()) / denom) if denom else np.nan)
    result = np.asarray(output, dtype=float)
    if result.shape != (24,):
        raise AssertionError("Window readout dimension changed")
    return result


def fit_column_scale(matrices: Sequence[np.ndarray]):
    """Fit train-only weighted column centers/scales; each session totals weight 1."""
    numerator = np.zeros(8, dtype=float)
    denominator = np.zeros(8, dtype=float)
    for values in matrices:
        weights = np.full(values.shape[0], 1.0 / values.shape[0])
        for col in range(8):
            valid = np.isfinite(values[:, col])
            if valid.any():
                numerator[col] += np.dot(weights[valid], values[valid, col])
                denominator[col] += weights[valid].sum()
    center = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    ss = np.zeros(8, dtype=float)
    denominator.fill(0)
    for values in matrices:
        weights = np.full(values.shape[0], 1.0 / values.shape[0])
        for col in range(8):
            valid = np.isfinite(values[:, col])
            if valid.any():
                ss[col] += np.dot(weights[valid], (values[valid, col] - center[col]) ** 2)
                denominator[col] += weights[valid].sum()
    scale = np.sqrt(np.divide(ss, denominator, out=np.ones_like(ss), where=denominator > 0))
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return center, scale


def transform_matrix(values: np.ndarray, center: np.ndarray, scale: np.ndarray):
    result = np.full_like(values, np.nan, dtype=float)
    valid = np.isfinite(values)
    z = (values - center) / scale
    result[valid] = z[valid]
    return result


def dtw_distance(left: np.ndarray, right: np.ndarray, min_shared_columns: int = 3):
    """Unbanded masked DTW; returns RMS local-column cost, or NaN if incomparable."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != 8 or right.shape[1] != 8:
        raise ValueError("DTW requires two J x 8 matrices")
    n, m = left.shape[0], right.shape[0]
    cost = np.full((n + 1, m + 1), np.inf, dtype=float)
    length = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            shared = np.isfinite(left[i - 1]) & np.isfinite(right[j - 1])
            if shared.sum() < min_shared_columns:
                continue
            local = float(np.mean((left[i - 1, shared] - right[j - 1, shared]) ** 2))
            candidates = [(cost[i - 1, j - 1], length[i - 1, j - 1]),
                          (cost[i - 1, j], length[i - 1, j]),
                          (cost[i, j - 1], length[i, j - 1])]
            prior_cost, prior_len = min(candidates, key=lambda item: (item[0], item[1]))
            if np.isfinite(prior_cost):
                cost[i, j] = prior_cost + local
                length[i, j] = prior_len + 1
    if not length[n, m]:
        return float("nan")
    return float(math.sqrt(cost[n, m] / length[n, m]))
