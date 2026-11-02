"""Versioned B/S/T behavioral fingerprints for ordered, partially timed traces.

The module deliberately keeps the frozen K1--K8 implementation untouched.
Symbols and action labels are adapter-specific; matrices from different
observation contracts are never compared unless an experiment explicitly says so.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from .traces import ATOM_GROUP

B_NAMES = tuple(f"B{i}" for i in range(1, 9))
S_NAMES = tuple(f"S{i}" for i in range(1, 9))
T_NAMES = tuple(f"T{i}" for i in range(1, 9))


def terminal_action(command: str) -> str | None:
    return ATOM_GROUP.get(str(command))


def _entropy(values: Sequence[str]) -> float:
    if not values:
        return float("nan")
    counts = Counter(values)
    n = len(values)
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


def _jsd(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    nl, nr = sum(left.values()), sum(right.values())
    if not nl or not nr:
        return float("nan")
    p = {k: left[k] / nl for k in keys}
    q = {k: right[k] / nr for k in keys}
    m = {k: (p[k] + q[k]) / 2 for k in keys}
    def kl(a):
        return sum(a[k] * math.log2(a[k] / m[k]) for k in keys if a[k] > 0 and m[k] > 0)
    return float((kl(p) + kl(q)) / 2)


def local8(symbols: Sequence[str], actions: Sequence[str | None]) -> np.ndarray:
    """B block: local diversity, entropy, action organization and repetition."""
    n = len(symbols)
    if n == 0:
        return np.full(8, np.nan)
    symbol_counts = Counter(symbols)
    pairs = [(a, b) for a, b in zip(actions, actions[1:]) if a is not None and b is not None]
    pair_counts = Counter(pairs)
    outgoing = Counter(a for a, _ in pairs)
    max_run = run = 0
    previous = None
    for action in actions:
        if action is None:
            run = 0
            previous = None
            continue
        run = run + 1 if action == previous else 1
        max_run = max(max_run, run)
        previous = action
    conditional_entropy = (
        -sum((count / len(pairs)) * math.log2(count / outgoing[a])
             for (a, _), count in pair_counts.items()) if pairs else float("nan")
    )
    return np.asarray([
        len(symbol_counts) / n,
        _entropy(list(symbols)),
        max(symbol_counts.values()) / n,
        sum(a != b for a, b in pairs) / len(pairs) if pairs else np.nan,
        conditional_entropy,
        max_run / n if max_run else np.nan,
        1 - len(pair_counts) / len(pairs) if pairs else np.nan,
        sum(a == b for a, b in zip(symbols, symbols[1:])) / (n - 1) if n > 1 else np.nan,
    ], dtype=float)


def _sequence8(symbols: Sequence[str], actions: Sequence[str | None], start: int,
               seen_symbols: dict[str, int], seen_bigrams: set[tuple[str, str]],
               seen_trigrams: set[tuple[str, str, str]],
               previous_action_counts: Counter | None) -> np.ndarray:
    """S block: prefix novelty, revisits, returns, fragment reuse and drift."""
    n = len(symbols)
    novelty, revisits, spans = [], [], []
    bigram_reuse, trigram_reuse = [], []
    for off, symbol in enumerate(symbols):
        idx = start + off
        prior = seen_symbols.get(symbol)
        novelty.append(prior is None)
        if prior is not None:
            lag = idx - prior
            revisits.append(lag >= 2)
            spans.append(math.log1p(lag) / math.log1p(128))
        if off >= 1:
            bigram = (symbols[off - 1], symbol)
            bigram_reuse.append(bigram in seen_bigrams)
            seen_bigrams.add(bigram)
        if off >= 2:
            trigram = (symbols[off - 2], symbols[off - 1], symbol)
            trigram_reuse.append(trigram in seen_trigrams)
            seen_trigrams.add(trigram)
        seen_symbols[symbol] = idx
    action_triplets = [actions[i:i + 3] for i in range(max(0, n - 2))]
    valid_aba = [(a, b, c) for a, b, c in action_triplets
                 if a is not None and b is not None and c is not None and a != b]
    action_quads = [actions[i:i + 4] for i in range(max(0, n - 3))]
    valid_lag3 = [(seq[0], seq[3]) for seq in action_quads
                  if len(seq) == 4 and seq[0] is not None and seq[3] is not None]
    current_counts = Counter(a for a in actions if a is not None)
    drift = _jsd(previous_action_counts, current_counts) if previous_action_counts is not None else np.nan
    values = np.asarray([
        float(np.mean(novelty)) if novelty else np.nan,
        float(np.mean(revisits)) if revisits else np.nan,
        float(np.mean(spans)) if spans else np.nan,
        sum(a == c for a, _, c in valid_aba) / len(valid_aba) if valid_aba else np.nan,
        sum(a == b for a, b in valid_lag3) / len(valid_lag3) if valid_lag3 else np.nan,
        float(np.mean(bigram_reuse)) if bigram_reuse else np.nan,
        float(np.mean(trigram_reuse)) if trigram_reuse else np.nan,
        drift,
    ], dtype=float)
    return values


def _time8(timestamps: Sequence[float | None], durations: Sequence[float | None],
           start: int, end: int) -> tuple[np.ndarray, dict[str, int]]:
    """T block: inter-submission gaps, event durations and observable idle time.

    T7/T8 stay missing unless start times and matched nonnegative durations exist.
    No response-generation latency is relabeled as elapsed inter-command time.
    """
    ts = [None if x is None else float(x) for x in timestamps[start:end]]
    ds = [None if x is None else float(x) for x in durations[start:end]]
    gaps = []
    bad = 0
    for a, b in zip(ts, ts[1:]):
        if a is None or b is None:
            continue
        delta = b - a
        if delta < 0:
            bad += 1
            continue
        gaps.append(delta)
    lg = np.log1p(np.asarray(gaps, dtype=float)) if gaps else np.asarray([], dtype=float)
    if gaps:
        mean, sd = float(np.mean(gaps)), float(np.std(gaps))
        burst = (sd - mean) / (sd + mean) if sd + mean else 0.0
        fast = float(np.mean(np.asarray(gaps) <= 1.0))
        slow = float(np.mean(np.asarray(gaps) >= 10.0))
        if len(lg) >= 3 and np.std(lg[:-1]) > 0 and np.std(lg[1:]) > 0:
            corr = float(np.corrcoef(lg[:-1], lg[1:])[0, 1])
        else:
            corr = np.nan
        median_gap = float(np.median(lg))
        q1, q3 = np.quantile(lg, [0.25, 0.75])
        iqr = float(q3 - q1)
    else:
        median_gap = iqr = burst = fast = slow = corr = np.nan
    valid_durations = [d for d in ds if d is not None and d >= 0 and math.isfinite(d)]
    duration_median = float(np.median(np.log1p(valid_durations))) if valid_durations else np.nan
    idle = []
    for t0, d0, t1 in zip(ts, ds, ts[1:]):
        if t0 is None or d0 is None or t1 is None or d0 < 0:
            continue
        delta = t1 - (t0 + d0)
        if delta >= 0:
            idle.append(delta)
        elif delta < 0:
            bad += 1
    idle_median = float(np.median(np.log1p(idle))) if idle else np.nan
    return np.asarray([median_gap, iqr, burst, fast, slow, corr, duration_median, idle_median]), {
        "timestamps": sum(x is not None for x in ts),
        "durations": len(valid_durations),
        "valid_gaps": len(gaps),
        "valid_idle_gaps": len(idle),
        "nonmonotone_or_overlapping": bad,
    }


def fingerprint_matrix(symbols: Sequence[str], actions: Sequence[str | None] | None = None,
                       timestamps: Sequence[float | None] | None = None,
                       durations: Sequence[float | None] | None = None,
                       window: int = 8, min_window_events: int = 4,
                       max_events: int = 128) -> tuple[dict[str, np.ndarray] | None, dict]:
    """Return variable-row B/S/T matrices, allowing short partial last windows."""
    symbols = [str(x) for x in symbols[:max_events]]
    if actions is None:
        actions = [terminal_action(x) for x in symbols]
    else:
        actions = list(actions[:max_events])
    if len(actions) != len(symbols):
        raise ValueError("symbols/actions must have equal length")
    timestamps = list(timestamps[:max_events]) if timestamps is not None else [None] * len(symbols)
    durations = list(durations[:max_events]) if durations is not None else [None] * len(symbols)
    if len(timestamps) != len(symbols) or len(durations) != len(symbols):
        raise ValueError("timing arrays must align one-to-one with symbols")
    starts = list(range(0, len(symbols), window))
    starts = [s for s in starts if len(symbols[s:s + window]) >= min_window_events]
    if not starts:
        return None, {"reason": "no_valid_window", "events": len(symbols), "windows": 0}
    b_rows, s_rows, t_rows = [], [], []
    seen_symbols: dict[str, int] = {}
    seen_bigrams: set[tuple[str, str]] = set()
    seen_trigrams: set[tuple[str, str, str]] = set()
    prev_counts = None
    time_audit = Counter()
    for start in starts:
        end = min(start + window, len(symbols))
        cs, ac = symbols[start:end], actions[start:end]
        b_rows.append(local8(cs, ac))
        s_rows.append(_sequence8(cs, ac, start, seen_symbols, seen_bigrams, seen_trigrams, prev_counts))
        current_counts = Counter(a for a in ac if a is not None)
        if current_counts:
            prev_counts = current_counts
        t_row, audit = _time8(timestamps, durations, start, end)
        t_rows.append(t_row)
        time_audit.update(audit)
    blocks = {"B": np.vstack(b_rows), "S": np.vstack(s_rows), "T": np.vstack(t_rows)}
    return blocks, {"reason": None, "events": len(symbols), "windows": len(starts),
                    "window_size": window, "min_window_events": min_window_events,
                    "time": dict(time_audit)}


def summarize(matrix: np.ndarray) -> np.ndarray:
    """Mean, population SD and normalized-position OLS slope per column."""
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("expected a non-empty J x D matrix")
    x = np.linspace(0.0, 1.0, matrix.shape[0]) if matrix.shape[0] > 1 else np.zeros(1)
    values = []
    for col in range(matrix.shape[1]):
        y = matrix[:, col]
        valid = np.isfinite(y)
        if not valid.any():
            values.extend([np.nan, np.nan, np.nan])
            continue
        xv, yv = x[valid], y[valid]
        center = float(np.mean(xv))
        denom = float(np.sum((xv - center) ** 2))
        slope = float(np.sum((xv - center) * (yv - np.mean(yv))) / denom) if valid.sum() >= 2 and denom else np.nan
        values.extend([float(np.mean(yv)), float(np.std(yv, ddof=0)), slope])
    return np.asarray(values, dtype=float)


def combine_summary(blocks: dict[str, np.ndarray], names: Sequence[str] = ("B", "S")) -> np.ndarray:
    """Concatenate selected block summaries while preserving feature order."""
    return np.concatenate([summarize(blocks[name]) for name in names])


def masked_dtw(left: np.ndarray, right: np.ndarray, min_shared_columns: int = 3) -> float:
    """Unbanded masked DTW, generalized to any block dimension."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("DTW matrices must have the same feature dimension")
    n, m = len(left), len(right)
    cost = np.full((n + 1, m + 1), np.inf)
    length = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            shared = np.isfinite(left[i - 1]) & np.isfinite(right[j - 1])
            if shared.sum() < min_shared_columns:
                continue
            local = float(np.mean((left[i - 1, shared] - right[j - 1, shared]) ** 2))
            prior = min(((cost[i - 1, j - 1], length[i - 1, j - 1]),
                         (cost[i - 1, j], length[i - 1, j]),
                         (cost[i, j - 1], length[i, j - 1])), key=lambda item: (item[0], item[1]))
            if np.isfinite(prior[0]):
                cost[i, j], length[i, j] = prior[0] + local, prior[1] + 1
    if not length[n, m]:
        return float("nan")
    return float(math.sqrt(cost[n, m] / length[n, m]))


def fit_scale(matrices: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Train-only per-session-weighted column center and scale."""
    if not matrices:
        raise ValueError("cannot fit scaling without training matrices")
    dims = matrices[0].shape[1]
    num, den = np.zeros(dims), np.zeros(dims)
    for mat in matrices:
        if mat.shape[1] != dims:
            raise ValueError("matrix dimensions differ")
        w = np.full(mat.shape[0], 1 / mat.shape[0])
        for col in range(dims):
            ok = np.isfinite(mat[:, col])
            if ok.any():
                num[col] += np.dot(w[ok], mat[ok, col])
                den[col] += w[ok].sum()
    center = np.divide(num, den, out=np.zeros(dims), where=den > 0)
    ss, den2 = np.zeros(dims), np.zeros(dims)
    for mat in matrices:
        w = np.full(mat.shape[0], 1 / mat.shape[0])
        for col in range(dims):
            ok = np.isfinite(mat[:, col])
            if ok.any():
                ss[col] += np.dot(w[ok], (mat[ok, col] - center[col]) ** 2)
                den2[col] += w[ok].sum()
    scale = np.sqrt(np.divide(ss, den2, out=np.ones(dims), where=den2 > 0))
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return center, scale


def apply_scale(matrix: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = np.full_like(matrix, np.nan, dtype=float)
    valid = np.isfinite(matrix)
    z = (matrix - center) / scale
    out[valid] = z[valid]
    return out
