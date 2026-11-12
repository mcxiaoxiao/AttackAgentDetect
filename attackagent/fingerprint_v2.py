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
from research.experiments.behavioral_fingerprint_v2.command_categories import (
    categorize_command,
    CATEGORY_LIST,
    NUM_CATEGORIES,
)
from research.experiments.behavioral_fingerprint_v2.command_profile import (
    canonicalize_command,
    command_profile_vector,
    command_presence_vector,
    COMMAND_PROFILE_TAXONOMY,
    NUM_PROFILE,
)

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


# =============================================================================
# Command-category-aware features (B_cat, S_cat, T_cat)
# =============================================================================

def category_distribution(commands: Sequence[str]) -> np.ndarray:
    """B_cat: command category distribution vector (8 features).

    Returns proportion of commands in each category. Unknown commands
    are excluded from the denominator (not counted as a category).
    """
    cats = [categorize_command(cmd) for cmd in commands]
    # Filter out "unknown" for proportion calculation
    known_cats = [c for c in cats if c != "unknown"]
    if not known_cats:
        return np.full(NUM_CATEGORIES, np.nan)
    counts = Counter(known_cats)
    total = len(known_cats)
    return np.asarray([counts.get(cat, 0) / total for cat in CATEGORY_LIST], dtype=float)


def category_transitions(commands: Sequence[str]) -> np.ndarray:
    """S_cat: category bigram transition frequencies (64 features).

    Returns flattened 8x8 matrix of category-to-category transition
    frequencies. Only transitions between known categories are counted.
    """
    cats = [categorize_command(cmd) for cmd in commands]
    # Filter to known categories only
    known = [(i, cats[i]) for i in range(len(cats)) if cats[i] != "unknown"]
    if len(known) < 2:
        return np.full(NUM_CATEGORIES * NUM_CATEGORIES, np.nan)
    # Build transition counts
    trans_counts = Counter()
    for i in range(len(known) - 1):
        idx1, cat1 = known[i]
        idx2, cat2 = known[i + 1]
        # Only count consecutive in original sequence
        if idx2 == idx1 + 1:
            trans_counts[(cat1, cat2)] += 1
    total = sum(trans_counts.values())
    if total == 0:
        return np.full(NUM_CATEGORIES * NUM_CATEGORIES, np.nan)
    # Flatten to vector
    result = np.zeros(NUM_CATEGORIES * NUM_CATEGORIES, dtype=float)
    for i, cat1 in enumerate(CATEGORY_LIST):
        for j, cat2 in enumerate(CATEGORY_LIST):
            result[i * NUM_CATEGORIES + j] = trans_counts.get((cat1, cat2), 0) / total
    return result


def category_timing(commands: Sequence[str], timestamps: Sequence[float | None]) -> np.ndarray:
    """T_cat: per-category median inter-command gap (8 features).

    For each category, compute median log-gap between consecutive
    commands in that category. Categories with <2 commands get NaN.
    """
    cats = [categorize_command(cmd) for cmd in commands]
    ts = [None if x is None else float(x) for x in timestamps]
    # Group timestamps by category
    cat_times: dict[str, list[float]] = {cat: [] for cat in CATEGORY_LIST}
    for cmd, cat, t in zip(commands, cats, ts):
        if cat != "unknown" and t is not None:
            cat_times[cat].append(t)
    # Compute median gap per category
    result = np.full(NUM_CATEGORIES, np.nan, dtype=float)
    for i, cat in enumerate(CATEGORY_LIST):
        times = sorted(cat_times[cat])
        if len(times) < 2:
            continue
        gaps = [times[j+1] - times[j] for j in range(len(times) - 1) if times[j+1] >= times[j]]
        if gaps:
            result[i] = float(np.median(np.log1p(np.asarray(gaps, dtype=float))))
    return result


def category_features(commands: Sequence[str], timestamps: Sequence[float | None] | None = None
                      ) -> dict[str, np.ndarray]:
    """Compute all category-aware features.

    Returns dict with keys 'B_cat', 'S_cat', 'T_cat'.
    B_cat: 8 features (category distribution)
    S_cat: 64 features (category transitions)
    T_cat: 8 features (per-category timing)
    """
    if timestamps is None:
        timestamps = [None] * len(commands)
    return {
        "B_cat": category_distribution(commands),
        "S_cat": category_transitions(commands),
        "T_cat": category_timing(commands, timestamps),
    }


def combine_category_summary(cat_features: dict[str, np.ndarray],
                             names: Sequence[str] = ("B_cat", "S_cat", "T_cat")) -> np.ndarray:
    """Concatenate category feature vectors.

    Unlike B/S/T matrices, category features are already per-trace
    summaries (not windowed), so we just concatenate directly.
    """
    return np.concatenate([cat_features[name] for name in names])


# =============================================================================
# Command-preference block (V): fixed-length vocabulary-aware readout
# =============================================================================
# The transferable execution-mode signal on AIT is command preference (which
# executables and adjacent command pairs an actor favours), not fine-grained
# windowed behaviour. This block absorbs that signal into a fixed-length
# fingerprint: k_cmd sublinear-TF-IDF command weights followed by k_bigram
# adjacent-command frequencies. The vocabulary and IDF are fit on training
# traces only, so held-out actors never contribute vocabulary.

def fit_command_vocabulary(command_sequences: Sequence[Sequence[str]],
                           k_cmd: int = 32, k_bigram: int = 16) -> dict:
    """Fit a command-preference vocabulary on training traces only.

    Returns a vocabulary artifact with the top-k_cmd command names and
    top-k_bigram adjacent command pairs (frequency ranked, lexical tie-break),
    plus smoothed IDF weights for the command unigrams.
    """
    cmd_counts: Counter = Counter()
    cmd_df: Counter = Counter()
    bigram_counts: Counter = Counter()
    n_docs = 0
    for commands in command_sequences:
        cmds = [str(c) for c in commands]
        n_docs += 1
        cmd_counts.update(cmds)
        cmd_df.update(set(cmds))
        bigram_counts.update(zip(cmds, cmds[1:]))
    cmd_items = sorted(cmd_counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    cmd_vocab = [t for t, _ in cmd_items[:k_cmd]]
    cmd_idf = {t: math.log((1 + n_docs) / (1 + cmd_df.get(t, 0))) + 1 for t in cmd_vocab}
    bi_items = sorted(bigram_counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    bigram_vocab = [t for t, _ in bi_items[:k_bigram]]
    return {"cmd_vocab": cmd_vocab, "cmd_idf": cmd_idf, "bigram_vocab": bigram_vocab,
            "k_cmd": k_cmd, "k_bigram": k_bigram}


def command_preference_vector(commands: Sequence[str], vocab: dict) -> np.ndarray:
    """Fixed-length command-preference readout for one trace.

    Layout: [k_cmd sublinear-TF-IDF command weights | k_bigram bigram frequencies].
    Commands absent from the training vocabulary contribute zero, so the vector
    is defined for any trace once the vocabulary artifact is fixed.
    """
    cmds = [str(c) for c in commands]
    cmd_vocab = vocab["cmd_vocab"]
    bigram_vocab = vocab["bigram_vocab"]
    cmd_idx = {t: j for j, t in enumerate(cmd_vocab)}
    bi_idx = {t: j for j, t in enumerate(bigram_vocab)}
    out = np.zeros(len(cmd_vocab) + len(bigram_vocab), dtype=float)
    n = len(cmds)
    if n:
        for token, count in Counter(cmds).items():
            if token in cmd_idx:
                out[cmd_idx[token]] = (1 + math.log(count)) * vocab["cmd_idf"][token]
    if n > 1:
        for token, count in Counter(zip(cmds, cmds[1:])).items():
            if token in bi_idx:
                out[len(cmd_vocab) + bi_idx[token]] = count / (n - 1)
    return out


def command_composition_profile(commands: Sequence[str]) -> np.ndarray:
    """Default corpus-independent command-composition block of the fingerprint.

    A fixed-taxonomy, single-log BINARY tool-repertoire profile (NUM_PROFILE
    presence slots). Presence/absence transfers across unseen actors better than
    frequency, and no training vocabulary or IDF is used, so these are genuine
    profile dimensions integrable into the B/S/T schema. The sublinear frequency
    variant remains available as ``command_profile_vector`` for ablation.
    """
    return command_presence_vector(commands)


def default_fingerprint_summary(blocks: dict[str, np.ndarray], commands: Sequence[str],
                                timestamps: Sequence[float | None] | None = None,
                                names: Sequence[str] = ("B", "S", "T")) -> np.ndarray:
    """Default integrated fingerprint: each block pairs 8 scalar slots with one vector element.

    Layout: [B scalar | B tool-repertoire (279 presence + 8 category mix) |
    S scalar | S category transitions (64) | T scalar | T category timing (8)].
    Every vector element is computed from a single log against the fixed command
    taxonomy, so the schema stays a genuine profile: one high-dimensional element
    per block, no external block, no scalar-element proliferation.
    """
    cat = category_features(commands, timestamps)
    parts = []
    for name in names:
        parts.append(combine_summary(blocks, (name,)))
        if name == "B":
            parts.append(command_composition_profile(commands))  # 279 presence
            parts.append(cat["B_cat"])                            # 8 category mix
        elif name == "S":
            parts.append(cat["S_cat"])                            # 64 transitions
        elif name == "T":
            parts.append(cat["T_cat"])                            # 8 per-category timing
    return np.concatenate(parts)


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
