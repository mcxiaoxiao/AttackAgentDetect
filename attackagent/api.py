"""Small, label-safe public API for trace normalization and comparison."""

from __future__ import annotations

import math
import pathlib
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .auditd import load_auditd
from .evidence import LabeledEvidence, ModeScore, RankedCandidate, TraceEvidence
from .methods import list_methods


@dataclass
class ModeDetector:
    source_format: str
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    training_rows: int
    calibrated: bool = False


def _feature_text(record: TraceEvidence) -> str:
    tokens = [f"cmd_{item}" for item in record.commands]
    tokens.extend(f"act_{item}" for item in record.actions)
    return " ".join(tokens)


def fit_detector(training_records: Iterable[LabeledEvidence], *, seed: int = 20261030) -> ModeDetector:
    """Fit a local binary detector from caller-provided, format-matched labels."""
    rows = list(training_records)
    if len(rows) < 8:
        raise ValueError("at least eight labeled traces are required")
    profiles = {row.evidence.source_format for row in rows}
    if len(profiles) != 1:
        raise ValueError("training traces must share one source_format")
    labels = [row.mode for row in rows]
    if set(labels) != {"human", "agent"}:
        raise ValueError("training data must contain both human and agent labels")
    texts = [_feature_text(row.evidence) for row in rows]
    if not any(text.strip() for text in texts):
        raise ValueError("training evidence has no command or action features")
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1)
    features = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed)
    classifier.fit(features, np.asarray([int(label == "agent") for label in labels]))
    return ModeDetector(
        source_format=next(iter(profiles)),
        vectorizer=vectorizer,
        classifier=classifier,
        training_rows=len(rows),
    )


def predict_modes(detector: ModeDetector, records: Iterable[TraceEvidence]) -> list[ModeScore]:
    """Score evidence only; annotations cannot be supplied to this interface."""
    rows = list(records)
    if not rows:
        return []
    texts = [_feature_text(row) for row in rows]
    features = detector.vectorizer.transform(texts)
    raw_scores = detector.classifier.predict_proba(features)[:, 1]
    nonzero = np.asarray(features.getnnz(axis=1)).ravel()
    output: list[ModeScore] = []
    for row, score, evidence_count in zip(rows, raw_scores, nonzero):
        reason = None
        if row.source_format != detector.source_format:
            reason = "incompatible_source_format"
        elif evidence_count == 0:
            reason = "empty_or_out_of_vocabulary_evidence"
        available = reason is None
        value = float(score) if available else None
        output.append(ModeScore(
            trace_id=row.trace_id,
            prediction=("agent" if score >= 0.5 else "human") if available else "unknown",
            agent_like_score=value,
            calibrated=bool(detector.calibrated),
            available=available,
            reason=reason,
            evidence_count=int(evidence_count),
            source_format=row.source_format,
        ))
    return output


def _representations(records: list[TraceEvidence], method: str):
    if method == "command_cosine":
        values = [" ".join(row.commands) for row in records]
        ngrams = (1, 1)
    elif method == "action_bigram":
        values = [" ".join(f"{a}__{b}" for a, b in zip(row.actions, row.actions[1:])) for row in records]
        ngrams = (1, 1)
    else:
        raise ValueError(f"unsupported behavior method: {method}")
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=ngrams, binary=True)
    try:
        return vectorizer.fit_transform(values)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        return csr_matrix((len(records), 0), dtype=float)


def _procgrep_similarity(records: list[TraceEvidence]) -> np.ndarray:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    local_src = project_root / "vendor" / "procgrep" / "src"
    try:
        from procgrep import Trace, encode, fit_bpe, jsd
    except ImportError:
        if local_src.is_dir():
            sys.path.insert(0, str(local_src))
            from procgrep import Trace, encode, fit_bpe, jsd
        else:
            raise RuntimeError("ProcGrep adapter is not installed")
    atoms = [list(row.actions or row.commands) for row in records]
    alphabet = {token for sequence in atoms for token in sequence}
    if not alphabet:
        return np.zeros((len(records), len(records)), dtype=float)
    vocab_size = len(alphabet) + min(32, max(0, 128 - len(alphabet)))
    vocab = fit_bpe(atoms, vocab_size=vocab_size, seed=20261030, min_pair_frequency=2)
    traces = [Trace(trace_id=row.trace_id, agent="unknown", atoms=atoms[i], group=row.trace_id)
              for i, row in enumerate(records)]
    fingerprints = encode(traces, vocab=vocab)
    similarity = np.eye(len(records), dtype=float)
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            value = 1.0 - jsd(fingerprints[i].distribution(), fingerprints[j].distribution())
            similarity[i, j] = similarity[j, i] = float(value)
    return similarity


def rank_candidates(
    query: TraceEvidence,
    candidates: Iterable[TraceEvidence],
    *,
    method: str = "command_cosine",
    detector: ModeDetector | None = None,
    base_method: str = "command_cosine",
    mode_weight: float = 0.25,
) -> list[RankedCandidate]:
    """Rank traces without consulting task, mode, source, or filename labels.

    The query itself is removed by trace ID. If a detector is requested but
    its format or evidence is incompatible for any gallery item, the entire
    query falls back to the behavior-only scores and retains the same gallery.
    Scores are similarities, not probabilities of common activity or actor.
    """
    rows = [row for row in candidates if row.trace_id != query.trace_id]
    if len({row.trace_id for row in rows}) != len(rows):
        raise ValueError("candidate trace_id values must be unique")
    if not 0.0 <= mode_weight <= 1.0:
        raise ValueError("mode_weight must be in [0, 1]")
    if method == "mode_adjusted":
        behavior_method = base_method
    else:
        behavior_method = method
    if behavior_method not in {"command_cosine", "action_bigram", "procgrep_jsd"}:
        raise ValueError(f"unsupported ranking method: {method}")
    corpus = [query, *rows]
    if behavior_method == "procgrep_jsd":
        matrix = _procgrep_similarity(corpus)
        baseline = matrix[0, 1:]
    else:
        matrix = _representations(corpus, behavior_method)
        baseline = np.asarray(matrix[0].multiply(matrix[1:]).sum(axis=1)).ravel()
    method_used = behavior_method
    semantics = "label-free cosine/Jensen-Shannon similarity"
    scores = np.asarray(baseline, dtype=float)
    if method == "mode_adjusted":
        if detector is None:
            semantics += "; detector unavailable, behavior-only fallback"
        else:
            modes = predict_modes(detector, corpus)
            if all(score.available and score.agent_like_score is not None for score in modes):
                probs = np.asarray([score.agent_like_score for score in modes], dtype=float)
                query_p = probs[0]
                candidate_p = probs[1:]
                same_mode = query_p * candidate_p + (1.0 - query_p) * (1.0 - candidate_p)
                scores = (1.0 - mode_weight) * scores + mode_weight * same_mode
                semantics = "exploratory weighted behavior/mode similarity; not a probability"
                method_used = "mode_adjusted"
            else:
                reasons = sorted({score.reason for score in modes if not score.available})
                semantics += "; detector unavailable for this query, behavior-only fallback (" + ", ".join(reasons) + ")"
    order = sorted(range(len(rows)), key=lambda index: (-float(scores[index]), rows[index].trace_id))
    return [RankedCandidate(
        trace_id=rows[index].trace_id,
        rank=rank,
        score=float(scores[index]),
        method=method_used,
        score_semantics=semantics,
    ) for rank, index in enumerate(order, start=1)]


__all__ = ["fit_detector", "list_methods", "load_auditd", "predict_modes", "rank_candidates"]
