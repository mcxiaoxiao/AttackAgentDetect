"""Label-free evidence objects used by the public library API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class TraceEvidence:
    """Observed behavior only; labels and provenance stay in evaluator sidecars."""

    trace_id: str
    source_format: str
    benchmark: str
    commands: tuple[str, ...]
    actions: tuple[str, ...]
    event_refs: tuple[str, ...] = ()
    quality: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id.strip():
            raise ValueError("trace_id must be non-empty")
        if not self.source_format.strip():
            raise ValueError("source_format must be non-empty")
        if not self.benchmark.strip():
            raise ValueError("benchmark must be non-empty")
        if any(not isinstance(value, str) for value in (*self.commands, *self.actions)):
            raise TypeError("commands and actions must contain only strings")


@dataclass(frozen=True)
class LabeledEvidence:
    """Training-only annotation bound to evidence by the caller."""

    evidence: TraceEvidence
    mode: str
    activity_id: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"human", "agent"}:
            raise ValueError("supported training labels are currently human and agent")


@dataclass(frozen=True)
class RankedCandidate:
    trace_id: str
    rank: int
    score: float
    method: str
    score_semantics: str


@dataclass(frozen=True)
class ModeScore:
    trace_id: str
    prediction: str
    agent_like_score: float | None
    calibrated: bool
    available: bool
    reason: str | None
    evidence_count: int
    source_format: str
