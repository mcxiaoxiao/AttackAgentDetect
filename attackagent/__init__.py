"""Open research tools for normalizing and comparing security behavior traces."""

from .api import fit_detector, list_methods, load_auditd, predict_modes, rank_candidates
from .evidence import LabeledEvidence, ModeScore, RankedCandidate, TraceEvidence

__version__ = "0.2.0.dev0"

__all__ = [
    "LabeledEvidence",
    "ModeScore",
    "RankedCandidate",
    "TraceEvidence",
    "__version__",
    "fit_detector",
    "list_methods",
    "load_auditd",
    "predict_modes",
    "rank_candidates",
]
