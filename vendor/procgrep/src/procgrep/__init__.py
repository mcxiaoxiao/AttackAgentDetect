"""Small research subset of ProcGrep pinned for reproducible study runs.

Upstream: https://github.com/hamidahoderinwale/procgrep
Pinned commit: a98b30e7391a1f42d93e05db7999b0ad21b881f5
See the adjacent LICENSE and vendor/procgrep/README.md.
"""

from .bpe import ProcedureVocabulary, fit_bpe
from .encode import Fingerprint, encode
from .jsd import jsd
from .types import Trace

__version__ = "0.1.3"

__all__ = [
    "Fingerprint",
    "ProcedureVocabulary",
    "Trace",
    "__version__",
    "encode",
    "fit_bpe",
    "jsd",
]
