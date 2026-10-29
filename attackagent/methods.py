"""Method inventory with provenance kept alongside implementations."""

from __future__ import annotations


METHODS = (
    {
        "id": "command_cosine",
        "kind": "implemented_baseline",
        "title": "Command-token cosine",
        "owner": "AttackAgentDetect maintainers",
        "reference": None,
        "license": "MIT (this repository)",
        "status": "available",
        "scope": "Unsupervised trace ranking; no source labels required.",
    },
    {
        "id": "action_bigram",
        "kind": "implemented_baseline",
        "title": "Action-bigram cosine",
        "owner": "AttackAgentDetect maintainers",
        "reference": None,
        "license": "MIT (this repository)",
        "status": "available",
        "scope": "Unsupervised comparison of adjacent coarse action labels.",
    },
    {
        "id": "procgrep_jsd",
        "kind": "external_method_adapter",
        "title": "ProcGrep procedure-frequency JSD",
        "owner": "Hamidah Oderinwale",
        "reference": "https://hamidah.me/procgrep/",
        "license": "MIT (upstream license retained in vendor/procgrep/LICENSE)",
        "status": "available",
        "scope": "Pinned ProcGrep BPE/JSD subset, adapted to ordered command/action tokens.",
    },
    {
        "id": "local_mode_detector",
        "kind": "experimental_baseline",
        "title": "TF-IDF + logistic-regression mode score",
        "owner": "AttackAgentDetect maintainers",
        "reference": None,
        "license": "MIT (this repository)",
        "status": "available_when_matching_labeled_training_data_is_supplied",
        "scope": "Binary human/Agent evidence score; uncalibrated unless a separate calibration study is run.",
    },
    {
        "id": "mode_aware_pair_ranker",
        "kind": "experimental_baseline",
        "title": "Mode-aware pair ranker",
        "owner": "AttackAgentDetect maintainers",
        "reference": None,
        "license": "MIT (this repository)",
        "status": "research_prototype",
        "scope": "Needs matched labels and held-out evaluation; does not attribute an actor or APT group.",
    },
    {
        "id": "holmes",
        "kind": "external_research",
        "title": "HOLMES: Real-time APT Detection through Correlation of Suspicious Information Flows",
        "owner": "Milajerdi et al.",
        "reference": "https://doi.org/10.1109/SP.2019.00026",
        "license": "Research publication; this project does not redistribute implementation",
        "status": "documented_not_implemented",
        "scope": "Audit/provenance evidence and information-flow correlation for campaign detection.",
    },
    {
        "id": "ostinato",
        "kind": "external_research",
        "title": "OSTINATO: Cross-host Attack Correlation Through Attack Activity Similarity Detection",
        "owner": "Ghosh et al.",
        "reference": "https://arxiv.org/abs/2312.09321",
        "license": "Research publication; this project does not redistribute implementation",
        "status": "documented_not_implemented",
        "scope": "Cross-host behavior/TTP similarity on provenance graphs.",
    },
)


def list_methods() -> list[dict[str, str | None]]:
    """Return provenance-aware entries for available and documented methods."""
    return [dict(row) for row in METHODS]
