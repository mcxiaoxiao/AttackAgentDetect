# Research papers and notes

## Current short paper

[`same-challenge-retrieval-shortpaper.tex`](same-challenge-retrieval-shortpaper.tex) is the current review-ready short paper. Its compiled PDF is five pages including references, within the eight-page limit. The accompanying [`same-challenge-retrieval-supplement.tex`](same-challenge-retrieval-supplement.tex) compiles to five pages and holds full feature definitions, threshold diagnostics, the five-seed mechanism control, exact-sequence-overlap sensitivity, complete source-specific results, and supplementary AIT/snapshot tests.

The revised study centers on same-task retrieval under changing recorded Human/Agent role composition. It does not treat task identity as actor, activity, or campaign attribution. The new CTH mechanism control finds no stable B/S gain beyond trace-volume and command-sequence controls; the CTH and NYU Agent-only panels show different error trade-offs. The main article therefore presents the evidence as a bounded measurement result.

Compile from this directory with:

```sh
tectonic --reruns 2 same-challenge-retrieval-shortpaper.tex
tectonic --reruns 2 same-challenge-retrieval-supplement.tex
```

## Previous drafts and reports

[`human-agent-fingerprint-draft.tex`](human-agent-fingerprint-draft.tex) is preserved as the preceding broader draft and is no longer the current short-paper version. Its local PDF and the historical overnight draft are retained. The site-hosted PDF and [source ZIP](../../site/assets/behavioral-fingerprint-v2/human-agent-fingerprint-draft-source.zip) are an older packaged snapshot and were not updated by this review.

The experiment protocol, scripts, manifests, frozen predictions, aggregate summaries, and provenance are under `research/experiments/matched_fingerprint_v2/` and `research/results/matched_fingerprint_v2/`. The two new short-paper analyses write to `shortpaper_frozen_20261105/` and `shortpaper_mechanism_20261105/`. Raw NYU transcripts and raw command content are not included in the local short-paper bundle.

The clean-extraction-verified local bundle is [`same-challenge-retrieval-20261106.zip`](../reproducibility/same-challenge-retrieval-20261106.zip). It contains frozen pair predictions and a standard-library script that re-aggregates the main reported point estimates without refitting. Full feature extraction and model reproduction still require the upstream source logs and pinned NYU revision. Review findings and evidence limits are in [`same-challenge-retrieval-shortpaper-review-2026-11-05.md`](../reports/same-challenge-retrieval-shortpaper-review-2026-11-05.md).

- [`fingerprint-target-and-ranker-2026-11-05.md`](../reports/fingerprint-target-and-ranker-2026-11-05.md) records the earlier ranking and snapshot studies.
- [`fingerprint-dimension-and-ranking-followup-2026-11-05.md`](../reports/fingerprint-dimension-and-ranking-followup-2026-11-05.md) records feature coverage and dimension-selection diagnostics.
- [`matched-fingerprint-v2.md`](../reports/matched-fingerprint-v2.md) preserves the broader experimental history and older protocols.
