# Research papers and notes

## Current draft

[`human-agent-fingerprint-draft.tex`](human-agent-fingerprint-draft.tex) and its compiled PDF are the current manuscript artifacts. The source bundle is published at [`site/assets/behavioral-fingerprint-v2/human-agent-fingerprint-draft-source.zip`](../../site/assets/behavioral-fingerprint-v2/human-agent-fingerprint-draft-source.zip). It presents the exact CTH mixture, same-pair learner comparison, unified CTH/AIT B/S/T ablation, and frozen NYU AA retrieval result. The main text and references end on page 6; the PDF has 11 pages total with appendices. The body preserves the original 10 pt type and 0.72 in margins.

The new CTH matched results do not show a common B/S gain across classifiers. The external NYU cohort shows tree and XGBoost BA gains with higher false-merge rates, while SVM declines. AIT's measured T increment is uncertain. Same-task labels remain retrieval proxies, not verified common attacks, operators, Agent instances, or campaigns. The prior [`human-agent-fingerprint-overnight.tex`](human-agent-fingerprint-overnight.tex) remains unchanged as an earlier manuscript snapshot; historical tables from it are copied verbatim into the new draft appendix.

Compile from this directory with `tectonic --reruns 2 human-agent-fingerprint-draft.tex`. The two generated result-table files are `matched-v2-main-tables.tex` and `matched-v2-appendix-tables.tex`; experiment inputs, scripts, analysis and figure sources are under `research/experiments/matched_fingerprint_v2/` and `research/results/matched_fingerprint_v2/`.

## Reproduction material

- [`research/experiments/behavioral_fingerprint_v2/`](../experiments/behavioral_fingerprint_v2/README.md) documents the current B/S/T protocol and reproduction commands.
- [`research/results/behavioral_fingerprint_v2/`](../results/behavioral_fingerprint_v2/) contains versioned outputs, splits, predictions, metrics, and provenance.
- [`overnight-fingerprint-v1.md`](../reports/overnight-fingerprint-v1.md) and [`research/results/overnight_fingerprint_v1/`](../results/overnight_fingerprint_v1/README.md) preserve the earlier frozen run and its results.
- [`human-agent-fingerprint-main.tex`](human-agent-fingerprint-main.tex) remains a broader research-design draft; proposed experiments in it are not completed results.
- `main.tex` and its included sections preserve the earlier command-and-artifact task-retrieval manuscript.

The overnight source bundle remains available as a historical artifact. The current source bundle includes all files needed to compile the draft independently, plus frozen pair predictions, manifests, aggregate analyses, and verification scripts; raw AIT logs and command text are excluded.
