# Research papers and notes

## Current draft

[`human-agent-fingerprint-draft.tex`](human-agent-fingerprint-draft.tex) and its compiled PDF are the current manuscript artifacts. The source bundle is published at [`site/assets/behavioral-fingerprint-v2/human-agent-fingerprint-draft-source.zip`](../../site/assets/behavioral-fingerprint-v2/human-agent-fingerprint-draft-source.zip). The main text includes the original three-step overview, the B/S/T formula atlas, and the CTH baseline triptych from the overnight draft. The evaluation section documents CTH, AIT, and NYU with the original Human/Agent icons and a separate NYU snapshot breakdown. The appendices group the remaining figures by measurement or evaluation protocol. The references end on page 8. The complete PDF has 13 pages with appendices and uses the original 10 pt type and 0.72 in margins.

On external NYU Agent-only challenge retrieval, B/S raises balanced accuracy by 5.71 percentage points for XGBoost and 3.64 points for the decision tree. Both models miss fewer links and falsely merge more different-challenge pairs. SVM loses 8.23 points on the same pairs. The CTH five-share comparison and AIT timing ablation leave their respective BA increments uncertain. The evaluated relation is benchmark task identity. The appendix preserves the historical tables verbatim from the earlier [`human-agent-fingerprint-overnight.tex`](human-agent-fingerprint-overnight.tex) snapshot.

Compile from this directory with `tectonic --reruns 2 human-agent-fingerprint-draft.tex`. The two generated association-table files are `matched-v2-main-tables.tex` and `matched-v2-appendix-tables.tex`; experiment inputs, scripts, analysis and figure sources are under `research/experiments/matched_fingerprint_v2/` and `research/results/matched_fingerprint_v2/`.

## Reproduction material

- [`research/experiments/behavioral_fingerprint_v2/`](../experiments/behavioral_fingerprint_v2/README.md) documents the current B/S/T protocol and reproduction commands.
- [`research/results/behavioral_fingerprint_v2/`](../results/behavioral_fingerprint_v2/) contains versioned outputs, splits, predictions, metrics, and provenance.
- [`overnight-fingerprint-v1.md`](../reports/overnight-fingerprint-v1.md) and [`research/results/overnight_fingerprint_v1/`](../results/overnight_fingerprint_v1/README.md) preserve the earlier frozen run and its results.
- [`human-agent-fingerprint-main.tex`](human-agent-fingerprint-main.tex) remains a broader research-design draft; proposed experiments in it are not completed results.
- `main.tex` and its included sections preserve the earlier command-and-artifact task-retrieval manuscript.

The overnight source bundle remains available as a historical artifact. The current source bundle includes all files needed to compile the draft independently, plus frozen pair predictions, manifests, aggregate analyses, and verification scripts; raw AIT logs and command text are excluded.
