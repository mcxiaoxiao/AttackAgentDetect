# Research papers and notes

## Current manuscript

[`human-agent-fingerprint-overnight.tex`](human-agent-fingerprint-overnight.tex) is the current concise research manuscript. It presents the log-derived B/S/T fingerprint, its execution-mode and trace-retrieval readouts, the multi-source data contracts, completed experiments, limitations, and reproducibility appendix. The current file is compiled in the desktop LaTeX editor; its source bundle with all referenced figures is available at [`site/assets/behavioral-fingerprint-v2/human-agent-fingerprint-overnight-source.zip`](../../site/assets/behavioral-fingerprint-v2/human-agent-fingerprint-overnight-source.zip).

The exploratory v3 follow-up adds exact mixture baselines and a source-held-out conditioned-fingerprint comparison; its paired scores and tree-path contributions are recorded under `research/results/association_fingerprint_v3/`. The manuscript is evidence-bounded: same-task or source-specific conditions are retrieval proxies, not verified common attacks, operators, or campaigns. The 17-task mixture experiment and 40-task cross-mode retrieval are reported separately from the earlier 97-trace command-and-artifact baseline. AIT recognition is a troubleshooting-log diagnostic with task and collection confounding; Honey and CyBiasBench are Agent-only source-local stress tests.

## Reproduction material

- [`research/experiments/behavioral_fingerprint_v2/`](../experiments/behavioral_fingerprint_v2/README.md) documents the current B/S/T protocol and reproduction commands.
- [`research/results/behavioral_fingerprint_v2/`](../results/behavioral_fingerprint_v2/) contains versioned outputs, splits, predictions, metrics, and provenance.
- [`overnight-fingerprint-v1.md`](../reports/overnight-fingerprint-v1.md) and [`research/results/overnight_fingerprint_v1/`](../results/overnight_fingerprint_v1/README.md) preserve the earlier frozen run and its results.
- [`human-agent-fingerprint-main.tex`](human-agent-fingerprint-main.tex) remains a broader research-design draft; proposed experiments in it are not completed results.
- `main.tex` and its included sections preserve the earlier command-and-artifact task-retrieval manuscript.

The current TeX source is compiled through the desktop editor so it uses the open document's project context and figures. The downloadable source bundle contains the TeX file and all referenced figures, including vector plots and the schematic illustration.
