# Paper

## Current round

[`human-agent-fingerprint-main.tex`](human-agent-fingerprint-main.tex) is the current-round manuscript: **Behavioral Fingerprint Matrices for Execution-Mode-Aware Trace Association Across Security Tasks**. It is a self-contained English TeX source with the literature synthesis, feasible log contract, eight core formulas, a proposed fingerprint matrix, the actual 1,142-record corpus, completed exploratory pilot outcomes, prospective association/transfer evaluation, and bibliography.

Status: research-design/progress draft. The 21-candidate literature inventory is contracted to eight mandatory measurements; richer telemetry is optional or deferred. The completed 95-run scalar pilot is reported separately; expanded-corpus matrix and association improvements are not claimed.

The [prepared overnight execution plan](../experiments/overnight_fingerprint_v1/program.md) specifies the next evidence-driven manuscript and Human/Agent-page revision. Its eligibility audit has run; the full experiment runner and the new results manuscript are not yet implemented. The future `human-agent-fingerprint-overnight.tex` becomes a new manuscript only after that execution, and is not an existing results paper.

To compile from this directory with a local LaTeX installation:

```sh
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build human-agent-fingerprint-main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build human-agent-fingerprint-main.tex
```

Alternatively, Tectonic can compile the same standalone source and resolve cross-references automatically.

## Earlier manuscript

`main.tex` is the earlier command-and-artifact task-retrieval manuscript. Its included sources are `pairwise-association-method.tex`, `pairwise-association-results.tex`, and `appendix.tex`. It is retained as historical work and is not the entry point for the new Human/Agent fingerprint round.
