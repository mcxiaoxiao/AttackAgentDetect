# Research workspace

The research workspace contains the current evaluation materials for AttackAgentDetect. The Python toolkit lives in [../attackagent/](../attackagent/); installation and usage are described in the [project README](../README.md).

## Current studies

- [Execution mode and behavior correlation](paper/mixed-gallery-main.tex) — task-held-out evaluation on 177 public CTF traces.
- [Pairwise trace association](reports/pairwise-trace-association.md) — current method, 97 traces, 40 task groups, and outer-fold results.
- [CTF trace set and provenance](../site/dataset.html) — source records, evaluation pairs, and trace-ID search.
- [Research website](https://mcxiaoxiao.github.io/AttackAgentDetect/).

CTF task IDs are retrieval labels only. They do not identify a shared operator, campaign, or real-world attack. Human terminal records and Agent tool-call records use different collection interfaces, so the available data cannot isolate execution mode from capture format.

## Reproduce

Create the environment and install the pinned dependencies from the repository root:

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -e . -r requirements.lock.txt
~~~

The current pair-association method, evaluation, and result files are described in the [study report](reports/pairwise-trace-association.md). The execution-mode analysis can be reproduced with the scripts under `scripts/` and the pinned source manifests under `data/manifests/`.

## Contents

- **data/** — source manifests and downloaded public traces.
- **experiments/** — evaluation protocols and runners.
- **reports/** — study results and limits.
- **scripts/** and **notebooks/** — data preparation, analysis, and figures.
- **results/** — aggregate metrics and derived artifacts.
- **paper/** — LaTeX sources and figures.
