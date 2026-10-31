# Research workspace

This folder contains the research records and reproducibility materials. The Python toolkit itself is in [../attackagent/](../attackagent/) and its usage is in the [project README](../README.md).

## Start here

- [Execution-mode and behavior-correlation study](reports/11-执行方式特征与行为关联-实验结果.md) — primary results, protocol, and limits.
- [Multidimensional profile study](reports/13-多维行为画像关联实验结果.md) — the later profile comparison and its external-set results.
- [Pair-classification study](reports/V16-增强型判别式分类.md) — development-set comparison and limitations.
- [Public data and local model comparison](reports/08-公开数据与模型对比.md).
- [Research website](https://mcxiaoxiao.github.io/AttackAgentDetect/) and [dataset provenance page](https://mcxiaoxiao.github.io/AttackAgentDetect/dataset.html).

The evidence is exploratory. CTF task IDs are same-challenge labels, not labels for a shared operator, campaign, or real-world attack.

## Reproduce an experiment

From the repository root, create the environment and install the pinned research dependencies:

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -e . -r requirements.lock.txt
~~~

For the mixed-gallery study, download the pinned public traces and run the analysis:

~~~bash
.venv/bin/python research/scripts/download_public_traces.py
.venv/bin/python research/scripts/download_crossbenchmark_traces.py
.venv/bin/python research/scripts/run_mixed_gallery.py --bootstraps 5000
~~~

For the feature and operating-point study:

~~~bash
.venv/bin/python research/scripts/run_feature_ablation.py
.venv/bin/python research/scripts/run_association_operating_points.py --bootstraps 5000 --threshold-grid 101
~~~

To open a notebook, run .venv/bin/jupyter lab research/notebooks. Notebook setup cells locate this research folder from either the repository root or the notebook directory.

Raw downloaded traces stay in the ignored research/data/raw/ directory. Checked-in results contain aggregate metrics and task identifiers; the research reports describe source revisions and known limits.

## Folder map

- **plans/** — chronological experiment plans and protocols, retained as historical records.
- **reports/** — written research results, audits, and classification reports.
- **data/manifests/** — pinned source manifests used to fetch public traces.
- **experiments/** — versioned experiment programs and frozen protocols.
- **scripts/** — runners, downloaders, analysis, and figure-generation tools.
- **notebooks/** — notebooks for inspecting and reproducing analyses.
- **results/** — machine-readable outputs and generated research figures.
- **paper/** — LaTeX papers and appendices.
- **state/** — research configuration snapshots.
