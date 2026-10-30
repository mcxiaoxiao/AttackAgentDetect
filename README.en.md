<h1 align="center">
  <img src="site/assets/attackagent-detect-logo.png" width="620" alt="AttackAgent Detect" />
</h1>

<p align="center"><a href="README.zh-CN.md">简体中文</a> | <strong>English</strong></p>

**[Visit the project website](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=en)** — research aims, results, data, methods, and related work.

[中文主页](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=zh) · [English website](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=en)

**An open-source research project and experimental toolkit for studying execution-mode signals in attack traces.**

The project is an open research library for normalizing security traces, comparing behavior, and testing whether execution-mode evidence helps link attack activity. Its motivating question is grounded in prior work: systems such as HOLMES, OSTINATO, POIROT, ATLAS, and ActMiner link audit behavior, sequences, TTPs, entities, or causal graphs; public threat records now also describe human operators coordinating AI agents. The library keeps external methods, local adaptations, and project baselines distinct.

The current reproducible study uses 177 human and Agent traces across 40 task groups from three public CTF benchmarks. In a mixed candidate gallery, mode-aware ranking reaches Hit@1 0.607 versus 0.571 for a task-supervised behavior ranker (paired change +0.036, 95% task-cluster CI [+0.006, +0.069]); versus command cosine, the change is only +0.008 (95% CI [−0.046, +0.059]). Across five alternate task splits, the mean change versus the task-supervised ranker is −0.007 (95% CI [−0.026, +0.012]). The result is split-sensitive and does not establish a stable association gain. The task ID is only a same-challenge proxy; human terminal capture and Agent tool-call capture are confounded with the labels, and scripts are absent. See the [current literature and experiment report](09-关联分析文献与混合候选实验.md).

The traces come from a pinned [Cyber Task Horizons](https://github.com/lyptus-research/cyber-task-horizons-data) release: PicoCTF-derived InterCode-CTF tasks, CyBench challenges from four competitions, and NYU CTF Bench tasks. The evaluated subset contains 40 benchmark-qualified task groups with both expert and Agent records. The current report documents provenance, filtering, hashes, labels, and the exact retrieval protocol.

For the current task-held-out mode-label diagnostic, the record-level confusion matrix is `[[43, 14], [8, 112]]` (Human/Agent rows); balanced accuracy is 0.844. Since capture path and label coincide, this is trace-label separability, not general operator identification. Older analyses remain in the repository as historical protocols and should not be pooled with the current mixed-gallery results.

**PyPI release planned for 2027.** A wheel can already be built and smoke-tested from this repository; the package has not been published to PyPI.

## Research and results

- [Research framing and eight-hour exploration plan](00-研究计划.md)
- [Primary experiment protocol](01-实验预设.md)
- [Related work and initial results](04-相关工作与初步实验.md)
- [Next experiments: identification features and association utility](10-下一轮实验计划-识别特征与关联效用.md)
- [Earlier controlled-experiment plan (historical)](07-后续实验推进计划.md)
- [Current related-work audit and mixed-gallery experiment](09-关联分析文献与混合候选实验.md)
- [Cross-benchmark prototype and specificity controls](06-原型与跨基准实验结果.md), [machine-readable control results](results/specificity_controls.json)
- [Public dataset audit and local model comparison](08-公开数据与模型对比.md), [model results JSON](results/local_model_comparison.json), [task-level metrics CSV](results/local_model_comparison_task_metrics.csv)
- [Conditional mode-label permutation diagnostic](results/mode_label_permutation.json) and [model comparison notebook](notebooks/06_local_model_comparison.ipynb)
- [Current LaTeX packet](paper/mixed-gallery-main.tex), [results section](paper/mixed-gallery-results.tex), and [appendix](paper/mixed-gallery-appendix.tex)
- [Earlier exploratory manuscript (historical)](paper/main.tex) and [its figure-generation source](scripts/generate_paper_figures.py)
- [Current mixed-gallery notebook](notebooks/07_mixed_gallery.ipynb), [result JSON](results/mixed_gallery.json), and [figure-generation source](scripts/run_mixed_gallery.py)
- [Executable preliminary-analysis notebook](notebooks/01_preliminary.ipynb)
- [Executable specificity-controls notebook](notebooks/03_specificity_controls.ipynb)
- [Executable permutation notebook](notebooks/04_mode_label_permutation.ipynb), [unseen-model notebook](notebooks/05_unseen_agent_model.ipynb)
- [Data audit](results/data-audit.json), [primary baselines](results/preliminary.json), [post-hoc diagnostics](results/sensitivity.json), and [Trace-style transfer check](results/trace-baseline.json)
- [Repeated split sensitivity check](results/split-diagnostics.json)
- [Compact aggregate output](results/quick-summary.json)

The experiment scripts parse known command names from public records; they never execute recorded commands. A compact, MIT-licensed subset of [ProcGrep](https://github.com/hamidahoderinwale/procgrep) is pinned in `vendor/procgrep/` so the published representation and JSD calculations can be reproduced without installing an unpublished dependency. See [third-party notices](THIRD_PARTY_NOTICES.md) for the owl icon and vendored code.

## Reproduce

Python 3.10 or newer is required. Conda is not needed. Create a local virtual environment, install the frozen analysis dependencies, and acquire the pinned public data:

```bash
git clone https://github.com/mcxiaoxiao/AttackAgentDetect.git
cd AttackAgentDetect
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/download_public_traces.py
```

Then run the complete audit, baselines, diagnostics, summary export, and consistency checks:

```bash
.venv/bin/python scripts/run_all.py
```

Reproduce the cross-benchmark association and the new mode-specificity controls with:

```bash
.venv/bin/python scripts/download_crossbenchmark_traces.py
.venv/bin/python scripts/run_crossbenchmark_experiments.py
.venv/bin/python scripts/run_mode_aware_association.py
.venv/bin/python scripts/run_specificity_controls.py
.venv/bin/python scripts/run_unseen_agent_model.py
.venv/bin/python scripts/run_mode_label_permutation.py --permutations 999 --seed 20262701
.venv/bin/python scripts/compare_trace_mode_models.py
.venv/bin/python scripts/plot_local_model_comparison.py
.venv/bin/python scripts/generate_paper_figures.py
```

The current mixed-gallery protocol and figures are reproduced with:

```bash
.venv/bin/python scripts/run_mixed_gallery.py --bootstraps 5000
.venv/bin/python -m unittest discover -s tests -v
```

The library accepts raw Linux auditd `SYSCALL`/`EXECVE` lines and normalizes them to versioned JSONL; it never executes recorded commands. It also ranks normalized traces using command cosine, action bigrams, or the vendored ProcGrep adapter:

```bash
.venv/bin/python -m attackagent.jsonl_cli normalize \
  --input examples/auditd/minimal.audit --output /tmp/normalized.jsonl
.venv/bin/python -m attackagent.jsonl_cli rank \
  --query examples/jsonl/queries.jsonl \
  --candidates examples/jsonl/candidates.jsonl --method command_cosine
.venv/bin/python -m attackagent.jsonl_cli methods
```

Human/Agent mode detection requires caller-provided, format-matched labels and remains experimental. Script has no supported detector yet; incompatible inputs abstain or keep the behavior-only ranking. Scores are not actor or APT attribution. The auditd fixture and JSONL rows are hand-written interface examples, not experimental data. See [`examples/auditd/`](examples/auditd/) and [`examples/jsonl/README.md`](examples/jsonl/README.md).

To open the notebooks, run `.venv/bin/jupyter lab`. `notebooks/07_mixed_gallery.ipynb` reproduces the current study and figures; earlier notebooks and reports preserve exploratory protocols for comparison.

The data downloader verifies the public source's file hashes. Raw trajectories and local environments are excluded from Git; the checked-in audit and aggregate metrics contain no raw command content. The dataset revision and known limitations are documented in the report.

## Open research

The related-work map and current evidence are presented on the [project page](https://mcxiaoxiao.github.io/AttackAgentDetect/). The repository is intended for public review and reproducible follow-up: results that fail to support the hypothesis remain documented, and no conclusion is generalized to real-world APT activity without actor and campaign ground truth collected through a common telemetry pipeline.

The research workflow also draws on the open procedural literature-review and experiment-design guidance in Kassis et al., [*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*](https://arxiv.org/abs/2609.00065) (2026).

## Project initiators

[sunyinggang](https://github.com/sunyinggang) and [OrlosZiming](https://mcxiaoxiao.github.io/).

## License

Project source code: [MIT](LICENSE). Dataset files remain under their upstream terms and are fetched locally rather than redistributed. The page owl and vendored ProcGrep code have separate notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The owl icon is by [Selina Bauder / OpenMoji](https://openmoji.org/library/emoji-1F989/), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
