<h1 align="center">
  <img src="site/assets/attackagent-detect-logo.png" width="620" alt="AttackAgent Detect" />
</h1>

<p align="center"><a href="README.zh-CN.md">简体中文</a> | <strong>English</strong></p>

**[Visit the project website](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=en)** — research aims, results, data, methods, and related work.

[中文主页](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=zh) · [English website](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=en)

**An open-source research project and experimental toolkit for studying execution-mode signals in attack traces.**

The project studies whether observable traces can distinguish human, scripted, and AI-agent execution, and whether execution mode affects attack activity correlation. The current reproducible experiment contains human operation records and model-agent tool calls only; it has no matched script-execution samples. The research protocol, analysis code, pinned data manifest, notebook, aggregate results, and limitations are published for checking and extension.

The current reproducible experiment covers three public CTF benchmarks, with 177 human and Agent traces across 40 matched task groups. On held-out task retrieval, Human-to-Agent Recall@1 is 0.414 against a 0.116 random baseline. Adding a mode score gives a +0.068 point-estimate change, while a command-length control gives +0.086; in the held-out-only candidate gallery, the mode-score change is −0.006. Current data therefore do not establish an association benefit from execution-mode identification: human terminal recordings and Agent tool-call logs use different capture paths, scripts are absent, and a shared task ID does not mean a shared attack activity. See the [cross-benchmark results](06-原型与跨基准实验结果.md) and [specificity controls](results/specificity_controls.json).

The traces come from a pinned [Cyber Task Horizons](https://github.com/lyptus-research/cyber-task-horizons-data) release: PicoCTF-derived InterCode-CTF tasks, CyBench challenges from four competitions, and competition tasks in NYU CTF Bench. We retain 40 task groups with both expert and Agent records. The [experiment report](06-原型与跨基准实验结果.md) describes the benchmark origins, trace collection, filtering, and file hashes.

The additional task-macro comparison puts command-name TF-IDF + logistic regression first on balanced accuracy (0.844; 95% task-cluster CI 0.766–0.915) among four CPU baselines. A 999-run conditional label-permutation diagnostic shows a positive point estimate only in the historical candidate gallery; the held-out-only gallery remains at −0.006 Recall@1. Neither result removes the label/capture-path confound or establishes an effect on real attack activity. Full model metrics, dataset-source decisions, unrun API comparisons, and the paper-ready figure are in [the data and model audit](08-公开数据与模型对比.md).

**PyPI release planned for 2027.** The package is not available to install yet; use the source checkout and reproduction steps below.

## Research and results

- [Research framing and eight-hour exploration plan](00-研究计划.md)
- [Primary experiment protocol](01-实验预设.md)
- [Related work and initial results](04-相关工作与初步实验.md)
- [Follow-up controlled-experiment plan](07-后续实验推进计划.md)
- [Cross-benchmark prototype and specificity controls](06-原型与跨基准实验结果.md), [machine-readable control results](results/specificity_controls.json)
- [Public dataset audit and local model comparison](08-公开数据与模型对比.md), [model results JSON](results/local_model_comparison.json), [task-level metrics CSV](results/local_model_comparison_task_metrics.csv)
- [Conditional mode-label permutation diagnostic](results/mode_label_permutation.json) and [model comparison notebook](notebooks/06_local_model_comparison.ipynb)
- [Paper results section](paper/experiment-results.tex), [appendix](paper/appendix.tex), and [figure-generation source](scripts/generate_paper_figures.py)
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

The scripts write aggregate JSON outputs, including `results/quick-summary.json`. A small research CLI now accepts normalized JSONL and emits a human/Agent trace score, evidence counts, abstention reason, and same-benchmark candidate ranking:

```bash
PYTHONPATH=. .venv/bin/python -m attackagent.jsonl_cli \
  --train examples/jsonl/train.jsonl \
  --queries examples/jsonl/queries.jsonl \
  --candidates examples/jsonl/candidates.jsonl
```

The bundled JSONL rows are interface-only examples, not experiment data. Training labels currently support `human` and `agent`; `script` is unsupported and is explicitly abstained. Scores are uncalibrated and rankings are not actor or APT attribution. The CLI expects normalized command names and coarse action categories; the project does not yet parse arbitrary auditd, Sysmon, or uploaded raw logs. See [`examples/jsonl/README.md`](examples/jsonl/README.md) for the schema and scope.

To open the notebooks, run `.venv/bin/jupyter lab`. `notebooks/01_preliminary.ipynb` rebuilds and audits the preliminary results; `notebooks/03_specificity_controls.ipynb` checks the fixed-fold controls; `notebooks/04_mode_label_permutation.ipynb` and `notebooks/05_unseen_agent_model.ipynb` audit the post-hoc robustness results; and `notebooks/06_local_model_comparison.ipynb` reruns the CPU classifier comparison and displays its publication figure.

The data downloader verifies the public source's file hashes. Raw trajectories and local environments are excluded from Git; the checked-in audit and aggregate metrics contain no raw command content. The dataset revision and known limitations are documented in the report.

## Open research

The related-work map and current evidence are presented on the [project page](https://mcxiaoxiao.github.io/AttackAgentDetect/). The repository is intended for public review and reproducible follow-up: results that fail to support the hypothesis remain documented, and no conclusion is generalized to real-world APT activity without actor and campaign ground truth collected through a common telemetry pipeline.

The research workflow also draws on the open procedural literature-review and experiment-design guidance in Kassis et al., [*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*](https://arxiv.org/abs/2609.00065) (2026).

## Project initiators

[sunyinggang](https://github.com/sunyinggang) and [OrlosZiming](https://mcxiaoxiao.github.io/).

## License

Project source code: [MIT](LICENSE). Dataset files remain under their upstream terms and are fetched locally rather than redistributed. The page owl and vendored ProcGrep code have separate notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The owl icon is by [Selina Bauder / OpenMoji](https://openmoji.org/library/emoji-1F989/), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
