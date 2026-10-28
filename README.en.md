<h1 align="center">
  <img src="site/assets/attackagent-detect-logo.png" width="620" alt="AttackAgent Detect" />
</h1>

<p align="center"><a href="README.md">中文</a> | <strong>English</strong></p>

**[Visit the project website](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=en)** — research aims, results, data, methods, and related work.

[中文主页](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=zh) · [English website](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=en)

**An open-source research project and experimental toolkit for studying execution-mode signals in attack traces.**

The project studies whether observable traces can distinguish human, scripted, and AI-agent execution, and whether execution mode affects attack activity correlation. The current reproducible experiment contains human operation records and model-agent tool calls only; it has no matched script-execution samples. The research protocol, analysis code, pinned data manifest, notebook, aggregate results, and limitations are published for checking and extension.

The current feasibility study uses public human and model-agent sessions from InterCode-CTF. A command n-gram classifier reached 0.889 balanced accuracy on six held-out tasks; cross-source same-task retrieval was near chance (AUC 0.511, task-cluster 95% interval 0.396–0.667). These are exploratory benchmark results, not evidence of real attacker identification or APT campaign correlation: human and model sessions use different logging pipelines, scripts are absent, and shared task IDs are not shared-actor or campaign labels.

**PyPI release planned for 2027.** The package is not available to install yet; use the source checkout and reproduction steps below.

## Research and results

- [Research framing and eight-hour exploration plan](00-研究计划.md)
- [Primary experiment protocol](01-实验预设.md)
- [Related work and initial results](04-相关工作与初步实验.md)
- [Executable preliminary-analysis notebook](notebooks/01_preliminary.ipynb)
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

The command writes aggregate JSON outputs, including `results/quick-summary.json`. It is a reproducible benchmark workflow, not a per-record prediction API. Current scripts accept the pinned InterCode-CTF archive layout; they do not yet accept arbitrary auditd, Sysmon, or uploaded log files.

To open the notebook, run `.venv/bin/jupyter lab` and open `notebooks/01_preliminary.ipynb`. After the result files are generated, the notebook verifies them, presents the diagnostic tables, and exports desktop and mobile charts. Set `REBUILD_RESULTS = True` in its first analysis cell for a full rebuild; the equivalent terminal command is `.venv/bin/python scripts/run_all.py`.

The data downloader verifies the public source's file hashes. Raw trajectories and local environments are excluded from Git; the checked-in audit and aggregate metrics contain no raw command content. The dataset revision and known limitations are documented in the report.

## Open research

The related-work map and current evidence are presented on the [project page](https://mcxiaoxiao.github.io/AttackAgentDetect/). The repository is intended for public review and reproducible follow-up: results that fail to support the hypothesis remain documented, and no conclusion is generalized to real-world APT activity without actor and campaign ground truth collected through a common telemetry pipeline.

The research workflow also draws on the open procedural literature-review and experiment-design guidance in Kassis et al., [*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*](https://arxiv.org/abs/2609.00065) (2026).

## Project initiators

[sunyinggang](https://github.com/sunyinggang) and [OrlosZiming](https://mcxiaoxiao.github.io/).

## License

Project source code: [MIT](LICENSE). Dataset files remain under their upstream terms and are fetched locally rather than redistributed. The page owl and vendored ProcGrep code have separate notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The owl icon is by [Selina Bauder / OpenMoji](https://openmoji.org/library/emoji-1F989/), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
