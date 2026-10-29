<h1 align="center">
  <img src="site/assets/attackagent-detect-logo.png" width="620" alt="AttackAgent Detect" />
</h1>

<p align="center"><strong>中文</strong> | <a href="README.md">English</a></p>

**[访问项目主页](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=zh)** — 研究目标、实验结果、数据与方法、相关工作。

[中文主页](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=zh) · [English website](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=en)

**AttackAgent Detect 是一项开放研究，探索智能体参与识别及其对攻击活动关联的影响。**

项目研究可观测轨迹能否区分人工、脚本与 AI Agent 执行，以及执行方式是否影响攻击活动关联。当前可复现实验仅包含人类操作记录与模型 Agent 工具调用，没有匹配的脚本执行样本。研究方案、分析代码、固定版本的数据清单、Notebook、汇总结果与局限均已公开，供检查和扩展。

当前可复现实验使用三个公开 CTF 基准，共 177 条人类和 Agent 轨迹、40 个同题组。留出任务检索中，Human→Agent Recall@1 为 0.414，随机基线 0.116；加入模式分数的点估计增量为 +0.068，但命令数量对照为 +0.086，严格留出候选库中的模式分数增量为 −0.006。现有数据未证明参与识别能改善关联：人类终端记录与 Agent 工具调用日志采集方式不同，缺少脚本组，同题标签也不代表攻击活动同源。细节见[跨基准结果](06-原型与跨基准实验结果.md)及[模式特异性对照](results/specificity_controls.json)。

数据来自固定版本的 [Cyber Task Horizons](https://github.com/lyptus-research/cyber-task-horizons-data)：InterCode-CTF 的 PicoCTF 衍生任务、来自四项竞赛的 CyBench 题目，以及 NYU CTF Bench 汇集的竞赛挑战。研究仅保留三套基准中同时有人类专家和 Agent 记录的 40 个题目组。任务来源、记录采集方式、样本筛选和逐文件哈希见[实验结果与数据说明](06-原型与跨基准实验结果.md)。

补充的等任务、等基准权重模型比较中，命令名 TF-IDF + 逻辑回归在四个 CPU 基线里平衡准确率最高（0.844；95% 任务簇区间 0.766–0.915）。999 次条件置换显示，历史候选库的模式增量为正，但严格留出候选库仍为 −0.006。两项结果都没有消除类别与采集方式的完全混杂，也没有证明真实攻击活动关联受到影响。数据筛选、未运行的 API 比较、完整指标和图表见[公开数据与模型对比](08-公开数据与模型对比.md)。

**PyPI 计划于 2027 年发布。** 软件包目前尚不可安装；页面中的安装与 API 调用为规划示意，当前请按下文从源码复现。

## 研究与结果

- [研究框架与八小时探索计划](00-研究计划.md)
- [主要实验预设](01-实验预设.md)
- [相关工作与初步实验](04-相关工作与初步实验.md)
- [后续受控实验推进计划](07-后续实验推进计划.md)
- [跨基准原型结果与特异性对照](06-原型与跨基准实验结果.md)、[机器可读对照结果](results/specificity_controls.json)
- [公开数据审计与本地模型比较](08-公开数据与模型对比.md)、[模型结果 JSON](results/local_model_comparison.json)、[逐任务 CSV](results/local_model_comparison_task_metrics.csv)
- [模式标签条件置换诊断](results/mode_label_permutation.json)、[本地模型比较 Notebook](notebooks/06_local_model_comparison.ipynb)
- [论文实验正文](paper/experiment-results.tex)、[附录](paper/appendix.tex)与[图形生成代码](scripts/generate_paper_figures.py)
- [可执行的初步分析 Notebook](notebooks/01_preliminary.ipynb)
- [可执行的特异性对照 Notebook](notebooks/03_specificity_controls.ipynb)
- [置换诊断 Notebook](notebooks/04_mode_label_permutation.ipynb)、[未见模型来源 Notebook](notebooks/05_unseen_agent_model.ipynb)
- [数据审计](results/data-audit.json)、[主要基线](results/preliminary.json)、[事后诊断](results/sensitivity.json)与 [Trace 风格迁移检查](results/trace-baseline.json)
- [重复划分敏感性检查](results/split-diagnostics.json)
- [精简汇总结果](results/quick-summary.json)

实验脚本从公开记录中解析已知命令名称，不执行其中记录的命令。仓库在 `vendor/procgrep/` 中固定保存了 [ProcGrep](https://github.com/hamidahoderinwale/procgrep) 的一个精简 MIT 许可子集，以便在不安装未发布依赖的情况下复现其轨迹表示与 JSD 计算。猫头鹰图标与引入代码的说明见[第三方声明](THIRD_PARTY_NOTICES.md)。

## 从源码复现

需要 Python 3.10 或更新版本，无需 Conda。获取仓库后，创建本地虚拟环境，安装固定版本的分析依赖，并下载指定版本的公开数据：

```bash
git clone https://github.com/mcxiaoxiao/AttackAgentDetect.git
cd AttackAgentDetect
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/download_public_traces.py
```

随后运行完整的数据审计、基线、诊断、汇总导出与一致性检查：

```bash
.venv/bin/python scripts/run_all.py
```

跨基准关联与新增的模式特异性对照按以下命令复现；它们输出任务级 CSV、汇总 JSON 与论文图形所需数据：

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

脚本会写入包括 `results/quick-summary.json` 在内的汇总 JSON。仓库也提供一个接收规范化 JSONL 的研究 CLI，可输出 human/Agent 轨迹得分、证据计数、拒判原因和同基准候选排序：

```bash
PYTHONPATH=. .venv/bin/python -m attackagent.jsonl_cli \
  --train examples/jsonl/train.jsonl \
  --queries examples/jsonl/queries.jsonl \
  --candidates examples/jsonl/candidates.jsonl
```

随仓库提供的 JSONL 是手工编写的接口演示，不是实验数据。训练标签目前只支持 `human` 和 `agent`；`script` 明确拒判。得分未校准，排序不代表攻击者或 APT 归因。CLI 要求输入已规范化的命令名和动作类别，尚未解析任意 auditd、Sysmon 或原始上传日志。输入格式与边界见 [`examples/jsonl/README.md`](examples/jsonl/README.md)。

运行 `.venv/bin/jupyter lab` 可打开 Notebook。`notebooks/01_preliminary.ipynb` 重建初步结果；`notebooks/03_specificity_controls.ipynb` 检查关联特异性对照；`notebooks/04_mode_label_permutation.ipynb` 和 `notebooks/05_unseen_agent_model.ipynb` 展示稳健性诊断；`notebooks/06_local_model_comparison.ipynb` 重跑 CPU 分类模型比较并展示论文图。

数据下载器会校验公开源文件的哈希。原始轨迹与本地环境不纳入 Git；仓库中的数据审计与汇总指标不含原始命令内容。数据版本与已知局限记录在研究报告中。

## 开放研究

[项目主页](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=zh) 展示相关工作脉络与当前证据。仓库面向公开检查与可复现的后续研究：不支持假设的结果也会保留记录。在缺少通过统一遥测流程采集的攻击者与攻击活动真值时，不将当前结论推广到真实 APT 活动。

研究流程也参考了 Kassis 等人的开放文献综述与实验设计程序指南：[*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*](https://arxiv.org/abs/2609.00065)（2026）。

## 项目发起

[sunyinggang](https://github.com/sunyinggang) 与 [OrlosZiming](https://mcxiaoxiao.github.io/)。

## 开源协议

项目源代码采用 [MIT 协议](LICENSE)，© 2026 AttackAgent Detect contributors。数据集遵循上游条款，仅下载到本地，不在本仓库重新分发。页面猫头鹰图标与引入的 ProcGrep 代码另见[第三方声明](THIRD_PARTY_NOTICES.md)。

猫头鹰图标由 [Selina Bauder / OpenMoji](https://openmoji.org/library/emoji-1F989/) 创作，采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 许可；上方品牌图使用同一图标。
