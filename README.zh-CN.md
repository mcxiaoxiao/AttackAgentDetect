<h1 align="center">
  <img src="site/assets/attackagent-detect-logo.png" width="620" alt="AttackAgent Detect" />
</h1>

<p align="center"><strong>中文</strong> | <a href="README.md">English</a></p>

**[访问项目主页](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=zh)** — 研究目标、实验结果、数据与方法、相关工作。

[中文主页](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=zh) · [English website](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=en)

**AttackAgent Detect 是一项开放研究，探索智能体参与识别及其对攻击活动关联的影响。**

项目提供安全日志标准化、行为比较与候选活动排序等可复用接口，并开放实验协议、代码和结果，供研究者检查和扩展。选题依据来自两类证据：HOLMES、OSTINATO、POIROT、ATLAS、ActMiner 等工作使用审计行为、序列、TTP、实体或因果图连接攻击证据；公开威胁报告也开始记录人类操作员通过 Agent 执行多阶段操作。库中会区分外部方法、项目适配与本项目实验基线。

已完成的 CPU 实验使用三个公开 CTF 基准，共 177 条人类与 Agent 轨迹、40 个题目组。命令 TF-IDF 是当前最强的单组识别特征（任务留出平衡准确率 0.85–0.87）；加入固定序列统计后的变化取决于分类器。在训练误合并目标为 20% 时，模式增强排序相对行为排序器使同题 Human–Agent 误拆分下降 12.7 个百分点、Hit@1 上升 3.6 个百分点；异题 Human–Agent 误合并同时上升 11.9 个百分点。测试集实际误合并率分别为 24.5% 和 23.3%。这支持把执行方式作为关联上下文继续评估，不支持把它当作普遍有效的修正项。同题标签只是 CTF 挑战代理；人类终端记录和 Agent 工具调用的采集方式与标签共线，也没有脚本组。完整协议、结果、图片和边界见[本轮实验记录](11-执行方式特征与行为关联-实验结果.md)。

此前的混合候选实验作为单独历史协议保留：预设主划分的模式感知 Hit@1 为 0.607，任务监督排序为 0.571；五个替代划分上的平均增量为 −0.007。其阈值目标不同，不能与上面的固定工作点实验合并。详见[早期关联文献与实验报告](09-关联分析文献与混合候选实验.md)。

数据来自固定版本的 [Cyber Task Horizons](https://github.com/lyptus-research/cyber-task-horizons-data)：PicoCTF 衍生的 InterCode-CTF、来自四项竞赛的 CyBench，以及 NYU CTF Bench。保留三套基准中同时有人类专家和 Agent 记录的 40 个 benchmark-qualified 题目组。来源、过滤、哈希、标签和切分协议见[本轮实验报告](11-执行方式特征与行为关联-实验结果.md)。

较早混合候选协议的主划分记录级混淆矩阵为 `[[43, 14], [8, 112]]`（Human/Agent 行，平衡准确率 0.844）。由于类别与采集路径一致，这只能说明该发布中的两类记录可区分，不能说明模型已识别真实操作者。该协议保留为历史结果，不与当前特征比较和固定工作点关联实验合并。

**PyPI 计划于 2027 年发布。** 当前已可从源码构建 wheel 并通过隔离安装目录的 CLI 冒烟检查，但尚未上传 PyPI。

## 研究与结果

- [研究框架与八小时探索计划](00-研究计划.md)
- [主要实验预设](01-实验预设.md)
- [相关工作与初步实验](04-相关工作与初步实验.md)
- [已完成的识别特征与关联效用实验](11-执行方式特征与行为关联-实验结果.md)
- [本轮实验预设与计划](10-下一轮实验计划-识别特征与关联效用.md)
- [上一轮受控实验计划（历史）](07-后续实验推进计划.md)
- [当前关联文献与混合候选实验](09-关联分析文献与混合候选实验.md)
- [跨基准原型结果与特异性对照](06-原型与跨基准实验结果.md)、[机器可读对照结果](results/specificity_controls.json)
- [公开数据审计与本地模型比较](08-公开数据与模型对比.md)、[模型结果 JSON](results/local_model_comparison.json)、[逐任务 CSV](results/local_model_comparison_task_metrics.csv)
- [模式标签条件置换诊断](results/mode_label_permutation.json)、[本地模型比较 Notebook](notebooks/06_local_model_comparison.ipynb)
- [当前 LaTeX 稿件](paper/mixed-gallery-main.tex)、[实验正文](paper/mixed-gallery-results.tex)与[附录](paper/mixed-gallery-appendix.tex)
- [早期探索稿件（历史）](paper/main.tex)与[对应图形生成代码](scripts/generate_paper_figures.py)
- [当前混合候选 Notebook](notebooks/07_mixed_gallery.ipynb)、[结果 JSON](results/mixed_gallery.json)与[绘图及实验脚本](scripts/run_mixed_gallery.py)
- [嵌套阈值关联错误实验](scripts/run_association_error_analysis.py)、[机器可读结果](results/association_error_analysis.json)、[逐对得分](results/association_error_pair_scores.csv)和[复现 Notebook](notebooks/08_association_error_analysis.ipynb)
- [可执行的初步分析 Notebook](notebooks/01_preliminary.ipynb)
- [可执行的特异性对照 Notebook](notebooks/03_specificity_controls.ipynb)
- [置换诊断 Notebook](notebooks/04_mode_label_permutation.ipynb)、[未见模型来源 Notebook](notebooks/05_unseen_agent_model.ipynb)
- [数据审计](results/data-audit.json)、[主要基线](results/preliminary.json)、[事后诊断](results/sensitivity.json)与 [Trace 风格迁移检查](results/trace-baseline.json)
- [重复划分敏感性检查](results/split-diagnostics.json)
- [特征消融](scripts/run_feature_ablation.py)、[配对对照](scripts/summarize_feature_contrasts.py)、[模型来源留出](scripts/run_agent_model_transfer.py)与[固定工作点关联实验](scripts/run_association_operating_points.py)
- [特征实验 Notebook](notebooks/09_feature_ablation_v1.ipynb)、[关联实验 Notebook](notebooks/10_association_operating_points_v1.ipynb)及保留的绘图脚本
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

早期混合候选协议按以下命令复现：

```bash
.venv/bin/python scripts/download_crossbenchmark_traces.py
.venv/bin/python scripts/run_mixed_gallery.py --bootstraps 5000
.venv/bin/python scripts/run_association_error_analysis.py --bootstraps 5000 --threshold-grid 101
.venv/bin/python -m unittest discover -s tests -v
```

重新运行已完成的特征比较与固定工作点关联实验：

```bash
.venv/bin/python scripts/run_feature_ablation.py
.venv/bin/python scripts/summarize_feature_contrasts.py
.venv/bin/python scripts/run_agent_model_transfer.py --bootstraps 5000
.venv/bin/python scripts/run_association_operating_points.py --bootstraps 5000 --threshold-grid 101
.venv/bin/python scripts/plot_feature_ablation.py
.venv/bin/python scripts/plot_association_operating_points.py
```

研究库可把 Linux auditd `SYSCALL`/`EXECVE` 原始记录规范化为版本化 JSONL，并按行为为候选轨迹排序；不会执行日志中的命令：

```bash
.venv/bin/python -m attackagent.jsonl_cli normalize \
  --input examples/auditd/minimal.audit --output /tmp/normalized.jsonl
.venv/bin/python -m attackagent.jsonl_cli rank \
  --query examples/jsonl/queries.jsonl \
  --candidates examples/jsonl/candidates.jsonl --method command_cosine
.venv/bin/python -m attackagent.jsonl_cli methods
```

Human/Agent 检测需要调用者提供与输入格式匹配的标注训练数据，当前仍为实验性功能；Script 类尚无支持检测器。格式不匹配时拒判或保留纯行为排序，得分和排序都不代表攻击者或 APT 归因。auditd 和 JSONL 样例均为手工接口演示，不是实验数据；格式见 [`examples/auditd/`](examples/auditd/) 与 [`examples/jsonl/README.md`](examples/jsonl/README.md)。

运行 `.venv/bin/jupyter lab` 可打开 Notebook。`notebooks/07_mixed_gallery.ipynb` 重跑混合候选排序；`notebooks/08_association_error_analysis.ipynb` 重跑嵌套阈值误差分析与分层图；较早的 Notebook 与报告保留先前探索协议，供对照追溯。

数据下载器会校验公开源文件的哈希。原始轨迹与本地环境不纳入 Git；仓库中的数据审计与汇总指标不含原始命令内容。数据版本与已知局限记录在研究报告中。

## 开放研究

[项目主页](https://mcxiaoxiao.github.io/AttackAgentDetect/?lang=zh) 展示相关工作脉络与当前证据。仓库面向公开检查与可复现的后续研究：不支持假设的结果也会保留记录。在缺少通过统一遥测流程采集的攻击者与攻击活动真值时，不将当前结论推广到真实 APT 活动。

研究流程也参考了 Kassis 等人的开放文献综述与实验设计程序指南：[*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*](https://arxiv.org/abs/2609.00065)（2026）。

## 项目发起

[sunyinggang](https://github.com/sunyinggang) 与 [OrlosZiming](https://mcxiaoxiao.github.io/)。

## 开源协议

项目源代码采用 [MIT 协议](LICENSE)，© 2026 AttackAgent Detect contributors。数据集遵循上游条款，仅下载到本地，不在本仓库重新分发。页面猫头鹰图标与引入的 ProcGrep 代码另见[第三方声明](THIRD_PARTY_NOTICES.md)。

猫头鹰图标由 [Selina Bauder / OpenMoji](https://openmoji.org/library/emoji-1F989/) 创作，采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 许可；上方品牌图使用同一图标。
