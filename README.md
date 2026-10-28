# 攻击活动关联中的 Agent 参与识别

本仓库旨在提供攻击 Agent 参与识别与活动关联工具，以及支撑其设计和适用范围的案例、实验与论文研究。研究从可观察的攻击轨迹出发，检验 LLM Agent 参与是否改变行为证据，以及参与信息对具体活动关联任务是否有用。

研究不预设 Agent 会使现有关联方法失效。复现实验、公开论文结论、模拟结果和未检验假设将分别标注。模拟攻击或相同任务的轨迹，不等于真实 APT 组织归因。

## 进度

- [约 8 小时自动科研计划](00-研究计划.md)：研究问题、数据路线、实验、复核与交付。
- [初步实验预设](01-实验预设.md)：公开轨迹、划分、关联基线与实验边界。
- 调研报告、实验 Notebook、结果和复现说明随研究推进补充；当前不代表实验已完成。

## 环境

实验使用本地隔离的 Python 环境；确切依赖见 [requirements.lock.txt](requirements.lock.txt)。无需 Conda。若安装了 Python 与 [uv](https://docs.astral.sh/uv/)，可在本目录运行：

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python scripts/setup_procgrep.py
.venv/bin/python scripts/download_public_traces.py
.venv/bin/jupyter lab
```

公开数据的来源、许可、版本、下载及预处理方式将在实验报告和代码中说明。本地虚拟环境、外部数据仓库和原始数据默认不提交；提交之前应检查样本和 Notebook 输出是否允许公开。
