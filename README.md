# 攻击活动关联中的 Agent 参与识别

本仓库记录一项早期研究：基于可观察的攻击轨迹，研究 LLM Agent 参与是否改变行为证据，并检验在具体活动关联任务中，参与信息是否有用。

研究不预设 Agent 会使现有关联方法失效。复现实验、公开论文结论、模拟结果和未检验假设将分别标注。模拟攻击或相同任务的轨迹，不等于真实 APT 组织归因。

## 进度

- [约 8 小时自动科研计划](00-研究计划.md)：研究问题、数据路线、实验、复核与交付。
- 调研报告、实验 Notebook、结果和复现说明随研究推进补充；当前不代表实验已完成。

## 环境

实验使用本地隔离的 Python 环境；确切依赖见 [requirements.lock.txt](requirements.lock.txt)。无需 Conda。若安装了 Python 与 [uv](https://docs.astral.sh/uv/)，可在本目录运行：

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/jupyter lab
```

公开数据的来源、许可、版本、下载及预处理方式将在实验报告和代码中说明。本地虚拟环境、外部数据仓库和原始数据默认不提交；提交之前应检查样本和 Notebook 输出是否允许公开。
