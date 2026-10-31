# AttackAgent Detect

<p align="center"><img src="site/assets/attackagent-detect-logo.png" width="500" alt="AttackAgent Detect" /></p>

<p align="center">简体中文 · <a href="README.md">English</a></p>

AttackAgent Detect 是一个 Python 工具库，用于把安全日志整理成统一轨迹，并比较轨迹中的操作行为。仓库另设研究工作区，使用公开 CTF 数据评估这些方法。

## 使用工具库

目前提供：

- 将 Linux auditd 的 SYSCALL 与 EXECVE 记录规范化为轨迹 JSONL。
- 根据命令和动作证据，对候选轨迹排序。
- 使用同一采集格式的带标签训练数据，尝试识别 Human / Agent（实验功能）。

工具只分析日志，不会执行日志中的命令。Human / Agent 分数不能证明实际操作者，也不能证明两条轨迹属于同一攻击。

### 从仓库安装

需要 Python 3.10 或更高版本。当前尚未发布到 PyPI。

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
~~~

规范化仓库自带的 auditd 样例：

~~~bash
.venv/bin/python -m attackagent.jsonl_cli normalize --input examples/auditd/minimal.audit --output /tmp/normalized.jsonl
~~~

对候选轨迹排序：

~~~bash
.venv/bin/python -m attackagent.jsonl_cli rank --query examples/jsonl/queries.jsonl --candidates examples/jsonl/candidates.jsonl --method command_cosine
~~~

运行 .venv/bin/python -m attackagent.jsonl_cli methods 可查看可用的排序方法。更多输入样例见 [examples/](examples/)。

## 研究工作区

研究材料与工具库代码分开放置。研究区包含实验方案、数据清单、实验代码、Notebook、结果和论文源文件。目前的公开 CTF 评估使用 40 个任务组、共 177 条轨迹。“同题”只表示来自同一道挑战题，不能据此认定是同一操作者、攻击活动或现实攻击。研究索引列出了主要报告、数据与复现步骤：[research/README.md](research/README.md)。

[项目网站](https://mcxiaoxiao.github.io/AttackAgentDetect/) 展示研究概况和数据来源。

## 目录

- **attackagent/** — Python 工具库。
- **examples/** — 输入与输出样例。
- **tests/** — 工具库检查。
- **research/** — 研究方案、实验、数据清单、结果、Notebook 与论文。
- **site/** — 项目网站源码。
- **vendor/procgrep/** — 第三方依赖及其许可证。

## 许可证

项目代码采用 MIT 许可证。上游数据许可和第三方声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
