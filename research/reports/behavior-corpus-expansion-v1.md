# 多源行为轨迹集扩充 v1

日期：2026-11-01。状态：数据接入与标准化已完成；未在扩充数据上声称新的分类、矩阵或关联效果。

## 已完成

目录共1,142条非空记录：79 Human、1,063 Agent。CTF核心664条，漏洞复现/命令补充478条。共有54个来源内任务组有双类记录，涉及207条轨迹；其中CTF双类任务组40个。

| 已接入来源 | 保留记录 | Human | Agent | 来源内任务/条件键 |
|---|---:|---:|---:|---:|
| antieval/cybench-trajectories | 189 | 0 | 189 | 32 |
| cyber-task-horizons | 615 | 79 | 536 | 282 |
| pagarsky/agent-trace | 338 | 0 | 338 | 331 |

CTH上游当前main仍是原固定版本178c0da1800f9c07311e1b8d6c61dd8104e7b4ae。本地已具备90个Human归档；本次重新导出全部已核实的选择清单，解码到87条Human样本，其中79条有公共命令视图。三个无样本归档仍记录在来源清单中。Agent重复retry快照按eval-set/任务/epoch确定性选择，不以旧快照重复增加样本。

HF Antieval六个模型JSONL共190条，保留189条。AgentTrace两份NL2Bash JSONL共400条，保留338条；模型和fixture版本均保留。跨fixture的相同题号不直接合并。本次新增下载约15.5MB，复用CTH已有原始文件，未重新下载18GB上游全部数据。

## 采集和标签的边界

- 标准化视图是显式shell输入中的公共命令头名称；它有词表过滤，不是完整主机执行日志。工具输出、答案、资源路径不用于猜测命令。
- CTF按来源内挑战身份评估；CVE可以按来源内漏洞目标评估，并保留one-day信息条件；NL2Bash带fixture条件。没有跨来源自动断言同名题目等价。
- 模型、模式、来源与任务是评估侧登记，不作为可见行为特征输入。任务标识不等于活动、操作者或APT组织。
- 新增HF来源没有Human组，不增加双类确认样本。短命令轨迹只适合对应的检索、质量/拒判或压力分析；不因在目录里就算作矩阵分类的有效样本。
- 原97条/40任务/1,558配对评估作为历史结果保留，未套用到扩充目录。

## 异构性的研究定位

关联工具面向多种行为检索与匹配问题。使用不同场景的数据可以检验表示接口、观测条件变化和方法失效边界，是合理的取样设计；“数据来自多个地方”本身不是泛化成功。

因此采用共享输入视图、分域标签和分域评估：先在各场景内说明何为关联，再做任务、模型、来源留出。具备双类标签的来源可研究Human/Agent及浓度；只有Agent的来源仍可研究任务检索、模型变化和表示迁移，但不能报告完整双类识别性能。跨源复用和泛化是要实验验证的目标，不通过改名或增加记录数推导。

## 筛选情况

共414条已解码/选择记录缺少显式保留命令视图，逐条登记于excluded_records.csv。按来源：{'cyber-task-horizons': 351, 'antieval/cybench-trajectories': 1, 'pagarsky/agent-trace': 62}。这表示当前公共视图不可用，不代表上游运行无意义。

Hugging Face补充检索还检查了LinuxArena、CyberAgentTrace和部分合成SFT集合。LinuxArena存在真实工具轨迹，但本轮尚未选择可核验的完整小批记录；CyberAgentTrace需要人工访问审批，未申请或导入；模拟工具输出的合成SFT轨迹未混入经验性核心数据。CTH的HF镜像主要是完成/估时表，不是新增终端轨迹来源。

## 复现与发布文件

- 首次获取：`python3 research/scripts/fetch_behavior_corpus_v1.py --download`。固定120份输入文件，总计1,443,787,470字节；已有文件校验后复用。去掉`--download`只检查本地缓存；`--source hf`只处理新增8份HF文件。
- HF固定下载清单已随代码保存到`research/data/manifests/behavior_corpus_hf_cybench_v1.json`与`behavior_corpus_hf_agenttrace_v1.json`，不依赖未发布的本地清单。
- 构建：`.venv/bin/python research/scripts/build_behavior_corpus_v1.py`。只读取数据，不执行远端代码或命令。
- 来源文件哈希：`research/results/behavior_corpus_v1/provenance.json`。
- 完整登记：`trace_registry.csv`；测量输入：`evidence.jsonl`（不含模式/任务标签）；剔除记录：`excluded_records.csv`；统计：`summary.json`。
- 页面目录：`site/data/behavior-corpus-v1.json`；公开登记：`site/data/behavior-corpus-v1.csv`。公开的是命令名称预览与登记，不复制原始提示/输出/flag。
- 逐源许可保留：CTH为CC BY 4.0；AgentTrace为Apache-2.0并保留上游题目条款；Antieval未声明统一许可，其状态如实登记，不套用其他来源的许可。

## 来源

- [Cyber Task Horizons](https://github.com/lyptus-research/cyber-task-horizons-data) · [Human日志说明](https://github.com/lyptus-research/cyber-task-horizons-data/blob/main/data/human/eval_logs/README.md)
- [HF Antieval Cybench](https://huggingface.co/datasets/antieval/cybench-trajectories)，固定f5696b7efe1a176d00e529ca91fc35dd3a9b5b0f。
- [HF AgentTrace](https://huggingface.co/datasets/pagarsky/agent-trace)，固定4b05b2f00eea267a5bb4d841c228059d1bf9ac0c。

文献可用性核查另见`public-dataset-availability-2026-11-01.md`；原始Zotero笔记在相关工作页保留，主数据候选清单已收缩到本研究实际能接入或有直接用途的来源。

追加检索还检查了Enigma、CheatBench与Wild-Trajectory。Enigma直接入口本轮返回401；Wild-Trajectory需接受联系方式共享条件，未申请或下载。CheatBench公开卡与目录可核对，包含Cybench派生记录，但它是作弊检测的二次整理；未完成会话级命令解析和与已有来源的重叠核对，本轮不计入目录规模。优先保留原始来源和明确适配的数据，避免用聚合库重复扩容。
