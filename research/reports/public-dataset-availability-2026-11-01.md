# 公开实验数据可用性核查（2026-11-01）

## 范围与方法

从用户提供的Zotero「attack」笔记出发，核对论文全文、官方代码/数据目录，并沿来源补检索arXiv、GitHub、Zenodo、Hugging Face及官方基准页面。这是单人、定向的数据可用性核查，不声称穷尽文献。检索词、目录SHA与文件证据保存在同目录的机器记录中。访问失败或未找到入口不等于数据不存在；“作者称发布”“列出下载文件”“已解析样例”“全量校验”分开报告。

本轮没有运行新分类或关联模型，也没有把论文准确率当作本项目复现结果。原始命令、提示与输出不复制到核查结果目录；上游源码仅阅读，未执行。

本轮随后完成了实际接入：CTH 615条、HF Cybench 189条、HF AgentTrace 338条，共1,142条非空公共命令视图。详见[扩充报告](behavior-corpus-expansion-v1.md)。下述样例核查与早期177条导出均保留其各自口径，不与当前目录相加。

## 选择结论

- CTF任务中的Human/Agent预实验仍以Cyber Task Horizons为起点；现有177条/40题导出中17题至少有2条Human和2条Agent。
- 优先扩充Agent侧：NYU CTF官方提交库（实际会话已读）和Honey for the Agent（归档目录与两份会话已解析）。
- 常见主机日志的双类机制检查：AIT/AECID已下载的109份原始排障运行。它不属于攻击数据。
- 已更正Known By Their Actions的可用性：找到Git LFS归档，并实际解析两份浏览器会话。不能再简单写“只有代码，未确认轨迹”。
- TraceML具有公开Human/Agent版本级轨迹，可作外域验证；InterCode公开的15个人类对照是SQL任务，不是CTF。
- SAGA、ATLASv2和DARPA类历史日志适合关联或溯源适配，不含可直接支撑Human/LLM Agent浓度实验的双类真值。
- 本轮未确认Trace、LIDAR、Synthetic APTs的完整逐会话下载包，也未定位Cross-Agent Campaign Attribution声称的独立补充归档；不将这些论文中的实验规模计为已拿到的数据。
- 不能将来自不同环境/采集器的两份数据分别标成Human和Agent直接训练；也不能将传统脚本、未知攻击流量或SAGA合成数据自动视为某个执行来源。

## 原相关工作逐项核查

### Synthetic APTs

**数据核实 · 2026-11-01：**全文可核对受控实验、TTP 对照表与提示设计，但本轮未定位到20次实验的独立逐会话原始日志归档，也没有 Human 对照。用于研究动机和条件设计；暂不列为可直接训练的数据集。[全文与附录](https://arxiv.org/html/2606.07158v1)。

### Trace

**数据核实 · 2026-11-01：**论文明确写明完整数据集稍后发布；评估代码指向匿名仓库，本轮该入口未能访问。2,028场是论文实验规模，不能当作已下载的公开样本量。当前作为词汇基线与格式参考，暂不计为现成数据。[全文的数据开放说明](https://arxiv.org/html/2605.01186v1) · [论文给出的代码入口](https://anonymous.4open.science/r/TRACE-A66F/)。

### CyBiasBench

**数据核实 · 2026-11-01：**已核对数据目录，并通过公开下载入口读取请求CSV表头：包含record_id、agent、target、request_index、HTTP方法/路径/状态及攻击家族等。可以做Agent侧画像和留一靶场验证；没有Human组，且HTTP请求不能改名成主机执行命令。[实际数据文件](https://github.com/Harry24k/CyBiasBench/tree/main/data) · [字段说明](https://github.com/Harry24k/CyBiasBench/blob/main/schema/columns_traces.md)。

### LIDAR

**数据核实 · 2026-11-01：**已核对全文中的探针和特征定义，未定位到作者提供的独立可下载逐会话轨迹入口。“36个模型”是实验设计规模，不是已获取的数据条数。适合作为主动探针和消融设计参考，暂不承诺直接复用训练数据。[论文全文](https://arxiv.org/html/2609.28559v1)。

### Known By Their Actions

**数据核实 · 2026-11-01：****更新原“未确认数据”的判断：已找到并读取归档样例。**src/traces.tar.xz是Git LFS指针，对应184,239,560字节的归档；本轮通过公开媒体入口读取2份会话JSON，确认含meta、model_name/model_family及dom_trace事件。全量会话数未统计。可用于浏览器Agent指纹与跨任务验证；未发现配套Human对照，也不属于攻击终端日志。[LFS归档入口](https://github.com/KabakaWilliam/known_actions/blob/main/src/traces.tar.xz) · [实际归档下载](https://media.githubusercontent.com/media/KabakaWilliam/known_actions/main/src/traces.tar.xz)。

### ProcGrep

**数据核实 · 2026-11-01：**代码与底层公开编码轨迹来源可核对。例如SWE-rebench OpenHands数据页直接提供trajectory_id、instance_id、trajectory和tools字段。ProcGrep README所列spines入口本轮返回401，暂不承诺该预计算集可用；底层轨迹与模型/框架标签仍须逐来源核验。适合外域方法验证，没有Human攻击对照。[代码与来源](https://github.com/hamidahoderinwale/procgrep) · [已确认的底层轨迹](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)。

### Cross-Agent Campaign Attribution

**数据核实 · 2026-11-01：**全文称补充材料包含SCD-v1与脱敏派生记录，但本轮未定位到独立可下载的补充数据包；因此“论文称发布”与“本地已拿到”分开记录，暂不按可立即训练处理。其persona/campaign为合成标签，针对Agent的攻击会话也不能直接充当Human/Agent执行来源标签。[全文与补充材料说明](https://arxiv.org/html/2607.18826v1)。

### CVE-Bench

**数据核实 · 2026-11-01：**已核对官方仓库，提供漏洞任务、运行与评分工具，适合在统一日志条件下重采。README中的“data and code”不自动等于论文所有逐会话轨迹已公开；本轮未确认完整历史评估轨迹下载包，也没有Human对照。[官方仓库](https://github.com/uiuc-kang-lab/cve-bench)。

### Diamond Model

**数据核实 · 2026-11-01：**这是组织入侵证据与关联关系的分析框架，不提供执行轨迹数据、Human/Agent标签或可直接训练的识别器。保留为字段与关系建模参考。

### ATLAS / HOLMES / ATLASv2

**数据核实 · 2026-11-01：**ATLASv2论文明确给出公开数据与REAPr攻击进程标签入口，含Windows安全、Sysmon等日志，可支持溯源图适配和攻击重建。真人日常背景活动不等于Human攻击会话；数据没有Agent组，不能单独做浓度实验。原ATLAS与HOLMES也不提供Human/Agent执行来源对照。[ATLASv2数据](https://bitbucket.org/sts-lab/atlasv2) · [攻击进程标签](https://bitbucket.org/sts-lab/reapr-ground-truth) · [开放说明](https://arxiv.org/html/2401.01341)。

### HERCULE / OSTINATO

**数据核实 · 2026-11-01：**OSTINATO使用OpTC和DARPA TC日志及红队记录构建关联评估，完整输入包含跨主机进程/对象与溯源关系。可参考关联基线，但不能仅给命令名就声称复现原算法。本轮未核实到其完整派生配对标签下载包；这类历史日志也不能默认全部标成Human攻击。[OSTINATO数据与真值说明](https://arxiv.org/html/2312.09321) · [DARPA TC官方数据入口](https://github.com/darpa-i2o/Transparent-Computing)。

### POIROT / SFM

POIROT与SFM需要事件/对象、因果关系或TTP/子图输入。SFM引用的[SAGA发布页](https://saga-tw.github.io/dataset/)有模拟campaign日志入口；这些标签不等于Human/Agent。仅有命令序列不能声称复现完整溯源图方法。

## 补充公开来源

### NYU CTF 官方提交轨迹库

**已读取实际会话样例，优先用于扩充CTF Agent侧。**官方目录有15份提交summary.json；已读JSON包含模型配置、对话、题目、成功状态和时间字段。不同提交格式不统一，提交数不等于独立模型或会话数。可按相同题目和不同模型/框架比较指纹，或增加Agent重复记录；与CTH Human记录拼接前必须核对题目版本和采集差异。没有配套Human终端对照。

[数据仓库](https://github.com/NYU-LLM-CTF/leaderboard_submissions) · [实际轨迹目录](https://github.com/NYU-LLM-CTF/leaderboard_submissions/tree/main/transcripts)

### Honey for the Agent

**归档目录及会话样例已解析。** 作者说明约16,200个会话，本轮ZIP目录实际有16,199个JSON文件；读取2份会话，确认初始化和逐轮交互的列表结构。归档157,739,317字节，解压总量约1.21GB；通过HTTP范围请求共读取约2.43MB完成检查，未下载全部数据或校验全量MD5。字段包含模型/提示/环境/轮数上限、命令、输出及响应耗时；无Human组。CC BY 4.0由Zenodo记录核对。适合模型/提示/环境对行为影响的Agent侧验证，环境/提示标签不能直接当活动身份。

[数据与字段说明](https://zenodo.org/records/20818246) · [日志归档](https://zenodo.org/records/20818246/files/Logs_final.zip?download=1) · [论文正式记录](https://orbit.dtu.dk/en/publications/honey-for-the-agent-cyber-deception-and-behavioral-fingerprinting/)

### Cyber Task Horizons：继续扩充现有安全任务轨迹

**同时提供Human与Agent记录，仍是现有CTF实验入口。**上游说明有90份Human终端记录和Agent评估日志，不能把完成记录或时间估计都算作可用轨迹。本地177条/40题导出中，有17题至少包含2条Human和2条Agent，能开展受限浓度预实验。题目一致是任务代理标签；人的script终端录制与Agent工具调用采集不同，归一化不等于消除了观测差异。

[公开数据](https://github.com/lyptus-research/cyber-task-horizons-data) · [Human采集说明](https://github.com/lyptus-research/cyber-task-horizons-data/blob/main/data/human/eval_logs/README.md)

### AIT/AECID：Human 与 AI 排障主机日志

**已下载并审计原始每次运行日志。**data.zip中有109份独立路径的原始audit.log，来源标签为61份Human、48份Agent，并有聚合副本；只使用原始每次运行版本。适合验证日志解析、画像提取和采集干扰；任务是Nextcloud/WordPress排障，不是攻击。V2/V3重复需按同一基础模型分组，不能当成新模型。

[数据与代码](https://github.com/ait-aecid/llm-user-shell-sim) · [原始日志归档](https://github.com/ait-aecid/llm-user-shell-sim/blob/main/data.zip)

### TraceML：Human 与 Agent 的版本级行为轨迹

**数据页提供实际表格记录，可作外域表示验证。**发布页paired子集为7个竞赛、430条Human轨迹与200条Agent轨迹；论文全语料、额外实验与配对子集的口径不同，不混加计数。记录是Kaggle代码版本/变化及动作标注，不是主机命令日志。部分Agent轨迹是共享节点的树分支，必须按原始run/谱系分组；不能把分支数当独立执行次数。

[数据与字段](https://huggingface.co/datasets/jerryyan/TraceML) · [论文](https://arxiv.org/abs/2608.26086)

### InterCode-SQL：小规模同机制 Human 对照

**已定位到公开Human轨迹文件。**原InterCode论文的人类对照是15个SQL任务，不是15个CTF攻击任务；仓库有ic_sql_multiturn_human_10_turns.json和Agent结果。可检查人/Agent在相近观测机制下的序列表示，但SQL操作需要单独适配，不能直接套用Linux命令维度。

[实际Human轨迹](https://github.com/princeton-nlp/intercode/tree/master/data/results/sql/human) · [官方仓库](https://github.com/princeton-nlp/intercode)

### SAGA：带campaign结构的模拟审计日志

**已核实发布页、版本说明与归档链接。**v2提供8个已知画像、20个生成campaign和10个复合campaign的模拟日志，适合检验关联流程、TTP或日志图表示。画像来自情报报告并不代表真实组织采集；数据没有独立Human/LLM Agent执行标签，不能把“SAGA合成”当作“LLM Agent”或直接拿来提高Agent浓度。本轮未下载完整归档。

[数据版本与下载](https://saga-tw.github.io/dataset/v2) · [SFM论文](https://arxiv.org/abs/2410.22602)

### Cybench / HackSynth：补采环境与运行框架

已核对官方仓库的任务、运行或配置材料。适合复用环境并自行记录相同契约的Human/Agent日志；“运行后会生成logs目录”不能作为历史实验日志已经公开的证明。本轮不把未核实的完整历史轨迹包计入可直接下载数据。

[Cybench](https://github.com/andyzorigin/cybench) · [HackSynth](https://github.com/aielte-research/HackSynth)

### Palisade LLM Honeypot：现场案例与主动探测参考

项目公开代码与仪表盘，可参考SSH事件采集和主动探测。现场“疑似Agent”依赖探测规则，未命中的流量也不自动等于Human；传统脚本、扫描器和未知来源必须保留。未核实可直接作为监督学习主数据的完整、独立核验Human/Agent标签集。

[官方代码](https://github.com/PalisadeResearch/llm-honeypot) · [项目说明](https://palisaderesearch.org/research/llm-honeypot)

## 抽样核查的具体证据

- NYU官方目录树SHA：69def02a79e8b1d890e1e0d5c11106a9f40e11e1。存在15份提交summary.json；抽查一份实际JSON，顶层包含args、messages、challenge、solved、rounds、start_time/end_time等。目录文件数不是独立会话数，pass@k也不是单次成功标签；需要逐提交适配。
- Known Actions目录树SHA：4f3df1851e0e64dbf835344372b02d1b9593cfab。LFS指针给出的归档大小184,239,560字节；媒体入口返回200并可以流式解压。两份JSON包含meta、result、verification、error、midscene_log和dom_trace，meta含episode_id、agent_id、model_name、model_family、timestamp、question、start_url、task_type。未统计全归档会话数，未校验完整LFS对象哈希。
- Honey：目录列出16,199份JSON；样例位于prompt_5/20/plain_cowrie/qwen3:4b，均21个列表元素，首元素包含初始化信息。样例解析不等于所有会话均通过质量筛选；精确原始规模与最终可用规模分别报告。
- CyBiasBench：实际CSV表头已通过206范围下载核对，包括record_id、scenario、target、agent、request_index、req_method、req_path、resp_status_code、attack_family、matched_rules、capec_id、cwe_id、success、success_evidence。
- InterCode：目录树SHA c3e46d827cfc9d4c704ec078f7abf9f41e3191d8；Human文件在data/results/sql/human/ic_sql_multiturn_human_10_turns.json。不能根据论文“human trajectories”字样将其算作CTF Human记录。
- TraceML：数据卡paired范围是430 Human + 200 Agent，论文全范围207 Agent还涉及其他口径；树分支共享原run的依赖必须保留。其动作是版本变更标注，与主机EXECVE观测粒度不同。

## 对下一步实验的建议

先补NYU和Honey的少量代表性会话，建立分别对应其观测类型的适配与质量审计，保留模型/框架/提示/环境/任务/原始run登记。用同题标签做CTF代理实验时，只在题目版本和观测契约可比较的交集上联合来源。Human不足、采集不一致不能通过多下载Agent轨迹解决。

页面继续保留原笔记，并在每篇后追加日期、状态与直接来源链接；核实补充优先于旧的可用性描述。没有把Agent-only、受害主机背景用户活动、派生版本轨迹或合成persona/campaign混成一个Human/Agent主训练集。

## 核查流程参考

使用literature-review技能组织可复现检索记录、来源核对和可用性分级；它是流程支持，不是数据有效性证据。参考：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner. *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. 2026. [arXiv最新记录](https://arxiv.org/abs/2609.00065) · [DOI](https://doi.org/10.48550/arXiv.2609.00065)。
