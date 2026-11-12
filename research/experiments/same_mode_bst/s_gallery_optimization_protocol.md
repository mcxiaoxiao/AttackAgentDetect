# S-维度与 Gallery 拒绝优化协议

冻结日期：2026-11-12。该实验是对已反复查看过的公开小样本队列进行的探索性复核，不作为独立确认。

## 问题与假设

现有结果中，S 单独加入时命中率和误接受均不稳定；S 的稀疏转移坐标可能把不同题目但执行模式相同的轨迹视为相似。另一个问题是旧阈值仅约束内折的配对误合并率，没有直接约束一个 gallery 中任一负例被接受的概率。

本轮检验两个有限假设：

1. 将稀疏 S 类别转移向量替换为低维顺序距离，或只保留回访与片段复用统计，能否比完整 S 更稳地提高同题同模式 Hit@1。
2. 在内折同时约束配对 FMR 和 no-match gallery 接受率不高于 E 基线，能否减少外折误接受；若外折不满足，不通过事后调阈值修补。

## 固定队列与切分

- CTH 使用 17 题、85 条轨迹；AIT 使用 12 题、109 条运行。沿用现有三折整题留出、128 事件截断、8 事件窗口，以及 CTH 最小窗口 2、AIT 最小窗口 4。
- 外折候选由同一随机种子、同一题折和同一 gallery 生成器创建，重复 100 次；与既有 E/BST 结果使用同一查询候选身份。每个查询还包含一个无匹配 gallery。
- 内层按外层训练题再做整题留出，用 20 次候选抽样进行选型和阈值选择。外折只用于最终报告。
- CTH 队列和折分文件在仓库内；AIT 原始归档沿用本地外部数据目录 `research/data/external/ait-shell-pilot-v1/data.zip`，不随结果重复打包。结果 JSON 记录归档 SHA-256，重跑 AIT 前需在该位置恢复相同归档。

## 特征臂

E 是命令与动作二元组 TF–IDF 相似度。CTH 的基准臂为 E+B，AIT 为 E+B+T；B 包含窗口行为摘要和固定命令出现向量重叠率，AIT 的 T 只使用已有日志中可观测且验证过的 T1–T6 启动间隔统计。

在相同基准上比较：

- `fullS`：现有 24 项 S 窗口均值、标准差、趋势，加 64 项类别转移向量。
- `S_scalar`：仅 24 项 S 窗口统计。
- `S_revisit_reuse`：仅 S2–S7 的 18 项窗口统计，覆盖回访跨度、返回和二元/三元片段复用。
- `S_smoothed_transition`：24 项窗口统计，加类别转移分布的对称 Jensen–Shannon 距离；转移计数按 Dirichlet α=0.5 平滑，并附缺失标记。
- `S_ordered_ngrams`：24 项窗口统计，加类别二元组和三元组集合的 Jaccard 距离及缺失标记。
- `fullS_hardneg2`、`S_smoothed_hardneg2`：对应 S 臂训练时将“异题同模式”负例权重乘 2；在原标签×H/A 配对类型单元内重归一化，保持单元总权重不变。

类别顺序距离使用固定命令类别表，不拟合标签或测试折词汇。训练折单独拟合标准化和分类器。缺失整个画像块时，该配对回退到 E；排序时若 gallery 中任一候选需要回退，则整组排名统一使用 E 分数。

## 校准与选型

每个内层验证折的训练题用于训练 XGBoost，验证题产生 out-of-fold 配对分数和固定 gallery 分数。E 阈值沿用既有内折规则。画像阈值在 gallery 分数上搜索，满足任务宏平均配对 FMR ≤ min(10%, E 的内折 FMR)，且 no-match gallery 接受率 ≤ E 的内折接受率；在可行阈值中选漏关联率最低者。若并列，选择更高阈值。

为分离“特征改变”和“阈值改变”的作用，E+B（AIT 为 E+B+T）及完整 S 臂还在相同外折候选上复测原先的内折配对 FMR≤10% 阈值。该对照只改变阈值选择方法。Gallery 阈值在分数排序上用单调二分搜索，约束及最优可行阈值与逐一枚举定义相同。

只在满足上述内折约束的臂中按任务宏平均 Hit@1 选型，依次以更少 S 维度、较高 BA 打破平局。选中的臂和阈值用于对应外折。主报告与 E 比较 Hit@1、BA、FMR、漏关联率和 no-match 接受率，不报告 gallery 抽样次数作为独立样本数。

## 解释边界

重复 gallery 抽样不会增加独立任务数。CTH 与 AIT 任务量有限，且特征设计和同类公开队列曾在开发中查看；所有结果均为探索性。配对 FMR 的外折升高或 no-match 接受率升高，均记为校准泛化失败。不得根据外折表现再选臂、改权重或调阈值。模式区分结果不代表共同操作者、攻击活动或组织归因。

## 重跑

从仓库根目录执行：

```bash
export DYLD_LIBRARY_PATH="$PWD/.venv/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
for DATASET in cth ait; do
  ./.venv/bin/python research/experiments/same_mode_bst/optimize_s_gallery.py \
    --dataset "$DATASET" \
    --out "research/results/same_mode_s_optimized" \
    --repetitions 100
done

./.venv/bin/python research/experiments/same_mode_bst/build_s_gallery_report.py \
  --input-dir research/results/same_mode_s_optimized \
  --output research/results/same_mode_s_optimized/report.md
./.venv/bin/python research/experiments/same_mode_bst/build_s_gallery_table.py \
  --input-dir research/results/same_mode_s_optimized \
  --output research/paper/same-mode-s-gallery-results.tex
```
