# S-维度与 Gallery 阈值优化结果

这是一轮对已反复查看过的公开队列所做的探索性复核。候选身份、三折整题切分和外折 100 次 gallery 抽样沿用同模式主实验；内层按训练题再划分，使用 20 次抽样完成特征臂选择和阈值校准。点数是任务宏平均百分比，不把重复 gallery 当作独立任务。

## 外折结果

Hit@1/BA 越高越好；FMR、FSR 和 No-match 越低越好。Base 为 CTH 的 E+B、AIT 的 E+B+T。gallery guard 在内层约束 FMR 不高于 min(10%, E) 且 no-match 接受不高于 E，再选最小可行 FSR；pair-threshold 对照复用原内层配对 FMR≤10% 规则。

### CTH

| Arm | Hit@1 | BA | FMR | FSR | No-match |
|---|---:|---:|---:|---:|---:|
| E | 44.40 | 52.04 | 6.47 | 89.46 | 18.45 |
| Base, pair threshold | 49.85 | 53.79 | 8.79 | 83.64 | 23.81 |
| Base, gallery guard | 49.85 | 53.24 | 8.69 | 84.84 | 22.99 |
| Full S, pair threshold | 47.74 | 54.87 | 7.78 | 82.48 | 23.91 |
| Full S, gallery guard | 47.74 | 54.35 | 7.58 | 83.72 | 22.98 |
| S scalars | 49.96 | 54.12 | 7.01 | 84.74 | 20.06 |
| S revisit/reuse | 50.87 | 53.47 | 8.34 | 84.72 | 21.18 |
| S smoothed transition | 50.42 | 54.12 | 7.01 | 84.74 | 20.06 |
| S category n-grams | 50.03 | 54.06 | 7.14 | 84.74 | 20.65 |
| Nested selected | 47.60 | 53.10 | 8.92 | 84.88 | 24.34 |

内层选择：fold 0: fullS, fold 1: fullS, fold 2: base。
阈值变化（gallery guard 减原配对阈值，百分点）：fmr -0.09, fsr +1.20, no_match_accept -0.82.
Nested-selected 的 no-match 误接受候选中，异题同模式占 2595/3601 (72.1%)。
### AIT

| Arm | Hit@1 | BA | FMR | FSR | No-match |
|---|---:|---:|---:|---:|---:|
| E | 49.50 | 57.05 | 5.69 | 80.22 | 21.90 |
| Base, pair threshold | 57.92 | 63.82 | 9.41 | 62.96 | 35.01 |
| Base, gallery guard | 57.92 | 63.32 | 6.88 | 66.47 | 27.86 |
| Full S, pair threshold | 55.54 | 61.33 | 7.93 | 69.40 | 29.65 |
| Full S, gallery guard | 55.54 | 61.29 | 6.32 | 71.10 | 25.82 |
| S scalars | 56.73 | 61.91 | 7.36 | 68.81 | 28.83 |
| S revisit/reuse | 56.39 | 63.64 | 7.38 | 65.34 | 28.75 |
| S smoothed transition | 57.22 | 61.24 | 7.50 | 70.02 | 28.69 |
| S category n-grams | 56.85 | 64.13 | 8.22 | 63.51 | 30.29 |
| Nested selected | 56.93 | 61.63 | 7.45 | 69.28 | 28.82 |

内层选择：fold 0: base, fold 1: S_smoothed_transition, fold 2: base。
阈值变化（gallery guard 减原配对阈值，百分点）：fmr -2.52, fsr +3.51, no_match_accept -7.15.
Nested-selected 的 no-match 误接受候选中，异题同模式占 2838/3961 (71.6%)。

## 解读

- 同题同模式 Hit@1 在所有新增 S 变体中多数高于 E，但没有一个变体在 CTH 和 AIT 上同时把 FMR 与 no-match 压到 E 以下；排序收益不能当作可靠接受阈值的证据。
- Gallery guard 在内层强制控制误接受，外折仍高于 E，表明当前任务量不足以稳定估计拒绝阈值；把阈值推到极高时虽能拒绝更多 gallery，也会增加漏关联。
- 回访/片段复用及平滑类别转移值得保留为候选测量，但本轮不足以据此替换主文 B/S/T 结果或宣称方法改进。2× 异题同模式负例权重臂在报告精度下与对应普通臂一致。
- 错误类型审计显示，无匹配误接受主要来自“异题、同模式”候选：跨题行为相似度过度泛化；每个 no-match gallery 同时含多个负候选，又把单对的小 FMR 放大为任一候选被接受的概率。上表按候选决策统计，不按 gallery 计数。
- 结论保持探索性：公开队列小、任务重复被查看，开发过程中比较了多个 S 方案。结果应作为方法诊断和后续数据需求，不作为独立确认。

## 可复核产物

每个数据集的 `*_metrics.json` 保存每折内层选择、阈值和外折聚合指标；`*_task_metrics.csv` 保存逐任务指标；`*_predictions.csv.gz` 保存所有候选臂、pair-threshold 对照和 nested-selected 的逐对分数、决策与回退标记。`optimize_s_gallery.py` 和 `s_gallery_optimization_protocol.md` 定义执行过程。
