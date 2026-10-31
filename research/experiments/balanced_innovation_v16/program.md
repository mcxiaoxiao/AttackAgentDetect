# V16 实验方案：增强型判别式分类与FSR优化

## 背景

V15通过革命性的选择逻辑（移除Pareto门控，直接优化balanced utility）实现了突破性性能：
- FMR: 78.1% → 17.9% (↓77%)
- BA: 57.4% → 88.8% (↑55%)
- Hit@1: 29.6% → 90.8% (↑207%)
- NDCG@5: 53.3% → 95.3% (↑79%)

但v15存在一个问题：FSR从基线的1.2%升高到4.5%（↑267%相对退化）。虽然这是可接受的trade-off，但v16应该尝试在保持FMR大幅降低的同时控制FSR。

## V16改进方向

### 1. 增强型分类器

**当前v15**: Logistic Regression (C=0.1, balanced)

**V16候选**:
- Gradient Boosting Classifier (n_estimators=100, max_depth=3)
- Random Forest (n_estimators=200, max_depth=5)
- SVM with RBF kernel (C=1.0, class_weight=balanced)

选择Gradient Boosting，因为：
- 对非线性特征交互建模能力强
- 内置特征重要性评估
- 对过拟合有较好的控制（通过max_depth和learning_rate）

### 2. FSR-aware Utility函数

**当前v15**:
```
balanced_utility = mean(balanced_accuracy, rank_quality, query_coverage)
```

**V16改进**:
```
fsr_penalty = max(0, FSR - FSR_baseline) * penalty_weight
adjusted_utility = balanced_utility - fsr_penalty
```

其中penalty_weight通过内层交叉验证选择（候选值：0.1, 0.2, 0.5, 1.0）。

### 3. 改进的硬负样本挖掘

**当前v15**: top-ranked错误任务对

**V16改进**:
- 动态硬负样本：每轮迭代重新计算最难的负样本
- 聚焦边界样本：选择分数在决策边界附近的负样本
- 增加负样本比例：从5%提升到10%

### 4. 特征工程优化

**当前v15**: 12维固定特征

**V16改进**:
- 特征标准化：Z-score标准化
- 特征选择：基于Gradient Boosting的特征重要性，保留top-10特征
- 特征交互：添加关键特征的二阶交互项

### 5. 集成策略

**V16新增**:
- 训练3个不同的分类器（Gradient Boosting, Random Forest, Logistic Regression）
- 使用soft voting集成
- 通过内层交叉验证选择最优集成权重

## 实验设计

### 嵌套交叉验证
- 外层：3折（与v15相同）
- 内层：3折（与v15相同）
- 种子：20261031（与v15相同）

### 选择逻辑
- 移除Pareto门控（继承v15革命性改进）
- 使用FSR-aware utility函数
- 选择最高adjusted utility的方法-阈值组合

### 评估指标
主要指标：
- False Merge Rate (FMR) - 越低越好
- False Split Rate (FSR) - 越低越好
- Balanced Accuracy (BA) - 越高越好

次要指标：
- Hit@1, MRR, NDCG@5 - 排序质量
- Query Coverage - 召回能力

### 对比基线
- Command TF-IDF（原始基线）
- V15 balanced_selected（当前最优）
- V15 discriminative_pair_score（方法级最优）

## 预期结果

V16应该实现：
1. FMR保持在20%以下（接近v15的17.9%）
2. FSR降低到3%以下（优于v15的4.5%）
3. BA保持在85%以上（接近v15的88.8%）
4. 整体utility提升（考虑FSR惩罚后）

## 成功标准

V16成功当且仅当：
- FSR相比v15降低至少1个百分点（从4.5%降至3.5%以下）
- FMR相比v15升高不超过2个百分点（从17.9%升至19.9%以下）
- Balanced utility（含FSR惩罚）高于v15

## 限制

- 开发集结果，非独立确认
- 任务标签为同任务代理，非演员/组织身份
- 97个trace，40个任务组，样本量有限
- 嵌套交叉验证结果，泛化性待验证
