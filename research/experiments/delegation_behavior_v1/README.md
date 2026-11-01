# Delegation behavior v1

研究目标：描述决策委托条件下的可观察行为指纹，检验其对有独立真值的受控活动关联有何影响，再评价预测参与信息的用途。

- [完整论证与实验协议草案](program.md)
- [评估登记表字段模板](evaluation_registry.template.csv)
- [已有输入审计](../../results/delegation_behavior_v1/input_audit.json)
- [24 项画像目录](../../results/delegation_behavior_v1/feature_catalog.csv)

复现输入审计，在仓库根目录运行：

```sh
python3 research/experiments/delegation_behavior_v1/audit_existing.py
```

脚本仅用标准库读取现有画像、评估侧 CSV 和特征定义，不训练模型，不读取历史性能结果，不下载数据。两套输入有重叠，不将其相加当作独立样本。输出记录输入文件和脚本 SHA-256；这一步是数据可用性审计，没有新实验效果指标。

登记模板用于评估侧元数据，不能作为画像模型的直接输入。controller_id、activity_id、任务与真实委托条件须与推理记录隔离；部署可见输入只包含约定的日志证据和质量信息。空模板不是已收集样本。

下一步需要统一采集、条件控制、episode 边界、交接及人工干预记录。实际活动与委托标签不足时，效果评估保持未运行；先完成协议所列的试点与冻结节点。
