#!/usr/bin/env python3
"""Create the experiment report from frozen aggregate metrics."""
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "research/results/matched_fingerprint_v1"
R = json.loads((OUT / "metrics.json").read_text())


def item(dataset, model, variant, frac, typ="ALL"):
    return next(x for x in R[dataset]["methods"] if x["learner"] == model
                and x["variant"] == variant and x["fraction"] == frac and x["pair_type"] == typ)


def p(v, signed=False):
    return f"{100*v:+.2f}" if signed else f"{100*v:.2f}"


def ci(v, signed=True):
    return "[" + ", ".join(p(x, signed) for x in v) + "]" if v else "不可估计"


sections = [
    "# 精确角色配比的匹配指纹关联：执行结果", "",
    "## 结论", "",
    "在相同 CTH 配对和训练侧选定的门槛下，三种分类器的五档平均变化方向不一致，且配对区间均跨零。加入指纹会改变误合并／漏关联取舍；本轮没有支持指纹稳定改善 CTH 关联的证据。AIT 排障关联中，完整指纹相对原证据有正的 BA 增量，但漏关联仍高，移除 T 的比较区间跨零。", "",
    "结果是在已用于开发的公开队列上进行的探索性实验。CTH 同题是基准代理标签，AIT 是排障场景；两者都不等于经独立核验的攻击活动或操作者真值。", "",
    "## CTH 五档等权对照", "",
    "| 模型 | BA 原证据 → 加 B/S | ΔBA [95% 区间]，百分点 | FMR 原 → 新 | FSR 原 → 新 |",
    "|---|---:|---:|---:|---:|",
]
for model, label in (("svm", "SVM"), ("tree", "决策树"), ("xgb", "XGBoost")):
    a = item("cth", model, "evidence", "average_5_fractions")
    b = item("cth", model, "BS", "average_5_fractions")
    d = b["vs_evidence"]["ba"]
    sections.append(f"| {label} | {p(a['metrics']['ba']['point'])}% → {p(b['metrics']['ba']['point'])}% | "
                    f"{p(d['point'], True)} {ci(d['ci95'])} | "
                    f"{p(a['metrics']['fmr']['point'])}% → {p(b['metrics']['fmr']['point'])}% | "
                    f"{p(a['metrics']['fsr']['point'])}% → {p(b['metrics']['fsr']['point'])}% |")
sections += [
    "", "完整决策树在三个外层折均选择了无可接纳正例的训练侧阈值，因此测试 FMR 为 0%、FSR 为 100%。训练侧 FMR≤10% 不保证测试侧同样满足，也不保证存在有用的工作点。", "",
    "## 统一消融：CTH B/S 与 AIT B/S/T", "",
    "| 数据 | XGBoost 配置 | BA [95% 区间] | FMR | FSR | 相对完整 ΔBA [95% 区间]，百分点 |",
    "|---|---|---:|---:|---:|---:|",
]
for dataset, fraction, configs in (
    ("cth", "average_5_fractions", (("BS", "E+B+S（完整）"), ("S", "移除 B"),
                                    ("B", "移除 S"), ("evidence", "仅 E"))),
    ("ait", "-1", (("BST", "E+B+S+T（完整）"), ("ST", "移除 B"), ("BT", "移除 S"),
                    ("BS", "移除 T"), ("evidence", "仅 E")))):
    for i, (config, label) in enumerate(configs):
        x = item(dataset, "xgb", config, fraction)
        ba, d = x["metrics"]["ba"], x["vs_xgb_full"]["ba"]
        dc = "参考" if i == 0 else p(d["point"], True) + " " + ci(d["ci95"])
        sections.append(f"| {dataset.upper() if i == 0 else ''} | {label} | "
                        f"{p(ba['point'])}% {ci(ba['ci95'], False)} | "
                        f"{p(x['metrics']['fmr']['point'])}% | {p(x['metrics']['fsr']['point'])}% | {dc} |")
sections += [
    "", "E 为命令与动作二元组相似度。CTH 为五档平均；AIT 是固定混合配对。两部分各自训练，不能直接以绝对数值排名。完整指标、各档和 HH／HA／AA 分层见 summary.csv 与 metrics.json。", "",
    "## 配对与数据审计", "",
]
cth = json.loads((OUT / "cth/input_audit.json").read_text())
ait = json.loads((OUT / "ait/input_audit.json").read_text())
baselines = json.loads((OUT / "baseline_reproduction.json").read_text())
full = item("cth", "xgb", "BS", "average_5_fractions")
sections += [
    f"- 旧两个相似度基线的逐题逐档复现最大误差 {baselines['max_error']:.2g}。",
    f"- CTH：{cth['task_count']} 题、{cth['trace_count']} 条轨迹、{cth['pair_count']} 条跨五档配对记录；同一对可重复出现。",
    f"- CTH 指纹模型回退：{full['fallback_records']}/{full['records']} 条配对记录（{p(full['fallback_records']/full['records'])}%）。",
    f"- AIT：{ait['task_count']} 题、{ait['trace_count']} 条轨迹、{ait['pair_count']} 条配对记录；原始运行内容哈希互不重复。",
    "- 题目留出后构造配对；Human/Agent、来源、题目 ID 均不进入关联模型。各折超参数、阈值和内层候选完整记录在 cth/ 与 ait/ 的 fold 审计文件中。",
    "- 阈值 JSON 中的 null 表示正无穷，即全部拒绝；逐对预测 CSV 将其写作 inf。",
    "- 置信区间对查询任务重采样 10,000 次，保留方法与档位配对。候选任务和轨迹共享导致区间为近似值；重复抽样不是新数据。", "",
    "## 敏感性、错例和限制", "",
]
for dataset in ("cth", "ait"):
    d = [x["delta_ba"] for x in R[dataset]["leave_one_task_out"]]
    sections.append(f"- {dataset.upper()}：每次从查询和候选两侧删除一个任务后，完整指纹相对 E 的 ΔBA 范围为 "
                    f"{p(min(d), True)} 至 {p(max(d), True)} 个百分点；此为敏感性检查。")
cases = list(csv.DictReader((OUT / "paired_score_cases.csv").open()))
sections += ["", "配对分数错例（分数是分类器判定分数，不是校准概率；不把分数变化解释为单一维度的因果作用）：", "",
             "| 数据 | 变化 | 配对类型 | 原分数／门槛 | 指纹分数／门槛 |",
             "|---|---|---|---:|---:|"]
for x in cases:
    sections.append(f"| {x['dataset'].upper()} | {x['error_transition']} | {x['pair_type']} | "
                    f"{float(x['evidence_score']):.3f} / {float(x['evidence_threshold']):.3f} | "
                    f"{float(x['full_score']):.3f} / {float(x['full_threshold']):.3f} |")
sections += [
    "", "AIT 的原证据模型在本组测试上没有误合并，因此不存在可列出的“原误合并被纠正”错例；不以样本重复制造案例。原始命令未写入报告。", "",
    "## 重现", "",
    "在仓库根目录运行 research/experiments/matched_fingerprint_v1/run.sh --dataset cth 和 --dataset ait，随后运行同目录 summarize.py、build_cases.py、build_figure.py、build_appendix.py、build_draft.py、build_page.py 和 package_draft.py。精确输入哈希、包版本、种子、抽样和候选评估见 research/results/matched_fingerprint_v1/。", "",
]
target = ROOT / "research/reports/matched-fingerprint-v1.md"
target.write_text("\n".join(sections))
print("Wrote", target)
