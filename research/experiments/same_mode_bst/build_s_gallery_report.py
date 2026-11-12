#!/usr/bin/env python3
"""Render the compact human-readable report from frozen experiment JSON."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import pathlib


ARMS = [
    ("E", "E"),
    ("base_pairwise", "Base, pair threshold"),
    ("base", "Base, gallery guard"),
    ("fullS_pairwise", "Full S, pair threshold"),
    ("fullS", "Full S, gallery guard"),
    ("S_scalar", "S scalars"),
    ("S_revisit_reuse", "S revisit/reuse"),
    ("S_smoothed_transition", "S smoothed transition"),
    ("S_ordered_ngrams", "S category n-grams"),
    ("selected_nested", "Nested selected"),
]
METRICS = ["hit1", "ba", "fmr", "fsr", "no_match_accept"]
HEADERS = ["Hit@1", "BA", "FMR", "FSR", "No-match"]


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    results = {
        dataset: json.loads((args.input_dir / f"{dataset}_metrics.json").read_text())
        for dataset in ("cth", "ait")
    }
    lines = [
        "# S-维度与 Gallery 阈值优化结果",
        "",
        "这是一轮对已反复查看过的公开队列所做的探索性复核。候选身份、三折整题切分和外折 100 次 gallery 抽样沿用同模式主实验；内层按训练题再划分，使用 20 次抽样完成特征臂选择和阈值校准。点数是任务宏平均百分比，不把重复 gallery 当作独立任务。",
        "",
        "## 外折结果",
        "",
        "Hit@1/BA 越高越好；FMR、FSR 和 No-match 越低越好。Base 为 CTH 的 E+B、AIT 的 E+B+T。gallery guard 在内层约束 FMR 不高于 min(10%, E) 且 no-match 接受不高于 E，再选最小可行 FSR；pair-threshold 对照复用原内层配对 FMR≤10% 规则。",
        "",
    ]
    for dataset, data in results.items():
        lines.extend([f"### {dataset.upper()}", "", "| Arm | " + " | ".join(HEADERS) + " |",
                      "|---|" + "---:|" * len(HEADERS)])
        metrics = data["results"]
        for key, label in ARMS:
            if key not in metrics:
                continue
            vals = " | ".join(pct(metrics[key][metric]) for metric in METRICS)
            lines.append(f"| {label} | {vals} |")
        selected = ", ".join(
            f"fold {row['outer_fold']}: {row['selected_arm']}"
            for row in data["outer_fold_selections"]
        )
        lines.extend(["", f"内层选择：{selected}。"])
        if "base_pairwise" in metrics and "base" in metrics:
            change = {
                metric: metrics["base"][metric] - metrics["base_pairwise"][metric]
                for metric in ("fmr", "fsr", "no_match_accept")
            }
            lines.append(
                "阈值变化（gallery guard 减原配对阈值，百分点）："
                + ", ".join(f"{name} {100 * value:+.2f}" for name, value in change.items())
                + "."
            )
        with gzip.open(args.input_dir / f"{dataset}_predictions.csv.gz", "rt",
                       newline="", encoding="utf-8") as stream:
            predictions = [row for row in csv.DictReader(stream)
                           if row["arm"] == "selected_nested"
                           and int(row["position"]) >= 4
                           and int(row["predicted_link"]) == 1]
        types: collections.Counter[str] = collections.Counter()
        for row in predictions:
            same_task = row["query_task"] == row["candidate_task"]
            same_mode = row["query_mode"] == row["candidate_mode"]
            kind = ("同题同模式" if same_task and same_mode else
                    "同题跨模式" if same_task else
                    "异题同模式" if same_mode else "异题跨模式")
            types[kind] += 1
        same_mode_share = 100 * types["异题同模式"] / len(predictions) if predictions else 0.0
        lines.append(
            f"Nested-selected 的 no-match 误接受候选中，异题同模式占 "
            f"{types['异题同模式']}/{len(predictions)} ({same_mode_share:.1f}%)。"
        )

    lines.extend([
        "",
        "## 解读",
        "",
        "- 同题同模式 Hit@1 在所有新增 S 变体中多数高于 E，但没有一个变体在 CTH 和 AIT 上同时把 FMR 与 no-match 压到 E 以下；排序收益不能当作可靠接受阈值的证据。",
        "- Gallery guard 在内层强制控制误接受，外折仍高于 E，表明当前任务量不足以稳定估计拒绝阈值；把阈值推到极高时虽能拒绝更多 gallery，也会增加漏关联。",
        "- 回访/片段复用及平滑类别转移值得保留为候选测量，但本轮不足以据此替换主文 B/S/T 结果或宣称方法改进。2× 异题同模式负例权重臂在报告精度下与对应普通臂一致。",
        "- 错误类型审计显示，无匹配误接受主要来自“异题、同模式”候选：跨题行为相似度过度泛化；每个 no-match gallery 同时含多个负候选，又把单对的小 FMR 放大为任一候选被接受的概率。上表按候选决策统计，不按 gallery 计数。",
        "- 结论保持探索性：公开队列小、任务重复被查看，开发过程中比较了多个 S 方案。结果应作为方法诊断和后续数据需求，不作为独立确认。",
        "",
        "## 可复核产物",
        "",
        "每个数据集的 `*_metrics.json` 保存每折内层选择、阈值和外折聚合指标；`*_task_metrics.csv` 保存逐任务指标；`*_predictions.csv.gz` 保存所有候选臂、pair-threshold 对照和 nested-selected 的逐对分数、决策与回退标记。`optimize_s_gallery.py` 和 `s_gallery_optimization_protocol.md` 定义执行过程。",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
