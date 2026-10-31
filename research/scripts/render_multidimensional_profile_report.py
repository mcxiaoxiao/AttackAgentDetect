#!/usr/bin/env python3
"""Render the Markdown and TeX result summaries from the frozen JSON output."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/multidimensional_profile_association_v1/results.json"
MARKDOWN = ROOT / "reports/13-多维行为画像关联实验结果.md"
TEX = ROOT / "paper/multidimensional-profile-results.tex"
MAIN = ROOT / "paper/main.tex"

METHODS = (
    "command_cosine", "behavior", "structure_aware", "mode_aware", "length_only",
    "profile_only", "tensor_flat_only", "tensor_tucker_only",
    "behavior_profile", "behavior_tensor_flat", "behavior_tensor_tucker",
    "behavior_mode_tucker",
)
LABELS = {
    "command_cosine": ("命令余弦", "Command cosine"),
    "behavior": ("行为排序器", "Behavior ranker"),
    "structure_aware": ("行为 + 旧结构", "Behavior + structure"),
    "mode_aware": ("行为 + 旧模式", "Behavior + mode"),
    "length_only": ("长度控制", "Length control"),
    "profile_only": ("24 维画像", "24D profile"),
    "tensor_flat_only": ("展平张量", "Flat tensor"),
    "tensor_tucker_only": ("Tucker 张量子空间", "Tucker tensor subspace"),
    "behavior_profile": ("行为 + 24 维画像", "Behavior + 24D profile"),
    "behavior_tensor_flat": ("行为 + 展平张量", "Behavior + flat tensor"),
    "behavior_tensor_tucker": ("行为 + Tucker 子空间", "Behavior + Tucker subspace"),
    "behavior_mode_tucker": ("行为 + 模式 + Tucker", "Behavior + mode + Tucker"),
}
MAIN_METRICS = (
    ("false_merge_rate", "FMR"), ("false_split_rate", "FSR"),
    ("false_merge_human_agent", "H–A FMR"),
    ("false_split_human_agent", "H–A FSR"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("query_has_accepted_match", "Query coverage"),
)
RANK_METRICS = (("hit_at_1", "Hit@1"), ("mrr", "MRR"), ("recall_at_5", "Recall@5"),
                ("ndcg_at_5", "NDCG@5"))
PAIR_METRICS = (
    ("false_merge_rate", "Overall FMR"), ("false_split_rate", "Overall FSR"),
    ("false_merge_human_agent", "H–A FMR"), ("false_split_human_agent", "H–A FSR"),
    ("hit_at_1", "Hit@1"), ("mrr", "MRR"), ("recall_at_5", "Recall@5"),
    ("ndcg_at_5", "NDCG@5"),
)


def value(record, percent=True):
    if record is None or record.get("estimate") is None:
        return "NA"
    scale = 100 if percent else 1
    estimate = record["estimate"] * scale
    low, high = [item * scale for item in record["ci_95"]]
    return f"{estimate:.1f} [{low:.1f}, {high:.1f}]"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(out)


def render_markdown(result):
    target_20 = result["false_merge_targets"]["20"]["summary"]
    ranking = result["ranking_summary"]
    rows = []
    for method in METHODS:
        if method not in target_20:
            continue
        rows.append([LABELS[method][0], *[value(target_20[method][name]) for name, _ in MAIN_METRICS],
                     *[value(ranking[method][name]) for name, _ in RANK_METRICS]])
    headers = ["方法", *[label for _, label in MAIN_METRICS], *[label for _, label in RANK_METRICS]]

    sensitivity = []
    for target in ("10", "30"):
        summary = result["false_merge_targets"][target]["summary"]
        for method in METHODS:
            if method in summary:
                sensitivity.append([target + "%", LABELS[method][0],
                                    value(summary[method]["false_merge_rate"]),
                                    value(summary[method]["false_merge_human_agent"]),
                                    value(summary[method]["false_split_rate"]),
                                    value(summary[method]["false_split_human_agent"]),
                                    value(summary[method]["query_has_accepted_match"])])

    paired = result["paired_deltas_at_20_percent_target_vs_behavior"]
    paired_rows = []
    for method in METHODS:
        if method == "behavior" or method not in paired.get("false_merge_human_agent", {}):
            continue
        cells = []
        for metric, _label in PAIR_METRICS:
            rec = paired.get(metric, {}).get(method)
            cells.append(value(rec))
        paired_rows.append([LABELS[method][0], *cells])

    evaluation = result["evaluation"]
    profile = evaluation["profile"]
    desc = result["descriptive_tensor_means_by_record_label"]
    lines = [
        "# 多维行为画像关联实验结果",
        "",
        "本页记录冻结计划 `plans/12-多维行为画像关联实验计划.md` 的执行结果。实验使用已被探索过的 CTH 固定队列；这些结果属于探索性复用，不是独立确认。",
        "",
        f"- 数据：{result['dataset']['records']} 条轨迹、{result['dataset']['benchmark_qualified_task_groups']} 个 `(基准, 题目)` 组；正例为同题 CTF 轨迹。",
        "- Human 终端记录与 Agent 工具调用的采集渠道不同，且与记录标签混杂；不据此推断稳定人格、真实活动归属或组织归因。",
        f"- 多维表示：24 个标量特征 + `{profile['tensor_axes']['phase']} × {len(profile['tensor_axes']['action_family'])} × {len(profile['tensor_axes']['channels'])}` 张量；通道为命令占比、操作类内词汇多样度、相邻同命令重复率。",
        "- Tucker/HOSVD 对照在各训练折内拟合固定 `(2, 4, 2)` 秩；这是结构保留的低秩对照，不是原始 TSA 算法的复现。",
        "- 切分：3 折按题目留出；阈值只在外层训练集内的 3 折 OOF 分数上选择；20% 宏误合并率是主目标，10%/30% 是预设敏感性工作点。",
        "- 区间：95% 配对基准分层题目簇 bootstrap 区间，5,000 次抽样；重复切分没有被当成独立样本。",
        "",
        "## 20% 训练误合并目标",
        "",
        "下表同时列出完整操作阈值指标和不依赖阈值的检索指标。数值为估计值 [95% 区间]，均以百分比表示。实际测试 FMR 不受训练目标保证；命中率下降或漏关联上升也完整保留。",
        "",
        md_table(headers, rows),
        "",
        "![固定 20% 训练误合并目标下的误合并—误拆分权衡](results/multidimensional_profile_association_v1/multidimensional_profile_tradeoff_v1.png)",
        "",
        "## 相对行为排序器的配对变化（20% 目标）",
        "",
        "负的误合并/误拆分变化分别表示该类错误减少；正的 Hit@1、MRR 和 Recall@5 变化表示排序改善。区间未作多重比较校正，不根据其中最小的区间或 p 值挑选赢家。",
        "",
        md_table(["方法", *[f"Δ {label}" for _, label in PAIR_METRICS]], paired_rows),
        "",
        "![新增方法相对旧行为排序器的跨模式错误与排序变化](results/multidimensional_profile_association_v1/multidimensional_profile_effects_v1.png)",
        "",
        "## 张量画像与特征保存",
        "",
        f"张量热图为全队列均值（Human {desc['human']['trace_count']} 条、Agent {desc['agent']['trace_count']} 条合并后计算），没有按记录标签分层，避免把采集渠道差异包装成画像差异。",
        "",
        "![4 阶段 × 8 操作类别 × 3 行为通道的总体均值](results/multidimensional_profile_association_v1/multidimensional_profile_tensor_v1.png)",
        "",
        "可复用文件：",
        "",
        "- `results/multidimensional_profile_association_v1/results.json`：完整方法、指标、区间、阈值和折信息。",
        "- `results/multidimensional_profile_association_v1/pair_scores.csv`、`task_metrics.csv`、`thresholds.csv`：配对分数、任务级结果和阈值审计。",
        "- `results/multidimensional_profile_association_v1/behavior_profiles_v3.jsonl`：逐轨迹 label-free 24 项画像、4 × 8 × 3 张量和外层留出模式预测；不含评估真值。",
        "- `results/multidimensional_profile_association_v1/evaluation_labels.csv`：单独保存的 Human/Agent 评估标签与题目分组。",
        "- 绘图和生成入口：`scripts/plot_multidimensional_profile_association.py`、`scripts/render_multidimensional_profile_report.py`、`scripts/build_tensor_demo_assets.py`。",
        "",
        "## 研究页动画",
        "",
        "研究页用三张通道切片呈现总体张量均值，动画按阶段逐步推进。动画是特征构造示意，不是分类表现或关联结果；静态 SVG 是回退显示，JavaScript 遵循系统减少动态效果设置。",
    ]
    return "\n".join(lines) + "\n"


def tex_num(record):
    if record is None or record.get("estimate") is None:
        return "--"
    estimate = record["estimate"] * 100
    low, high = [item * 100 for item in record["ci_95"]]
    return f"{estimate:.1f} [{low:.1f}, {high:.1f}]"


def render_tex(result):
    rates = result["false_merge_targets"]["20"]["summary"]
    ranks = result["ranking_summary"]
    rate_rows = []
    rank_rows = []
    for method in METHODS:
        if method not in rates:
            continue
        label = LABELS[method][1]
        rate_rows.append(" & ".join([label,
            tex_num(rates[method]["false_merge_rate"]),
            tex_num(rates[method]["false_split_rate"]),
            tex_num(rates[method]["false_merge_human_agent"]),
            tex_num(rates[method]["false_split_human_agent"]),
            tex_num(rates[method]["balanced_accuracy"]),
            tex_num(rates[method]["query_has_accepted_match"]),
        ]) + r" \\")
        rank_rows.append(" & ".join([label, *[tex_num(ranks[method][name]) for name, _ in RANK_METRICS]]) + r" \\")
    rate_table_rows = "\n".join(rate_rows)
    rank_table_rows = "\n".join(rank_rows)

    sensitivity_rows = []
    for target in ("10", "30"):
        target_summary = result["false_merge_targets"][target]["summary"]
        for method in METHODS:
            if method not in target_summary:
                continue
            summary = target_summary[method]
            sensitivity_rows.append(" & ".join([
                rf"{target}\%", LABELS[method][1],
                tex_num(summary["false_merge_rate"]),
                tex_num(summary["false_split_rate"]),
                tex_num(summary["false_merge_human_agent"]),
                tex_num(summary["false_split_human_agent"]),
                tex_num(summary["query_has_accepted_match"]),
            ]) + r" \\")

    paired = result["paired_deltas_at_20_percent_target_vs_behavior"]
    selected_method = "behavior_profile"
    def delta_text(metric):
        record = paired[metric][selected_method]
        estimate = record["estimate"] * 100
        low, high = [item * 100 for item in record["ci_95"]]
        return f"{estimate:+.1f} [{low:+.1f}, {high:+.1f}]"

    profile_deltas = "; ".join(
        f"{label} {delta_text(metric)}"
        for metric, label in PAIR_METRICS
    )
    tex = r"""\section{Multidimensional behavior profiles}
\label{sec:multidimensional-profiles}

The preserved baseline represents execution-mode evidence as a predicted scalar score. This extension stores a 24-feature label-free profile and an ordered third-order tensor with axes \emph{phase} (four contiguous command-index bins), \emph{action family} (eight fixed categories), and \emph{behavior channel} (within-phase command share, within-family command vocabulary diversity, and adjacent exact-command repeat rate). Empty cells are zero; the command-share channel distinguishes absent events from observed non-repetition. The schema and per-trace records are saved separately from the Human/Agent evaluation labels.

For each fold, we fit profile imputers, scalers, and tensor factors on training traces only. We compare vector-valued absolute profile differences and flat-tensor differences with a train-fitted Tucker/HOSVD projection (fixed ranks $(2,4,2)$). The projection uses mode-unfolding SVD factors and a Tucker core; it is a TSA-inspired structured representation rather than a reproduction of the original image-focused TSA algorithm. The original TSA paper treats images as second-order tensors and learns a tensor subspace; HOSVD generalizes SVD through mode-wise unfoldings and a core tensor (\href{https://papers.nips.cc/paper_files/paper/2005/hash/332647f433a1c10fa2e2ae04abfdf83e-Abstract.html}{TSA, NeurIPS 2005}; \href{https://jmlr.csail.mit.edu/papers/volume22/21-0287/21-0287.pdf}{mode-wise tensor decomposition, JMLR 2022}).

The evaluation reuses the pinned 177-trace, 40-task CTH cohort. Outer and nested folds keep complete benchmark-qualified tasks together. Every ranker uses the same query/candidate sets, with no mode-based candidate filtering. Thresholds are selected within outer training data under the prespecified 20\% macro false-merge ceiling; 10\% and 30\% are sensitivity points. Tables~\ref{tab:multidimensional-profile-results}--\ref{tab:multidimensional-profile-ranking} report all prespecified methods. Entries are task-macro percentages with 95\% paired benchmark-stratified task-cluster bootstrap intervals (5,000 draws). Ranking measures do not depend on the threshold.

\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\renewcommand{\arraystretch}{1.18}
\begin{tabular}{lrrrrrr}
\toprule
Method & FMR & FSR & H--A FMR & H--A FSR & Bal. Acc. & Coverage \\
\midrule
""" + rate_table_rows + r"""
\bottomrule
\end{tabular}
\caption{Association results at the 20\% training false-merge target. FMR is false merge; FSR is false split; coverage is the proportion of queries with an accepted candidate. Intervals are task-cluster bootstrap intervals.}
\label{tab:multidimensional-profile-results}
\end{table}

\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
Method & Hit@1 & MRR & Recall@5 & NDCG@5 \\
\midrule
""" + rank_table_rows + r"""
\bottomrule
\end{tabular}
\caption{Threshold-independent retrieval results at the same held-out queries. Higher values indicate better ranking.}
\label{tab:multidimensional-profile-ranking}
\end{table}

\begin{table}[p]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{rlrrrrr}
\toprule
Target & Method & FMR & FSR & H--A FMR & H--A FSR & Coverage \\
\midrule
""" + "\n".join(sensitivity_rows) + r"""
\bottomrule
\end{tabular}
\caption{Sensitivity analyses at the prespecified 10\% and 30\% training false-merge targets. Values are outer-test estimates; a training target does not guarantee the corresponding test false-merge rate.}
\label{tab:multidimensional-profile-sensitivity}
\end{table}

For the additive 24D profile ranker, paired changes versus the behavior ranker at the 20\% target (percentage points [95\% interval]) are: """ + profile_deltas + r""". These are unadjusted descriptive intervals among the prespecified comparisons, not a multiplicity-corrected winner claim.

\begin{figure}[tb]
  \centering
  \includegraphics[width=\linewidth]{multidimensional_profile_tradeoff_v1.pdf}
  \caption{Test false-merge and false-split estimates at a common training target. Each point is a prespecified method; bars show task-cluster bootstrap intervals. The target is not a guarantee about test-set false merges.}
  \label{fig:multidimensional-profile-tradeoff}
\end{figure}

\begin{figure}[tb]
  \centering
  \includegraphics[width=\linewidth]{multidimensional_profile_tensor_v1.pdf}
  \caption{Descriptive cohort mean of the phase--action--channel tensor. Channels are shown as aligned slices. Values pool Human and Agent records because capture pathway is confounded with the record label; this figure does not establish behavioral identity differences.}
  \label{fig:multidimensional-profile-tensor}
\end{figure}

\paragraph{Interpretation.}
The point estimates and intervals are reported together, including increases in false splits and methods that do not improve retrieval. The CTH labels denote same-task challenge matching, not common attack activity, operator, or organization. Human terminal input and Agent tool calls use different capture pathways, so observed feature differences cannot be separated from instrumentation in this cohort. These results are exploratory reuse, not independent confirmation.

\begin{figure}[tb]
  \centering
  \includegraphics[width=\linewidth]{multidimensional_profile_effects_v1.pdf}
\caption{Paired change versus the behavior ranker at the 20\% target. The panels show overall FMR, overall FSR, Hit@1, and MRR changes. Intervals are not multiplicity-adjusted tests.}
  \label{fig:multidimensional-profile-effects}
\end{figure}
"""
    return tex


def main():
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    MARKDOWN.write_text(render_markdown(result), encoding="utf-8")
    TEX.write_text(render_tex(result), encoding="utf-8")
    main_text = MAIN.read_text(encoding="utf-8")
    target = "\\input{experiment-results}"
    addition = target + "\n\\input{multidimensional-profile-results}"
    if "\\input{multidimensional-profile-results}" not in main_text:
        if target not in main_text:
            raise SystemExit("Could not find the experiment-results input in paper/main.tex")
        MAIN.write_text(main_text.replace(target, addition, 1), encoding="utf-8")
    print(json.dumps({"markdown": str(MARKDOWN), "tex": str(TEX), "main_tex": str(MAIN)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
