#!/usr/bin/env python3
"""Insert matched experimental results without reformatting the existing page."""
import html
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "research/results/matched_fingerprint_v1"
PAGE = ROOT / "site/behavior-based-trace-association.html"
ASSETS = ROOT / "site/assets/behavioral-fingerprint-v2"
results = json.loads((OUT / "metrics.json").read_text())


def item(dataset, learner, variant, fraction, typ="ALL"):
    return next(x for x in results[dataset]["methods"]
                if x["learner"] == learner and x["variant"] == variant
                and x["fraction"] == fraction and x["pair_type"] == typ)


def p(v, signed=False):
    if v is None:
        return "—"
    return f"{100*v:+.2f}" if signed else f"{100*v:.2f}"


def ci(v, signed=False):
    return "[" + ", ".join(p(x, signed) for x in v) + "]" if v else "—"


def d(m):
    x = m["vs_evidence"]["ba"]
    return p(x["point"], True) + " " + ci(x["ci95"], True)


def td(value):
    return "<td>" + html.escape(value) + "</td>"


def bilingual(zh, en, tag="p"):
    return f'<{tag} data-en="{html.escape(en, quote=True)}">{html.escape(zh)}</{tag}>'


cth = {v: item("cth", "xgb", v, "average_5_fractions")
       for v in ("BS", "S", "B", "evidence")}
ait = {v: item("ait", "xgb", v, "-1") for v in ("BST", "ST", "BT", "BS", "evidence")}
svm = item("cth", "svm", "BS", "average_5_fractions")
tree = item("cth", "tree", "BS", "average_5_fractions")
t = ait["BS"]["vs_xgb_full"]["ba"]
t_gain, t_ci = -t["point"], [-t["ci95"][1], -t["ci95"][0]]

rows = []
for dataset, lookup, variants in (
    ("CTH", cth, (("BS", "E + B + S（完整）", "E + B + S (full)"),
                  ("S", "移除 B", "Remove B"), ("B", "移除 S", "Remove S"),
                  ("evidence", "仅 E", "E only"))),
    ("AIT", ait, (("BST", "E + B + S + T（完整）", "E + B + S + T (full)"),
                  ("ST", "移除 B", "Remove B"), ("BT", "移除 S", "Remove S"),
                  ("BS", "移除 T", "Remove T"), ("evidence", "仅 E", "E only")))):
    for i, (variant, zh, en) in enumerate(variants):
        m = lookup[variant]
        ba = m["metrics"]["ba"]
        delta = m["vs_xgb_full"]["ba"]
        dc = "参考" if i == 0 else p(delta["point"], True) + " " + ci(delta["ci95"], True)
        de = "Reference" if i == 0 else dc
        values = [p(ba["point"]) + " " + ci(ba["ci95"]),
                  p(m["metrics"]["fmr"]["point"]), p(m["metrics"]["fsr"]["point"])]
        rows.append("<tr>" + (f'<th scope="row">{dataset}</th>' if i == 0 else "<td></td>")
                    + f'<td data-en="{html.escape(en, quote=True)}">{html.escape(zh)}</td>'
                    + "".join(td(x) for x in values)
                    + f'<td data-en="{html.escape(de, quote=True)}">{html.escape(dc)}</td></tr>')

cthbase, cthfull = cth["evidence"], cth["BS"]
ha0 = item("cth", "xgb", "evidence", "0.5", "HA")
ha1 = item("cth", "xgb", "BS", "0.5", "HA")
paragraph_zh = (
    "在同一批 17 题查询与候选上，三种模型分别比较仅原证据与加入 B／S。"
    f"五档平均 BA 的配对变化：SVM {d(svm)}、决策树 {d(tree)}、"
    f"XGBoost {d(cthfull)} 个百分点；区间均跨零。"
    f"XGBoost 的误合并率从 {p(cthbase['metrics']['fmr']['point'])}% 升至 "
    f"{p(cthfull['metrics']['fmr']['point'])}%，漏关联率从 "
    f"{p(cthbase['metrics']['fsr']['point'])}% 降至 {p(cthfull['metrics']['fsr']['point'])}%。"
    "完整决策树在训练侧选定的门槛下拒绝了所有测试配对，误合并为 0%、漏关联为 100%。")
paragraph_en = (
    "On identical 17-task queries and galleries, each learner compares original evidence with B/S. "
    f"Mean BA changes across five role shares are {d(svm)} pp for SVM, {d(tree)} pp for the "
    f"decision tree and {d(cthfull)} pp for XGBoost; all intervals span zero. "
    f"XGBoost false merges rise from {p(cthbase['metrics']['fmr']['point'])}% to "
    f"{p(cthfull['metrics']['fmr']['point'])}% while missed links fall from "
    f"{p(cthbase['metrics']['fsr']['point'])}% to {p(cthfull['metrics']['fsr']['point'])}%. "
    "At its training-selected threshold, the full decision tree rejects every test pair: "
    "zero false merges and 100% missed links.")

pair_zh = (f"在 50% Agent 角色占比下，XGBoost 的人–Agent 配对漏关联率从 "
           f"{p(ha0['metrics']['fsr']['point'])}% 降至 {p(ha1['metrics']['fsr']['point'])}%，"
           f"误合并率从 {p(ha0['metrics']['fmr']['point'])}% 升至 {p(ha1['metrics']['fmr']['point'])}%。"
           "这是同一批日志的描述性分层，不能单独归因于执行方式。")
pair_en = (f"At 50% Agent role share, XGBoost Human–Agent missed links change from "
           f"{p(ha0['metrics']['fsr']['point'])}% to {p(ha1['metrics']['fsr']['point'])}%, "
           f"while false merges change from {p(ha0['metrics']['fmr']['point'])}% to "
           f"{p(ha1['metrics']['fmr']['point'])}%. These descriptive strata reuse the same "
           "traces and do not isolate execution mode from collection.")

time_zh = (f"AIT 中，完整 B／S／T 相比仅原证据的 BA 变化为 {d(ait['BST'])} 个百分点；"
           f"只比较加入 T 的增量为 {p(t_gain, True)} {ci(t_ci, True)} 个百分点，"
           "区间跨零。AIT 是排障任务，不是攻击活动；本轮的来源和采集流程仍可能与执行方式混杂。")
time_en = (f"On AIT, full B/S/T versus E changes BA by {d(ait['BST'])} pp; the isolated "
           f"increment of adding T to B/S is {p(t_gain, True)} {ci(t_ci, True)} pp and "
           "the interval spans zero. AIT is troubleshooting rather than attack activity, "
           "and collection workflows remain confounded with execution mode.")

fragment = """
<!-- matched-fingerprint-v1:start -->
<div class="association-result-block" id="matched-fingerprint-v1">
<h3 data-en="Matched classifiers across exact role shares">同一批配对：三种模型的指纹增量</h3>
""" + bilingual(paragraph_zh, paragraph_en) + bilingual(pair_zh, pair_en) + """
<p class="matched-scroll-hint" data-en="Dashed blue: original evidence; solid teal: evidence plus B/S. Swipe horizontally to view all three models.">蓝色虚线为原证据，绿色实线为加入 B／S；左右滑动可查看全部三个模型。</p>
<figure class="association-wide-figure scientific-evidence-figure">
<div class="scientific-figure-viewport" tabindex="0" role="region" aria-label="三种模型在五档配比下的关联对照" data-en-aria-label="Matched association for three classifiers across five role shares">
<a class="scientific-figure-source" href="assets/behavioral-fingerprint-v2/matched-mixture-v1.svg">
<img src="assets/behavioral-fingerprint-v2/matched-mixture-v1.png" alt="同一候选集下三种模型加入指纹前后的平衡准确率、误合并率和漏关联率" data-en-alt="Balanced accuracy, false merge and missed-link rates before and after fingerprints for three classifiers on identical galleries" loading="lazy"/></a>
</div>
<figcaption data-en="Bands are 95% task-cluster intervals. BA and missed-link axes start near 45%; the false-merge axis displays 0–30%. Each metric uses one shared scale across the three models.">阴影为 95% 任务簇区间。为显示变化，BA 和漏关联纵轴从约 45% 开始，误合并纵轴显示 0–30%；每项指标在三个模型中使用相同尺度。</figcaption>
<a class="scientific-figure-open" href="assets/behavioral-fingerprint-v2/matched-mixture-v1.svg" data-en="Open vector original">打开矢量原图</a>
</figure>
<h3 data-en="Joint CTH / AIT association ablation">CTH／AIT 关联消融：B、S 与 T</h3>
""" + bilingual(time_zh, time_en) + """
<p class="matched-scroll-hint" data-en="Swipe horizontally to view all metrics.">左右滑动可查看全部指标。</p>
<div class="association-table-scroll" tabindex="0" role="region" aria-label="CTH 和 AIT 关联消融表" data-en-aria-label="CTH and AIT association ablation table"><table class="association-results-table">
<thead><tr><th data-en="Dataset">数据集</th><th data-en="Configuration">配置</th><th>BA [95% CI]</th><th data-en="False merge %">误合并 %</th><th data-en="Missed links %">漏关联 %</th><th data-en="ΔBA vs. full [95% CI], pp">相对完整 ΔBA [95% CI]，百分点</th></tr></thead>
<tbody>
""" + "\n".join(rows) + """
</tbody></table></div>
<p class="association-transfer-note" data-en="E is command plus action-bigram evidence. CTH averages the five role shares; AIT uses fixed mixed-mode pairs. Compare rows within each dataset, not absolute rates across datasets. The same XGBoost family is fitted independently for each dataset.">E 为命令与动作二元组证据。CTH 汇总五档配比，AIT 使用固定混合配对；应在各数据集内部比较，不直接比较两个数据集的绝对分数。两组分别训练同一类 XGBoost 模型。</p>
""" + bilingual(
    "CTH 保留短日志：2.00% 的测试配对因指纹不可提取，回退到同折原证据模型。执行方式标签仅用于抽样和分层，不输入关联模型。全部结果属于探索性评估。",
    "Short CTH traces are retained: 2.00% of evaluated pairs fall back to the evidence model because a fingerprint cannot be extracted. Execution-mode labels are used only for sampling and stratification, not as model inputs. All comparisons are exploratory.") + """
</div>
<!-- matched-fingerprint-v1:end -->
"""
text = PAGE.read_text()
start, end = "<!-- matched-fingerprint-v1:start -->", "<!-- matched-fingerprint-v1:end -->"
if start in text:
    a, b = text.index(start), text.index(end) + len(end)
    text = text[:a] + fragment.strip() + text[b:]
else:
    anchor = '<div class="association-result-block" id="association-ba-v3">'
    assert text.count(anchor) == 1
    text = text.replace(anchor, fragment.strip() + "\n" + anchor, 1)
text = text.replace(
    '<h2 data-en="How fingerprints improve association decisions">指纹如何改善关联判断</h2>',
    '<h2 data-en="How fingerprints affect association decisions">指纹对关联判断的影响</h2>')
text = text.replace(
    '<h3 data-en="Conditioned fingerprints improve held-out association">条件化指纹提升留出关联表现</h3>',
    '<h3 data-en="Earlier multi-source comparison (different protocol)">既有多来源比较（不同实验协议）</h3>')
text = text.replace(
    '<h3 data-en="CTH fingerprint ablation">CTH 指纹消融</h3>',
    '<h3 data-en="Earlier 40-task CTH ablation">既有 40 题 CTH 指纹消融</h3>')
oldlink = '<a href="assets/behavioral-fingerprint-v2/human-agent-fingerprint-overnight.tex" data-en="Fingerprint experiment TeX source">行为指纹实验 TeX 源码</a>'
newlinks = ('<a href="assets/behavioral-fingerprint-v2/human-agent-fingerprint-draft.pdf" data-en="Research draft PDF">研究初稿 PDF</a>\n'
            '<a href="assets/behavioral-fingerprint-v2/human-agent-fingerprint-draft.tex" data-en="Research draft TeX source">研究初稿 TeX 源码</a>\n'
            '<a href="assets/behavioral-fingerprint-v2/human-agent-fingerprint-draft-source.zip" data-en="Research draft source bundle">研究初稿源码包 ZIP</a>\n')
if 'human-agent-fingerprint-draft.pdf' not in text:
    assert text.count(oldlink) == 1
    text = text.replace(oldlink, newlinks + oldlink, 1)
PAGE.write_text(text)

# Public export contains aggregate results only, with no pair-level or session-level records.
public = {dataset: {"methods": [x for x in results[dataset]["methods"] if x["pair_type"] == "ALL"],
                    "task_count": len(results[dataset]["tasks"]),
                    "bootstrap_draws": results[dataset]["bootstrap_draws"]}
          for dataset in ("cth", "ait")}
ASSETS.mkdir(parents=True, exist_ok=True)
(ASSETS / "matched-fingerprint-v1.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n")
print("Updated page with matched figure, ablation and draft links")
