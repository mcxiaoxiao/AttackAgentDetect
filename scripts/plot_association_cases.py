#!/usr/bin/env python3
"""Draw audited worked examples and render matching paper/page passages."""
from __future__ import annotations

import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/association_cases_v1.json'
FIG = ROOT / 'results/figures/association_cases_v1'
INK, BLUE, PALE, LINE, MUTED = '#17324D', '#287BC1', '#EDF5FC', '#B7CFE3', '#53697C'


def number(value):
    return f'{value:.4f}'


def preview(trace):
    names = trace['command_names']
    return ' → '.join(names[:8]) + (' …' if len(names) > 8 else '')


def draw(data, language):
    zh = language == 'zh'
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    cjk = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if not cjk.exists():
        cjk = Path('/Library/Fonts/Arial Unicode.ttf')
    if zh and not cjk.exists():
        raise FileNotFoundError('A CJK font is required for the Chinese figure.')
    font = FontProperties(fname=str(cjk)) if zh else FontProperties(family='Arial')
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.4))
    fig.subplots_adjust(left=.025, right=.98, top=.91, bottom=.09, hspace=.27)

    def text(ax, x, y, value, size=8, **kw):
        return ax.text(x, y, value, fontsize=size, color=kw.pop('color', INK),
                       fontproperties=font, transform=ax.transAxes, **kw)

    def box(ax, x, y, w, h, fill='white', edge=LINE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.007,rounding_size=0.015',
                                    facecolor=fill, edgecolor=edge, linewidth=.8, transform=ax.transAxes))

    for index, (ax, case) in enumerate(zip(axes, data['cases'][:2])):
        ax.set_axis_off()
        recovered = index == 0
        title = ('A  同题轨迹：漏关联得到纠正' if recovered else 'B  异题轨迹：产生错误关联') if zh else (
            'A  Same-challenge link recovered' if recovered else 'B  Different-challenge link introduced')
        text(ax, 0, 1.04, title, 10, fontweight='bold')
        for item, y, label in [(case['query'], .53, 'Human'), (case['candidate'], .10, 'Agent')]:
            box(ax, .005, y, .51, .33, fill='#F8FBFE')
            task = item['task_for_evaluation_only'].split('/', 1)[1]
            text(ax, .02, y+.25, f'{label} · {task}', 8.1, fontweight='bold')
            text(ax, .02, y+.14, preview(item), 7.1)
            desc = (f"{item['command_count']} 条命令 · 识别器 Agent 分数 {number(item['predicted_agent_score_raw'])}" if zh else
                    f"{item['command_count']} commands · detector Agent score {number(item['predicted_agent_score_raw'])}")
            text(ax, .02, y+.04, desc, 7.2, color=MUTED)
        ax.annotate('', xy=(.585, .49), xytext=(.53, .49), xycoords='axes fraction',
                    arrowprops={'arrowstyle': '-|>', 'color': BLUE, 'lw': 1})
        text(ax, .554, .61, '评分' if zh else 'Score', 7, ha='center', color=MUTED)
        for method, y in [('behavior', .53), ('mode_aware', .10)]:
            d = case['decisions'][method]
            enhanced = method == 'mode_aware'
            box(ax, .60, y, .385, .33, fill=PALE if enhanced else 'white', edge=BLUE if enhanced else LINE)
            name = ('行为 + 执行方式分数' if enhanced else '仅行为特征') if zh else (
                'Behavior + mode scores' if enhanced else 'Behavior features only')
            text(ax, .62, y+.25, name, 8.2, fontweight='bold')
            symbol = '≥' if d['linked'] else '<'
            text(ax, .62, y+.13, f"{number(d['score'])} {symbol} {number(d['threshold'])}", 12, fontweight='bold')
            if zh:
                status = '判相关' if d['linked'] else '判不相关'
                effect = ('纠正漏关联' if enhanced else '漏关联') if recovered else ('错误关联' if enhanced else '正确拒绝')
            else:
                status = 'Link' if d['linked'] else 'No link'
                effect = ('recovered' if enhanced else 'missed link') if recovered else ('spurious link' if enhanced else 'correct rejection')
            text(ax, .62, y+.035, f'{status} · {effect}', 7.5)
    title = '执行方式分数如何改变同题检索判断' if zh else 'How execution-mode scores change same-challenge matching'
    fig.text(.025, .974, title, fontsize=11, fontproperties=font, fontweight='bold', color=INK, va='top')
    note = ('每个数值框：关联分数与训练侧选定的判断门槛。左侧仅展示命令名前 8 项；全部评分使用完整轨迹。\n'
            '公开 CTF 留出题目的两个决策案例；使用同题评测标签与固定模型。整体效果见任务留出实验。') if zh else (
            'Each comparison: association score vs training-selected threshold. Command previews show the first 8 names; scoring uses full traces.\n'
            'Two decisions from held-out public CTF tasks; same-challenge labels and fixed models. Aggregate effects are reported separately.')
    fig.text(.025, .027, note, fontsize=6.8, fontproperties=font, color=MUTED, linespacing=1.5)
    stem = Path(str(FIG) + ('_zh' if zh else ''))
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=360, facecolor='white')
    fig.savefig(stem.with_suffix('.svg'), facecolor='white')
    shutil.copy2(stem.with_suffix('.png'), ROOT / 'site/assets' / stem.with_suffix('.png').name)
    if not zh:
        shutil.copy2(stem.with_suffix('.pdf'), ROOT / 'paper/figures' / stem.with_suffix('.pdf').name)
    plt.close(fig)


def render(data):
    good, bad, persistent = data['cases']
    def comparison(case, method, latex=False):
        d = case['decisions'][method]
        symbol = r'\geq' if d['linked'] and latex else ('≥' if d['linked'] else '<')
        return f"{number(d['score'])} {symbol} {number(d['threshold'])}"
    tex = r'''% Generated by scripts/plot_association_cases.py from audited JSON.
\subsection{Audited examples of changed decisions}

In the held-out CyBench challenge \texttt{back\_to\_the\_past}, a Human trace with 48 extracted commands and an Agent trace with 47 commands share the task label but have command-cosine similarity 0.215. The behavior ranker rejects the pair ($GOOD_BEFORE$); the mode-aware ranker accepts it ($GOOD_AFTER$). The mode detector's raw Agent scores are GOOD_HUMAN and GOOD_AGENT, respectively. This pair illustrates how predicted mode information can change a borderline matching decision.

The same representation also introduces a cross-challenge link: a Human \texttt{back\_to\_the\_past} trace and an Agent \texttt{glacier-exchange} trace change from rejection ($BAD_BEFORE$) to acceptance ($BAD_AFTER$). Figure~\ref{fig:association-cases} locates both the value and the design challenge: execution-mode context can recover cross-mode matches, while distinguishing the underlying task remains essential to controlling spurious links. Case selection and full trace identifiers are documented in the appendix.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{association_cases_v1.pdf}
\caption{Audited Human--Agent pairs from held-out CTF tasks. Each numeric comparison shows an association score against its training-selected threshold. The common 20\% training false-merge target permits method-specific thresholds. Only command-name previews are displayed; scoring uses complete normalized trajectories. Source labels are shown for interpretation, while the ranker uses predicted mode scores. These examples explain a recovered same-challenge link and a new cross-challenge error.}
\label{fig:association-cases}
\end{figure}
'''
    for key, value in {'GOOD_BEFORE': comparison(good, 'behavior', True), 'GOOD_AFTER': comparison(good, 'mode_aware', True),
                       'GOOD_HUMAN': number(good['query']['predicted_agent_score_raw']), 'GOOD_AGENT': number(good['candidate']['predicted_agent_score_raw']),
                       'BAD_BEFORE': comparison(bad, 'behavior', True), 'BAD_AFTER': comparison(bad, 'mode_aware', True)}.items():
        tex = tex.replace(key, value)
    (ROOT / 'paper/association-cases.tex').write_text(tex)
    body = ['<!-- association-cases:start -->', '<div class="study-table-wrap" id="case-study">',
            '<h3 data-en="Audited cases: how the decision changes">可核查案例：关联判断如何改变</h3>',
            '<p data-en="On the held-out CyBench challenge back_to_the_past, Human and Agent records have 48 and 47 extracted commands. The behavior ranker rejects the pair; adding predicted mode scores changes it to a link. Another pair from different challenges changes to an erroneous link. These cases explain the observed trade-off.">在留出的 CyBench 题目 back_to_the_past 中，人类与 Agent 轨迹分别包含 48、47 条提取后的命令。行为排序器拒绝将两者关联，加入预测的执行方式分数后改为关联；另一个来自不同题目的配对则被新方法误连。下图展示这两种判断变化。</p>',
            '<figure class="diagnostic-figure"><img src="assets/association_cases_v1_zh.png" data-en-src="assets/association_cases_v1.png" alt="两组公开 CTF 留出轨迹的评分与门槛：同题漏关联被纠正，同时出现一例异题错误关联。" data-en-alt="Scores and frozen thresholds for a recovered same-challenge link and an introduced cross-challenge link."/><figcaption data-en="Raw detector scores and method-specific association thresholds are shown explicitly. Command previews omit arguments and outputs. Same challenge is the evaluation label; it does not denote one intrusion. The diagram explains individual decisions, not a general causal effect.">图中列出识别器原始分数、关联分数与各方法的判断门槛。命令预览不含参数或输出。同题是本轮评测标签，不表示同一次入侵。图解释个例决策，不代表普遍的因果效果。</figcaption></figure>',
            '<table class="study-results"><thead><tr><th data-en="Case">案例</th><th data-en="Task relation">题目关系</th><th data-en="Behavior score / threshold">行为分数 / 门槛</th><th data-en="Mode-aware score / threshold">增强分数 / 门槛</th></tr></thead><tbody>']
    labels = [('纠正漏关联', 'Recovered link', '同题', 'Same challenge'), ('新增错误关联', 'Introduced error', '异题', 'Different challenges'), ('仍未关联', 'Persistent miss', '同题', 'Same challenge')]
    for c, (zh, en, rel, en_rel) in zip(data['cases'], labels):
        body.append(f'<tr><td data-en="{en}">{zh}</td><td data-en="{en_rel}">{rel}</td><td>{html.escape(comparison(c,"behavior"))}</td><td>{html.escape(comparison(c,"mode_aware"))}</td></tr>')
    body.extend(['</tbody></table>',
                 '<p class="caption" data-en="Post-hoc worked examples: the lexicographically first Human-query/Agent-candidate pair in each decision-change category. The full catalog, trace IDs, frozen thresholds, and independent score-refit checks are available below. Case counts use unweighted directed pairs and must not be substituted for task-macro error rates.">事后案例分析：每类取按基准、题目、轨迹 ID 字典序排列的第一组 Human 查询—Agent 候选配对。下方提供完整目录、轨迹 ID、固定门槛及评分复算检查。个例不是随机代表样本，配对数量也不等于按题目平均的错误率。</p>',
                 '<p class="result-links"><a href="https://github.com/mcxiaoxiao/AttackAgentDetect/blob/main/results/association_cases_v1.json" data-en="Audited case data">案例与复算记录</a><a href="https://github.com/mcxiaoxiao/AttackAgentDetect/blob/main/results/association_case_catalog_v1.csv" data-en="Full pair catalog">全部配对目录</a><a href="https://github.com/mcxiaoxiao/AttackAgentDetect/blob/main/scripts/extract_association_cases.py" data-en="Extraction and verification">提取与核验代码</a><a href="https://github.com/mcxiaoxiao/AttackAgentDetect/blob/main/scripts/plot_association_cases.py" data-en="Figure source">绘图代码</a></p>',
                 '</div>', '<!-- association-cases:end -->'])
    path = ROOT / 'site/execution-mode-behavior-correlation.html'
    page = path.read_text()
    block = '\n'.join(body) + '\n'
    start, end = '<!-- association-cases:start -->', '<!-- association-cases:end -->'
    if start in page:
        a, b = page.index(start), page.index(end) + len(end)
        page = page[:a] + block.rstrip() + page[b:]
    else:
        anchor = '<div class="metric-explain">'
        page = page.replace(anchor, block + anchor, 1)
    path.write_text(page)


def main():
    data = json.loads(SOURCE.read_text())
    assert len(data['cases']) == 3 and all(a['max_absolute_score_error'] < 1e-10 for a in data['refit_audit'])
    for language in ('en', 'zh'):
        draw(data, language)
    render(data)
    print('Rendered audited cases to English/Chinese vector and raster figures, paper fragment, and research page.')


if __name__ == '__main__':
    main()
