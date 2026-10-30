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


def clean_svg(path):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')


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
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.1))
    fig.subplots_adjust(left=.035, right=.985, top=.97, bottom=.105, hspace=.23)

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
        text(ax, .003, .965, title, 9.5, fontweight='bold')
        for item, y, label in [(case['query'], .53, 'Human'), (case['candidate'], .10, 'Agent')]:
            box(ax, .012, y, .50, .31, fill='#F8FBFE')
            task = item['task_for_evaluation_only'].split('/', 1)[-1]
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
    note = ('数值框比较关联分数与训练侧选定的判断门槛；每条命令仅展示命令名，评分使用完整轨迹。\n'
            '这是两个阈值判定案例；各自查询的 Hit@1 均未改变。首位候选变化见下一图。') if zh else (
            'Each comparison: association score vs training-selected threshold. Command previews show the first 8 names; scoring uses full traces.\n'
            'These two held-out decisions leave each query Hit@1 unchanged. The top-ranked candidate change is shown separately below.')
    fig.text(.025, .027, note, fontsize=6.8, fontproperties=font, color=MUTED, linespacing=1.5)
    stem = Path(str(FIG) + ('_zh' if zh else ''))
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=360, facecolor='white')
    fig.savefig(stem.with_suffix('.svg'), facecolor='white')
    clean_svg(stem.with_suffix('.svg'))
    shutil.copy2(stem.with_suffix('.png'), ROOT / 'site/assets' / stem.with_suffix('.png').name)
    if not zh:
        shutil.copy2(stem.with_suffix('.pdf'), ROOT / 'paper/figures' / stem.with_suffix('.pdf').name)
    plt.close(fig)


def draw_ranking(data, language):
    zh = language == 'zh'
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    cjk = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if not cjk.exists():
        cjk = Path('/Library/Fonts/Arial Unicode.ttf')
    if zh and not cjk.exists():
        raise FileNotFoundError('A CJK font is required for the Chinese figure.')
    font = FontProperties(fname=str(cjk)) if zh else FontProperties(family='Arial')
    correct = next(c for c in data['cases'] if c['category'] == 'ranking_correct_agent')
    previous = next(c for c in data['cases'] if c['category'] == 'ranking_previous_top')
    fig, ax = plt.subplots(figsize=(7.4, 3.45))
    fig.subplots_adjust(left=.025, right=.985, top=.98, bottom=.105)
    ax.set_axis_off()

    def text(x, y, value, size=8, color=INK, **kw):
        return ax.text(x, y, value, fontsize=size, color=color, fontproperties=font,
                       transform=ax.transAxes, **kw)

    def box(x, y, w, h, fill='white', edge=LINE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.015',
                                    facecolor=fill, edgecolor=edge, linewidth=.9,
                                    transform=ax.transAxes))

    q = correct['query']
    task = q['task_for_evaluation_only'].split('/', 1)[-1]
    text(.025, .955, '同一 Human 查询在混合候选库中的首位结果' if zh else
         'One Human query in the mixed candidate gallery', 10, fontweight='bold')
    box(.025, .69, .70, .19, fill='#F8FBFE')
    text(.045, .805, f"Human · {task}", 8.5, fontweight='bold')
    text(.045, .735, f"{preview(q)}  ·  {q['command_count']} 条命令" if zh else
         f"{preview(q)}  ·  {q['command_count']} commands", 7.4, color=MUTED)
    box(.755, .69, .22, .19, fill=PALE, edge=BLUE)
    hit_before = correct['decisions']['behavior']['query_hit_at_1']
    hit_after = correct['decisions']['mode_aware']['query_hit_at_1']
    text(.77, .795, 'Hit@1', 8, color=MUTED, fontweight='bold')
    text(.77, .725, f'{hit_before:.0f}  →  {hit_after:.0f}', 13, color=BLUE, fontweight='bold')

    box(.025, .16, .46, .44, fill='white', edge=LINE)
    box(.515, .16, .46, .44, fill=PALE, edge=BLUE)
    text(.045, .535, '仅行为特征' if zh else 'Behavior features only', 9, fontweight='bold')
    text(.535, .535, '行为 + 预测执行方式分数' if zh else 'Behavior + predicted mode scores', 9, fontweight='bold')

    prev_rank = previous['decisions']['behavior']['candidate_rank_low']
    prev_score = previous['decisions']['behavior']['score']
    right_rank_before = correct['decisions']['behavior']['candidate_rank_low']
    right_score_before = correct['decisions']['behavior']['score']
    right_rank_after = correct['decisions']['mode_aware']['candidate_rank_low']
    right_score_after = correct['decisions']['mode_aware']['score']
    prev_rank_after = previous['decisions']['mode_aware']['candidate_rank_low']
    prev_score_after = previous['decisions']['mode_aware']['score']
    old_task = previous['candidate']['task_for_evaluation_only'].split('/', 1)[-1]
    new_task = correct['candidate']['task_for_evaluation_only'].split('/', 1)[-1]
    before_lines = [
        (f'#{prev_rank}  Human · {old_task} · {number(prev_score)}',
         '异题首位' if zh else 'cross-challenge top result', True),
        (f'#{right_rank_before}  Agent · {new_task} · {number(right_score_before)}',
         '同题候选' if zh else 'same-challenge candidate', False),
    ]
    after_lines = [
        (f'#{right_rank_after}  Agent · {new_task} · {number(right_score_after)}',
         '同题首位' if zh else 'same-challenge top result', True),
        (f'#{prev_rank_after}  Human · {old_task} · {number(prev_score_after)}',
         '异题候选' if zh else 'cross-challenge candidate', False),
    ]
    for x, lines in ((.045, before_lines), (.535, after_lines)):
        for i, (main, sub, highlight) in enumerate(lines):
            y = .425 - i * .15
            text(x, y, main, 7.3, color=BLUE if highlight else INK,
                 fontweight='bold' if highlight else 'normal')
            text(x + .018, y - .065, sub, 7.1, color=MUTED)
    note = ('展示固定 39 条候选中的两条；相关性标签按是否解答同一道题定义，不表示同一次攻击。') if zh else (
            'Two of 39 fixed candidates are shown. Relevance means the same challenge, not the same attack.')
    fig.text(.025, .035, note, fontsize=6.9, fontproperties=font, color=MUTED)
    stem = ROOT / 'results/figures/association_rank_case_v1'
    stem.parent.mkdir(parents=True, exist_ok=True)
    output = stem.with_name(stem.name + ('_zh' if zh else ''))
    fig.savefig(output.with_suffix('.pdf'), facecolor='white')
    fig.savefig(output.with_suffix('.png'), dpi=360, facecolor='white')
    fig.savefig(output.with_suffix('.svg'), facecolor='white')
    clean_svg(output.with_suffix('.svg'))
    shutil.copy2(output.with_suffix('.png'), ROOT / 'site/assets' / output.with_suffix('.png').name)
    if not zh:
        shutil.copy2(output.with_suffix('.pdf'), ROOT / 'paper/figures' / output.with_suffix('.pdf').name)
    plt.close(fig)


def render(data):
    good, bad, persistent = data['cases'][:3]
    right = next(c for c in data['cases'] if c['category'] == 'ranking_correct_agent')
    old = next(c for c in data['cases'] if c['category'] == 'ranking_previous_top')
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

One held-out Human query gives a direct view of Hit@1. Its gallery contains 39 traces. The behavior-only ranker places a different-challenge Human trace first (score RANK_OLD_BEFORE) and the same-challenge Agent at rank RANK_RIGHT_BEFORE. With predicted execution-mode scores, the Agent trace moves to rank RANK_RIGHT_AFTER and the previous top result moves to rank RANK_OLD_AFTER; query Hit@1 changes from 0 to 1. This selected example illustrates the retrieval metric; the aggregate estimate uses every held-out query.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{association_rank_case_v1.pdf}
\caption{A held-out Human query whose top-ranked candidate changes from a different-challenge Human trace to a same-challenge Agent trace after adding predicted execution-mode scores. Two candidates are shown from a fixed gallery of 39. Challenge relevance is an evaluation label, not an attack-activity label.}
\label{fig:association-ranking-case}
\end{figure}
'''
    for key, value in {'GOOD_BEFORE': comparison(good, 'behavior', True), 'GOOD_AFTER': comparison(good, 'mode_aware', True),
                       'GOOD_HUMAN': number(good['query']['predicted_agent_score_raw']), 'GOOD_AGENT': number(good['candidate']['predicted_agent_score_raw']),
                       'BAD_BEFORE': comparison(bad, 'behavior', True), 'BAD_AFTER': comparison(bad, 'mode_aware', True),
                       'RANK_OLD_BEFORE': number(old['decisions']['behavior']['score']),
                       'RANK_RIGHT_BEFORE': str(right['decisions']['behavior']['candidate_rank_low']),
                       'RANK_RIGHT_AFTER': str(right['decisions']['mode_aware']['candidate_rank_low']),
                       'RANK_OLD_AFTER': str(old['decisions']['mode_aware']['candidate_rank_low'])}.items():
        tex = tex.replace(key, value)
    (ROOT / 'paper/association-cases.tex').write_text(tex)
    body = ['<!-- association-cases:start -->', '<div class="study-table-wrap" id="case-study">',
            '<h3 data-en="Examples from held-out traces">留出轨迹中的实际例子</h3>',
            '<p data-en="In case A, behavior features miss a same-challenge Human–Agent pair; adding mode scores links it. In case B, the same change wrongly links traces from different challenges. The next figure shows how a top-ranked candidate changes.">案例 A 中，只看行为特征会漏掉同题 Human–Agent 配对；加入模式分数后，两条轨迹被关联。案例 B 中，同样的变化把不同题目的轨迹误连。下一图展示候选排名如何变化。</p>',
            '<figure class="diagnostic-figure"><img src="assets/association_cases_v1_zh.png" data-en-src="assets/association_cases_v1.png" alt="两组公开 CTF 留出轨迹的评分与门槛：同题漏关联被纠正，同时出现一例异题错误关联。" data-en-alt="Scores and frozen thresholds for a recovered same-challenge link and an introduced cross-challenge link."/><figcaption data-en="Raw detector scores and method-specific association thresholds are shown explicitly. Command previews omit arguments and outputs. The two panels show a recovered cross-mode match and a cross-challenge link introduced by the mode-aware score.">图中列出识别器原始分数、关联分数与各方法的判断门槛。命令预览不含参数或输出。两组案例分别展示跨模式同题配对恢复和模式分数引入的异题关联。</figcaption></figure>',
            '<p data-en="A separate held-out query shows what Hit@1 measures: among 39 candidates, the behavior ranker puts a different-challenge Human trace first. Adding predicted execution-mode scores moves a same-challenge Agent trace to rank 1, and this query\'s Hit@1 changes from 0 to 1. The aggregate +3.6-point estimate comes from all held-out queries, not this selected example.">另一条留出查询展示 Hit@1 的含义：在 39 条候选中，行为排序器把异题 Human 轨迹排在首位；加入预测的执行方式分数后，同题 Agent 轨迹升至第 1 位，该查询 Hit@1 从 0 变为 1。总体 +3.6 个百分点来自全部留出查询，不由这个个例单独推出。</p>',
            '<figure class="diagnostic-figure"><img src="assets/association_rank_case_v1_zh.png" data-en-src="assets/association_rank_case_v1.png" alt="一个留出 Human 查询的候选排名由异题 Human 转为同题 Agent，Hit@1 从 0 变为 1。" data-en-alt="A held-out Human query changes its top candidate from a different-challenge Human trace to a same-challenge Agent trace; Hit@1 changes from 0 to 1."/><figcaption data-en="Two candidates are shown from the fixed 39-trace gallery. Same-challenge relevance is an evaluation label and does not imply one shared intrusion.">图中展示固定 39 条候选中的两条。同题相关性只是评测标签，不表示属于同一次入侵。</figcaption></figure>',
            '<table class="study-results"><thead><tr><th data-en="Case">案例</th><th data-en="Task relation">题目关系</th><th data-en="Behavior score / threshold">行为分数 / 门槛</th><th data-en="Mode-aware score / threshold">增强分数 / 门槛</th></tr></thead><tbody>']
    labels = [('纠正漏关联', 'Recovered link', '同题', 'Same challenge'), ('新增错误关联', 'Introduced error', '异题', 'Different challenges'), ('仍未关联', 'Persistent miss', '同题', 'Same challenge')]
    for c, (zh, en, rel, en_rel) in zip(data['cases'][:3], labels):
        body.append(f'<tr><td data-en="{en}">{zh}</td><td data-en="{en_rel}">{rel}</td><td>{html.escape(comparison(c,"behavior"))}</td><td>{html.escape(comparison(c,"mode_aware"))}</td></tr>')
    body.extend(['</tbody></table>',
                 '<p class="caption" data-en="Threshold cases are the lexicographically first Human-query/Agent-candidate pair in each decision category. The ranking case is the first of two held-out Human queries that meet the selection criterion. Full trace IDs, thresholds, candidate ranks, and independent score-refit checks are available below.">阈值案例按每类中字典序第一组 Human 查询—Agent 候选配对选取；排名案例是两个符合筛选条件的留出 Human 查询中按序第一例。完整轨迹 ID、门槛、排名和评分复算记录见下方。</p>',
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
    assert len(data['cases']) == 5 and all(a['max_absolute_score_error'] < 1e-10 for a in data['refit_audit'])
    for language in ('en', 'zh'):
        draw(data, language)
        draw_ranking(data, language)
    render(data)
    print('Rendered audited cases to English/Chinese vector and raster figures, paper fragment, and research page.')


if __name__ == '__main__':
    main()
