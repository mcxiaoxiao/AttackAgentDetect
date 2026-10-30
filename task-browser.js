/* A small static parser for the committed, machine-readable pair results. */
(() => {
  const root = document.querySelector('[data-task-browser]');
  if (!root) return;

  const $ = selector => root.querySelector(selector);
  const refs = {
    dataset: $('[data-browser-dataset]'),
    method: $('[data-browser-method]'),
    task: $('[data-browser-task]'),
    outcome: $('[data-browser-outcome]'),
    pairType: $('[data-browser-pair-type]'),
    search: $('[data-browser-search]'),
    status: $('[data-browser-status]'),
    metrics: $('[data-browser-metrics]'),
    rows: $('[data-browser-rows]'),
    page: $('[data-browser-page]'),
    previous: $('[data-browser-prev]'),
    next: $('[data-browser-next]'),
    detail: $('[data-browser-detail]'),
    intro: $('[data-browser-intro]'),
    jsonLink: $('[data-browser-json-link]'),
    scoresLink: $('[data-browser-scores-link]'),
    summaryLink: $('[data-browser-summary-link]'),
  };
  const PAGE_SIZE = 16;
  const state = {data: null, page: 0, pairId: null, loading: false, loadId: 0};

  const copy = {
    zh: {
      allTasks: '全部任务', allOutcomes: '全部结果', allTypes: '全部类型',
      type: {'H-A': 'Human—Agent', 'H-H': 'Human—Human', 'A-A': 'Agent—Agent'},
      outcome: {TP: '正确关联', FP: '误合并', FN: '漏关联', TN: '正确拒绝'},
      positive: '同题（检索代理标签）', negative: '异题', accepted: '接纳', rejected: '拒绝',
      metrics: {false_merge_rate: '误合并 FMR', false_split_rate: '漏关联 FSR', balanced_accuracy: '平衡准确率'},
      tasks: '任务组', pairs: '配对', records: '当前筛选',
      loaded: (pairs, tasks, traces) => `已载入 ${pairs} 个候选配对 · ${tasks} 道题 · ${traces} 条轨迹。每项指标按任务组汇总；表格展示逐对结果。`,
      failed: '结果 JSON 暂时无法读取。可直接打开下方 GitHub 产物。',
      found: (shown, total) => `显示 ${shown} / ${total} 条`,
      empty: '没有符合筛选条件的配对。',
      open: '查看', close: '收起详情', previous: '上一页', next: '下一页',
      page: (current, total) => `${current} / ${total} 页`,
      task: '题目', pair: '配对', query: '查询轨迹', candidate: '候选轨迹',
      label: '真值标签', prediction: '方法判断', method: '方法', score: '分数',
      threshold: '阈值（开发集冻结）', rank: '候选排名',
      scoreTitle: '各方法：分数与内层阈值',
      profile: '轨迹画像', dims: '维统计画像', tensor: '阶段 × 操作 × 通道张量',
      phases: ['早期', '中早', '中晚', '后期'],
      channels: ['命令占比', '类内词汇多样性', '相邻重复率'],
      features: {
        log_command_count: 'log 命令数', command_ttr: '词汇丰富度', command_entropy_bits: '命令熵',
        command_top1_share: '最高频命令占比', adjacent_command_repeat: '相邻命令重复',
        command_bigram_repeat: '命令二元组重复', command_bigram_entropy_bits: '二元组熵',
        action_coverage: '操作类别覆盖', action_entropy_bits: '操作熵',
        command_half_jsd: '前后半段命令差异', action_half_jsd: '前后半段操作差异',
      },
      actions: {build: '构建', discover: '发现', environment: '环境', execute: '执行', modify: '修改', network: '网络', read: '读取', search: '搜索'},
      sameTaskWarning: '“同题”只表示共享 CTF 任务，不代表同一人员、Agent 身份或组织。',
      micro: counts => `逐对计数 TP ${counts.TP} · FP ${counts.FP} · FN ${counts.FN} · TN ${counts.TN}`,
      noData: '画像数据缺失',
      noCommands: '没有可展示的命令特征。', commands: '提取的命令', artifacts: '共享产物词',
      loading: '正在读取所选结果…',
    },
    en: {
      allTasks: 'All tasks', allOutcomes: 'All outcomes', allTypes: 'All pair types',
      type: {'H-A': 'Human–Agent', 'H-H': 'Human–Human', 'A-A': 'Agent–Agent'},
      outcome: {TP: 'Correct link', FP: 'False merge', FN: 'Missed link', TN: 'Correct reject'},
      positive: 'Same task (retrieval proxy)', negative: 'Different tasks', accepted: 'Accepted', rejected: 'Rejected',
      metrics: {false_merge_rate: 'False merges · FMR', false_split_rate: 'Missed links · FSR', balanced_accuracy: 'Balanced accuracy'},
      tasks: 'task groups', pairs: 'pairs', records: 'filtered pairs',
      loaded: (pairs, tasks, traces) => `${pairs} candidate pairs · ${tasks} tasks · ${traces} traces loaded. Metrics are task-group summaries; the table shows pair-level outcomes.`,
      failed: 'The result JSON could not be loaded. Open the GitHub artifact below.',
      found: (shown, total) => `Showing ${shown} / ${total}`,
      empty: 'No pairs match these filters.',
      open: 'Inspect', close: 'Close detail', previous: 'Previous', next: 'Next',
      page: (current, total) => `Page ${current} / ${total}`,
      task: 'Task', pair: 'Pair', query: 'Query trace', candidate: 'Candidate trace',
      label: 'Reference label', prediction: 'Decision', method: 'Method', score: 'Score',
      threshold: 'Threshold (frozen on development)', rank: 'Candidate rank',
      scoreTitle: 'Method scores and inner-fold cutoffs',
      profile: 'Trace profile', dims: 'statistical dimensions', tensor: 'phase × action × channel tensor',
      phases: ['Early', 'Early-mid', 'Late-mid', 'Late'],
      channels: ['Command share', 'Within-family diversity', 'Adjacent repeat rate'],
      features: {
        log_command_count: 'Log command count', command_ttr: 'Vocabulary richness', command_entropy_bits: 'Command entropy',
        command_top1_share: 'Top-command share', adjacent_command_repeat: 'Adjacent repeats',
        command_bigram_repeat: 'Bigram repetition', command_bigram_entropy_bits: 'Bigram entropy',
        action_coverage: 'Action coverage', action_entropy_bits: 'Action entropy',
        command_half_jsd: 'Command half-shift', action_half_jsd: 'Action half-shift',
      },
      actions: {build: 'Build', discover: 'Discover', environment: 'Environment', execute: 'Execute', modify: 'Modify', network: 'Network', read: 'Read', search: 'Search'},
      sameTaskWarning: '“Same task” denotes a shared CTF challenge, not a shared person, Agent identity, or organization.',
      micro: counts => `Pair counts: TP ${counts.TP} · FP ${counts.FP} · FN ${counts.FN} · TN ${counts.TN}`,
      noData: 'Profile unavailable',
      noCommands: 'No command features are available.', commands: 'Extracted commands', artifacts: 'Artifact tokens',
      loading: 'Loading selected result…',
    },
  };

  const language = () => document.documentElement.lang.toLowerCase().startsWith('en') ? 'en' : 'zh';
  const words = () => copy[language()];
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined && text !== null) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  };
  const pct = value => value === null || value === undefined ? '—' : `${(value * 100).toFixed(1)}%`;
  const num = value => value === null || value === undefined ? '—' : Number(value).toFixed(3);
  const activeMethod = () => state.data.methods.find(item => item.key === refs.method.value);

  function option(select, value, label) {
    const item = node('option', label);
    item.value = value;
    select.append(item);
  }

  function fillControls() {
    const data = state.data;
    const text = words();
    const selectedMethod = refs.method.value;
    refs.method.replaceChildren();
    data.methods.forEach(method => option(
      refs.method,
      method.key,
      language() === 'en' ? method.name_en : method.name_zh,
    ));
    const initialMethod = data.inspection_default_method || data.selected_method;
    refs.method.value = data.methods.some(item => item.key === selectedMethod) ? selectedMethod : initialMethod;

    const selectedTask = refs.task.value;
    refs.task.replaceChildren();
    option(refs.task, '', text.allTasks);
    [...new Set(data.pairs.map(pair => pair.query_task_for_evaluation_only))].forEach(task => option(refs.task, task, task));
    if ([...refs.task.options].some(item => item.value === selectedTask)) refs.task.value = selectedTask;

    const selectedOutcome = refs.outcome.value;
    refs.outcome.replaceChildren();
    option(refs.outcome, '', text.allOutcomes);
    for (const code of ['TP', 'FP', 'FN', 'TN']) option(refs.outcome, code, text.outcome[code]);
    if ([...refs.outcome.options].some(item => item.value === selectedOutcome)) refs.outcome.value = selectedOutcome;

    const selectedType = refs.pairType.value;
    refs.pairType.replaceChildren();
    option(refs.pairType, '', text.allTypes);
    for (const code of ['H-A', 'H-H', 'A-A']) option(refs.pairType, code, text.type[code]);
    if ([...refs.pairType.options].some(item => item.value === selectedType)) refs.pairType.value = selectedType;
    refs.search.placeholder = language() === 'en' ? 'Task name or trace ID' : '输入题目名或轨迹 ID';
  }

  function filteredPairs() {
    const query = refs.search.value.trim().toLowerCase();
    const method = refs.method.value;
    return state.data.pairs.filter(pair => {
      const matchesTask = !refs.task.value || pair.query_task_for_evaluation_only === refs.task.value;
      const matchesType = !refs.pairType.value || pair.pair_type_for_evaluation_only === refs.pairType.value;
      const matchesOutcome = !refs.outcome.value || pair.predictions[method].outcome === refs.outcome.value;
      const haystack = [pair.pair_id, pair.benchmark, pair.query_task_for_evaluation_only,
        pair.candidate_task_for_evaluation_only, pair.query_trace_id, pair.candidate_trace_id].join(' ').toLowerCase();
      return matchesTask && matchesType && matchesOutcome && (!query || haystack.includes(query));
    });
  }

  function renderMetrics(filteredCount) {
    const text = words();
    const method = activeMethod();
    refs.metrics.replaceChildren();
    for (const key of ['false_merge_rate', 'false_split_rate', 'balanced_accuracy']) {
      const metric = method.metrics[key];
      const card = node('article', undefined, 'task-browser-metric');
      card.append(node('span', text.metrics[key]));
      card.append(node('strong', pct(metric?.estimate)));
      const ci = metric?.ci95;
      const detail = ci ? `${text.tasks}: ${metric.task_count} · 95% CI ${pct(ci[0])}–${pct(ci[1])}` : `${text.tasks}: ${metric?.task_count ?? '—'}`;
      card.append(node('small', detail));
      refs.metrics.append(card);
    }
    const stats = node('div', undefined, 'task-browser-counts');
    stats.append(node('span', text.micro(method.pair_counts)));
    stats.append(node('span', text.found(filteredCount, state.data.pairs.length)));
    refs.metrics.append(stats);
  }

  function outcomeBadge(code) {
    const badge = node('span', words().outcome[code], `task-browser-badge outcome-${code.toLowerCase()}`);
    return badge;
  }

  function renderTable() {
    if (!state.data) return;
    const data = state.data;
    const text = words();
    const method = activeMethod();
    const pairs = filteredPairs();
    const pageCount = Math.max(1, Math.ceil(pairs.length / PAGE_SIZE));
    state.page = Math.min(state.page, pageCount - 1);
    const visible = pairs.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE);
    refs.rows.replaceChildren();
    for (const pair of visible) {
      const prediction = pair.predictions[method.key];
      const row = node('tr');
      const taskCell = node('td');
      taskCell.append(node('b', pair.pair_id));
      const candidateTask = pair.candidate_task_for_evaluation_only || pair.query_task_for_evaluation_only;
      taskCell.append(node('small', pair.query_task_for_evaluation_only === candidateTask
        ? pair.query_task_for_evaluation_only
        : `${pair.query_task_for_evaluation_only} → ${candidateTask}`));
      const traceCell = node('td');
      traceCell.append(node('code', `${pair.query_trace_id} → ${pair.candidate_trace_id}`));
      traceCell.append(node('small', text.type[pair.pair_type_for_evaluation_only] || pair.pair_type_for_evaluation_only));
      const labelCell = node('td', pair.same_task_proxy_for_evaluation_only ? text.positive : text.negative);
      labelCell.className = pair.same_task_proxy_for_evaluation_only ? 'is-positive' : 'is-negative';
      const decisionCell = node('td');
      decisionCell.append(outcomeBadge(prediction.outcome));
      const scoreCell = node('td');
      const threshold = pair.thresholds?.[method.key] ?? method.threshold;
      scoreCell.append(node('span', `${num(pair.scores[method.key])} / ${num(threshold)} / #${prediction.rank}`, 'task-browser-score'));
      const detailCell = node('td');
      const open = node('button', text.open, 'task-browser-open');
      open.type = 'button';
      open.dataset.pairId = pair.pair_id;
      detailCell.append(open);
      row.append(taskCell, traceCell, labelCell, decisionCell, scoreCell, detailCell);
      refs.rows.append(row);
    }
    refs.status.textContent = text.loaded(data.dataset.pair_count, data.dataset.task_count, Object.keys(data.traces).length);
    refs.page.textContent = text.page(state.page + 1, pageCount);
    refs.previous.disabled = state.page === 0;
    refs.next.disabled = state.page >= pageCount - 1;
    renderMetrics(pairs.length);
    if (state.pairId && !pairs.some(pair => pair.pair_id === state.pairId)) state.pairId = null;
    if (state.pairId) renderDetail(state.pairId);
    else refs.detail.hidden = true;
    if (!visible.length) {
      const emptyRow = node('tr');
      const emptyCell = node('td', text.empty);
      emptyCell.colSpan = 6;
      emptyRow.append(emptyCell);
      refs.rows.append(emptyRow);
    }
  }

  function renderScoreChart(pair) {
    const text = words();
    const wrapper = node('div', undefined, 'task-browser-score-list');
    for (const method of state.data.methods) {
      const score = pair.scores[method.key];
      const threshold = pair.thresholds?.[method.key] ?? method.threshold;
      const prediction = pair.predictions[method.key];
      const row = node('div', undefined, `task-browser-score-row${method.key === refs.method.value ? ' is-selected' : ''}`);
      const label = node('div', undefined, 'task-browser-score-label');
      label.append(node('b', language() === 'en' ? method.name_en : method.name_zh));
      label.append(outcomeBadge(prediction.outcome));
      const bar = node('div', undefined, 'task-browser-meter');
      const fill = node('i');
      fill.style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
      const cutoff = node('b');
      cutoff.style.left = `${Math.max(0, Math.min(100, threshold * 100))}%`;
      bar.append(fill, cutoff);
      const values = node('small', `${text.score} ${num(score)} · ${text.threshold} ${num(threshold)} · ${text.rank} #${prediction.rank}`);
      row.append(label, bar, values);
      wrapper.append(row);
    }
    return wrapper;
  }

  function featureLabel(name) {
    return words().features[name] || name.replaceAll('_', ' ');
  }

  function renderTensor(trace) {
    const text = words();
    const container = node('div', undefined, 'task-browser-tensor-channels');
    if (!trace?.tensor) {
      container.append(node('p', text.noData));
      return container;
    }
    for (let channel = 0; channel < trace.tensor_channels.length; channel += 1) {
      const panel = node('div', undefined, 'task-browser-tensor-panel');
      panel.append(node('h6', text.channels[channel]));
      const grid = node('div', undefined, 'task-browser-tensor-grid');
      grid.style.setProperty('--phase-count', String(trace.tensor_shape[0]));
      grid.append(node('span', ''));
      for (let phase = 0; phase < trace.tensor_shape[0]; phase += 1) grid.append(node('b', text.phases[phase]));
      for (let action = 0; action < trace.tensor_action_labels.length; action += 1) {
        const actionName = trace.tensor_action_labels[action];
        grid.append(node('span', text.actions[actionName] || actionName, 'tensor-action-label'));
        for (let phase = 0; phase < trace.tensor_shape[0]; phase += 1) {
          const value = Number(trace.tensor[phase][action][channel]) || 0;
          const cell = node('span', `${Math.round(value * 100)}` , 'tensor-heat-cell');
          const alpha = 0.12 + Math.max(0, Math.min(1, value)) * 0.74;
          cell.style.backgroundColor = `rgba(25, 111, 168, ${alpha})`;
          cell.style.color = value > 0.62 ? '#fff' : '#17344a';
          cell.title = `${text.actions[actionName] || actionName} · ${text.phases[phase]} · ${text.channels[channel]}: ${pct(value)}`;
          cell.setAttribute('aria-label', cell.title);
          grid.append(cell);
        }
      }
      panel.append(grid);
      container.append(panel);
    }
    return container;
  }

  function renderTrace(traceId, label) {
    const text = words();
    const trace = state.data.traces[traceId];
    const card = node('article', undefined, 'task-browser-trace-card');
    const header = node('header');
    header.append(node('span', label, 'task-browser-trace-role'));
    header.append(node('code', traceId));
    card.append(header);
    if (!trace) {
      card.append(node('p', text.noData));
      return card;
    }
    const meta = node('p', `${trace.benchmark} · ${trace.task_for_evaluation_only} · ${trace.mode_for_evaluation_only} · ${trace.feature_dimensions} ${text.dims}`, 'task-browser-trace-meta');
    card.append(meta);
    const featureKeys = [
      'log_command_count', 'command_ttr', 'command_entropy_bits', 'command_top1_share',
      'adjacent_command_repeat', 'command_bigram_repeat', 'action_coverage', 'action_entropy_bits',
    ];
    const featureList = node('dl', undefined, 'task-browser-features');
    for (const key of featureKeys) {
      const entry = node('div');
      entry.append(node('dt', featureLabel(key)));
      entry.append(node('dd', num(trace.features[key])));
      featureList.append(entry);
    }
    card.append(node('h5', text.profile), featureList, node('h5', text.tensor), renderTensor(trace));
    const commandList = node('p', undefined, 'task-browser-command-list');
    commandList.append(node('b', `${text.commands}: `));
    commandList.append(node('code', trace.commands?.length ? trace.commands.join(' → ') : text.noCommands));
    card.append(commandList);
    if (trace.artifact_tokens?.length) {
      const artifactList = node('p', undefined, 'task-browser-command-list');
      artifactList.append(node('b', `${text.artifacts}: `));
      artifactList.append(node('code', trace.artifact_tokens.join(' · ')));
      card.append(artifactList);
    }
    return card;
  }

  function renderDetail(pairId) {
    const pair = state.data.pairs.find(item => item.pair_id === pairId);
    if (!pair) return;
    const text = words();
    const method = activeMethod();
    const currentPrediction = pair.predictions[method.key];
    const panel = node('div', undefined, 'task-browser-detail-inner');
    const head = node('header', undefined, 'task-browser-detail-head');
    const heading = node('div');
    heading.append(node('span', pair.pair_id, 'tensor-kicker'));
    const taskLabel = pair.query_task_for_evaluation_only === pair.candidate_task_for_evaluation_only
      ? pair.query_task_for_evaluation_only
      : `${pair.query_task_for_evaluation_only} → ${pair.candidate_task_for_evaluation_only}`;
    heading.append(node('h4', `${taskLabel} · ${pair.pair_type_for_evaluation_only}`));
    heading.append(node('p', `${text.query}: ${pair.query_trace_id} → ${text.candidate}: ${pair.candidate_trace_id}`));
    const verdict = node('div', undefined, 'task-browser-verdict');
    verdict.append(outcomeBadge(currentPrediction.outcome));
    verdict.append(node('small', `${pair.same_task_proxy_for_evaluation_only ? text.positive : text.negative} · ${currentPrediction.accepted ? text.accepted : text.rejected}`));
    head.append(heading, verdict);
    panel.append(head, node('p', text.sameTaskWarning, 'task-browser-warning'));

    const scores = node('section', undefined, 'task-browser-score-section');
    scores.append(node('h5', text.scoreTitle), renderScoreChart(pair));
    panel.append(scores);

    const profiles = node('div', undefined, 'task-browser-traces');
    profiles.append(renderTrace(pair.query_trace_id, text.query), renderTrace(pair.candidate_trace_id, text.candidate));
    panel.append(profiles);
    refs.detail.replaceChildren(panel);
    refs.detail.hidden = false;
  }

  async function load() {
    if (state.data || state.loading) return;
    const loadId = ++state.loadId;
    state.loading = true;
    refs.status.textContent = words().loading;
    try {
      const response = await fetch(refs.dataset.value, {headers: {Accept: 'application/json'}});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (loadId !== state.loadId) return;
      state.data = data;
      const lang = language();
      refs.intro.textContent = [
        state.data.dataset[lang === 'en' ? 'description_en' : 'description_zh']
          || state.data.dataset.threshold_boundary,
        state.data.dataset[lang === 'en' ? 'zero_miss_sensitivity_en' : 'zero_miss_sensitivity_zh'],
      ].filter(Boolean).join(' ');
      refs.jsonLink.href = state.data.sources.github_json || (refs.dataset.value.includes('v2')
        ? 'https://github.com/mcxiaoxiao/AttackAgentDetect/blob/main/site/data/task-association-v2.json'
        : 'https://github.com/mcxiaoxiao/AttackAgentDetect/blob/main/site/data/task-association-v1.json');
      refs.scoresLink.href = state.data.sources.github_pair_scores || refs.scoresLink.href;
      refs.summaryLink.href = state.data.sources.github_summary || refs.summaryLink.href;
      fillControls();
      renderTable();
    } catch (error) {
      if (loadId === state.loadId) {
        refs.status.textContent = words().failed;
        console.error('Task association result parser could not read its JSON artifact.', error);
      }
    } finally {
      if (loadId === state.loadId) state.loading = false;
    }
  }

  root.addEventListener('toggle', () => { if (root.open) load(); });
  refs.dataset.addEventListener('change', () => {
    state.loadId += 1;
    state.loading = false;
    state.data = null;
    state.page = 0;
    state.pairId = null;
    refs.detail.hidden = true;
    refs.rows.replaceChildren();
    refs.metrics.replaceChildren();
    load();
  });
  for (const control of [refs.method, refs.task, refs.outcome, refs.pairType]) {
    control.addEventListener('change', () => { state.page = 0; renderTable(); });
  }
  refs.search.addEventListener('input', () => { state.page = 0; renderTable(); });
  refs.previous.addEventListener('click', () => { state.page = Math.max(0, state.page - 1); renderTable(); });
  refs.next.addEventListener('click', () => { state.page += 1; renderTable(); });
  refs.rows.addEventListener('click', event => {
    const button = event.target.closest('[data-pair-id]');
    if (!button) return;
    if (state.pairId === button.dataset.pairId) {
      state.pairId = null;
      refs.detail.hidden = true;
    } else {
      state.pairId = button.dataset.pairId;
      renderDetail(state.pairId);
      refs.detail.scrollIntoView({behavior: 'smooth', block: 'nearest'});
    }
  });
  refs.method.addEventListener('change', () => { if (state.pairId) renderDetail(state.pairId); });
  document.addEventListener('pagelanguagechange', () => {
    if (state.data) {
      fillControls();
      const lang = language();
      refs.intro.textContent = [
        state.data.dataset[lang === 'en' ? 'description_en' : 'description_zh']
          || state.data.dataset.threshold_boundary,
        state.data.dataset[lang === 'en' ? 'zero_miss_sensitivity_en' : 'zero_miss_sensitivity_zh'],
      ].filter(Boolean).join(' ');
      renderTable();
    }
  });
})();
