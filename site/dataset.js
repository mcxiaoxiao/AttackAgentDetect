(() => {
  const sourceFilter = document.getElementById('sourceFilter');
  const labelFilter = document.getElementById('labelFilter');
  const traceSearch = document.getElementById('traceSearch');
  const status = document.getElementById('dataStatus');
  const detail = document.getElementById('pairDetail');
  const counter = document.getElementById('caseCounter');
  const previous = document.getElementById('prevBtn');
  const next = document.getElementById('nextBtn');
  const exportButton = document.getElementById('exportBtn');
  if (!sourceFilter || !labelFilter || !detail) return;

  const sourceNames = {
    'intercode-ctf': 'InterCode-CTF',
    cybench: 'CyBench',
    nyuctf: 'NYU CTF Bench',
  };
  const messages = {
    zh: {
      loading: '正在读取公开配对文件…',
      loaded: count => `已载入 ${count.toLocaleString('zh-CN')} 条配对记录。`,
      loadError: '配对文件暂时无法读取。请从仓库运行 python3 -m http.server --directory site 8000，再打开 http://localhost:8000/dataset.html；也可直接下载上方 CSV。',
      noResults: '当前筛选没有配对记录。',
      pair: '配对', task: '参考标签', accepted: '方法判断', method: '入选方法',
      methods: {command_tfidf: '命令 TF–IDF', artifact_containment_command: '命令 + 产物包含度'},
      score: '分数', threshold: '阈值', queryTrace: '查询轨迹', candidateTrace: '候选轨迹',
      queryTask: '查询任务', candidateTask: '候选任务', modePair: '执行方式',
      same: '同题（检索代理标签）', different: '异题', acceptedValue: '接纳', rejectedValue: '拒绝',
      filtered: (count, total) => `显示 ${count.toLocaleString('zh-CN')} / ${total.toLocaleString('zh-CN')} 条`,
      page: (current, total) => `${current.toLocaleString('zh-CN')} / ${total.toLocaleString('zh-CN')}`,
      exported: '已导出当前筛选结果。', exportError: '导出失败。',
      searchPlaceholder: '输入 trace ID 定位样例',
      traceMatch: id => `已定位到轨迹 ${id}，显示包含该轨迹的配对。`,
      flow: '展开两条轨迹的操作流程',
      flowNote: '按记录中的命令名称与动作类别还原；参数和命令输出未展示。动画同步展示步骤顺序，不表示两侧步骤一一对应；这是 CTF 解题流程，不代表真实攻击链或共同操作者。',
      traceSteps: '条命令步骤', noSteps: '该轨迹没有可展示的解析命令。',
      playFlow: '播放步骤动画', pauseFlow: '暂停动画', querySide: '查询轨迹', candidateSide: '候选轨迹',
      phases: {discover: '发现与定位', read: '读取与解析', search: '搜索', execute: '执行与分析', build: '构建', network: '网络交互', modify: '文件操作', environment: '环境检查'},
    },
    en: {
      loading: 'Loading the published pair file…',
      loaded: count => `${count.toLocaleString('en-US')} pair records loaded.`,
      loadError: 'The pair file could not be read. From the repository, run python3 -m http.server --directory site 8000 and open http://localhost:8000/dataset.html, or download the CSV above.',
      noResults: 'No pairs match these filters.',
      pair: 'Pair', task: 'Reference label', accepted: 'Method decision', method: 'Selected method',
      methods: {command_tfidf: 'Command TF–IDF', artifact_containment_command: 'Command + artifact containment'},
      score: 'Score', threshold: 'Threshold', queryTrace: 'Query trace', candidateTrace: 'Candidate trace',
      queryTask: 'Query task', candidateTask: 'Candidate task', modePair: 'Execution modes',
      same: 'Same task (retrieval proxy)', different: 'Different tasks', acceptedValue: 'Accepted', rejectedValue: 'Rejected',
      filtered: (count, total) => `Showing ${count.toLocaleString('en-US')} / ${total.toLocaleString('en-US')}`,
      page: (current, total) => `${current.toLocaleString('en-US')} / ${total.toLocaleString('en-US')}`,
      exported: 'Filtered records exported.', exportError: 'Export failed.',
      searchPlaceholder: 'Enter a trace ID to jump to its records',
      traceMatch: id => `Showing pairs that contain trace ${id}.`,
      flow: 'Expand both trace workflows',
      flowNote: 'Reconstructed from recorded command names and action categories; arguments and outputs are omitted. The animation synchronizes step order, not one-to-one semantic alignment. These are CTF task-solving steps, not a real-world attack chain or shared actor.',
      traceSteps: 'command steps', noSteps: 'No parsed command steps are available for this trace.',
      playFlow: 'Play step animation', pauseFlow: 'Pause animation', querySide: 'Query trace', candidateSide: 'Candidate trace',
      phases: {discover: 'Discover', read: 'Read / parse', search: 'Search', execute: 'Execute / analyze', build: 'Build', network: 'Network', modify: 'Modify', environment: 'Environment'},
    },
  };
  let rows = [];
  let columns = [];
  let visible = [];
  let current = 0;
  let traceDetails = {};
  let playbackTimer = null;
  let playbackStep = -1;

  const words = () => messages[document.documentElement.lang.toLowerCase().startsWith('en') ? 'en' : 'zh'];
  const text = (tag, value, className) => {
    const node = document.createElement(tag);
    node.textContent = value ?? '—';
    if (className) node.className = className;
    return node;
  };

  function parseCSV(source) {
    const matrix = [];
    let row = [];
    let field = '';
    let quoted = false;
    const input = source.replace(/^\uFEFF/, '');
    for (let i = 0; i < input.length; i++) {
      const char = input[i];
      if (quoted) {
        if (char === '"' && input[i + 1] === '"') { field += '"'; i++; }
        else if (char === '"') quoted = false;
        else field += char;
      } else if (char === '"' && field === '') quoted = true;
      else if (char === ',') { row.push(field); field = ''; }
      else if (char === '\n' || char === '\r') {
        if (char === '\r' && input[i + 1] === '\n') i++;
        row.push(field); field = '';
        if (row.some(value => value !== '')) matrix.push(row);
        row = [];
      } else field += char;
    }
    if (field !== '' || row.length) { row.push(field); matrix.push(row); }
    const [header, ...records] = matrix;
    if (!header?.length) throw new Error('CSV has no header');
    return {columns: header, rows: records.map(values => Object.fromEntries(header.map((name, index) => [name, values[index] ?? ''])))};
  }

  function update() {
    const benchmark = sourceFilter.value;
    const label = labelFilter.value;
    const query = (traceSearch?.value || '').trim().toLowerCase();
    visible = rows.filter(row =>
      (benchmark === 'all' || row.benchmark === benchmark) &&
      (label === 'all' || (label === 'same' ? row.same_task_proxy_for_evaluation_only === '1' : row.same_task_proxy_for_evaluation_only === '0')) &&
      (!query || [row.query_trace_id, row.candidate_trace_id].some(id => (id || '').toLowerCase().includes(query))),
    );
    current = Math.min(current, Math.max(visible.length - 1, 0));
    status.classList.remove('is-error');
    const exactMatch = query && visible.some(row => [row.query_trace_id, row.candidate_trace_id]
      .some(id => (id || '').toLowerCase() === query));
    status.textContent = exactMatch
      ? words().traceMatch(traceSearch.value.trim())
      : words().filtered(visible.length, rows.length);
    exportButton.disabled = visible.length === 0;
    previous.disabled = visible.length === 0 || current === 0;
    next.disabled = visible.length === 0 || current >= visible.length - 1;
    counter.textContent = visible.length ? words().page(current + 1, visible.length) : '—';
    renderPair();
    if (exactMatch) {
      const url = new URL(location.href);
      url.searchParams.set('trace', traceSearch.value.trim());
      url.hash = 'browser';
      history.replaceState(null, '', url);
      detail.scrollIntoView({behavior: 'smooth', block: 'start'});
    }
  }

  function addField(grid, label, value, code = false) {
    const wrapper = document.createElement('dl');
    wrapper.className = 'pair-card-field';
    wrapper.append(text('dt', label));
    const result = text('dd', value);
    if (code && value) {
      const codeNode = text('code', value);
      result.replaceChildren(codeNode);
    }
    wrapper.append(result);
    grid.append(wrapper);
  }

  function renderTraceFlow(traceId, sideLabel, lang) {
    const trace = traceDetails[traceId];
    const panel = document.createElement('section');
    panel.className = `trace-flow-panel is-${trace?.mode || 'unknown'}`;
    const heading = document.createElement('div');
    heading.className = 'trace-flow-heading';
    heading.append(text('h4', sideLabel), text('span', `${trace?.mode || '—'} · ${trace?.steps?.length || 0} ${lang.traceSteps}`));
    const list = document.createElement('ol');
    list.className = 'trace-flow-steps';
    for (const [index, step] of (trace?.steps || []).entries()) {
      const item = document.createElement('li');
      item.append(text('span', String(index + 1).padStart(2, '0'), 'trace-step-index'));
      const body = document.createElement('div');
      body.append(text('code', step.command), text('span', lang.phases[step.phase] || step.phase_label, 'trace-step-phase'));
      item.append(body);
      list.append(item);
    }
    if (!list.children.length) list.append(text('li', lang.noSteps, 'trace-flow-empty'));
    panel.append(heading, list);
    return panel;
  }

  function addPairWorkflows(record, lang) {
    const disclosure = document.createElement('details');
    disclosure.className = 'pair-workflows';
    const summary = document.createElement('summary');
    summary.textContent = lang.flow;
    const note = text('p', lang.flowNote, 'pair-workflow-note');
    const controls = document.createElement('div');
    controls.className = 'trace-flow-controls';
    const play = text('button', lang.playFlow, 'dataset-action dataset-action-secondary');
    play.type = 'button';
    controls.append(play);
    const tracks = document.createElement('div');
    tracks.className = 'trace-flow-compare';
    tracks.append(renderTraceFlow(record.query_trace_id, lang.querySide, lang), renderTraceFlow(record.candidate_trace_id, lang.candidateSide, lang));
    disclosure.append(summary, note, controls, tracks);
    play.addEventListener('click', () => {
      const querySteps = [...tracks.children[0].querySelectorAll('.trace-flow-steps li:not(.trace-flow-empty)')];
      const candidateSteps = [...tracks.children[1].querySelectorAll('.trace-flow-steps li:not(.trace-flow-empty)')];
      const stepCount = Math.max(querySteps.length, candidateSteps.length);
      if (playbackTimer) {
        clearInterval(playbackTimer); playbackTimer = null; play.textContent = lang.playFlow; return;
      }
      if (playbackStep >= stepCount - 1) playbackStep = -1;
      play.textContent = lang.pauseFlow;
      const advance = () => {
        querySteps.forEach(step => step.classList.remove('is-active'));
        candidateSteps.forEach(step => step.classList.remove('is-active'));
        playbackStep += 1;
        if (playbackStep >= stepCount) {
          clearInterval(playbackTimer); playbackTimer = null; playbackStep = -1; play.textContent = lang.playFlow; return;
        }
        const active = [querySteps[playbackStep], candidateSteps[playbackStep]].filter(Boolean);
        active.forEach(step => step.classList.add('is-active'));
        active[0]?.scrollIntoView({block: 'nearest', behavior: 'smooth'});
      };
      advance(); playbackTimer = setInterval(advance, 650);
    });
    disclosure.addEventListener('toggle', () => {
      if (!disclosure.open && playbackTimer) {
        clearInterval(playbackTimer); playbackTimer = null; playbackStep = -1; play.textContent = lang.playFlow;
      }
    });
    return disclosure;
  }

  function renderPair() {
    if (playbackTimer) clearInterval(playbackTimer);
    playbackTimer = null; playbackStep = -1;
    const lang = words();
    const record = visible[current];
    if (!record) {
      detail.replaceChildren(text('p', lang.noResults));
      return;
    }
    const same = record.same_task_proxy_for_evaluation_only === '1';
    const accepted = record.selected_accepted === '1';
    const header = document.createElement('div');
    header.className = 'pair-card-header';
    const title = text('h3', `${sourceNames[record.benchmark] || record.benchmark} · ${lang.pair} ${String(current + 1).padStart(4, '0')}`);
    const tags = document.createElement('div');
    tags.className = 'pair-tags';
    tags.append(text('span', same ? lang.same : lang.different, `pair-tag ${same ? 'is-positive' : 'is-negative'}`));
    tags.append(text('span', accepted ? lang.acceptedValue : lang.rejectedValue, 'pair-tag'));
    header.append(title, tags);

    const grid = document.createElement('div');
    grid.className = 'pair-card-grid';
    const method = lang.methods[record.selected_method] || record.selected_method || '—';
    const score = Number.parseFloat(record.selected_score);
    const threshold = Number.parseFloat(record.selected_threshold);
    const format = value => Number.isFinite(value) ? value.toFixed(4) : '—';
    addField(grid, lang.queryTrace, record.query_trace_id, true);
    addField(grid, lang.candidateTrace, record.candidate_trace_id, true);
    addField(grid, lang.queryTask, record.query_task);
    addField(grid, lang.candidateTask, record.candidate_task);
    addField(grid, lang.modePair, `${record.query_mode_for_evaluation_only || '—'} → ${record.candidate_mode_for_evaluation_only || '—'}`);
    addField(grid, lang.method, method);
    addField(grid, lang.score, format(score));
    addField(grid, lang.threshold, format(threshold));
    detail.replaceChildren(header, grid, addPairWorkflows(record, lang));
  }

  function exportCSV() {
    if (!visible.length || !columns.length) return;
    const quote = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
    const csv = [columns, ...visible.map(row => columns.map(name => row[name] ?? ''))]
      .map(record => record.map(quote).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob(['\uFEFF', csv], {type: 'text/csv;charset=utf-8'}));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'ctf-behavior-trace-pairs-filtered.csv';
    link.click();
    URL.revokeObjectURL(url);
    status.textContent = words().exported;
  }

  sourceFilter.addEventListener('change', () => { current = 0; update(); });
  labelFilter.addEventListener('change', () => { current = 0; update(); });
  traceSearch?.addEventListener('input', () => {
    current = 0;
    update();
    const query = traceSearch.value.trim();
    if (!query || !visible.some(row => [row.query_trace_id, row.candidate_trace_id]
      .some(id => (id || '').toLowerCase() === query.toLowerCase()))) {
      const url = new URL(location.href);
      url.searchParams.delete('trace');
      history.replaceState(null, '', url);
    }
  });
  previous.addEventListener('click', () => { current = Math.max(0, current - 1); update(); });
  next.addEventListener('click', () => { current = Math.min(visible.length - 1, current + 1); update(); });
  exportButton.addEventListener('click', exportCSV);
  document.addEventListener('pagelanguagechange', () => {
    if (traceSearch) traceSearch.placeholder = words().searchPlaceholder;
    if (rows.length) update();
  });

  async function load() {
    if (traceSearch) {
      traceSearch.placeholder = words().searchPlaceholder;
      traceSearch.value = new URL(location.href).searchParams.get('trace') || '';
    }
    status.textContent = words().loading;
    try {
      const [response, traceResponse] = await Promise.all([
        fetch(new URL('data/ctf-behavior-trace-pairs.csv?revision=association-20261101', document.baseURI)),
        fetch(new URL('data/trace-details.json?revision=trace-flow-v1', document.baseURI)),
      ]);
      if (!response.ok || !traceResponse.ok) throw new Error(`HTTP ${response.status}/${traceResponse.status}`);
      const [csvText, traceJson] = await Promise.all([response.text(), traceResponse.json()]);
      const parsed = parseCSV(csvText);
      columns = parsed.columns;
      rows = parsed.rows;
      traceDetails = traceJson;
      update();
    } catch (error) {
      console.error('Unable to load CTF behavior trace pairs:', error);
      status.classList.add('is-error');
      status.textContent = words().loadError;
      detail.replaceChildren(text('p', words().loadError));
    }
  }

  load();
})();
