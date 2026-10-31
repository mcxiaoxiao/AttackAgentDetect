(() => {
  const sourceFilter = document.getElementById('sourceFilter');
  const labelFilter = document.getElementById('labelFilter');
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
      score: '分数', threshold: '阈值', queryTrace: '查询轨迹', candidateTrace: '候选轨迹',
      queryTask: '查询任务', candidateTask: '候选任务', modePair: '执行方式',
      same: '同题（检索代理标签）', different: '异题', acceptedValue: '接纳', rejectedValue: '拒绝',
      filtered: (count, total) => `显示 ${count.toLocaleString('zh-CN')} / ${total.toLocaleString('zh-CN')} 条`,
      page: (current, total) => `${current.toLocaleString('zh-CN')} / ${total.toLocaleString('zh-CN')}`,
      exported: '已导出当前筛选结果。', exportError: '导出失败。',
    },
    en: {
      loading: 'Loading the published pair file…',
      loaded: count => `${count.toLocaleString('en-US')} pair records loaded.`,
      loadError: 'The pair file could not be read. From the repository, run python3 -m http.server --directory site 8000 and open http://localhost:8000/dataset.html, or download the CSV above.',
      noResults: 'No pairs match these filters.',
      pair: 'Pair', task: 'Reference label', accepted: 'Method decision', method: 'Selected method',
      score: 'Score', threshold: 'Threshold', queryTrace: 'Query trace', candidateTrace: 'Candidate trace',
      queryTask: 'Query task', candidateTask: 'Candidate task', modePair: 'Execution modes',
      same: 'Same task (retrieval proxy)', different: 'Different tasks', acceptedValue: 'Accepted', rejectedValue: 'Rejected',
      filtered: (count, total) => `Showing ${count.toLocaleString('en-US')} / ${total.toLocaleString('en-US')}`,
      page: (current, total) => `${current.toLocaleString('en-US')} / ${total.toLocaleString('en-US')}`,
      exported: 'Filtered records exported.', exportError: 'Export failed.',
    },
  };
  let rows = [];
  let columns = [];
  let visible = [];
  let current = 0;

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
    visible = rows.filter(row =>
      (benchmark === 'all' || row.benchmark === benchmark) &&
      (label === 'all' || (label === 'same' ? row.same_task_proxy_for_evaluation_only === '1' : row.same_task_proxy_for_evaluation_only === '0')),
    );
    current = Math.min(current, Math.max(visible.length - 1, 0));
    status.classList.remove('is-error');
    status.textContent = words().filtered(visible.length, rows.length);
    exportButton.disabled = visible.length === 0;
    previous.disabled = visible.length === 0 || current === 0;
    next.disabled = visible.length === 0 || current >= visible.length - 1;
    counter.textContent = visible.length ? words().page(current + 1, visible.length) : '—';
    renderPair();
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

  function renderPair() {
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
    const method = record.selected_method || '—';
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
    detail.replaceChildren(header, grid);
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
  previous.addEventListener('click', () => { current = Math.max(0, current - 1); update(); });
  next.addEventListener('click', () => { current = Math.min(visible.length - 1, current + 1); update(); });
  exportButton.addEventListener('click', exportCSV);
  document.addEventListener('pagelanguagechange', () => { if (rows.length) update(); });

  async function load() {
    status.textContent = words().loading;
    try {
      const response = await fetch(new URL('data/ctf-behavior-trace-pairs.csv', document.baseURI));
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const parsed = parseCSV(await response.text());
      columns = parsed.columns;
      rows = parsed.rows;
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
