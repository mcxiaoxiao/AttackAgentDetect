(() => {
  const root = document.getElementById('corpusBrowser');
  if (!root) return;
  const source = document.getElementById('corpusSource');
  const benchmark = document.getElementById('corpusBenchmark');
  const mode = document.getElementById('corpusMode');
  const track = document.getElementById('corpusTrack');
  const search = document.getElementById('corpusSearch');
  const matched = document.getElementById('corpusMatched');
  const body = document.getElementById('corpusRows');
  const status = document.getElementById('corpusStatus');
  const previous = document.getElementById('corpusPrevious');
  const next = document.getElementById('corpusNext');
  let records = [], page = 0;
  const size = 12;
  const english = () => document.documentElement.dataset.language === 'en';
  const text = (tag, value) => { const e = document.createElement(tag); e.textContent = value; return e; };
  function populate(select, values) {
    for (const value of [...new Set(values)].sort()) {
      const option = text('option', value); option.value = value; select.append(option);
    }
  }
  function render(reset = false) {
    if (reset) page = 0;
    const term = search.value.trim().toLowerCase();
    const rows = records.filter(r =>
      (source.value === 'all' || r.source_dataset === source.value) &&
      (benchmark.value === 'all' || r.benchmark === benchmark.value) &&
      (mode.value === 'all' || r.mode === mode.value) &&
      (track.value === 'all' || r.track === track.value) &&
      (!matched.checked || r.has_human_agent_peer) &&
      (!term || [r.trace_id, r.task_id, r.model].some(x => String(x).toLowerCase().includes(term))));
    page = Math.min(page, Math.max(0, Math.ceil(rows.length / size) - 1));
    const fragment = document.createDocumentFragment();
    for (const r of rows.slice(page * size, (page + 1) * size)) {
      const tr = document.createElement('tr');
      const identity = document.createElement('td');
      const details = document.createElement('details');
      details.append(text('summary', r.trace_id));
      details.append(text('p', r.commands_preview.join(' → ')));
      details.append(text('p', `${r.observation_format} · ${r.normalization}`));
      identity.append(details); tr.append(identity);
      tr.append(text('td', `${r.source_dataset} / ${r.benchmark}`));
      tr.append(text('td', r.task_id));
      tr.append(text('td', `${r.mode === 'human' ? 'Human' : 'Agent'} · ${r.model}`));
      tr.append(text('td', String(r.command_count)));
      const cell = document.createElement('td');
      const a = text('a', english() ? 'Source record ↗' : '来源记录 ↗');
      const url = new URL(r.source_url);
      if (url.protocol === 'https:') a.href = url.href;
      cell.append(a); tr.append(cell); fragment.append(tr);
    }
    body.replaceChildren(fragment);
    status.textContent = english()
      ? `${rows.length.toLocaleString()} matching traces · page ${page + 1} / ${Math.max(1, Math.ceil(rows.length / size))}`
      : `${rows.length.toLocaleString()} 条匹配轨迹 · 第 ${page + 1} / ${Math.max(1, Math.ceil(rows.length / size))} 页`;
    previous.disabled = page === 0;
    next.disabled = (page + 1) * size >= rows.length;
  }
  for (const control of [source, benchmark, mode, track, matched]) control.addEventListener('change', () => render(true));
  search.addEventListener('input', () => render(true));
  previous.addEventListener('click', () => { page--; render(); });
  next.addEventListener('click', () => { page++; render(); });
  document.addEventListener('pagelanguagechange', () => { if (records.length) render(); });
  fetch(new URL('data/behavior-corpus-v1.json?revision=20261101-expansion-v1', document.baseURI))
    .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
    .then(data => {
      records = data.traces;
      populate(source, records.map(r => r.source_dataset));
      populate(benchmark, records.map(r => r.benchmark));
      render();
    })
    .catch(error => {
      status.textContent = english() ? 'Unable to load the trajectory catalog.' : '轨迹目录暂时无法加载。';
      console.error('Trajectory catalog load failed:', error);
    });
})();
