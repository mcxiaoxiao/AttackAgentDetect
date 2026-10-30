/* Native stage-by-stage playback for the research page's 4 × 8 × 3 profile. */
(() => {
  const host = document.querySelector('[data-profile-tensor]');
  if (!host) return;
  const chart = host.querySelector('[data-tensor-chart]');
  const fallback = host.querySelector('.tensor-demo-fallback');
  const button = host.querySelector('[data-tensor-play]');
  const status = host.querySelector('[data-tensor-status]');
  const stageTrack = host.querySelector('[data-tensor-stage-track]');
  const svgNS = 'http://www.w3.org/2000/svg';
  const chineseActions = {
    build: '构建', discover: '发现', environment: '环境', execute: '执行',
    modify: '修改', network: '网络', read: '读取', search: '搜索',
  };
  const englishActions = {
    build: 'build', discover: 'discover', environment: 'environment', execute: 'execute',
    modify: 'modify', network: 'network', read: 'read', search: 'search',
  };
  const channelText = {
    zh: ['命令占比', '词汇多样度', '相邻重复率'],
    en: ['Command share', 'Vocabulary diversity', 'Adjacent repeat rate'],
  };
  let values = null;
  let frame = 0;
  let activePhase = -1;
  let playbackStarted = 0;
  let playing = false;

  function language() {
    return document.documentElement.lang === 'en' ? 'en' : 'zh';
  }

  function words() {
    return language() === 'en' ? {
      play: 'Play phase sequence', replay: 'Replay phase sequence',
      playing: phase => `Filling phase ${phase + 1} of 4 across the three behavior channels.`,
      complete: 'All four ordered phases are visible. Values are cohort means across 177 traces.',
      reduced: 'Reduced motion is enabled; the complete tensor is shown without playback.',
      loadError: 'The animated data could not be loaded. The static tensor figure is shown instead.',
      alt: 'Three aligned heatmaps show the cohort-mean tensor: four ordered phases by eight action families, one heatmap per behavior channel.',
      actionAxis: 'Action family', stageAxis: 'Ordered phase',
    } : {
      play: '播放阶段流转', replay: '再次播放阶段流转',
      playing: phase => `正在填入第 ${phase + 1}/4 阶段的三个行为通道。`,
      complete: '四个有序阶段已全部显示；数值为 177 条轨迹的队列均值。',
      reduced: '已启用“减少动态效果”，直接显示完整张量。',
      loadError: '动画数据未能加载，当前显示静态张量图。',
      alt: '三张对齐热图展示队列平均张量：四个有序阶段乘八类操作，每张对应一个行为通道。',
      actionAxis: '操作类别', stageAxis: '有序阶段',
    };
  }

  function element(tag, attrs = {}, parent = null, text = null) {
    const node = document.createElementNS(svgNS, tag);
    for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
    if (text !== null) node.textContent = text;
    if (parent) parent.append(node);
    return node;
  }

  function color(value) {
    const palette = ['#f2f6fa', '#d8e8f2', '#a8ccdf', '#6ba6c8', '#277eae'];
    return palette[Math.max(0, Math.min(4, Math.round(Math.max(0, Math.min(1, value)) * 4)))];
  }

  function render() {
    if (!values || !chart) return;
    const copy = words();
    chart.replaceChildren();
    const svg = element('svg', {
      class: 'tensor-svg', viewBox: '0 0 930 324', role: 'img',
      'aria-labelledby': 'profile-tensor-title profile-tensor-desc',
      focusable: 'false',
    }, chart);
    element('title', {id: 'profile-tensor-title'}, svg,
      language() === 'en' ? 'Multidimensional behavior profile tensor' : '多维行为画像张量');
    element('desc', {id: 'profile-tensor-desc'}, svg, copy.alt);

    const actionMap = language() === 'en' ? englishActions : chineseActions;
    const panelWidth = 310;
    const gridX = 101;
    const gridY = 67;
    const cellW = 42;
    const cellH = 23;
    const gapX = 4;
    const gapY = 4;
    const phaseLabels = language() === 'en' ? ['01', '02', '03', '04'] : ['一', '二', '三', '四'];

    for (let channel = 0; channel < 3; channel += 1) {
      const panelX = channel * panelWidth;
      const group = element('g', {transform: `translate(${panelX},0)`, 'data-channel-panel': channel}, svg);
      element('text', {x: 7, y: 21, class: 'tensor-panel-title'}, group, channelText[language()][channel]);
      element('text', {x: gridX, y: 44, class: 'tensor-axis-label'}, group, copy.stageAxis);
      element('text', {x: 7, y: 57, class: 'tensor-axis-label'}, group, copy.actionAxis);
      element('line', {x1: 7, y1: 29, x2: panelWidth - 12, y2: 29, class: 'tensor-plane-rule'}, group);

      for (let phase = 0; phase < 4; phase += 1) {
        element('text', {
          x: gridX + phase * (cellW + gapX) + cellW / 2,
          y: 61, class: 'tensor-phase-label', 'text-anchor': 'middle',
        }, group, phaseLabels[phase]);
      }

      for (let action = 0; action < 8; action += 1) {
        const y = gridY + action * (cellH + gapY);
        element('text', {x: 7, y: y + 15, class: 'tensor-axis-label'}, group,
          actionMap[values.action_labels[action]] || values.action_labels[action]);
        for (let phase = 0; phase < 4; phase += 1) {
          const value = Number(values.mean_tensor[phase][action][channel]) || 0;
          const x = gridX + phase * (cellW + gapX);
          const rect = element('rect', {
            class: 'tensor-cell', 'data-phase': phase, 'data-action': action,
            x, y, width: cellW, height: cellH, rx: 4,
            fill: color(value), opacity: '.76',
            style: `--pulse-delay:${action * 24}ms`,
          }, group);
          rect.setAttribute('aria-hidden', 'true');
          const label = `${Math.round(value * 100)}%`;
          element('text', {
            x: x + cellW / 2, y: y + 15,
            class: value >= .75 ? 'tensor-cell-text is-bright' : 'tensor-cell-text',
            'text-anchor': 'middle',
          }, group, label);
        }
      }
      element('rect', {
        class: 'tensor-column-focus', 'data-phase-focus': '',
        x: gridX - 3, y: 63, width: cellW + 6,
        height: (cellH + gapY) * 8 - gapY + 8, rx: 7,
      }, group);
      if (channel < 2) {
        element('line', {x1: panelWidth - 3, y1: 10, x2: panelWidth - 3, y2: 308,
          class: 'tensor-plane-rule'}, svg);
      }
    }
    chart.hidden = false;
    fallback?.setAttribute('hidden', '');
    setPhase(activePhase, false);
  }

  function setPhase(phase, animate = false) {
    if (!chart) return;
    activePhase = phase;
    for (const panel of chart.querySelectorAll('[data-channel-panel]')) {
      panel.querySelectorAll('[data-phase-focus]').forEach(item => {
        item.setAttribute('x', String(98 + phase * 46));
        item.classList.toggle('is-current', animate);
      });
      panel.querySelectorAll('.tensor-cell').forEach(cell => {
        const cellPhase = Number(cell.dataset.phase);
        cell.classList.toggle('is-current', animate && cellPhase === phase);
        cell.classList.toggle('is-visited', animate && cellPhase < phase);
        cell.setAttribute('opacity', animate && cellPhase > phase ? '.28' : '.88');
      });
    }
    stageTrack?.querySelectorAll('[data-stage-step]').forEach(step => {
      const stepPhase = Number(step.dataset.stageStep);
      step.classList.toggle('is-current', animate && stepPhase === phase);
      step.classList.toggle('is-visited', animate && stepPhase < phase);
    });
    if (stageTrack) stageTrack.style.setProperty('--phase-progress', animate ? `${phase * 25}%` : '0%');
    if (status) status.textContent = animate ? words().playing(phase) : words().complete;
    if (button && animate && playing) button.textContent = words().playing(phase);
  }

  function finish(reduced = false) {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    playing = false;
    button?.removeAttribute('aria-busy');
    chart?.classList.remove('is-playing');
    for (const cell of chart?.querySelectorAll('.tensor-cell') || []) {
      cell.classList.remove('is-current');
      cell.classList.add('is-visited');
      cell.setAttribute('opacity', '.88');
    }
    for (const focus of chart?.querySelectorAll('.tensor-column-focus') || []) focus.classList.remove('is-current');
    stageTrack?.querySelectorAll('[data-stage-step]').forEach(step => {
      step.classList.remove('is-current');
      step.classList.add('is-visited');
    });
    if (stageTrack) stageTrack.style.setProperty('--phase-progress', '75%');
    if (status) status.textContent = reduced ? words().reduced : words().complete;
    if (button) button.textContent = words().replay;
  }

  function resetPlayback() {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    playing = false;
    button?.removeAttribute('aria-busy');
    chart?.classList.remove('is-playing');
    for (const cell of chart?.querySelectorAll('.tensor-cell') || []) {
      cell.classList.remove('is-current', 'is-visited');
      cell.setAttribute('opacity', '.76');
    }
    for (const focus of chart?.querySelectorAll('.tensor-column-focus') || []) focus.classList.remove('is-current');
    stageTrack?.querySelectorAll('[data-stage-step]').forEach(step => step.classList.remove('is-current', 'is-visited'));
    if (stageTrack) stageTrack.style.setProperty('--phase-progress', '0%');
    activePhase = -1;
    if (status) status.textContent = language() === 'en'
      ? 'Click play to follow the phase axis.' : '点击播放，沿阶段轴查看张量切片。';
    if (button) button.textContent = words().play;
  }

  function play() {
    if (!values || !button || !chart) return;
    if (frame) cancelAnimationFrame(frame);
    chart.classList.add('is-playing');
    playing = true;
    button.setAttribute('aria-busy', 'true');
    button.textContent = words().playing(0);
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setPhase(3, true);
      finish(true);
      return;
    }
    playbackStarted = 0;
    function tick(now) {
      if (!playbackStarted) playbackStarted = now;
      const elapsed = now - playbackStarted;
      const phase = Math.min(3, Math.floor(elapsed / 850));
      if (phase !== activePhase || elapsed === 0) setPhase(phase, true);
      if (elapsed >= 4 * 850) {
        finish(false);
        return;
      }
      frame = requestAnimationFrame(tick);
    }
    frame = requestAnimationFrame(tick);
  }

  button?.addEventListener('click', play);
  const panel = host.closest('.demo-panel');
  if (panel) {
    new MutationObserver(() => {
      if (panel.hidden && playing) resetPlayback();
    }).observe(panel, {attributes: true, attributeFilter: ['hidden']});
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && playing) resetPlayback();
  });
  document.addEventListener('pagelanguagechange', () => {
    if (playing) finish(false);
    button.textContent = words().play;
    if (values) render();
    if (status) status.textContent = words().complete;
  });

  fetch('assets/profile-tensor-means-v1.json')
    .then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then(data => {
      values = data;
      render();
      if (status) status.textContent = language() === 'en'
        ? `${data.trace_count} traces; click play to move through four ordered stages.`
        : `${data.trace_count} 条轨迹；点击播放，依次查看四个有序阶段。`;
    })
    .catch(() => {
      if (button) button.disabled = true;
      if (status) status.textContent = words().loadError;
    });
})();
