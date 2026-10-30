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
  let hasAutoPlayed = false;

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
      alt: 'Three offset data planes form the cohort-mean tensor. The same phase slice is highlighted across all three behavior channels.',
      actionAxis: 'Action family', stageAxis: 'Ordered phase',
    } : {
      play: '播放阶段流转', replay: '再次播放阶段流转',
      playing: phase => `正在填入第 ${phase + 1}/4 阶段的三个行为通道。`,
      complete: '四个有序阶段已全部显示；数值为 177 条轨迹的队列均值。',
      reduced: '已启用“减少动态效果”，直接显示完整张量。',
      loadError: '动画数据未能加载，当前显示静态张量图。',
      alt: '三片错层数据板组成队列平均张量；同一阶段切片在三个行为通道中同步高亮。',
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
      class: 'tensor-svg tensor-svg-stack', viewBox: '0 0 930 520', role: 'img',
      'aria-labelledby': 'profile-tensor-title profile-tensor-desc',
      focusable: 'false',
    }, chart);
    element('title', {id: 'profile-tensor-title'}, svg,
      language() === 'en' ? 'Multidimensional behavior profile tensor' : '多维行为画像张量');
    element('desc', {id: 'profile-tensor-desc'}, svg, copy.alt);

    const actionMap = language() === 'en' ? englishActions : chineseActions;
    const gridX = 200;
    const cellW = 43;
    const cellH = 14;
    const gapX = 4;
    const gapY = 3;
    const stepX = cellW + gapX;
    const stepY = cellH + gapY;
    const gridW = cellW * 4 + gapX * 3;
    const gridH = cellH * 8 + gapY * 7;
    const frontY = 359;
    const channelDX = 42;
    const channelDY = 145;
    const phaseLabels = language() === 'en' ? ['01', '02', '03', '04'] : ['一', '二', '三', '四'];

    const positions = [0, 1, 2].map(channel => ({
      channel, x: gridX + channel * channelDX, y: frontY - channel * channelDY,
    }));
    const far = positions[2];
    const front = positions[0];
    for (const corner of [[0, 0], [gridW, 0], [0, gridH], [gridW, gridH]]) {
      element('line', {
        x1: front.x + corner[0], y1: front.y + corner[1],
        x2: far.x + corner[0], y2: far.y + corner[1], class: 'tensor-channel-edge',
      }, svg);
    }
    element('text', {x: 202, y: 30, class: 'tensor-stack-caption'}, svg,
      language() === 'en' ? 'PHASE  →' : '阶段  →');
    element('text', {x: 610, y: 40, class: 'tensor-stack-caption'}, svg,
      language() === 'en' ? 'PROFILE TENSOR · 4 × 8 × 3' : '画像张量 · 4 × 8 × 3');
    element('text', {x: 610, y: 62, class: 'tensor-axis-label'}, svg,
      language() === 'en' ? 'phase × action family × channel' : '阶段 × 操作类别 × 行为通道');

    for (const {channel, x, y} of [...positions].reverse()) {
      const group = element('g', {class: 'tensor-channel-plane', 'data-channel-panel': channel}, svg);
      element('rect', {x: x - 7, y: y - 8, width: gridW + 14, height: gridH + 16,
        rx: 10, class: `tensor-plane-surface tensor-plane-${channel}`}, group);
      element('text', {x, y: y - 21, class: 'tensor-plane-phase-axis'}, group,
        language() === 'en' ? 'ORDERED PHASE' : '有序阶段');
      for (let phase = 0; phase < 4; phase += 1) {
        element('text', {
          x: x + phase * stepX + cellW / 2, y: y - 8,
          class: 'tensor-phase-label', 'text-anchor': 'middle',
        }, group, phaseLabels[phase]);
      }
      for (let action = 0; action < 8; action += 1) {
        const cy = y + action * stepY;
        if (channel === 0) {
          element('text', {x: 102, y: cy + 11, class: 'tensor-axis-label', 'text-anchor': 'end'}, group,
            actionMap[values.action_labels[action]] || values.action_labels[action]);
        }
        for (let phase = 0; phase < 4; phase += 1) {
          const value = Number(values.mean_tensor[phase][action][channel]) || 0;
          const cx = x + phase * stepX;
          const rect = element('rect', {
            class: 'tensor-cell', 'data-phase': phase, 'data-action': action,
            x: cx, y: cy, width: cellW, height: cellH, rx: 3,
            fill: color(value), opacity: '.8',
            style: `--pulse-delay:${action * 24 + channel * 72}ms`,
          }, group);
          rect.setAttribute('aria-hidden', 'true');
          element('text', {
            x: cx + cellW / 2, y: cy + 10.5,
            class: value >= .75 ? 'tensor-cell-text is-bright' : 'tensor-cell-text',
            'text-anchor': 'middle',
          }, group, `${Math.round(value * 100)}%`);
        }
      }
      element('rect', {
        class: 'tensor-column-focus', 'data-phase-focus': '',
        x: x - 3, y: y - 4, width: cellW + 6, height: gridH + 8, rx: 6,
        'data-origin-x': x,
      }, group);
      element('text', {x: 508, y: y + 8, class: 'tensor-channel-label'}, svg,
        channelText[language()][channel]);
      element('line', {x1: x + gridW + 8, y1: y + 3, x2: 500, y2: y + 3,
        class: 'tensor-channel-leader'}, svg);
    }
    element('text', {x: 20, y: 503, class: 'tensor-axis-label'}, svg,
      language() === 'en' ? 'action family ↓' : '操作类别 ↓');
    const legend = element('g', {class: 'tensor-value-legend', transform: 'translate(612 116)'}, svg);
    element('text', {x: 0, y: 0, class: 'tensor-legend-title'}, legend,
      language() === 'en' ? 'CELL VALUE' : '格内数值');
    const gradient = element('linearGradient', {id: 'tensor-value-gradient', x1: '0%', x2: '100%'}, svg);
    ['#f2f6fa', '#d8e8f2', '#a8ccdf', '#6ba6c8', '#277eae'].forEach((stop, index) => {
      element('stop', {offset: `${index * 25}%`, 'stop-color': stop}, gradient);
    });
    element('rect', {x: 0, y: 13, width: 190, height: 12, rx: 6,
      fill: 'url(#tensor-value-gradient)'}, legend);
    element('text', {x: 0, y: 41, class: 'tensor-axis-label'}, legend, '0%');
    element('text', {x: 190, y: 41, class: 'tensor-axis-label', 'text-anchor': 'end'}, legend, '100%');
    element('text', {x: 0, y: 78, class: 'tensor-axis-label'}, legend,
      language() === 'en' ? 'One phase moves through all channels.' : '一个阶段切片同步穿过三个通道。');
    chart.hidden = false;
    fallback?.setAttribute('hidden', '');
    setPhase(activePhase, false);
  }

  function setPhase(phase, animate = false) {
    if (!chart) return;
    activePhase = phase;
    for (const panel of chart.querySelectorAll('[data-channel-panel]')) {
      panel.querySelectorAll('[data-phase-focus]').forEach(item => {
        item.setAttribute('x', String(Number(item.dataset.originX) + Math.max(phase, 0) * 47));
        item.classList.toggle('is-current', animate && phase >= 0);
      });
      panel.querySelectorAll('.tensor-cell').forEach(cell => {
        const cellPhase = Number(cell.dataset.phase);
        cell.classList.toggle('is-current', animate && cellPhase === phase);
        cell.classList.toggle('is-visited', animate && phase >= 0 && cellPhase < phase);
        cell.setAttribute('opacity', animate && phase >= 0 && cellPhase > phase ? '.28' : '.88');
      });
    }
    stageTrack?.querySelectorAll('[data-stage-step]').forEach(step => {
      const stepPhase = Number(step.dataset.stageStep);
      step.classList.toggle('is-current', animate && stepPhase === phase);
      step.classList.toggle('is-visited', animate && phase >= 0 && stepPhase < phase);
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

  function scheduleFirstPlayback() {
    if (hasAutoPlayed || matchMedia('(prefers-reduced-motion: reduce)').matches ||
        !('IntersectionObserver' in window)) return;
    const observer = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting && entry.intersectionRatio >= .25)) return;
      hasAutoPlayed = true;
      observer.disconnect();
      window.setTimeout(play, 180);
    }, {threshold: [.25]});
    observer.observe(host);
  }

  button?.addEventListener('click', () => {
    hasAutoPlayed = true;
    play();
  });
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
      scheduleFirstPlayback();
      if (status) status.textContent = language() === 'en'
        ? `${data.trace_count} traces; click play to move through four ordered stages.`
        : `${data.trace_count} 条轨迹；点击播放，依次查看四个有序阶段。`;
    })
    .catch(() => {
      if (button) button.disabled = true;
      if (status) status.textContent = words().loadError;
    });
})();
