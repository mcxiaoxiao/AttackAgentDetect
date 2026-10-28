(() => {
  const section = document.querySelector('.product-demo');
  if (!section) return;
  const clockAtOpen = Date.now();
  const pad = value => String(value).padStart(2, '0');
  const stamp = offset => {
    const date = new Date(clockAtOpen - offset * 1000);
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  };
  const past = hours => Math.random() * hours * 3600;
  const id = () => `sim_${Math.floor(Math.random() * 65536).toString(16).padStart(4, '0')}`;
  const humanLogs = () => {
    const base = 60 + past(71);
    return [
      `[${stamp(base)}] [INFO] [bash] tty=pts/0 command="find /srv -name config.yml" exit=0`,
      `[${stamp(base - 25)}] [INFO] [audit] type=EXECVE uid=1000 exe=/usr/bin/grep`,
      `[${stamp(base - 46)}] [NOTICE] [sudo] user=analyst command=/usr/bin/id`,
    ];
  };
  const scriptLogs = () => {
    const base = 60 + past(71);
    return [
      `[${stamp(base)}] [INFO] [cron] job=inventory pid=${1000 + Math.floor(Math.random() * 8000)} schedule="*/5 * * * *"`,
      `[${stamp(base - 25)}] [DEBUG] [python.worker] batch=07 targets=32 retries=0`,
      `[${stamp(base - 46)}] [INFO] [bash] command="find /srv -name config.yml" exit=0`,
    ];
  };
  const agentLogs = () => {
    const call = id();
    const base = 60 + past(71);
    return [
      `[${stamp(base)}] [DEBUG] [agent.plan] role=assistant step=03 action=inspect_target`,
      `[${stamp(base - 2)}] [INFO] [agent.tool_call] tool=bash call_id=${call} command="find /srv -name config.yml"`,
      `[${stamp(base - 5)}] [WARN] [agent.tool_result] call_id=${call} exit=1 retry=feedback_driven`,
    ];
  };
  const modeLogs = {human: humanLogs, script: scriptLogs, agent: agentLogs};
  const logOutput = section.querySelector('[data-demo-log]');
  const verdicts = [...section.querySelectorAll('[data-mode-case]')];
  const modeButtons = [...section.querySelectorAll('[data-demo-mode]')];
  function selectMode(name) {
    const makeLogs = modeLogs[name];
    if (!makeLogs) return;
    logOutput.textContent = makeLogs().join('\n');
    for (const button of modeButtons) button.setAttribute('aria-pressed', String(button.dataset.demoMode === name));
    for (const verdict of verdicts) verdict.hidden = verdict.dataset.modeCase !== name;
  }
  modeButtons.forEach(button => button.addEventListener('click', () => selectMode(button.dataset.demoMode)));
  selectMode('agent');

  const tabs = [...section.querySelectorAll('[role="tab"]')];
  const panels = tabs.map(tab => document.getElementById(tab.getAttribute('aria-controls')));
  function showTab(index, focus = false) {
    tabs.forEach((tab, number) => {
      tab.setAttribute('aria-selected', String(number === index));
      tab.tabIndex = number === index ? 0 : -1;
      panels[number].hidden = number !== index;
    });
    stopSequence();
    if (index !== 2 && typeof clearPlayback === 'function') clearPlayback();
    if (focus) tabs[index].focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => showTab(index));
    tab.addEventListener('keydown', event => {
      let next = index;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else return;
      event.preventDefault(); showTab(next, true);
    });
  });

  const sequenceButton = section.querySelector('[data-play-sequence]');
  const sequenceStatus = section.querySelector('[data-sequence-status]');
  const sequenceSteps = [...section.querySelectorAll('[data-signal-step]')];
  const progress = section.querySelector('.demo-time-rule');
  const sequenceCopy = {
    zh: ['研究流程待播放', '锁定 InterCode-CTF 版本…', '提取人类终端输入与 Agent bash 调用…', '映射命令名与动作类别；排除 Python 调用…', '按任务留出评估分类与同题检索…', '研究流程演示完成'],
    en: ['Research workflow ready', 'Pinning the InterCode-CTF snapshot…', 'Extracting human terminal input and Agent bash calls…', 'Mapping command names to action groups; excluding Python calls…', 'Evaluating classification and same-task retrieval on held-out tasks…', 'Workflow preview complete'],
  };
  let sequenceTimers = [];
  function stopSequence() {
    sequenceTimers.forEach(clearTimeout); sequenceTimers = [];
    sequenceSteps.forEach(step => step.classList.remove('is-current', 'is-done'));
    progress?.style.setProperty('--demo-progress', '0');
    sequenceButton?.removeAttribute('aria-busy');
    const language = document.documentElement.lang === 'en' ? 'en' : 'zh';
    if (sequenceStatus) sequenceStatus.textContent = sequenceCopy[language][0];
  }
  sequenceButton?.addEventListener('click', () => {
    stopSequence(); sequenceButton.setAttribute('aria-busy', 'true');
    const language = document.documentElement.lang === 'en' ? 'en' : 'zh';
    const words = sequenceCopy[language];
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      sequenceSteps.forEach(step => step.classList.add('is-done'));
      progress?.style.setProperty('--demo-progress', '1');
      sequenceStatus.textContent = words.at(-1);
      sequenceButton.removeAttribute('aria-busy');
      return;
    }
    sequenceSteps.forEach((step, index) => sequenceTimers.push(setTimeout(() => {
      sequenceSteps.forEach((item, n) => item.classList.toggle('is-current', n === index));
      if (index > 0) sequenceSteps[index - 1].classList.add('is-done');
      progress?.style.setProperty('--demo-progress', String((index + 1) / sequenceSteps.length));
      sequenceStatus.textContent = words[index + 1];
      if (index === sequenceSteps.length - 1) {
        step.classList.add('is-done'); step.classList.remove('is-current');
        sequenceStatus.textContent = words.at(-1); sequenceButton.removeAttribute('aria-busy');
        sequenceTimers = [];
      }
    }, index * 700)));
  });

  const comparisonVariants = {
    shared: {
      human: [
        {at: 0, text: 'tty · pwd'}, {at: 2.4, text: 'bash · find config'},
        {at: 5.8, text: 'grep · config.yml'}, {at: 10.8, text: 'cat · result'},
      ],
      script: [
        {at: 0, text: 'cron · job start'}, {at: 1.2, text: 'bash · pwd'},
        {at: 2.4, text: 'find · config'}, {at: 3.6, text: 'grep · config.yml'},
      ],
      agent: [
        {at: 0, text: 'tool_call · bash pwd'}, {at: .5, text: 'tool_result · exit=0'},
        {at: 1.6, text: 'tool_call · find'}, {at: 2, text: 'tool_result · exit=0'},
        {at: 3.2, text: 'tool_call · grep'}, {at: 3.7, text: 'tool_result · exit=1'},
        {at: 4.5, text: 'agent.retry · attempt=2'}, {at: 5, text: 'tool_call · cat'},
      ],
    },
    different: {
      human: [
        {at: 0, text: 'tty · whoami'}, {at: 2.8, text: 'bash · id'},
        {at: 6.3, text: 'find · /opt'}, {at: 10.9, text: 'cat · notes'},
      ],
      script: [
        {at: 0, text: 'cron · job start'}, {at: 1.2, text: 'python.worker · batch=07'},
        {at: 2.4, text: 'tar · extract'}, {at: 3.6, text: 'grep · token'},
      ],
      agent: [
        {at: 0, text: 'tool_call · curl'}, {at: .4, text: 'tool_result · 403'},
        {at: 1.5, text: 'tool_call · grep'}, {at: 2, text: 'tool_result · empty'},
        {at: 4, text: 'tool_call · ssh'}, {at: 4.6, text: 'tool_result · timeout'},
        {at: 6.2, text: 'agent.retry · attempt=2'}, {at: 6.8, text: 'tool_call · scp'},
      ],
    },
  };
  const compareDuration = 12;
  const compareSpeed = 5;
  const compareButtons = [...section.querySelectorAll('[data-compare-case]')];
  const compareLists = [...section.querySelectorAll('[data-compare-events]')];
  const comparePlay = section.querySelector('[data-compare-play]');
  const compareClock = section.querySelector('[data-compare-clock]');
  const compareResult = section.querySelector('[data-compare-result] span');
  let currentComparison = 'shared';
  let playbackFrame = 0;
  let playbackState = 'idle';
  function compareWords() {
    const english = document.documentElement.lang === 'en';
    return english ? {
      play: 'Play at 5×', replay: 'Replay at 5×', playing: 'Playing at 5×…',
      ready: currentComparison === 'shared' ? 'Shared-task illustration. Play the 12-second timeline at 5×.' : 'Different-task illustration. Play the 12-second timeline at 5×.',
      complete: '12 simulated seconds played in about 2.4 seconds.',
      reduced: 'Playback shown at its final state to respect reduced-motion settings.',
    } : {
      play: '播放 5× 回放', replay: '再次播放 5×', playing: '正在以 5× 回放…',
      ready: currentComparison === 'shared' ? '同题示意，点击播放 5× 回放 12 秒时间轴。' : '异题示意，点击播放 5× 回放 12 秒时间轴。',
      complete: '12 秒模拟过程已用约 2.4 秒回放。',
      reduced: '已显示完整时间线，并遵循减少动态效果设置。',
    };
  }
  function updateCompareCopy() {
    const words = compareWords();
    if (comparePlay) comparePlay.textContent = playbackState === 'playing' ? words.playing : playbackState === 'idle' ? words.play : words.replay;
    if (compareResult) {
      compareResult.textContent = playbackState === 'complete' ? words.complete
        : playbackState === 'reduced' ? words.reduced
        : playbackState === 'playing' ? words.playing
        : words.ready;
    }
  }
  function clearPlayback() {
    if (playbackFrame) cancelAnimationFrame(playbackFrame);
    playbackFrame = 0;
    playbackState = 'idle';
    comparePlay?.removeAttribute('aria-busy');
    for (const list of compareLists) {
      for (const item of list.children) item.classList.remove('is-current', 'is-done');
      const lane = list.closest('[data-mode-lane]');
      lane?.querySelector('[data-tempo-progress]')?.style.setProperty('--playback-progress', '0');
      lane?.querySelectorAll('.tempo-marker').forEach(marker => marker.classList.remove('is-fired', 'is-current'));
    }
    if (compareClock) compareClock.textContent = 't+00.0s / 12.0s';
    updateCompareCopy();
  }
  function renderComparison(name) {
    currentComparison = name;
    clearPlayback();
    compareButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.compareCase === name)));
    for (const list of compareLists) {
      const mode = list.dataset.compareEvents;
      const events = comparisonVariants[name][mode];
      const markers = list.closest('[data-mode-lane]')?.querySelector('[data-tempo-markers]');
      const markerNodes = [];
      list.replaceChildren(...events.map(event => {
        const item = document.createElement('li');
        item.dataset.at = String(event.at);
        const time = document.createElement('time');
        time.textContent = `t+${event.at.toFixed(1).padStart(4, '0')}s`;
        const code = document.createElement('code');
        code.textContent = event.text;
        item.append(time, code);
        const marker = document.createElement('i');
        marker.className = 'tempo-marker';
        marker.setAttribute('aria-hidden', 'true');
        marker.style.setProperty('--at', `${event.at / compareDuration * 100}%`);
        markerNodes.push(marker);
        return item;
      }));
      markers?.replaceChildren(...markerNodes);
    }
    updateCompareCopy();
  }
  function drawComparisonTime(seconds) {
    const progressRatio = seconds / compareDuration;
    for (const list of compareLists) {
      const lane = list.closest('[data-mode-lane]');
      lane?.querySelector('[data-tempo-progress]')?.style.setProperty('--playback-progress', String(progressRatio));
      const items = [...list.children];
      const markers = lane?.querySelectorAll('.tempo-marker') ?? [];
      items.forEach((item, index) => {
        const at = Number(item.dataset.at);
        const nextAt = Number(items[index + 1]?.dataset.at ?? Infinity);
        const fired = seconds >= at;
        const active = fired && seconds < nextAt;
        item.classList.toggle('is-current', active);
        item.classList.toggle('is-done', fired && !active);
        markers[index]?.classList.toggle('is-fired', fired);
        markers[index]?.classList.toggle('is-current', active);
      });
    }
    if (compareClock) compareClock.textContent = `t+${seconds.toFixed(1).padStart(4, '0')}s / 12.0s`;
  }
  function playComparison() {
    clearPlayback();
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      playbackState = 'reduced';
      drawComparisonTime(compareDuration);
      updateCompareCopy();
      return;
    }
    playbackState = 'playing';
    comparePlay?.setAttribute('aria-busy', 'true');
    updateCompareCopy();
    const startedAt = performance.now();
    const tick = now => {
      const elapsed = Math.min(compareDuration, (now - startedAt) / 1000 * compareSpeed);
      drawComparisonTime(elapsed);
      if (elapsed >= compareDuration) {
        playbackFrame = 0;
        playbackState = 'complete';
        comparePlay?.removeAttribute('aria-busy');
        updateCompareCopy();
      } else playbackFrame = requestAnimationFrame(tick);
    };
    playbackFrame = requestAnimationFrame(tick);
  }
  compareButtons.forEach(button => button.addEventListener('click', () => renderComparison(button.dataset.compareCase)));
  comparePlay?.addEventListener('click', playComparison);
  renderComparison('shared');
  document.addEventListener('pagelanguagechange', updateCompareCopy);

  const sourceExamples = {
    human: () => `[${stamp(past(72))}] [INFO] [bash] tty=pts/0 command="cat README.md" exit=0`,
    script: () => `[${stamp(past(72))}] [INFO] [cron] job=scan.sh pid=${1000 + Math.floor(Math.random() * 8000)} schedule="*/5 * * * *"`,
    agent: () => `[${stamp(past(72))}] [INFO] [agent.tool_call] call_id=${id()} tool=bash finish_reason=tool_calls`,
    audit: () => `[${stamp(past(72))}] [INFO] [audit] type=EXECVE uid=1000 ppid=${1000 + Math.floor(Math.random() * 8000)} exe=/usr/bin/find`,
  };
  const sourceOutput = section.querySelector('[data-source-log]');
  section.querySelectorAll('[data-source-preview]').forEach(button => button.addEventListener('click', () => {
    sourceOutput.textContent = sourceExamples[button.dataset.sourcePreview]();
  }));
})();
