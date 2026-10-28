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
    zh: ['模拟 Agent 工具调用序列', '正在检查目标…', '正在调用终端…', '读取命令结果…', '依据反馈重试…', '序列已完成'],
    en: ['A simulated agent tool-use sequence', 'Inspecting the target…', 'Calling the shell…', 'Reading the command result…', 'Retrying with feedback…', 'Sequence complete'],
  };
  let sequenceTimers = [];
  function stopSequence() {
    sequenceTimers.forEach(clearTimeout); sequenceTimers = [];
    sequenceSteps.forEach(step => step.classList.remove('is-current', 'is-done'));
    progress?.style.setProperty('--demo-progress', '0');
    sequenceButton?.removeAttribute('aria-busy');
  }
  sequenceButton?.addEventListener('click', () => {
    stopSequence(); sequenceButton.setAttribute('aria-busy', 'true');
    const language = document.documentElement.lang === 'en' ? 'en' : 'zh';
    const words = sequenceCopy[language];
    sequenceSteps.forEach((step, index) => sequenceTimers.push(setTimeout(() => {
      sequenceSteps.forEach((item, n) => item.classList.toggle('is-current', n === index));
      if (index > 0) sequenceSteps[index - 1].classList.add('is-done');
      progress?.style.setProperty('--demo-progress', String((index + 1) / sequenceSteps.length));
      sequenceStatus.textContent = words[index + 1];
      if (index === sequenceSteps.length - 1) {
        step.classList.add('is-done'); step.classList.remove('is-current');
        sequenceStatus.textContent = words.at(-1); sequenceButton.removeAttribute('aria-busy');
      }
    }, index * 700)));
  });

  const comparisonVariants = {
    shared: ['ls', 'cat', 'grep', 'cat'],
    different: ['curl', 'tar', 'ssh', 'scp'],
  };
  const comparisonButtons = [...section.querySelectorAll('[data-compare-case]')];
  const agentTrace = section.querySelector('[data-compare-commands]');
  let currentComparison = 'shared';
  comparisonButtons.forEach(button => button.addEventListener('click', () => {
    currentComparison = button.dataset.compareCase;
    comparisonButtons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    agentTrace.replaceChildren(...comparisonVariants[currentComparison].map(command => {
      const token = document.createElement('b'); token.dataset.command = command === 'ls' || command === 'find' ? 'inspect' : command === 'cat' ? 'read' : command === 'grep' ? 'search' : 'other'; token.textContent = command; return token;
    }));
    section.querySelector('[data-compare-result]').textContent = document.documentElement.lang === 'en' ? (currentComparison === 'shared' ? 'Ready to compare this simulated pair' : 'Comparison example changed. Run to review the overlap.') : (currentComparison === 'shared' ? '等待比较这组模拟轨迹' : '示例已切换，运行对比查看重合行为。');
    section.querySelectorAll('.demo-compare-traces b').forEach(token => token.classList.remove('is-match'));
  }));
  section.querySelector('[data-compare-run]')?.addEventListener('click', () => {
    const first = [...section.querySelectorAll('.demo-compare-traces > div:first-child b')];
    const second = [...section.querySelectorAll('.demo-compare-traces > div:last-child b')];
    const common = new Set(first.map(token => token.dataset.command));
    for (const token of [...first, ...second]) {
      const matched = token.dataset.command !== 'other' && common.has(token.dataset.command);
      token.classList.toggle('is-match', matched);
    }
    const count = new Set([...first, ...second].map(token => token.dataset.command).filter(value => value !== 'other' && common.has(value))).size;
    section.querySelector('[data-compare-result]').textContent = document.documentElement.lang === 'en'
      ? `${count} shared action groups in this simulated example. Review context before drawing conclusions.`
      : `这组模拟轨迹有 ${count} 类重合行为，仍需结合上下文判断。`;
  });

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
