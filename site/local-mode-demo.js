(() => {
  const demo = document.querySelector('[data-mode-trace-demo]');
  if (!demo) return;
  const play = demo.querySelector('[data-trace-play]');
  const reset = demo.querySelector('[data-trace-reset]');
  const status = demo.querySelector('[data-trace-status]');
  const lanes = [...demo.querySelectorAll('.mode-trace-lane')];
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let index = -1;
  let timer;
  const steps = lanes.map(lane => [...lane.querySelectorAll('li')]);
  const paint = () => steps.forEach(items => items.forEach((item, i) => item.classList.toggle('is-active', i === index)));
  const stop = () => { clearInterval(timer); timer = undefined; play.textContent = document.documentElement.lang.startsWith('zh') ? '▶ 播放' : '▶ Play'; };
  const tick = () => {
    index += 1;
    if (index >= Math.max(...steps.map(items => items.length))) { stop(); index = -1; paint(); return; }
    paint();
    const zh = document.documentElement.lang.startsWith('zh');
    status.textContent = zh ? `播放中 · 第 ${index + 1} 个序号位置` : `Playing · sequence position ${index + 1}`;
  };
  play.addEventListener('click', () => {
    if (timer) { stop(); return; }
    play.textContent = document.documentElement.lang.startsWith('zh') ? 'Ⅱ 暂停' : 'Ⅱ Pause';
    if (index < 0) tick();
    timer = setInterval(tick, reducedMotion ? 1100 : 720);
  });
  reset.addEventListener('click', () => {
    stop(); index = -1; paint();
    status.textContent = document.documentElement.lang.startsWith('zh') ? '就绪 · Human 7 条命令，Agent 4 次工具调用' : 'Ready · 7 Human commands, 4 Agent tool calls';
  });
})();
