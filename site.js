/* Preserve previously published section links after splitting the documentation. */
if (/\/(?:index\.html)?$/.test(location.pathname)) {
  const legacyPages = { '#question': 'research.html', '#results': 'research.html', '#related': 'research.html', '#source-reproduction': 'research.html' };
  const destination = legacyPages[location.hash];
  if (destination) location.replace(destination + location.search + location.hash);
}

/* Lightweight navigation state; all content remains usable without JavaScript. */
const links = [...document.querySelectorAll(".rail nav a")];
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        links.forEach((link) => link.classList.toggle("active", link.hash === "#" + entry.target.id));
      }
    }
  }, {rootMargin: "-15% 0px -72% 0px"});
  document.querySelectorAll("main section[id]").forEach((section) => observer.observe(section));


}

for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copy);
    const text = target?.textContent?.trim().replace(/^\$\s*/, "");
    if (!text) return;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const field = document.createElement("textarea");
        field.value = text;
        field.setAttribute("readonly", "");
        field.style.position = "fixed";
        field.style.opacity = "0";
        document.body.append(field);
        field.select();
        document.execCommand("copy");
        field.remove();
      }
      const original = button.textContent;
      const english = () => document.documentElement.lang === "en";
      const toast = document.getElementById("copy-toast");
      if (toast && button.dataset.copy !== "ai-brief-markdown") {
        toast.textContent = english()
          ? (button.dataset.copySuccessEn || "Copied")
          : (button.dataset.copySuccess || "已复制");
        toast.hidden = false;
        window.clearTimeout(toast.hideTimer);
        toast.hideTimer = window.setTimeout(() => { toast.hidden = true; }, 2400);
      }
      button.textContent = button.dataset.copy === "ai-brief-markdown"
        ? (english() ? button.dataset.copySuccessEn : button.dataset.copySuccess)
        : (english() ? "Copied" : "已复制");
      window.setTimeout(() => { button.textContent = original; }, 1400);
    } catch {
      const english = document.documentElement.lang === "en";
      const toast = document.getElementById("copy-toast");
      if (toast && button.dataset.copy !== "ai-brief-markdown") {
        toast.textContent = english ? "Copy failed — select the text manually" : "复制失败，请手动选择文本";
        toast.hidden = false;
        window.clearTimeout(toast.hideTimer);
        toast.hideTimer = window.setTimeout(() => { toast.hidden = true; }, 3000);
      }
      button.textContent = english ? "Select and copy" : "请手动复制";
      window.setTimeout(() => { button.textContent = button.dataset.en || (english ? "Copy" : "复制"); }, 1600);
    }
  });
}

/* Cobalt reference: https://www.usehallmark.com/themes/cobalt/.
   This field illustrates Human / Script / Model-agent execution labels; examples are schematic, not live verdicts. */
(() => {
  const hero = document.querySelector('.hero');
  const canvas = hero?.querySelector('.field');
  const context = canvas?.getContext('2d');
  if (!context) return;
  const css = getComputedStyle(document.documentElement);
  const token = name => css.getPropertyValue(name).trim();
  const colors = {
    paper: token('--paper'), ink: token('--ink'), muted: token('--muted'),
    blue: token('--cobalt'), deep: token('--cobalt-deep'),
    glyphs: [1, 2, 3, 4].map(n => token('--glyph-' + n)),
  };
  const mono = token('--mono');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const ramp = ' ·:-=+*#%@';
  const typed = [' ', '⠁', '⠃', '⠇', '⡇', '⣇', '⣧', '⣷', '⣿', '⣿'];
  const logCharacters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[]{}():/._= ,\"-';
  const glyphs = [...new Set([...ramp, ...typed, ...logCharacters, '┊', '┼', '┬', '┴'])];
  const glyphIndex = new Map(glyphs.map((glyph, index) => [glyph, index]));
  const tints = [...colors.glyphs, colors.blue, colors.deep];
  const tintAt = [0, 0, 0, 1, 1, 2, 3, 4, 4, 5];
  // Simulated sessions use the visitor's local clock at page open, never a fixed date.
  const openedAt = Date.now();
  const templates = [
    {channel: 'bash', level: 'INFO', body: 'tty=pts/0 pid={pid} cmd="pwd" exit=0', mode: '🥷 Human'},
    {channel: 'agent.tool_call', level: 'INFO', body: 'step=03 tool=bash call={id}', mode: '🤖 AI agent'},
    {channel: 'audit', level: 'INFO', body: 'type=EXECVE uid=1000 exe=/usr/bin/find', mode: '🥷 Human'},
    {channel: 'agent.tool_result', level: 'INFO', body: 'call={id} exit=0 elapsed_ms={ms}', mode: '🤖 AI agent'},
    {channel: 'agent.plan', level: 'DEBUG', body: 'role=assistant step=04 action=inspect_file', mode: '🤖 AI agent'},
    {channel: 'sshd', level: 'INFO', body: 'pid={pid} user=analyst session=pts/0', mode: '🥷 Human'},
    {channel: 'agent.retry', level: 'WARN', body: 'step=04 attempt=2 backoff_ms=500', mode: '🤖 AI agent'},
    {channel: 'sudo', level: 'NOTICE', body: 'user=analyst tty=pts/0 command=/usr/bin/id', mode: '🥷 Human'},
    {channel: 'agent.request', level: 'INFO', body: 'POST /tools/bash status=200 request={id}', mode: '🤖 AI agent'},
    {channel: 'bash', level: 'INFO', body: 'session=pts/0 cmd="cat README.md" exit=0', mode: '🥷 Human'},
    {channel: 'agent.completion', level: 'DEBUG', body: 'finish_reason=tool_calls tokens=184', mode: '🤖 AI agent'},
    {channel: 'cron', level: 'INFO', body: 'job=scan.sh pid={pid} schedule="*/5 * * * *" exit=0', mode: '⚙️ Script'},
    {channel: 'python.worker', level: 'DEBUG', body: 'batch=07 targets=32 retries=0 elapsed_ms={ms}', mode: '⚙️ Script'},
  ];
  function sampleLog(index) {
    const template = templates[index % templates.length];
    const date = new Date(openedAt - Math.floor(Math.random() * 72 * 60 * 60 * 1000));
    const pad = value => String(value).padStart(2, '0');
    const stamp = `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
    const id = Math.floor(Math.random() * 65536).toString(16).padStart(4, '0');
    const body = template.body.replace('{id}', 'sim_' + id).replace('{pid}', String(1000 + Math.floor(Math.random() * 8000))).replace('{ms}', String(30 + Math.floor(Math.random() * 1500)));
    return {...template, raw: `[${stamp}] [${template.level}] [${template.channel}] ${body}`};
  }
  const ambientLogs = Array.from({length: 24}, (_, index) => sampleLog(index));
  const cw = 8, ch = 16;
  let width, height, columns, rows, gate, laneY, origins, atlas, calm, quiet;
  let particles = [], time = 37, frameId = 0, previous = 0, nextId = 0;
  let visible = true, ready = false;
  const pointer = {x: -1000, y: -1000, strength: 0, target: 0};

  // Seeded gradient noise preserves the reference's broad, soft-edged flow shapes.
  const permutation = new Uint8Array(512);
  {
    const values = Array.from({length: 256}, (_, index) => index);
    let seed = 1042;
    for (let index = 255; index > 0; index--) {
      seed = seed * 16807 % 2147483647;
      const other = Math.floor(seed / 2147483647 * (index + 1));
      [values[index], values[other]] = [values[other], values[index]];
    }
    for (let index = 0; index < 512; index++) permutation[index] = values[index & 255];
  }
  const gradientX = [1, -1, 1, -1, 1, -1, 0, 0];
  const gradientY = [1, 1, -1, -1, 0, 0, 1, -1];
  function noise(x, y) {
    const ix = Math.floor(x), iy = Math.floor(y);
    const dx = x - ix, dy = y - iy, px = ix & 255, py = iy & 255;
    const u = dx * dx * dx * (dx * (dx * 6 - 15) + 10);
    const v = dy * dy * dy * (dy * (dy * 6 - 15) + 10);
    const aa = permutation[permutation[px] + py] & 7;
    const ab = permutation[permutation[px] + py + 1] & 7;
    const ba = permutation[permutation[px + 1] + py] & 7;
    const bb = permutation[permutation[px + 1] + py + 1] & 7;
    const top = (gradientX[aa] * dx + gradientY[aa] * dy) * (1 - u) + (gradientX[ba] * (dx - 1) + gradientY[ba] * dy) * u;
    const bottom = (gradientX[ab] * dx + gradientY[ab] * (dy - 1)) * (1 - u) + (gradientX[bb] * (dx - 1) + gradientY[bb] * (dy - 1)) * u;
    return top * (1 - v) + bottom * v;
  }
  function seed() {
    particles = [
      {lane: 0, x: gate - 43, example: 1},
      {lane: 1, x: gate + (columns - gate) * .24, example: 2, resolved: true},
      {lane: 2, x: width >= 900 ? gate - 55 : gate + (columns - gate) * .2, example: width >= 900 ? 0 : 11, resolved: width < 900},
      {lane: 3, x: gate + (columns - gate) * .22, example: 3, resolved: true},
      {lane: 4, x: gate + (columns - gate) * .4, example: 11, resolved: true},
    ].filter(p => p.lane < laneY.length).map(p => {
      const sample = sampleLog(p.example);
      const offset = Math.max(2, Math.floor((sample.raw.length - sample.mode.length) / 2));
      return {...p, x: p.x - (p.x >= gate ? offset : 0), sample, speed: 9 + p.lane * .7};
    });
  }
  function resize() {
    const rect = hero.getBoundingClientRect();
    width = Math.round(rect.width); height = Math.round(rect.height);
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = width * dpr; canvas.height = height * dpr;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    columns = Math.ceil(width / cw); rows = Math.ceil(height / ch);
    atlas = document.createElement('canvas');
    atlas.width = glyphs.length * cw * dpr; atlas.height = tints.length * ch * dpr;
    const pen = atlas.getContext('2d');
    pen.scale(dpr, dpr); pen.font = `400 13.333px ${mono}`;
    pen.textAlign = 'center'; pen.textBaseline = 'middle';
    tints.forEach((tint, row) => {
      pen.fillStyle = tint;
      glyphs.forEach((glyph, col) => pen.fillText(glyph, col * cw + cw / 2, row * ch + ch / 2 + 1));
    });
    const boxes = [];
    let textBottom = 0;
    for (const element of hero.querySelector('.hero-copy').children) {
      let rectangles = [element.getBoundingClientRect()];
      if (/^(H1|P)$/.test(element.tagName)) {
        const range = document.createRange(); range.selectNodeContents(element);
        rectangles = [...range.getClientRects()];
      }
      for (const box of rectangles) {
        if (!box.width) continue;
        boxes.push({left: box.left - rect.left - 20, right: box.right - rect.left + 28,
          top: box.top - rect.top - 12, bottom: box.bottom - rect.top + 12});
        textBottom = Math.max(textBottom, box.bottom - rect.top);
      }
    }
    calm = new Float32Array(columns * rows); quiet = new Float32Array(columns * rows);
    for (let row = 0; row < rows; row++) for (let col = 0; col < columns; col++) {
      let mask = 0, near = 0;
      for (const box of boxes) {
        const distance = Math.hypot(Math.max(box.left - col * cw, 0, col * cw - box.right),
          Math.max(box.top - row * ch, 0, row * ch - box.bottom));
        mask = Math.max(mask, Math.max(0, 1 - distance / 64));
        near = Math.max(near, Math.max(0, 1 - distance / 20));
      }
      calm[row * columns + col] = mask * mask * (3 - 2 * mask);
      quiet[row * columns + col] = near;
    }
    const wide = width >= 900;
    gate = Math.round(width * (wide ? .585 : .55) / cw);
    const count = wide ? 5 : 4;
    const top = wide ? height * .245 : textBottom + 90;
    const spacing = wide ? Math.min(80, height * .105) : 48;
    laneY = Array.from({length: count}, (_, i) => Math.round((top + spacing * i) / ch) * ch + ch / 2);
    origins = Array.from({length: count}, (_, i) => height * .1 + height * .8 * (i + .5) / count);
    if (!wide) origins = laneY.map((y, i) => y + Math.sin(i * 2) * 36);
    seed(); draw();
  }
  function band(lane, x) {
    let y, spread, amplitude;
    if (x >= gate * cw) {
      y = laneY[lane]; spread = 7;
      amplitude = .4 + .12 * Math.sin((x - time * 70) / 46 + lane);
    } else {
      let blend = Math.max(0, Math.min(1, (x - gate * cw * .3) / (gate * cw * .62)));
      blend = blend * blend * (3 - 2 * blend);
      const wave = noise((x - time * 26) / 260, lane * 3.7 + .5) * 120 * (1 - blend);
      y = origins[lane] + (laneY[lane] - origins[lane]) * blend + wave;
      spread = (1 - blend) * (26 + 20 * (.5 + noise((x - time * 40) / 180, lane * 5.1 + 9))) + blend * 9;
      amplitude = Math.min(1, .5 + .75 * (.5 + noise((x - time * 48) / 150, lane * 7.3 + 3))) * (1 - blend * .25) + blend * .25;
    }
    const distance = Math.hypot(x - pointer.x, y - pointer.y);
    if (distance < 200) {
      const force = (1 - distance / 200) ** 2 * pointer.strength;
      y += (y >= pointer.y ? 1 : -1) * force * 46;
      amplitude *= 1 + force * .7;
    }
    return {y, spread, amplitude};
  }
  function label(text, x, y, color = colors.ink) {
    context.font = `500 13.333px ${mono}`; context.textBaseline = 'middle';
    context.fillStyle = colors.paper; context.fillRect(x - 6, y - 9, text.length * cw + 12, ch + 2);
    context.fillStyle = color; context.fillText(text, x, y + 1);
  }
  function draw() {
    context.clearRect(0, 0, width, height);
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const bands = laneY.map((_, lane) => Array.from({length: columns}, (_, col) => band(lane, col * cw + cw / 2)));
    for (let row = 1; row < rows - 1; row++) for (let col = 1; col < columns - 1; col++) {
      const y = row * ch, mask = calm[row * columns + col];
      let glyph, tint, alpha = 1 - mask * .7;
      if (col === gate && y >= laneY[0] - 32 && y <= laneY.at(-1) + 32) {
        glyph = y < laneY[0] - 16 ? '┬' : y > laneY.at(-1) + 16 ? '┴' : laneY.includes(y + ch / 2) ? '┼' : '┊';
        tint = 3;
      } else {
        let density = 0;
        for (const lane of bands) {
          const b = lane[col];
          density = Math.max(density, b.amplitude * Math.exp(-((y + ch / 2 - b.y) ** 2) / (2 * b.spread ** 2)));
        }
        density *= 1 - mask * .93;
        if (density < .09) continue;
        const level = Math.min(9, 1 + Math.floor((density - .09) / .91 * 9));
        tint = tintAt[level];
        if (col > gate) {
          glyph = typed[level];
        } else {
          // The reference's broad clouds remain intact. Their denser cells contain
          // coherent log fragments, with pale punctuation noise along the edges.
          const record = ambientLogs[row % ambientLogs.length].raw;
          const drift = Math.floor((time - 37) * (2 + row % 3));
          const offset = (col + drift + Math.floor(row / 3) * 13) % (record.length + 12);
          glyph = level >= 5 && row % 9 === 1 ? record[offset] || ' ' : ramp[level];
        }
        if (col < 8) alpha *= (col + 1) / 9;
      }
      context.globalAlpha = alpha;
      context.drawImage(atlas, glyphIndex.get(glyph) * cw * dpr, tint * ch * dpr, cw * dpr, ch * dpr, col * cw, y, cw, ch);
    }
    context.globalAlpha = 1;
    for (const packet of particles) {
      const example = packet.sample;
      const col = Math.floor(packet.x);
      const sourceHead = Math.max(0, Math.min(gate - 1, col + example.raw.length - 1));
      const inputY = Math.round((band(packet.lane, sourceHead * cw).y - ch / 2) / ch) * ch + ch / 2;
      context.font = `500 13.333px ${mono}`; context.textBaseline = 'middle';
      const outputStart = Math.max(2, Math.floor((example.raw.length - example.mode.length) / 2));
      const modeVisible = col + outputStart >= gate;
      for (let i = 0; i < example.raw.length; i++) {
        const x = (col + i) * cw;
        if (packet.resolved || col + i >= gate) continue;
        const y = inputY;
        const row = Math.max(0, Math.min(rows - 1, Math.floor(y / ch)));
        const cell = Math.max(0, Math.min(columns - 1, col + i));
        context.globalAlpha = (1 - quiet[row * columns + cell]) * Math.min(1, Math.max(0, (col + example.raw.length) / example.raw.length)) * .72;
        context.fillStyle = colors.paper; context.fillRect(x, y - ch / 2, cw, ch);
        context.fillStyle = colors.muted; context.fillText(example.raw[i], x, y + 1);
      }
      if (modeVisible) {
        context.save();
        context.beginPath(); context.rect(gate * cw, 0, width - gate * cw, height); context.clip();
        context.globalAlpha = 1;
        const labelWidth = context.measureText(example.mode).width;
        // Keep a clear break between the classifier seam and the mode label; no leader bar.
        // On narrow canvases clamp the label so its final glyphs stay inside the viewport.
        const outputX = Math.min(
          Math.max((col + outputStart) * cw, gate * cw + 64),
          width - labelWidth - 18,
        );
        const labelY = laneY[packet.lane] - ch / 2;
        context.fillStyle = colors.paper;
        context.fillRect(gate * cw + 1, labelY, Math.max(0, outputX - gate * cw - 34), ch);
        context.fillRect(outputX - 34, labelY, labelWidth + 44, ch);
        context.fillStyle = colors.ink; context.fillText(example.mode, outputX, laneY[packet.lane] + 1);
        context.restore();
      }

    }
    context.globalAlpha = 1;
    const captionY = laneY[0] - 48;
    label(width >= 900 ? 'Session traces' : 'Traces', gate * cw - (width >= 900 ? 56 : 24), captionY, colors.muted);
    const caption = width >= 900 ? 'Execution mode' : 'Mode';
    label(caption, width - caption.length * cw - 24, captionY, colors.muted);
  }
  function stop() { cancelAnimationFrame(frameId); frameId = 0; previous = 0; }
  function start() {
    if (!ready || frameId || reduce.matches || !visible || document.hidden) return;
    frameId = requestAnimationFrame(frame);
  }
  function frame(ms) {
    frameId = 0;
    if (previous && ms - previous < 1000 / 30) { frameId = requestAnimationFrame(frame); return; }
    const dt = previous ? Math.min(.1, (ms - previous) / 1000) : 0;
    previous = ms; time += dt;
    pointer.strength += (pointer.target - pointer.strength) * .12;
    for (const packet of particles) packet.x += packet.speed * dt;
    particles = particles.filter(packet => packet.x < columns + 5);
    for (const lane of [0, 2]) {
      if (!particles.some(packet => packet.lane === lane && packet.x < gate * .6)) {
        particles.push({lane, x: -52 - lane * 6, sample: sampleLog(nextId++), speed: 9 + lane * .7});
      }
    }
    draw(); frameId = requestAnimationFrame(frame);
  }
  hero.addEventListener('pointermove', event => {
    const rect = hero.getBoundingClientRect();
    pointer.x = event.clientX - rect.left; pointer.y = event.clientY - rect.top; pointer.target = 1;
  }, {passive: true});
  hero.addEventListener('pointerleave', () => { pointer.target = 0; });
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; if (visible) start(); else stop(); }).observe(hero);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); else start(); });
  reduce.addEventListener('change', () => { if (reduce.matches) { stop(); seed(); draw(); } else start(); });
  new ResizeObserver(() => { if (ready) resize(); }).observe(hero);
  document.addEventListener('pagelanguagechange', () => { if (ready) requestAnimationFrame(resize); });
  const fonts = document.fonts ? Promise.race([document.fonts.ready, new Promise(resolve => setTimeout(resolve, 1500))]) : Promise.resolve();
  fonts.then(() => { resize(); ready = true; document.documentElement.classList.add('has-field'); start(); });
  document.fonts?.addEventListener('loadingdone', () => { if (ready) resize(); });
})();

// Reveal the runnable source workflow when its link is followed from the API preview.
for (const link of document.querySelectorAll('a[href="#source-reproduction"]')) {
  link.addEventListener('click', () => {
    const details = document.getElementById('source-reproduction')?.closest('details');
    if (details) details.open = true;
  });
}

if (location.hash === "#source-reproduction") {
  document.getElementById("source-reproduction")?.closest("details")?.setAttribute("open", "");
}
