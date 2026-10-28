/* Authored native copy. Code, trace examples, citations and numeric results stay intact. */
(() => {
  const root = document.documentElement;
  const copy = [...document.querySelectorAll('[data-en], [data-en-html]')].map(element => ({
    element,
    html: element.hasAttribute('data-en-html'),
    zh: element.hasAttribute('data-en-html') ? element.innerHTML : element.textContent,
    en: element.dataset.enHtml ?? element.dataset.en,
  }));
  const attributes = [];
  for (const name of ['aria-label', 'alt', 'title']) {
    for (const element of document.querySelectorAll(`[data-en-${name}]`)) {
      attributes.push({element, name, zh: element.getAttribute(name), en: element.getAttribute(`data-en-${name}`)});
    }
  }
  function choose(language, persist = true) {
    const english = language === 'en';
    root.lang = english ? 'en' : 'zh-CN';
    root.dataset.language = english ? 'en' : 'zh';
    for (const item of copy) {
      if (item.html) item.element.innerHTML = english ? item.en : item.zh;
      else item.element.textContent = english ? item.en : item.zh;
    }
    for (const item of attributes) item.element.setAttribute(item.name, english ? item.en : item.zh);
    for (const button of document.querySelectorAll('button[data-language]')) {
      button.setAttribute('aria-pressed', String(button.dataset.language === (english ? 'en' : 'zh')));
    }
    if (persist) {
      try { localStorage.setItem('attackagent-language', english ? 'en' : 'zh'); } catch {}
    }
    document.dispatchEvent(new CustomEvent('pagelanguagechange', {detail: {language: english ? 'en' : 'zh'}}));
  }
  for (const button of document.querySelectorAll('button[data-language]')) {
    button.addEventListener('click', () => choose(button.dataset.language));
  }
  let saved = 'zh';
  try { saved = localStorage.getItem('attackagent-language') || 'zh'; } catch {}
  const requested = new URL(location.href).searchParams.get('lang');
  choose(requested === 'en' || requested === 'zh' ? requested : saved, false);
})();
