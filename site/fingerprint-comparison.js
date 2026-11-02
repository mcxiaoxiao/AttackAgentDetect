/* Two equivalent views of the same source-backed fingerprint values. */
(() => {
  for (const comparison of document.querySelectorAll('[data-fingerprint-comparison]')) {
    const tabs = [...comparison.querySelectorAll('[data-fingerprint-view]')];
    const panels = [...comparison.querySelectorAll('[data-fingerprint-panel]')];
    const controls = comparison.querySelector('[role="tablist"]');
    const show = (view, moveFocus = false) => {
      for (const tab of tabs) {
        const selected = tab.dataset.fingerprintView === view;
        tab.setAttribute('aria-selected', String(selected));
        tab.tabIndex = selected ? 0 : -1;
        if (selected && moveFocus) tab.focus();
      }
      for (const panel of panels) panel.hidden = panel.dataset.fingerprintPanel !== view;
    };
    for (const [index, tab] of tabs.entries()) {
      tab.addEventListener('click', () => show(tab.dataset.fingerprintView));
      tab.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
        else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = tabs.length - 1;
        else return;
        event.preventDefault();
        show(tabs[next].dataset.fingerprintView, true);
      });
    }
    show('stack');
    controls.hidden = false;
  }
})();
