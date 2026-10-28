(() => {
  const link = document.querySelector('[data-pages-commit]');
  if (!link) return;
  const endpoint = 'https://api.github.com/repos/mcxiaoxiao/AttackAgentDetect/commits/gh-pages';
  fetch(endpoint, {headers: {Accept: 'application/vnd.github+json'}})
    .then(response => { if (!response.ok) throw new Error('GitHub revision unavailable'); return response.json(); })
    .then(({sha}) => {
      if (!/^[0-9a-f]{40}$/i.test(sha || '')) return;
      const short = sha.slice(0, 7);
      link.href = `https://github.com/mcxiaoxiao/AttackAgentDetect/commit/${sha}`;
      link.title = sha;
      link.setAttribute('aria-label', `GitHub Pages deployed commit ${sha}`);
      link.replaceChildren(Object.assign(document.createElement('code'), {textContent: short}));
    })
    .catch(() => {});
})();
