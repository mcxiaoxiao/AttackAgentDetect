(() => {
  const link = document.querySelector('[data-pages-commit]');
  if (!link) return;
  const repository = 'https://github.com/mcxiaoxiao/AttackAgentDetect';
  const apiEndpoint = 'https://api.github.com/repos/mcxiaoxiao/AttackAgentDetect/commits/gh-pages';

  const applyRevision = (sha, metadata = {}) => {
    if (!/^[0-9a-f]{40}$/i.test(sha || '')) return false;
    const short = sha.slice(0, 7);
    const deployedAt = metadata.deployedAt ? String(metadata.deployedAt).slice(0, 10) : '';
    link.href = `${repository}/commit/${sha}`;
    link.title = deployedAt ? `${sha} · source ${deployedAt}` : sha;
    link.setAttribute('aria-label', `Source revision ${sha}${deployedAt ? ` updated on ${deployedAt}` : ''}`);
    const code = document.createElement('code');
    code.textContent = short;
    link.replaceChildren(code);
    return true;
  };

  const readDeploymentFile = async () => {
    const response = await fetch(`deployment.json?v=${Date.now()}`, {
      cache: 'no-store',
      headers: {Accept: 'application/json'},
    });
    if (!response.ok) throw new Error('deployment metadata unavailable');
    const metadata = await response.json();
    return applyRevision(metadata.commit || metadata.sha, {
      deployedAt: metadata.deployed_at || metadata.deployedAt,
    });
  };

  const readGitHubRevision = async () => {
    const response = await fetch(apiEndpoint, {
      headers: {Accept: 'application/vnd.github+json'},
    });
    if (!response.ok) throw new Error('GitHub revision unavailable');
    const {sha} = await response.json();
    return applyRevision(sha);
  };

  readDeploymentFile()
    .then(applied => (applied ? undefined : readGitHubRevision()))
    .catch(() => readGitHubRevision())
    .catch(() => {});
})();
