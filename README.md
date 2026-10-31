<h1 align="center">
  <img src="site/assets/attackagent-detect-logo.png" width="500" alt="AttackAgent Detect" />
</h1>

<p align="center"><a href="README.zh-CN.md">简体中文</a> · English</p>

AttackAgent Detect is a Python toolkit for turning security logs into consistent traces and comparing their behavior. The repository also contains a separate research workspace for evaluating these methods on public CTF data.

## Use the toolkit

The library can:

- Normalize Linux auditd SYSCALL and EXECVE records into trace JSONL.
- Rank candidate traces using command and action evidence.
- Run an experimental Human / Agent detector when labeled training traces use the same source format.

The tools analyze records. They do not execute logged commands. Human / Agent scores are not proof of who performed an action or whether traces belong to the same attack.

### Install from this repository

Python 3.10 or later is required. The package is not published on PyPI yet.

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
~~~

Normalize the included auditd example:

~~~bash
.venv/bin/python -m attackagent.jsonl_cli normalize --input examples/auditd/minimal.audit --output /tmp/normalized.jsonl
~~~

Rank candidate traces:

~~~bash
.venv/bin/python -m attackagent.jsonl_cli rank --query examples/jsonl/queries.jsonl --candidates examples/jsonl/candidates.jsonl --method command_cosine
~~~

Run .venv/bin/python -m attackagent.jsonl_cli methods to see the available ranking methods. More input examples are in [examples/](examples/).

## Research, kept separate

The research workspace contains study plans, datasets and manifests, experiment code, notebooks, results, and paper sources. Its current evidence uses 177 traces from 40 public CTF task groups. A shared task means the same challenge; it does not establish a shared operator, campaign, or real-world attack. See the [research index](research/README.md) for the studies, data, and reproduction steps.

The [project website](https://mcxiaoxiao.github.io/AttackAgentDetect/) presents the main findings and dataset provenance.

## Repository layout

- **attackagent/** — Python toolkit.
- **examples/** — small input and output examples.
- **tests/** — toolkit checks.
- **research/** — research plans, experiments, data manifests, results, notebooks, and papers.
- **site/** — project website source.
- **vendor/procgrep/** — vendored dependency and its license.

## License

The project code is MIT licensed. Upstream dataset terms and third-party notices are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
