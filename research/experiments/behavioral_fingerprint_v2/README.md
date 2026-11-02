# Behavioral fingerprint v2 experiment protocol

This version evaluates a log-derived fingerprint with three blocks: local behavior (B), sequence evolution (S), and observed timing (T). It has separate Human/Agent-label and pairwise-retrieval readouts. Source IDs, task labels, model names, capture-format labels, and outcomes are evaluation metadata and are not fingerprint inputs.

## Reproduce

Run from the repository root with the project's `.venv`:

```sh
.venv/bin/python research/experiments/behavioral_fingerprint_v2/run_ait_recognition.py
.venv/bin/python research/experiments/behavioral_fingerprint_v2/run_ait_ablation.py
.venv/bin/python research/experiments/behavioral_fingerprint_v2/run_cth_mixture.py
.venv/bin/python research/experiments/behavioral_fingerprint_v2/run_cth_cross_mode.py
.venv/bin/python research/experiments/behavioral_fingerprint_v2/run.py
.venv/bin/python research/experiments/behavioral_fingerprint_v2/run_cybias_retrieval.py
.venv/bin/python research/experiments/behavioral_fingerprint_v2/build_figures.py
.venv/bin/python research/experiments/behavioral_fingerprint_v2/plot_ait_ablation.py
```

The AIT source archive is read through the existing pilot loader. The CTH experiment reads the pinned source manifest and local normalized trace views. Honey and CyBiasBench scripts read their pinned public inputs; their session-level rows are not copied into the website bundle. Exact input revisions, hashes, split definitions, and seeds are recorded under `research/results/behavioral_fingerprint_v2/`.

## AIT controls

`run_ait_ablation.py` compares the full B+S summary with (1) the first valid eight-event window only, (2) randomly permuted eight-event chunks with B contents retained and S recomputed, (3) B+S+T, and (4) B+S+T after permuting observed gaps within each eight-event window while retaining each window's gap multiset. Twenty perturbations are averaged per trace. These are diagnostics on this AIT collection; they do not establish a general execution-mode signature.

## Evidence and licensing boundaries

- CTH same-challenge identity is a task-level retrieval proxy, not shared activity or actor truth.
- AIT execution labels co-vary with troubleshooting tasks and capture workflows.
- Honey contains Agent-only SSH sessions; its durations are not inter-command start-time gaps.
- CyBiasBench contains Agent-only HTTP sessions, no precise event timestamps, and has source-specific reuse terms.
- Results are aggregated by the experiment's valid grouping unit. Repeated candidate draws do not become independent samples.

Raw external records remain governed by their upstream terms. The public site distributes aggregate analysis and source references, not a repackaged copy of Honey or CyBiasBench sessions.

## Completed bounded refinements

Run `run_refinement.py` for the four-family, three-C nested comparison on the unchanged 40-task galleries; `run_mode_assisted.py` cross-fits the fingerprint mode readout; `run_error_strata.py` computes the fixed-50% task-cluster contrasts; `run_kypo_retrieval.py` evaluates Human-only retrieval with participant-disjoint folds. Their results are separately versioned under `refinement/` and `kypo_human/`. These follow-ups were designed after inspection of the core results and are exploratory. `plot_error_strata.py` renders the error-structure diagnostic.

All figures are exported at approximately 7.2-inch final manuscript width with complete error intervals, explicit units, shared heatmap scales, and readable local-scrolling web viewports. `package_manuscript.py` rebuilds the current source-and-figure ZIP from the TeX references.
