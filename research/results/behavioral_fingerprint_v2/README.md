# Behavioral fingerprint v2 results

This directory contains versioned aggregate metrics, out-of-fold predictions, fold assignments, data audits, and figure inputs for the B/S/T fingerprint study. Results are separated by source and by task, mode-label, or source-specific condition; labels are not interchangeable across datasets.

## Main outputs

- `cth_mixture/`: exact 0/25/50/75/100% Agent-role mixtures on the 17-task, 85-trace core; uncertainty is task-clustered.
- `cth_cross_mode/`: held-out task-gallery retrieval over all 40 mixed CTH challenges and 175 feature-eligible traces.
- `ait_recognition/`: frozen 89-run and expanded 109-run AIT recognition cohorts, task-group and source-group splits.
- `ait_ablation/`: AIT first-window, window-order, and within-window gap-order controls.
- `honey/`: Agent-only SSH retrieval by prompt/cap/environment condition and execution-duration sensitivity.
- `cybiasbench/`: Agent-only HTTP route retrieval with target-held-out evaluation; only three target clusters are available.
- `legacy_association/`: the original 97-trace command/artifact comparison, kept separate from B/S/T increments.
- `figures/`: publication figures rendered from these result files.

Read each subdirectory's `provenance.json` and the matching experiment README before interpreting or reusing a result. Site-level aggregate values are mirrored in `site/data/behavior-trace-analysis-v2.json`.

- `refinement/`: complete nested family/C comparison, cross-fitted mode assistance, and 50% mixture task-cluster error contrasts.
- `kypo_human/`: participant-disjoint Human-only known-task retrieval; 197 eligible traces, 190 evaluated queries from 97 people across eight represented task levels.

The CTH HA–HH task-macro FSR gap is +35.5 pp [12.8, 58.4]. B/S fusion reduces HA missed links by 6.0 pp while increasing HA false merges by 4.6 pp. Nested refinement retains all attempted methods. General ranking gains remain unsupported; the clearest positive feature-block increment is AIT timing under task holdout (+15.8 pp BA). Later contrasts and refinements are exploratory.
