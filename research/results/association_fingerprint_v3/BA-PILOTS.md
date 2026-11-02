# Exploratory BA study audit

All experiments follow inspection of earlier results. Intervals are unadjusted; variants and held-out scores must not be treated as a fresh confirmatory test. CTH labels are same benchmark task; execution mode and capture path are confounded. The task-held-out CTH galleries are identical across these pilots.

| Experiment | Method | BA (%) | Difference from its matched evidence baseline (pp; 95% task interval) |
| --- | --- | ---: | --- |
| ba_pilot | evidence | 63.36 | +0.00 [+0.00, +0.00] |
| ba_pilot | additive | 65.74 | +2.38 [-2.30, +6.96] |
| ba_pilot | gated | 64.72 | +1.35 [-0.70, +3.63] |
| ba_pilot | gated_components | 64.63 | +1.26 [-3.84, +6.10] |
| ba_matched_negatives | evidence | 46.80 | +0.00 [+0.00, +0.00] |
| ba_matched_negatives | additive | 51.83 | +5.03 [+0.81, +9.32] |
| ba_matched_negatives | gated | 50.24 | +3.43 [-2.69, +9.44] |
| ba_matched_negatives | gated_components | 54.31 | +7.51 [+1.55, +13.77] |
| ba_blend_25 | evidence | 63.18 | +0.00 [+0.00, +0.00] |
| ba_blend_25 | additive | 61.74 | -1.44 [-5.80, +3.00] |
| ba_blend_25 | gated | 63.69 | +0.51 [-2.53, +3.42] |
| ba_blend_25 | gated_components | 63.20 | +0.02 [-4.76, +4.71] |
| ba_blend_50 | evidence | 58.33 | +0.00 [+0.00, +0.00] |
| ba_blend_50 | additive | 59.48 | +1.15 [-3.30, +5.76] |
| ba_blend_50 | gated | 60.50 | +2.17 [-1.90, +6.49] |
| ba_blend_50 | gated_components | 57.92 | -0.41 [-5.63, +4.50] |
| ba_local_alignment | evidence | 63.36 | +0.00 [+0.00, +0.00] |
| ba_local_alignment | lexical_local | 63.36 | -0.01 [-3.50, +3.41] |
| ba_local_alignment | fingerprint_local | 64.82 | +1.45 [-2.63, +5.38] |
| ba_action_profiles | evidence | 63.36 | +0.00 [+0.00, +0.00] |
| ba_action_profiles | phase_hist | 61.65 | -1.72 [-6.76, +3.53] |
| ba_action_profiles | transition_hist | 55.49 | -7.88 [-14.07, -1.42] |
| ba_action_profiles | combined_profile | 59.79 | -3.57 [-9.91, +2.78] |
| ba_trees | evidence | 65.07 | +0.00 [+0.00, +0.00] |
| ba_trees | additive | 64.72 | -0.35 [-4.07, +3.35] |
| ba_trees | gated | 67.51 | +2.45 [+0.22, +4.78] |
| ba_trees | gated_components | 67.10 | +2.03 [+0.08, +3.98] |
| ba_tree_window4 | evidence | 65.07 | +0.00 [+0.00, +0.00] |
| ba_tree_window4 | additive | 67.19 | +2.13 [-1.15, +5.33] |
| ba_tree_window4 | gated | 65.08 | +0.02 [-2.51, +2.41] |
| ba_tree_window4 | gated_components | 67.40 | +2.33 [-1.39, +5.74] |
| ba_tree_window16 | evidence | 65.07 | +0.00 [+0.00, +0.00] |
| ba_tree_window16 | additive | 67.00 | +1.93 [-1.29, +4.98] |
| ba_tree_window16 | gated | 68.12 | +3.05 [+0.48, +5.47] |
| ba_tree_window16 | gated_components | 65.43 | +0.37 [-3.32, +3.81] |
| ba_tree_ablation | evidence | 65.07 | +0.00 [+0.00, +0.00] |
| ba_tree_ablation | length_control | 65.78 | +0.71 [-2.34, +3.74] |
| ba_tree_ablation | gated_B | 65.63 | +0.57 [-2.28, +3.22] |
| ba_tree_ablation | gated_S | 63.96 | -1.11 [-4.32, +2.03] |
| ba_tree_ablation | gated_components | 67.10 | +2.03 [+0.08, +3.98] |
| ba_tree_window_shuffle | evidence | 65.07 | +0.00 [+0.00, +0.00] |
| ba_tree_window_shuffle | additive | 64.82 | -0.25 [-3.85, +3.25] |
| ba_tree_window_shuffle | gated | 65.86 | +0.79 [-1.42, +3.01] |
| ba_tree_window_shuffle | gated_components | 66.37 | +1.30 [-1.69, +4.00] |

The matched-hard-negative-only baseline drops to 46.80% BA, below the original evidence baseline (63.36%). Its apparent +7.51 pp fingerprint increase is therefore not evidence of a net improvement over the stronger prior baseline. Mixed-negative, action-profile, and local-alignment attempts are retained above.

## Frozen-family holdouts

The gated-component shallow-tree family is evaluated with identical candidates and train-side thresholds in each comparison. Within-source windows are chosen on inner folds; source-held-out comparisons use an eight-event window. Known normalized CyBench task-name overlap is purged from outer and inner source-held-out training.

| Source | Holdout | Evidence BA | Fingerprint BA | Paired delta [95% interval], pp |
| --- | --- | ---: | ---: | --- |
| CTH | task or participant | 65.07% | 68.00% | +2.94 [+0.43, +5.34] |
| CTH | entire source | 51.98% | 56.69% | +4.71 [+0.69, +8.83] |
| KYPO | task or participant | 55.83% | 60.08% | +4.25 [+1.03, +8.02] |
| KYPO | entire source | 66.40% | 66.61% | +0.21 [-1.06, +1.36] |
| ANTIEVAL | task or participant | 70.24% | 70.38% | +0.13 [-2.47, +2.61] |
| ANTIEVAL | entire source | 71.90% | 68.63% | -3.27 [-6.28, -0.22] |

CTH source holdout has 175 queries and 40 task units: FSR 85.75% → 72.88%, FMR 10.29% → 13.75%. Its BA increase coexists with substantial remaining missed links. KYPO intervals resample query participants while candidate galleries remain fixed. Matching challenge names are a conservative proxy for overlap; unrecognized task/environment overlap may remain.

Each source-held-out candidate score, threshold, and grouped tree-path contribution is recorded in `ba_leave_source_out_purged_pairs.csv`. Tree-path contributions sum to the model probability and are descriptive model decompositions, not causal attribution.
