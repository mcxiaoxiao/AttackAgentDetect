# Matched CTH iteration: role-balanced training and abstract command arguments

Status: exploratory development evidence. This report records two targeted attempts to diagnose the weak/unstable B/S increment. Neither iteration is independent confirmation, and no manuscript or public research-page result has been updated from it.

## Frozen evaluation

- CTH same-task proxy cohort: 17 tasks and 85 traces (34 Human, 51 Agent).
- Three outer task folds; all preprocessing and parameter selection remain inside the training tasks.
- Exact query/gallery role shares: 0%, 25%, 50%, 75%, and 100% Agent, with 170 coupled draws. Each query has one same-task and three different-task candidates.
- Each model/version receives the same 54,400 query-candidate pairs. The combined output contains 598,400 per-pair predictions across three classifiers and their controls.
- Metrics are task-macro averages. Confidence intervals use 10,000 paired resamples of the 17 task clusters; shared candidate tasks make them approximate.
- The decision threshold is selected on inner task-held-out scores to maximize recall subject to pooled FMR at most 10%. It is one threshold per outer fold and model version, not one threshold per mixture share or pair type.
- The task label is a retrieval proxy. It does not identify an attack activity, operator, Agent instance, or organization.

## Five-share CTH matched classifier comparison

The original evidence E and E+B/S were compared on the same 54,400 query-candidate pairs per learner across five exact Agent shares. Metrics first average within task and share; the headline averages the five shares equally and then gives each of the 17 tasks equal weight. Intervals use 10,000 paired task-cluster resamples. The 0.5-share figures in older descriptive reports used different baselines and must not be substituted for these learned-model results.

| Learner | Evidence | BA % [95% task interval] | FMR % | FSR % |
|---|---|---:|---:|---:|
| SVM | E | 52.57 [49.79, 55.77] | 6.52 | 88.35 |
| SVM | E+B/S | 53.95 [50.29, 57.89] | 13.50 | 78.60 |
| Decision tree | E | 52.80 [49.43, 56.61] | 6.59 | 87.81 |
| Decision tree | E+B/S | 50.00 [50.00, 50.00] | 0.00 | 100.00 |
| XGBoost | E | 56.06 [52.49, 59.86] | 5.97 | 81.91 |
| XGBoost | E+B/S | 53.68 [50.73, 56.99] | 15.43 | 77.21 |

The paired equal-share ΔBA was +1.38 pp [−3.83, +6.53] for SVM, −2.80 pp [−6.61, +0.57] for the tree, and −2.38 pp [−7.13, +2.25] for XGBoost. SVM and XGBoost lower FSR while raising FMR; the tree version rejects all candidates. Across XGBoost's individual shares, ΔBA was −8.67, −4.21, −1.16, +1.10, and +1.04 pp from 0% through 100% Agent. This is a local cohort pattern, not evidence that all models improve as Agent share rises.

XGBoost pair-type averages use only shares where that type exists and give those shares equal weight within each task. For HH across 0–75%, E+B/S changed FMR by +5.92 pp and FSR by +11.76 pp (ΔBA −8.84 pp, 95% interval [−20.34, +1.67]). For HA across 25–75%, FMR changed +15.66 pp and FSR −18.23 pp (ΔBA +1.29 pp [−1.78, +4.71]). For AA across 25–100%, FMR changed +9.77 pp and FSR −12.54 pp (ΔBA +1.39 pp [−5.80, +8.03]). At 50%, the pair-decision counts were HH 662 positive/2,004 negative, HA 1,396/4,161, and AA 662/1,995. These are repeated candidate decisions, not independent observations. The within-type error pattern differs, but the intervals do not establish a reliable BA improvement for any XGBoost stratum.

The complete per-share and per-type metrics, CIs, pair counts, paired error transitions, and equal-weight summaries are in `research/results/matched_fingerprint_v2/five_share/five_share_analysis.json` and `five_share_rows.csv`; `analyze_cth_five_share.py` verifies that every classifier and feature version uses the same ordered pair manifest. Aggregate HH/HA/AA results in earlier reports used different protocols, so the five-share output is the reference for this main comparison.

## Attempt 1: balance HH/HA/AA training weights

The training pair sample was balanced overall by positive/negative label, but the observed HH/HA/AA counts differed within each label. For example, one outer fold had 15/38/35 positive HH/HA/AA pairs and 9/54/25 negative pairs. I gave each observed pair type equal total weight within each label. The pair sample, features, classifiers, folds, model-selection budget, test galleries, and threshold rule stayed fixed.

The XGBoost E+B/S version changed by +1.32 pp BA (95% paired task interval [−4.84,+7.92]) relative to the earlier unweighted E+B/S version; FMR fell 1.29 pp and FSR fell 1.34 pp. Neither SVM nor XGBoost reached the planned gain threshold, and the decision tree still rejected every outer candidate. Under the balanced training itself, XGBoost E+B/S had +1.51 pp BA over E (95% interval [−3.09,+6.43]), with FMR up 8.91 pp and FSR down 11.94 pp. Training balance helps explain some score movement but does not remove the false-merge/false-split trade-off.

Full predictions, weights, task-fold audits, and paired transitions are under `research/results/matched_fingerprint_v2/role_balanced/`. `research/experiments/matched_fingerprint_v2/verify_role_balanced.py` checks the shared pair list, exact shares, task isolation, weight totals, score thresholds, fallback behavior, and BA arithmetic.

## Attempt 2: restore abstract command syntax to both arms

The current command adapter keeps known command heads and discards their arguments. Raw-log inspection found 578 known Agent command invocations, of which 576 included arguments and 370 included options; among 240 Human invocations, 163 included arguments and 31 included options. The command-head sequence for every one of the 85 traces matches the frozen feature corpus. Because the Human and Agent capture interfaces differ, these counts establish a data-coverage gap, not an execution-mode fingerprint.

I built a value-redacted profile containing option names, coarse argument types, and argument-count buckets. It excludes literal paths, addresses, hostnames, filenames, numeric values, task IDs, and labels. The same training-only fitted profile contributes to E and E+B/S. The model also emits a head-only E control so the base-evidence effect can be separated from the B/S increment. All three controls use the same selected learner parameter, representation, training pairs, and gallery within each fold.

### Five-share average

Each metric is followed by its 95% task-cluster interval. E is command-head and action-bigram evidence; E+Args adds abstract argument syntax; E+Args+B/S adds the fingerprint.

| Learner | Version | BA % | FMR % | FSR % |
|---|---|---:|---:|---:|
| XGBoost | E | 56.06 [52.50, 59.94] | 5.97 [3.71, 8.45] | 81.91 [73.45, 89.76] |
| XGBoost | E+Args | 53.57 [49.46, 57.55] | 12.77 [7.04, 19.14] | 80.10 [70.82, 88.60] |
| XGBoost | E+Args+B/S | 54.66 [50.42, 59.64] | 10.11 [5.93, 14.58] | 80.58 [66.87, 92.22] |
| SVM | E | 52.23 [49.02, 55.90] | 5.73 [3.65, 8.02] | 89.81 [82.01, 96.35] |
| SVM | E+Args | 50.25 [47.68, 53.64] | 7.15 [5.69, 8.69] | 92.35 [85.29, 97.67] |
| SVM | E+Args+B/S | 54.03 [50.85, 57.45] | 9.23 [7.43, 11.07] | 82.70 [74.68, 90.32] |
| Decision tree | E | 50.00 [50.00, 50.00] | 0.00 [0.00, 0.00] | 100.00 [100.00, 100.00] |
| Decision tree | E+Args | 50.00 [50.00, 50.00] | 0.00 [0.00, 0.00] | 100.00 [100.00, 100.00] |
| Decision tree | E+Args+B/S | 50.00 [50.00, 50.00] | 0.00 [0.00, 0.00] | 100.00 [100.00, 100.00] |

For the primary XGBoost comparison, E+Args+B/S minus E+Args is +1.09 pp BA (95% interval [−4.38,+7.58]), −2.66 pp FMR ([−5.93,+0.38]), and +0.49 pp FSR ([−11.04,+10.90]). The arguments-only increment from E to E+Args is −2.49 pp BA ([−5.50,+0.02]); FMR rises 6.80 pp ([+3.08,+10.92]). Thus argument recovery alone does not improve the five-share average.

### Pair-type findings

Each pair-type estimate averages the tiers where that type exists (HH: 0–75%; HA: 25–75%; AA: 25–100%). BA includes its 95% task interval. FMR/FSR entries are point changes unless an interval is printed.

| Learner and comparison | Pair type | ΔBA pp [95% interval] | ΔFMR pp | ΔFSR pp |
|---|---|---:|---:|---:|
| XGBoost, E+Args vs E | HH | −4.64 [−10.41, −1.07] | +3.41 | +5.88 |
| XGBoost, E+Args vs E | HA | −1.66 [−4.03, +0.79] | +8.22 | −4.89 |
| XGBoost, E+Args vs E | AA | −1.07 [−5.26, +2.92] | +9.07 | −6.93 |
| XGBoost, E+Args+B/S vs E+Args | HH | −2.03 [−11.81, +7.77] | −1.82 | +5.88 |
| XGBoost, E+Args+B/S vs E+Args | HA | **+3.95 [+0.14, +8.72]** | **−4.16 [−6.44, −1.85]** | −3.74 [−12.26, +3.48] |
| XGBoost, E+Args+B/S vs E+Args | AA | +2.09 [−4.92, +9.13] | −2.06 | −2.12 |
| SVM, E+Args+B/S vs E+Args | HH | **+10.54 [+3.36, +19.41]** | −3.43 | −17.65 |
| SVM, E+Args+B/S vs E+Args | HA | −0.38 [−2.77, +2.25] | +11.78 | −11.03 |
| SVM, E+Args+B/S vs E+Args | AA | +0.16 [−5.95, +6.13] | +0.54 | −0.86 |

For XGBoost, the HA stratum is a promising local result: B/S lowers both error-rate point estimates and the BA interval is above zero. The FMR interval is also below zero, while the FSR interval crosses zero. SVM’s clearer gain is in HH and comes with a different HA trade-off. At the five-share aggregate, neither family has a confident BA improvement, and the decision tree remains an all-reject classifier. HH is not present at the 100% tier; HA is absent at 0% and 100%; AA is absent at 0%.

This rules out the simple explanation that the aggregate result changes only because the number of pair types changes: within-type effects also differ. It does not tell us why. Similar-looking behavior across distinct tasks, execution-path changes within one task, capture-interface effects, and threshold calibration remain plausible explanations. Pair types describe trace sources, not real attack activities or actors.

## Attempt 3: expand training-pair coverage

The sampled training protocol used only a small number of positive pairs per task. I compared it with every distinct available positive and negative pair in each CTH training fold. Both conditions used identical test galleries, thresholds selected on inner tasks, equal effective training-weight totals, and equal weight mass across labels and observed HH/HA/AA cells.

With all available training pairs, XGBoost E+B/S versus E changed five-share BA by +3.29 pp (95% interval [−1.35, +8.66]); FMR rose +0.70 pp and FSR fell −7.29 pp. Under the same expanded pairing, the abstract-argument comparison E+Args+B/S versus E+Args changed BA by +6.25 pp [+1.17, +12.24], FMR +0.56 pp, and FSR −13.06 pp. This is a clearer recall-side signal, but it still introduces some additional false merges and is not independent confirmation. More pairs help explain sensitivity to training support; they do not establish that B/S lowers both error types.

The pair manifest, weights, fold support, and paired predictions are in `research/results/matched_fingerprint_v2/pair_support/`. The experiment is recorded in `research/experiments/matched_fingerprint_v2/run_pair_support.py`.

## Attempt 4: frozen cross-source confirmation on NYU CTF Agent–Agent traces

The confirmation protocol and eligible task manifest were frozen before model fitting. Models were trained only on the 17-task mixed-role CTH development cohort; NYU contains 160 previously unused challenge groups and 585 distinct D-CIPHER runs. Each run was a query with one same-challenge run and three different-challenge candidates, sampled 100 times from a fixed manifest. Each of the three learners and two feature arms used the same 234,000 pairs. The full prediction file has 1,404,000 rows; every NYU pair had complete B/S features, so the fallback rate was zero. Task-cluster bootstrap intervals resampled the 160 challenge groups and averaged repeated galleries within each group.

| Learner | Evidence | BA % [95% task interval] | FMR % | FSR % |
|---|---|---:|---:|---:|
| SVM | E | 58.16 [56.21, 60.15] | 26.58 | 57.09 |
| SVM | E+B/S | 49.93 [49.91, 49.95] | 0.13 | 100.00 |
| Decision tree | E | 49.69 [49.09, 50.32] | 3.85 | 96.78 |
| Decision tree | E+B/S | 53.33 [51.85, 54.90] | 8.54 | 84.80 |
| XGBoost | E | 51.16 [49.89, 52.58] | 10.72 | 86.96 |
| XGBoost | E+B/S | 56.87 [55.04, 58.69] | 25.39 | 60.86 |

The paired BA change was −8.23 pp for SVM (95% interval [−10.22, −6.28]), +3.64 pp for the tree ([+1.97, +5.37]), and +5.71 pp for XGBoost ([+3.63, +7.73]). The tree and XGBoost BA gains remained positive under leave-one-task-out sensitivity, with ranges [+3.36, +3.78] and [+5.46, +5.91] pp. They did not reduce both error types: XGBoost FMR rose by 14.67 pp while FSR fell by 26.10 pp; the tree FMR rose by 4.69 pp while FSR fell by 11.98 pp. SVM with B/S rejected almost all pairs and reached 100% FSR.

On the identical pairs, XGBoost B/S corrected 19,708 same-task false splits and introduced 4,329, but corrected 11,241 different-task false merges and introduced 37,348. The tree corrected 8,959 same-task misses and introduced 1,926, while correcting 6,732 false merges and introducing 15,244. XGBoost task-macro Hit@1 improved +3.20 pp [0.45, 5.90], MRR +1.94 pp [0.06, 3.74], and AUC +1.82 pp [−0.38, 3.97]. The rank metrics therefore indicate a modest ordering signal, while the fixed CTH threshold does not transfer its FMR≤10% operating point: external XGBoost FMR is 10.72% for E and 25.39% for E+B/S. Thresholds were not changed after opening NYU.

This is a cross-source CTF task-retrieval confirmation for four D-CIPHER snapshots, not independent confirmation across Agent frameworks or Human/Agent pairing types. The same challenge filename is a retrieval proxy; it does not establish a common activity, operator, Agent instance, or campaign. It is evidence of a classifier-dependent recall/false-merge trade-off and does not satisfy the predeclared requirement that both FMR and FSR decrease. NYU must not be used for additional model or threshold selection.

Machine-readable task-macro metrics, cluster intervals, ranking results, error transitions, leave-one-task-out checks, thresholds, hashes, and software versions are in `research/results/matched_fingerprint_v2/independent_nyu_agent/analysis.json` and `run_manifest.json`. The frozen panel is described by `research/experiments/matched_fingerprint_v2/independent_nyu_aa_protocol.md` and `research/results/matched_fingerprint_v2/independent_nyu_task_manifest.json`.

## Unified CTH/AIT feature ablation

The table below is recomputed from the stored matched predictions. CTH reports the equal-weight mean across five exact role shares; AIT reports a task-macro result over its 12 task groups. Each dataset uses XGBoost and the same pair list across rows. BA and ΔBA intervals use 10,000 paired task-cluster resamples; FMR and FSR are point estimates. “E” is the original evidence with all fingerprint blocks removed. The pair counts are repeated candidate decisions, not independent observations.

| Dataset | Configuration | B | S | T | BA % [95% task interval] | FMR % | FSR % | ΔBA vs complete, pp [95% interval] |
|---|---|:---:|:---:|:---:|---:|---:|---:|---:|
| CTH | E+B+S | ✓ | ✓ | — | 53.68 [50.68, 56.96] | 15.43 | 77.21 | 0.00 [0.00, 0.00] |
| CTH | E+S | — | ✓ | — | **56.10 [51.99, 60.39]** | 14.47 | 73.34 | +2.42 [−3.09, +8.05] |
| CTH | E+B | ✓ | — | — | 52.75 [49.67, 55.96] | 13.13 | 81.37 | −0.93 [−2.52, +0.59] |
| CTH | E | — | — | — | 56.06 [52.42, 59.91] | 5.97 | 81.91 | +2.38 [−2.28, +7.29] |
| AIT | E+B+S+T | ✓ | ✓ | ✓ | 51.92 [50.58, 53.18] | 5.37 | 90.79 | 0.00 [0.00, 0.00] |
| AIT | E+S+T | — | ✓ | ✓ | **52.01 [51.08, 53.03]** | 3.53 | 92.45 | +0.09 [−1.05, +1.14] |
| AIT | E+B+T | ✓ | — | ✓ | 51.37 [50.15, 52.84] | 5.24 | 92.02 | −0.55 [−1.66, +0.51] |
| AIT | E+B+S | ✓ | ✓ | — | 50.94 [49.38, 52.58] | 5.28 | 92.84 | −0.98 [−2.10, +0.21] |
| AIT | E | — | — | — | 50.12 [50.00, 50.36] | 0.00 | 99.76 | −1.80 [−3.14, −0.37] |

CTH has 17 task groups and 13,600 positive/40,800 negative pair decisions per row across the five shares; 1,089 fingerprint-arm decisions used the prespecified E fallback because at least one trace lacked a complete fingerprint. AIT has 12 tasks and 10,900 positive/32,700 negative pair decisions per row, with no fallback. The AIT T increment is the full E+B+S+T row minus E+B+S: +0.98 pp BA [−0.21, +2.10], with FMR +0.09 pp and FSR −2.05 pp. Its interval crosses zero. This supports including T in the ablation, but not a reliable independent T gain. AIT time features come from recorded start times; the dataset's execution-mode labels remain confounded with task and collection workflow.

The machine-readable table and source hashes are in `research/results/matched_fingerprint_v2/unified_ablation/unified_bst_ablation.json` and `.csv`; regenerate with `build_unified_bst_ablation.py`. The CTH five-share comparison is separately preserved at `research/results/matched_fingerprint_v2/five_share/five_share_analysis.json`.

## Reproducibility and next checks

The frozen predictions, summary, thresholds, role assignments, training/validation pairs, code, and verification are under `research/results/matched_fingerprint_v2/argument_profile/` and `research/experiments/matched_fingerprint_v2/`. The raw argument profile is regenerated from local public CTH logs and never written as trace text. The experiment uses Python 3.10.16, NumPy 2.2.6, scikit-learn 1.7.2, and XGBoost 2.1.4.

Pair-type-specific thresholds have already been tested on CTH validation tasks. They reduce some false merges while increasing false splits, and sparse pair-type validation groups can yield all-reject behavior. This does not resolve the trade-off and does not support deployment of type-specific thresholds.

The next work is a protocol audit and a common result table, not another threshold search on NYU: verify the exact five-share CTH E vs E+B/S comparisons for all three learners; report HH/HA/AA counts, FMR, FSR, BA, paired changes and task intervals from the shared predictions; then verify the existing CTH B/S and AIT B/S/T ablation predictions, especially the AIT T increment and its timestamp coverage. Older CTH 40-task local-alignment and window explorations are development evidence and must remain labeled as such. If no eligible independent Human/Agent source remains, report that the NYU AA confirmation cannot validate the HA finding.
