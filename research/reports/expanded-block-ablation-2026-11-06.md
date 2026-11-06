# Expanded feature-block ablation (2026-11-06)

## Question and frozen design

This fixed batch completes the CTH single-block comparisons for all three learners and the AIT XGBoost factorial. Feature arms use the original task-disjoint folds, selected representations, learner parameters, candidate pairs, inner out-of-fold threshold rule, and pairwise E fallback. No representation, parameter, threshold rule, feature subset, or NYU result was used to choose an arm.

| Cohort | Task clusters | Trace/run count | Pair decisions per arm | Arm grid |
|---|---:|---:|---:|---|
| CTH mixed roles | 17 | 85 | 54,400 | E, E+B, E+S, E+B+S; RBF-SVM, tree, XGBoost |
| AIT audit logs | 12 | 109 | 43,600 | E plus all seven B/S/T1--T6 subsets; XGBoost |
| NYU Agent-only | 160 | 585 | 234,000 | Frozen E and E+B+S; three learners |

B denotes local behavior, S sequence, and T denotes only the six observed AIT launch-gap summaries T1--T6. T7--T8 remain unobserved. NYU predictions are unchanged from the frozen external analysis.

## Absolute task-macro outcomes

Cells are point estimate [paired 95% task-cluster interval], in percent. CTH first averages the five role shares within each task. AIT and NYU average within task. The same task clusters and frozen pair identities are used across arms.

| Source | Learner | Arm | BA | FMR | FSR | Hit@1 | Gallery AUC | Pair AP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| AIT | xgb | E | 50.12 [50.00, 50.36] | 0.00 [0.00, 0.00] | 99.76 [99.28, 100.00] | 40.20 [33.48, 47.27] | 63.34 [57.32, 69.39] | 38.25 [32.46, 44.88] |
| AIT | xgb | E+B | 50.94 [49.39, 52.68] | 2.78 [2.16, 3.39] | 95.34 [91.74, 98.44] | 34.20 [28.62, 39.50] | 57.47 [51.95, 62.44] | 33.35 [29.32, 37.27] |
| AIT | xgb | E+B+S | 50.94 [49.37, 52.58] | 5.28 [3.86, 6.87] | 92.84 [88.95, 96.57] | 33.33 [28.46, 37.70] | 57.87 [52.81, 62.29] | 32.60 [28.89, 36.20] |
| AIT | xgb | E+B+S+T1--T6 | 51.92 [50.59, 53.18] | 5.37 [3.65, 6.97] | 90.79 [87.81, 93.87] | 34.01 [29.13, 38.31] | 58.80 [53.85, 63.28] | 34.11 [30.63, 37.47] |
| AIT | xgb | E+B+T | 51.37 [50.17, 52.83] | 5.24 [3.15, 7.34] | 92.02 [87.72, 95.89] | 35.53 [30.00, 40.16] | 59.46 [53.78, 64.13] | 33.96 [30.23, 37.32] |
| AIT | xgb | E+S | 53.18 [50.78, 56.74] | 3.71 [1.90, 5.69] | 89.94 [81.83, 95.50] | 37.05 [32.83, 41.29] | 60.31 [55.60, 64.81] | 35.80 [32.63, 39.38] |
| AIT | xgb | E+S+T | 52.01 [51.08, 53.07] | 3.53 [2.29, 4.84] | 92.45 [90.23, 94.32] | 38.02 [33.47, 42.22] | 61.70 [56.90, 66.05] | 36.76 [33.42, 39.96] |
| AIT | xgb | E+T | 52.42 [50.60, 54.45] | 4.84 [3.09, 6.61] | 90.32 [85.77, 94.53] | 39.02 [34.15, 43.66] | 62.78 [58.21, 67.16] | 36.73 [32.84, 40.69] |
| CTH | svm | E | 52.57 [49.85, 55.85] | 6.52 [2.85, 10.59] | 88.35 [79.21, 95.80] | 33.75 [26.20, 41.66] | 58.73 [52.36, 65.16] | 39.78 [34.22, 45.83] |
| CTH | svm | E+B | 54.08 [50.62, 57.85] | 14.00 [8.75, 19.87] | 77.83 [68.32, 86.86] | 36.93 [31.52, 42.80] | 60.63 [56.82, 64.45] | 40.90 [36.65, 45.41] |
| CTH | svm | E+B+S | 53.95 [50.33, 57.91] | 13.50 [8.18, 19.22] | 78.60 [68.51, 88.10] | 38.45 [31.32, 45.62] | 62.22 [56.17, 68.21] | 42.66 [37.24, 48.28] |
| CTH | svm | E+S | 52.45 [49.62, 55.42] | 13.57 [9.83, 17.11] | 81.54 [73.66, 89.06] | 35.52 [28.20, 42.95] | 61.67 [55.48, 68.00] | 40.57 [35.41, 45.83] |
| CTH | tree | E | 52.80 [49.46, 56.75] | 6.59 [2.46, 10.88] | 87.81 [77.54, 96.98] | 34.91 [29.09, 40.82] | 60.30 [55.39, 65.12] | 35.21 [31.47, 39.34] |
| CTH | tree | E+B | 50.31 [49.71, 51.09] | 0.93 [0.21, 1.83] | 98.44 [96.14, 100.00] | 27.49 [22.11, 33.18] | 53.62 [48.04, 59.20] | 30.90 [27.83, 34.35] |
| CTH | tree | E+B+S | 50.00 [50.00, 50.00] | 0.00 [0.00, 0.00] | 100.00 [100.00, 100.00] | 36.27 [29.63, 42.87] | 59.77 [54.84, 64.64] | 36.44 [31.94, 41.09] |
| CTH | tree | E+S | 51.79 [49.74, 54.75] | 0.89 [0.16, 1.75] | 95.53 [88.87, 100.00] | 32.08 [24.88, 39.70] | 56.14 [50.70, 61.97] | 34.07 [29.57, 39.07] |
| CTH | xgb | E | 56.06 [52.45, 59.83] | 5.97 [3.69, 8.43] | 81.91 [73.65, 89.77] | 34.40 [27.92, 40.83] | 58.92 [53.47, 64.35] | 40.97 [36.41, 45.81] |
| CTH | xgb | E+B | 52.75 [49.74, 56.00] | 13.13 [10.36, 16.28] | 81.37 [72.84, 89.25] | 33.53 [27.77, 39.35] | 58.81 [53.39, 64.14] | 38.98 [34.35, 43.70] |
| CTH | xgb | E+B+S | 53.68 [50.67, 57.05] | 15.43 [11.45, 19.39] | 77.21 [67.81, 86.11] | 37.01 [30.60, 43.09] | 61.73 [55.81, 67.31] | 41.06 [35.93, 46.12] |
| CTH | xgb | E+S | 56.10 [51.92, 60.39] | 14.47 [8.89, 19.84] | 73.34 [61.61, 84.83] | 36.95 [30.87, 43.00] | 61.49 [56.45, 66.36] | 40.68 [35.39, 46.15] |
| NYU | svm | E | 58.16 [56.21, 60.22] | 26.58 [25.22, 27.88] | 57.09 [52.81, 61.18] | 39.73 [37.10, 42.47] | 65.75 [63.55, 67.94] | 42.77 [40.69, 44.97] |
| NYU | svm | E+B+S | 49.93 [49.91, 49.95] | 0.13 [0.10, 0.17] | 100.00 [100.00, 100.00] | 38.67 [36.04, 41.44] | 65.56 [63.35, 67.76] | 41.35 [39.37, 43.49] |
| NYU | tree | E | 49.69 [49.10, 50.32] | 3.85 [3.60, 4.11] | 96.78 [95.44, 98.01] | 36.92 [34.93, 39.03] | 63.94 [62.20, 65.69] | 35.29 [33.86, 36.83] |
| NYU | tree | E+B+S | 53.33 [51.83, 54.90] | 8.54 [7.66, 9.47] | 84.80 [81.30, 88.22] | 31.33 [29.10, 33.72] | 56.80 [54.89, 58.77] | 32.29 [30.77, 34.01] |
| NYU | xgb | E | 51.16 [49.89, 52.55] | 10.72 [10.15, 11.30] | 86.96 [84.16, 89.53] | 34.74 [32.06, 37.59] | 61.29 [59.12, 63.54] | 39.55 [37.64, 41.66] |
| NYU | xgb | E+B+S | 56.87 [55.06, 58.77] | 25.39 [23.91, 26.85] | 60.86 [56.55, 65.04] | 37.94 [35.46, 40.52] | 63.10 [60.84, 65.38] | 41.62 [39.76, 43.63] |

## Paired differences

Changes are percentage points; values are mean [paired 95% task-cluster interval]. Each feature arm is paired against the same learner/source E baseline. The separate AIT timing row compares E+B+S+T1--T6 against E+B+S.

| Source | Learner | Contrast | ΔBA | ΔFMR | ΔFSR | ΔHit@1 | ΔGallery AUC | ΔPair AP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| AIT | xgb | E->E+B | +0.82 [-0.81, +2.64] | +2.78 [+2.16, +3.39] | -4.42 [-8.12, -1.15] | -6.00 [-10.39, -1.89] | -5.86 [-9.07, -2.54] | -4.90 [-9.39, -0.80] |
| AIT | xgb | E->E+B+S | +0.82 [-0.92, +2.54] | +5.28 [+3.86, +6.87] | -6.92 [-10.95, -2.96] | -6.88 [-12.37, -1.46] | -5.47 [-9.19, -1.49] | -5.65 [-10.73, -0.95] |
| AIT | xgb | E->E+B+S+T1--T6 | +1.80 [+0.33, +3.16] | +5.37 [+3.65, +6.97] | -8.97 [-12.05, -5.78] | -6.19 [-12.31, -0.36] | -4.53 [-7.74, -1.17] | -4.14 [-10.45, +1.48] |
| AIT | xgb | E->E+B+T | +1.25 [-0.05, +2.78] | +5.24 [+3.15, +7.34] | -7.74 [-12.11, -3.81] | -4.67 [-10.62, +0.76] | -3.88 [-7.82, -0.14] | -4.29 [-10.00, +0.67] |
| AIT | xgb | E->E+S | +3.06 [+0.69, +6.61] | +3.71 [+1.90, +5.69] | -9.82 [-17.89, -4.38] | -3.15 [-7.63, +1.12] | -3.02 [-6.06, -0.34] | -2.45 [-6.65, +1.46] |
| AIT | xgb | E->E+S+T | +1.89 [+0.83, +3.03] | +3.53 [+2.29, +4.84] | -7.31 [-9.61, -5.36] | -2.18 [-7.08, +2.56] | -1.64 [-4.56, +1.18] | -1.49 [-7.46, +3.99] |
| AIT | xgb | E->E+T | +2.30 [+0.50, +4.32] | +4.84 [+3.09, +6.61] | -9.44 [-13.84, -5.33] | -1.19 [-5.89, +3.20] | -0.56 [-3.58, +2.04] | -1.52 [-7.20, +3.59] |
| CTH | svm | E->E+B | +1.51 [-3.46, +6.14] | +7.49 [-0.76, +16.16] | -10.51 [-24.27, +4.18] | +3.18 [-5.29, +11.53] | +1.90 [-3.95, +8.18] | +1.13 [-3.99, +5.97] |
| CTH | svm | E->E+B+S | +1.38 [-3.81, +6.50] | +6.99 [-1.50, +15.68] | -9.74 [-24.65, +5.60] | +4.70 [-2.92, +13.04] | +3.49 [-2.40, +10.84] | +2.88 [-2.51, +8.78] |
| CTH | svm | E->E+S | -0.12 [-3.94, +3.54] | +7.05 [+0.75, +13.15] | -6.81 [-17.21, +3.72] | +1.77 [-6.64, +10.92] | +2.94 [-3.48, +10.53] | +0.79 [-4.96, +7.08] |
| CTH | tree | E->E+B | -2.49 [-6.51, +1.00] | -5.66 [-10.36, -1.24] | +10.63 [+1.22, +21.49] | -7.41 [-13.02, -1.85] | -6.69 [-12.02, -1.49] | -4.31 [-7.86, -0.87] |
| CTH | tree | E->E+B+S | -2.80 [-6.75, +0.54] | -6.59 [-10.88, -2.46] | +12.19 [+3.02, +22.46] | +1.37 [-6.16, +9.24] | -0.53 [-6.26, +5.51] | +1.23 [-2.73, +5.19] |
| CTH | tree | E->E+S | -1.01 [-5.75, +3.54] | -5.70 [-10.39, -1.29] | +7.72 [-4.25, +19.93] | -2.83 [-9.78, +4.07] | -4.17 [-9.39, +1.42] | -1.15 [-5.64, +3.13] |
| CTH | xgb | E->E+B | -3.31 [-7.96, +1.49] | +7.16 [+3.90, +10.64] | -0.54 [-10.18, +8.91] | -0.87 [-7.84, +6.23] | -0.11 [-5.34, +5.25] | -1.99 [-6.99, +3.09] |
| CTH | xgb | E->E+B+S | -2.38 [-7.15, +2.41] | +9.46 [+5.55, +13.66] | -4.70 [-15.36, +6.02] | +2.61 [-5.15, +10.96] | +2.81 [-3.74, +9.93] | +0.08 [-5.03, +5.55] |
| CTH | xgb | E->E+S | +0.04 [-3.11, +3.53] | +8.50 [+4.73, +12.22] | -8.57 [-17.29, -0.51] | +2.55 [-3.17, +9.03] | +2.57 [-2.79, +8.24] | -0.30 [-4.22, +3.90] |
| NYU | svm | E->E+B+S | -8.23 [-10.29, -6.27] | -26.45 [-27.75, -25.09] | +42.91 [+38.82, +47.19] | -1.06 [-3.73, +1.43] | -0.18 [-2.05, +1.55] | -1.41 [-3.36, +0.42] |
| NYU | tree | E->E+B+S | +3.64 [+1.94, +5.36] | +4.69 [+3.68, +5.74] | -11.98 [-15.88, -8.10] | -5.59 [-8.15, -3.07] | -7.13 [-9.29, -4.99] | -3.00 [-4.66, -1.38] |
| NYU | xgb | E->E+B+S | +5.71 [+3.63, +7.81] | +14.67 [+13.30, +16.06] | -26.10 [-30.80, -21.34] | +3.20 [+0.49, +5.87] | +1.82 [-0.39, +4.02] | +2.07 [+0.03, +4.05] |
| AIT | xgb | E+B+S->E+B+S+T1--T6 | +0.98 [-0.19, +2.13] | +0.09 [-1.34, +1.58] | -2.05 [-4.41, +0.35] | +0.68 [-1.50, +2.71] | +0.93 [-0.79, +2.66] | +1.50 [-0.67, +3.43] |

## Fold configurations and thresholds

Representations and parameters below are the original frozen selections. Added arms use those settings; only their inner OOF thresholds are fit separately under the same pooled FMR ≤10% rule. `∞` means the maximum-recall rule selected the all-reject threshold. OOF FMR includes the corresponding E fallback.

| Source | Fold | Train/test tasks | Representation | Learner | Parameter | Arm thresholds (OOF FMR; test fallback pairs; all reject) |
|---|---:|---:|---|---|---:|---|
| CTH | 0 | 11/6 | 4/mean_std/conditional | xgb | 2 | evidence: t=0.78046143055, fallback=0; BS: t=0.800316035748, fallback=1089; B: t=0.76122123003, fallback=1089; S: t=0.682949244976, fallback=1089 |
| CTH | 0 | 11/6 | 4/mean_std/conditional | svm | 10 | evidence: t=2.002646748, fallback=0; BS: t=0.914564361142, fallback=1089; B: t=1.00044209244, OOF FMR=7.27%, fallback=1089, all-reject=False; S: t=1.05298812122, OOF FMR=9.09%, fallback=1089, all-reject=False |
| CTH | 0 | 11/6 | 4/mean_std/conditional | tree | 3 | evidence: t=+∞, fallback=0; BS: t=+∞, fallback=1089; B: t=1, OOF FMR=3.03%, fallback=1089, all-reject=False; S: t=0.928571428571, OOF FMR=7.27%, fallback=1089, all-reject=False |
| CTH | 1 | 11/6 | 4/mean_std_trend/direct | xgb | 3 | evidence: t=0.702958643436, fallback=0; BS: t=0.695187151432, fallback=0; B: t=0.712531864643, fallback=0; S: t=0.724367678165, fallback=0 |
| CTH | 1 | 11/6 | 4/mean_std_trend/direct | svm | 0.1 | evidence: t=0.284884886039, fallback=0; BS: t=0.35829427589, fallback=0; B: t=0.306636419817, OOF FMR=8.48%, fallback=0, all-reject=False; S: t=0.362614554008, OOF FMR=7.88%, fallback=0, all-reject=False |
| CTH | 1 | 11/6 | 4/mean_std_trend/direct | tree | 8 | evidence: t=1, fallback=0; BS: t=+∞, fallback=0; B: t=+∞, OOF FMR=1.82%, fallback=0, all-reject=True; S: t=+∞, OOF FMR=1.82%, fallback=0, all-reject=True |
| CTH | 2 | 12/5 | 4/mean_std_trend/direct | xgb | 3 | evidence: t=0.731237053871, fallback=0; BS: t=0.78592556715, fallback=0; B: t=0.804004192352, fallback=0; S: t=0.825489342213, fallback=0 |
| CTH | 2 | 12/5 | 4/mean_std_trend/direct | svm | 10 | evidence: t=1.27431908227, fallback=0; BS: t=0.710048727838, fallback=0; B: t=0.784253232239, OOF FMR=10.00%, fallback=0, all-reject=False; S: t=0.657695696331, OOF FMR=7.78%, fallback=0, all-reject=False |
| CTH | 2 | 12/5 | 4/mean_std_trend/direct | tree | 3 | evidence: t=0.842105263158, fallback=0; BS: t=+∞, fallback=0; B: t=0.928571428571, OOF FMR=8.89%, fallback=0, all-reject=True; S: t=+∞, OOF FMR=0.00%, fallback=0, all-reject=True |
| AIT | 0 | 8/4 | 4/mean_std/conditional | xgb | 3 | evidence: t=0.884359478951, fallback=0; BST: t=0.83963906765, fallback=0; ST: t=0.873307108879, fallback=0; BT: t=0.838442862034, fallback=0; BS: t=0.864110469818, fallback=0; B: t=0.850774884224, OOF FMR=9.39%, fallback=0, all-reject=False; S: t=0.864318966866, OOF FMR=8.92%, fallback=0, all-reject=False; T: t=0.839698374271, OOF FMR=8.92%, fallback=0, all-reject=False |
| AIT | 1 | 8/4 | 4/mean_std/conditional | xgb | 2 | evidence: t=0.835576653481, fallback=0; BST: t=0.856556415558, fallback=0; ST: t=0.869694530964, fallback=0; BT: t=0.848543941975, fallback=0; BS: t=0.865038335323, fallback=0; B: t=0.858592569828, OOF FMR=9.13%, fallback=0, all-reject=False; S: t=0.854014754295, OOF FMR=9.59%, fallback=0, all-reject=False; T: t=0.868388593197, OOF FMR=8.22%, fallback=0, all-reject=False |
| AIT | 2 | 8/4 | 4/mean_std/conditional | xgb | 3 | evidence: t=0.85781621933, fallback=0; BST: t=0.836960554123, fallback=0; ST: t=0.850146591663, fallback=0; BT: t=0.801808357239, fallback=0; BS: t=0.790339589119, fallback=0; B: t=0.810515522957, OOF FMR=9.46%, fallback=0, all-reject=False; S: t=0.801614403725, OOF FMR=9.46%, fallback=0, all-reject=False; T: t=0.821697771549, OOF FMR=9.91%, fallback=0, all-reject=False |

## Fallback, ties, and all-reject checks

Decision metrics use pairwise E fallback as in the frozen classifier. Ranking uses a separate gallery-level rule: if one of the four candidate pairs lacks a required block, all four ranking scores are replaced by matched E scores so a gallery is never ranked on mixed score scales. Hit@1 awards fractional credit for ties; gallery AUC gives half credit to equal positive/negative scores. Pair AP is calculated within query task and, for CTH, role share.

| Source | Learner | Arm | Pair fallback rows | Gallery fallbacks / galleries | Positive tied galleries | Any-score-tie galleries | Threshold rejects all? |
|---|---|---|---:|---:|---:|---:|---|
| AIT | xgb | E | 0 | 0 / 10900 | 454 | 1155 | False |
| AIT | xgb | E+B | 0 | 0 / 10900 | 9 | 15 | False |
| AIT | xgb | E+B+S | 0 | 0 / 10900 | 0 | 0 | False |
| AIT | xgb | E+B+S+T1--T6 | 0 | 0 / 10900 | 0 | 0 | False |
| AIT | xgb | E+B+T | 0 | 0 / 10900 | 0 | 2 | False |
| AIT | xgb | E+S | 0 | 0 / 10900 | 0 | 0 | False |
| AIT | xgb | E+S+T | 0 | 0 / 10900 | 0 | 3 | False |
| AIT | xgb | E+T | 0 | 0 / 10900 | 6 | 8 | False |
| CTH | svm | E | 0 | 0 / 13600 | 2803 | 4307 | False |
| CTH | svm | E+B | 1089 | 723 / 13600 | 2803 | 4305 | False |
| CTH | svm | E+B+S | 1089 | 723 / 13600 | 2803 | 4305 | False |
| CTH | svm | E+S | 1089 | 723 / 13600 | 2803 | 4305 | False |
| CTH | tree | E | 0 | 0 / 13600 | 8342 | 11625 | False |
| CTH | tree | E+B | 1089 | 723 / 13600 | 8836 | 12682 | False |
| CTH | tree | E+B+S | 1089 | 723 / 13600 | 8114 | 11992 | True |
| CTH | tree | E+S | 1089 | 723 / 13600 | 8921 | 12302 | False |
| CTH | xgb | E | 0 | 0 / 13600 | 3449 | 5536 | False |
| CTH | xgb | E+B | 1089 | 723 / 13600 | 2938 | 4609 | False |
| CTH | xgb | E+B+S | 1089 | 723 / 13600 | 2948 | 4661 | False |
| CTH | xgb | E+S | 1089 | 723 / 13600 | 3197 | 4912 | False |
| NYU | svm | E | 0 | 0 / 58500 | 280 | 1131 | False |
| NYU | svm | E+B+S | 0 | 0 / 58500 | 442 | 1366 | False |
| NYU | tree | E | 0 | 0 / 58500 | 45069 | 56836 | False |
| NYU | tree | E+B+S | 0 | 0 / 58500 | 42687 | 56782 | False |
| NYU | xgb | E | 0 | 0 / 58500 | 4378 | 8101 | False |
| NYU | xgb | E+B+S | 0 | 0 / 58500 | 388 | 1604 | False |

## Integrity checks and reproducibility

- Every arm was checked against the same pair identity set, including fold, pair ID, query/candidate IDs and tasks, label, and candidate position.
- Outer training and test task lists were checked disjoint; held-out candidate tasks equal that fold's frozen test tasks.
- Recreated E scores and thresholds were checked against the frozen per-fold E files before added predictions were accepted.
- Added-arm OOF pooled FMR was checked against the 10% constraint, with the original E fallback applied.
- Vectorized aggregation was checked against the prior aggregation implementation on one task each from CTH, AIT, and NYU; all six outcomes matched exactly.
- NYU is evaluation only: no new fitting, feature choice, or threshold selection was performed on it.

Bootstrap: 10,000 paired task-cluster resamples, seed `20261106`. Model/pair seed: `20261103`. Repeated decisions are not independent samples.

### Provenance hashes

- `attackagent/fingerprint_v2.py`: `9c3141e1a8d5b7f463807131120b9c1209d5e3741080d93da9c2eabdf928d2e7`
- `research/data/external/ait-shell-pilot-v1/data.zip`: `f3ff78d7fcb7db9068d6b0c6057659412c725830d0b05fb64cea319f0e638914`
- `research/experiments/matched_fingerprint_v1/run.py`: `6a377798928cbf1217bec0568366840a99fa2ca4aaf556065d122ee5630c617a`
- `research/experiments/matched_fingerprint_v2/analyze_frozen_ranking_metrics.py`: `a63e7bca9294868c604d36cb260a1bf7d3a192d9ff04fc81938091c5daf29d21`
- `research/experiments/matched_fingerprint_v2/expanded_block_ablation_20261106_protocol.md`: `f231cf07c34f15dcc3b948af231714f6e1c815b8d933f71fc0af981ff30221c8`
- `research/results/behavior_corpus_v1/evidence.jsonl`: `090e2f90c30dbf19db6e56e4ecafea81a356b9db98e455a303f18ebce910e01c`
- `research/results/behavior_corpus_v1/trace_registry.csv`: `571527d9acb8161488a39dd7e497b94bd433edc58452f497663705360e0ec2e5`
- `research/results/matched_fingerprint_v1/ait/fold-0-audit.json`: `e548e066afe6115ea91b497d5870bb474751b20dc5011269b0d9e5e5418b38e4`
- `research/results/matched_fingerprint_v1/ait/fold-1-audit.json`: `52d12a66987c54de6e102856c2a0457d643df849169be097b196dc0e5110c342`
- `research/results/matched_fingerprint_v1/ait/fold-2-audit.json`: `b5a4db9f650e3812b8075e45a6d895eb3f08ab09fbdafa5d1830be5b9a5ef336`
- `research/results/matched_fingerprint_v1/ait/pairs.csv.gz`: `b7ec9b6dcb4e5f0b6384f55a8e3455caee2cb58b70476f7682b1a8c80724a97a`
- `research/results/matched_fingerprint_v1/cth/fold-0-audit.json`: `8f3a7dc824910b28493881034875b853b94bc1b99c89c01a301a8fda1cb5aa6d`
- `research/results/matched_fingerprint_v1/cth/fold-1-audit.json`: `40a207768a818044c3cc55f660492cffcad916e01f73ab2ab416029f315b5594`
- `research/results/matched_fingerprint_v1/cth/fold-2-audit.json`: `5312aee29ad6a8639a9abb7f4e29af20723fe64c01db8c096c83d177cb6f24fa`
- `research/results/matched_fingerprint_v1/cth/pairs.csv.gz`: `e52c4f22c06e99983f30ec63f66d92d3e8454245d00260adf6388e406873acab`
- `research/results/matched_fingerprint_v1/pair_predictions.csv.gz`: `c6af21f2111738decbb32cdbbeb56ec7ef31e956d256576e1069b9e5bb234d90`
- `research/results/matched_fingerprint_v2/independent_nyu_agent/predictions.csv.gz`: `208df72a15180c221ed63312cd3bbd0f12a592f37d58c1230bd1f4fce90adc8d`
- `research/results/overnight_fingerprint_v1/task_folds.csv`: `1c07850d20282ed84b6ac2be14ecde9f0981d368cfb1dba0d5b64520bd48868d`

- Analysis script: `research/experiments/matched_fingerprint_v2/run_expanded_block_ablation.py` — `4de4fc1ba4842463f53cec01fb72e9248d4f5652b9f293126037919fb9a63df7`
- Frozen runner: `research/experiments/matched_fingerprint_v1/run.py` — `6a377798928cbf1217bec0568366840a99fa2ca4aaf556065d122ee5630c617a`
- Protocol: `research/experiments/matched_fingerprint_v2/expanded_block_ablation_20261106_protocol.md` — `f231cf07c34f15dcc3b948af231714f6e1c815b8d933f71fc0af981ff30221c8`
- Merged per-arm predictions (research/results/matched_fingerprint_v2/expanded_ablation_20261106/expanded_predictions.csv.gz): `314552956a7d561942eeb131adfec87895b586864903e916dbba17400a722167`
- Per-fold added prediction files, their hashes, all thresholds, and task-level estimates are recorded in `research/results/matched_fingerprint_v2/expanded_ablation_20261106/analysis.json` and `run_manifest.json`.

## Interpretation

The expanded comparison sharpens rather than overturns the existing result. CTH B/S effects vary by learner; thresholded improvements exchange FMR for FSR in different ways, and the decision tree's full arm rejects all candidates. AIT's T1--T6 increment over B/S is small and its paired intervals include zero; all factorial AIT arms have lower ranking point estimates than E. The frozen NYU XGBoost comparison improves ranking and BA while exceeding the CTH 10% calibration target on FMR. These are task-retrieval observations under the named collection and candidate protocols, not evidence of actor or campaign identity.
