# Matched fingerprint study: final acceptance record

Date: 2026-11-04
Branch at experiment start: `main`
Starting commit: `7b93a87667f6145ed89d1eed82b7154663f99f4f`

## Outcome

On the frozen external NYU Agent–Agent evaluation, B/S raises balanced accuracy by 5.71 percentage points for XGBoost and 3.64 points for the decision tree, with fewer missed links and more false merges. SVM loses 8.23 points on the same pairs. The CTH five-share comparison and AIT timing ablation leave their respective BA increments uncertain. These results do not meet the prespecified requirement for stable gains with both error rates decreasing. The evaluated relation is benchmark task identity.

## Experimental evidence and provenance

- **CTH:** 17 task groups, 85 traces, 170 coupled samples, exact 0/25/50/75/100% Agent-role shares, and one same-task plus three different-task candidates per query. Matched classifier comparisons used the same 54,400 candidate pairs. Results and 10,000 task-cluster resamples are in `research/results/matched_fingerprint_v2/five_share/`.
- **AIT:** 109 independent runs across 12 tasks; same-task retrieval uses distinct runs. The 218,000 frozen predictions cover 43,600 candidate pairs shared across the ablation. T uses observed time fields only. Labels remain confounded with task and collection workflow.
- **External NYU AA:** frozen before fitting, with 160 task groups, 585 runs, 234,000 candidate pairs, 58,500 query galleries, and 1,404,000 predictions. The task-group split, pair equality, zero fallback, threshold arithmetic, and prediction manifest passed verification. This is a one-shot same-challenge retrieval proxy, not Human–Agent or verified attack-activity confirmation.
- **Key result files:** CTH five-share analysis SHA-256 `026600a75fb39a0316c8f0cae9b311e5148a1f5c9880604e41198875ffae6e1d`; unified CTH/AIT ablation SHA-256 `9c0f4279bc38f18c72561e7536dcd346de299a7282f111e801675446afd96a0e`; NYU analysis SHA-256 `86a65cdb6a90b27699be48791b39a1dfc315ebe25a251f14da16ab28365b8451`.
- **NYU frozen artifacts:** pair manifest SHA-256 `a25ad991e2d20f27e23f48d4e56d6e63e225921eacb6f8888805df6c3b6a0625`; predictions SHA-256 `208df72a15180c221ed63312cd3bbd0f12a592f37d58c1230bd1f4fce90adc8d`.
- **Result builder:** `research/experiments/matched_fingerprint_v2/build_paper_results.py`, SHA-256 `7a9aa4acc860f8d4f34ea980175d16dbbdf773324463df1150e84e2ce1fe98aa`. It derives the manuscript tables, plots, and bilingual page data from frozen result JSON; SVG output keeps vector geometry while trimming redundant path whitespace.

## Manuscript and page checks

- `research/paper/human-agent-fingerprint-draft.tex` retains the original 10 pt type and 0.72 in margins. The references end on page 8. The complete PDF has 13 pages. Three main-text tables give the evaluation cohorts, available fields, and the four NYU submissions. The matched ablation result is reported in prose and the full results are available in the research artifacts.
- The draft uses all eight original figure assets and the owl logo from the overnight draft. The main text contains the overview, B/S/T formula atlas, and CTH baseline triptych. The appendices group the remaining figures with their feature definitions and evaluation protocols. The overnight source preserves its original table blocks. The author line reads AttackAgent Detect.
- The PDF was compiled with Tectonic and rendered page-by-page for visual inspection. No missing figures, unresolved references, clipped tables, or visible overflow were found. Two bibliography underfull-box notices remain; they do not affect rendered legibility.
- The source ZIP was extracted into a clean temporary directory and compiled independently. It contains the manuscript sources, referenced figures, frozen predictions/manifests, analyses, verifiers, and research notes; raw AIT logs and raw command text are excluded.
- The local bilingual page was checked in Chinese and English at desktop and narrow viewport sizes. Browser console returned no errors or warnings. Document width matched the viewport; wide tables remain horizontally scrollable within their containers.
- Static scan found 52 valid local file references, 16 external references, and no broken local files or anchors. The local page, PDF, TeX, and ZIP download endpoints each returned HTTP 200.

## Automated verification

- `PYTHONPATH="$PWD/research:$PWD" .venv/bin/python -m unittest discover -s tests`: 50 tests passed. The required Homebrew `libomp` runtime was installed after an initial environment-only OpenMP import failure.
- v1 verification checked 435,200 CTH and 218,000 AIT predictions; 850 exact role allocations; task separation; and baseline metric arithmetic (maximum floating-point discrepancy below `6.7e-16`).
- NYU verification checked the frozen task/pair manifests, no fallback, candidate isolation, and threshold calculations.
- The source-bundle build independently recompiles the extracted archive and completed successfully.

## Final artifact digests

The current PDF and TeX digests are SHA-256 `33044637543d95ff38186aa4e5099c95b7aae5321986d0b1a8cc27b41a22b635` and `2ba685c8b6c0d3c50ba871dc8dac91f76563f7132850d0019e2d7c6d7080bf03`. The independently compilable 111-file source ZIP is 58,922,574 bytes. Its SHA-256 is `80cb6a1a5865fd40d2a6db65dfbaf2cf6ce9c850fb377dfab6d1782b5164c0e4`.

## Limits retained in the paper

CTH and AIT labels support task-level retrieval only. HH/HA/AA identify the roles of the two traces and do not establish relationships between real attack activities or actors. NYU supplies only a frozen Agent–Agent challenge-retrieval check; independent Human–Agent confirmation is unavailable. The observed gains and error costs vary by classifier and source, so neither general improvement nor general failure is claimed.
