# Same-challenge retrieval short-paper review — 2026-11-05

## Review decision

The manuscript is now framed as a bounded measurement study of same-task retrieval. It does not claim that benchmark task identity establishes a shared operator, Agent, campaign, or real-world activity. Its central result is a role-composition-associated shift in fixed-threshold errors; ranking changes are reported separately. The evidence does not support a source-independent behavioral-fingerprint claim.

## Main measurements

- Across 17 CTH tasks and five paired role shares, the action-bigram baseline's task-macro FMR moves from 32.27% to 9.18%, while FSR moves from 52.94% to 97.87%. The paired endpoint changes are −23.09 pp FMR [−39.88, −8.36], +44.93 pp FSR [+21.40, +68.46], and −10.92 pp BA [−20.56, −2.90]. Hit@1 changes by −0.05 pp [−12.96, +12.91] and gallery AUC by −11.48 pp [−25.29, +3.83]. The threshold error shift is the clearest measurement; the ranking contrasts remain uncertain.
- The CTH XGBoost B/S arm shifts errors in a different direction from the cosine baseline: its equal-share FMR rises 9.46 pp and FSR falls 4.70 pp, for a BA change of −2.38 pp [−7.29, +2.30].
- In the single frozen five-arm mechanism batch, adding command and record counts to E changes BA by −3.04 pp. Adding normalized command LCS to that control lowers FMR by 3.21 pp [−5.28, −1.35], but BA, Hit@1, and AUC intervals include zero. Adding real B/S to the count control changes BA by −0.79 pp [−4.96, +3.51], Hit@1 by −0.93 pp [−5.09, +3.30], and AUC by −0.84 pp [−6.78, +5.04]. No shuffled-order seed separates from real B/S on BA, Hit@1, or AUC at the equal-share mean.
- Frozen Agent-only comparisons have different error trade-offs. CTH AA B/S decreases FMR by 6.14 pp and increases FSR by 13.03 pp, with an uncertain BA change of −3.44 pp. NYU AA increases FMR by 14.67 pp and decreases FSR by 26.10 pp, for a BA gain of 5.71 pp; its test FMR is 25.39% despite CTH's 10% calibration constraint. The two evaluations also use different training-pair construction, so their contrast is not attributable to source alone.

## Audit findings and limits

- The CTH mixture uses 17 tasks, 85 traces, 170 paired allocations, five role shares, and 54,400 pair decisions. Human terminal input and Agent tool calls use different collection interfaces. The data do not establish that Human participants never used AI, so role share is not a causal execution-mode manipulation.
- Leave-one-task-out analysis keeps the action-bigram FMR decrease and FSR increase in all 17 omissions. Task-disjoint folds nevertheless share exact normalized command-head sequences: 9/30, 5/30, and 16/25 held-out traces by fold. Removing every complete gallery containing an overlap retains 2,114/13,600 galleries and 15 tasks overall, but only five tasks are available at both endpoint shares. The filtered action-bigram FMR endpoint difference is zero and FSR rises 20 pp with wide intervals. This sensitivity is too small to resolve the full-cohort pattern.
- The task-cluster intervals resample query tasks. Candidate tasks recur across query galleries, so the bootstrap does not capture all dependence. The 17-task CTH sample remains the effective unit for the mixture claim.
- The 24-value B/S/T number is the count of per-window measurements, not a universally observed or validated model dimension. CTH and NYU have no event times and use B/S only; pooling and pairwise differences make the fitted input wider. AIT's timing increment is uncertain. Nested deletion selects different measurements by fold and does not justify a stable reduced subset.
- The main paper omits tests with a different target relation or incompatible protocol. The supplement reports the frozen snapshot and dimension checks, AIT, full mixture curves, threshold diagnostics, overlap sensitivity, mechanism seeds, and source-specific outcomes.

## Deliverables

- Main article: `research/paper/same-challenge-retrieval-shortpaper.tex` and its four-page PDF.
- Supplement: `research/paper/same-challenge-retrieval-supplement.tex` and its four-page PDF.
- Frozen analysis protocol: `research/experiments/matched_fingerprint_v2/shortpaper_review_protocol.md`.
- Point-estimate verifier: `research/experiments/matched_fingerprint_v2/verify_shortpaper_tables.py` re-aggregates the included frozen predictions without refitting.
- Local source archive: `research/reproducibility/same-challenge-retrieval-20261105.zip`. It excludes raw NYU transcripts and raw command content. Rebuilding predictions from source logs still requires the upstream source data and pinned repositories.

## Verification performed

Both TeX files compile independently with Tectonic; the main manuscript is four pages including references and the supplement is four pages. The figures and all pages were rendered for visual inspection. A standard-library re-aggregation script checks the paper's main mixture values, CTH matched-model rates, Agent-only task-confirmation rates, and five-arm mechanism point contrasts from frozen pair predictions. The archive is also extracted to a clean temporary directory, recompiled, and re-aggregated before delivery.
