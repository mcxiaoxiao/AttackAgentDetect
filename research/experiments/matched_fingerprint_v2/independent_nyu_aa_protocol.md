# Independent NYU CTF Agent–Agent confirmation

## Freeze and purpose

This is a one-time cross-source confirmation of the CTH-trained association models on previously unused NYU CTF challenge groups. The held-out source is the public `NYU-LLM-CTF/leaderboard_submissions` repository at revision `69def02a79e8b1d890e1e0d5c11106a9f40e11e1`. The external results must not be used to select features, model parameters, thresholds, or task exclusions. The eligible task and trace list is frozen separately in `research/results/matched_fingerprint_v2/independent_nyu_task_manifest.json` before model fitting.

The evaluation label is whether two submitted traces have the same challenge filename. It is a challenge-retrieval proxy, not ground truth for a shared attack activity, operator, Agent instance, or campaign. Every record is Agent-generated (AA); this dataset cannot estimate HH or HA errors.

## External trace cohort and exclusions

Use only four D-CIPHER submission snapshots with a shared transcript structure: `dcipher_claude3.5_sonnet`, `dcipher_claude3.7_sonnet`, `dcipher_gpt4.1`, and `dcipher_gpt4o`. A trace is the ordered sequence of `run_command` tool calls from one challenge file. `create_file`, `finish_task`, tool outputs, summary success labels, and model/submission identifiers are not model features. The same command-head parser and action map used for CTH are applied; command values are not retained in outputs. Each trace is capped at the first 128 recognized command heads and must have at least two events with complete B/S features under a four-event window.

Before fitting, exclude challenge slugs found in any prior CTH association output, including the already opened 26-task CTH AA confirmation. Within each remaining challenge, remove exact duplicate command-head sequences. The audit identified 196 challenge slugs, 36 previously used slugs, and 160 eligible challenge groups with 585 distinct traces after removing one exact duplicate sequence. The deterministic SHA-256 partition counts are 48, 50, and 62 tasks. These counts and every source file SHA-256 are recorded in the frozen manifest. Raw submitter logs are read from a local checkout and are not redistributed.

## Models and training

The development set remains the 17 mixed-role CTH tasks and 85 traces. For each training fold, generate every distinct within-task positive pair and every distinct cross-task negative pair. Apply sample weights that give equal total mass to the two labels and equal mass to each observed HH/HA/AA label cell. The total effective weight in a fold equals the original sampled-pair count; the two feature arms share the same pairs and weights. This tests broader pair coverage without changing the class or pair-type prior.

Use the frozen four-event, mean-plus-standard-deviation, command-conditioned B/S representation. Classifier settings are fixed from the majority of the prior CTH fold selections: RBF-SVM C=1, decision tree depth 3, and XGBoost depth 3. Fit one model with E (command-head and action-bigram cosine evidence) and one with E+B/S. For a missing external B/S pair, use the corresponding E decision; the manifest and predictions record fallback.

For each classifier and feature arm, choose the decision threshold only from three-fold task-held-out CTH scores: maximize positive-pair recall subject to pooled validation FMR at most 10%. Refit on all 17 CTH development tasks after fixing thresholds. No NYU challenge or outcome enters fitting, threshold selection, or model choice.

## Candidate panel and statistics

For each of 100 seeded gallery draws, every eligible trace is a query. Select one different submission trace from the same challenge as the positive candidate and three traces from three different challenges in the query's SHA-256 task partition as negatives. All models and feature arms share the exact pair manifest. This produces 234,000 pairs per version and 1,404,000 predictions across three classifiers and two feature arms.

Report task-macro FMR, FSR, and BA at the frozen thresholds. Also report task-macro Hit@1, MRR, and pairwise AUC to separate score ordering from the selected operating point. Use 10,000 paired bootstrap resamples of the 160 challenge groups; the 100 repeated galleries are averaged within task and are not independent samples. Save corrected and introduced error counts on the identical candidate pairs.

## Interpretation limits

This confirms transfer only to new CTF challenge groups from the NYU submission repository and only for four model snapshots of the D-CIPHER framework. It does not establish transfer across Agent frameworks, Human/Agent mixtures, attack activity identity, or real-world actor attribution. The CTH external AA panel is not reused to confirm the all-pairs training revision.
