# Research

The current direction is interpretable behavioral fingerprint matrices for trace association across security tasks, with Human/Agent estimation as a related endpoint. The normalized catalog now contains 1,142 records across a CTF core and vulnerability/command supplements. Coverage is documented separately from measured transfer performance; matrix-association results remain prospective. SVD, HOSVD, and TSA are outside this plan.

- [Redesigned dimensions, formulas, required telemetry, and literature sources](plans/human-agent-fingerprint-dimensions-v1.md)
- [Current-round complete TeX manuscript](paper/human-agent-fingerprint-main.tex)
- [Literature, log-feasibility decisions, and completed progress](reports/human-agent-fingerprint-round1-progress.md)
- [Expanded corpus, source contracts, and ingestion results](reports/behavior-corpus-expansion-v1.md)
- [Prepared overnight execution plan, fixed protocol, and delivery criteria](experiments/overnight_fingerprint_v1/program.md)
- [Actual overnight-study eligibility audit; no new model scores](results/overnight_fingerprint_v1/preflight.json)
- [Completed exploratory common-log pilot](reports/log-fingerprint-pilot-v1.md)
- [Public dataset availability checks and additional experimental sources](reports/public-dataset-availability-2026-11-01.md)

The literature inventory defines six feature families and 21 candidate scalars. The current primary log contract contracts this to eight core measurements: executable diversity, entropy, dominant share, action switching, conditional action entropy, longest action-run share, action-pair reuse, and adjacent executable repetition. Argument-count/option summaries are optional; shell syntax, completion feedback, resource-access behavior, and decision timing are deferred until matching fields are available. Feature selection and the inference schema must be fixed after development checks and before final evaluation. Neither a heatmap nor a missing field is evidence of Agent execution by itself.

## Longer-term controlled extension

Decision delegation and activity correlation remain a separate controlled-study draft, with no effectiveness results. They are not the current fingerprint-classification endpoint.

- [Delegation, fingerprinting, and correlation protocol](experiments/delegation_behavior_v1/program.md)
- [Experiment workspace and input audit](experiments/delegation_behavior_v1/README.md)
- [Existing-data eligibility audit](results/delegation_behavior_v1/input_audit.json)
- [24-dimension feature inventory](results/delegation_behavior_v1/feature_catalog.csv)

This extension asks whether matched delegation conditions change observable behavior and activity correlation. A shared CTF task is not an activity label or a delegation label.

## Earlier exploratory studies

The existing manuscript below concerns task-level retrieval of Human and Agent traces from public CTF tasks. It combines command TF--IDF similarity with IDF-weighted containment of distinctive artifact names. It is not a report of the new controlled study. Historical experiment outputs remain available; v15/v16 performance claims are not carried into the new protocol.

- [Method and evaluation report](reports/pairwise-trace-association.md)
- [Existing task-retrieval manuscript](paper/main.tex)
- [CTF trace set and pair browser](../site/dataset.html)
- [Research site](https://mcxiaoxiao.github.io/AttackAgentDetect/)

The evaluation covers 97 traces and 40 task groups. A same-task label means the records belong to the same CTF challenge; it does not identify a shared operator, campaign, or real-world attack. Human terminal records and Agent tool-call records use different collection interfaces, so execution mode cannot be separated from capture format in this dataset.
