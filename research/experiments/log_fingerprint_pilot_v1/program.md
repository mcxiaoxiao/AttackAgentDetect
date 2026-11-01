# Common-log Human/Agent fingerprint pilot v1

Frozen before classification: 2026-11-01 (Asia/Shanghai).

## Objective and bounded batch

Test whether the proposed eight observable statistics warrant further work on common execution logs. Primary readout: out-of-fold balanced accuracy on held-out application/scenario pairs, compared with length and lexical baselines. Secondary readouts: macro-F1, confusion matrix, Human false-positive rate, Agent false-negative rate, AUROC where both classes exist, and descriptive Brier score of uncalibrated estimates. No parameter search, feature selection, threshold fitting, or external final-test claim. One batch of three fixed methods and four observation views; no paid APIs; maximum 20 minutes of model computation. Stop on malformed units, split leakage, or invalid feature definitions rather than silently changing the protocol.

## Inputs, versions, and independent units

- Public AIT/AECID `llm-user-shell-sim/data.zip`, downloaded from main on 2026-11-01. Archive SHA256: `f3ff78d7fcb7db9068d6b0c6057659412c725830d0b05fb64cea319f0e638914` (13,692,715 bytes). A content hash pins this copy even if an upstream commit lookup fails.
- Use only original `data/{Nextcloud,WordPress}/LOGS_Result_<Actor>/expN/audit.log` files, never aggregated copies. Inventory: 109 raw run intervals, 61 Human and 48 Agent folder labels. These are troubleshooting runs, not attack sessions.
- Ground-truth provenance follows the upstream documented folder convention (`GPT` means AI; other listed participant folders mean Human). This is source annotation, not independent verification of each process's decision-maker. No folder text or identity enters features.
- Agent suffixes `_V2` and `_V3` are grouped with their base model name. Repetitions are not independent unseen models. Human aliases stay grouped across applications; whether they represent the same participant is treated conservatively by retaining the grouping.
- Unit = original per-run interval. It may contain background processes; it is not automatically a clean actor-only session. Join audit records by node/timestamp/serial, remove duplicate records, retain successful EXECVE/execveat event launches only. Never count every syscall as a command. No final exit status is inferred.
- Prefer `SYSCALL.exe` as actual executable image; fall back to EXECVE a0, decode audit hex, take lowercase POSIX basename. Keep unknown command names in place. Use the existing `attackagent.traces.ATOM_GROUP` map without edits or new outcome-selected aliases.

## Fixed observation views and admission

1. **login_exec** (primary): successful executable events with a non-unset audit login ID and session ID. Their values do not enter the classifier. This is an attribution heuristic, not proof of actor ownership.
2. **all_exec** (sensitivity): all successful executable events in the original interval, including background processes.
3. **login_no_wrappers** (sensitivity): primary stream excluding a fixed set `sudo, su, sh, bash, dash, env, sshd, systemd-executor`. These exclusions test launch-plumbing dependence; they do not establish complete nuisance removal.
4. **login_first50** (budget check): primary streams of at least 50 executable events truncated to the first 50 before feature computation; no session contributes multiple windows.

For all views admit only streams with at least 10 executable events and at least five adjacent pairs with two mapped actions. Report exclusions by class/application. Do not lower these limits to improve retention. A view with insufficient two-class training data is reported unavailable.

## Fixed features and models

- K1 executable TTR; K2 executable Shannon entropy; K3 maximum executable frequency.
- K4 action-switch rate; K5 next-action conditional entropy using the outgoing marginal over eligible pairs; K6 longest known contiguous action run divided by all executable events; K7 action bigram reuse; K8 adjacent executable-name repetition.
- Unknown actions break runs and pairs, never bridge them. Empty denominators are missing. Counts/coverage/missingness/identity are not inputs to the eight-feature model.
- **length**: log1p(event count), standard scaler, balanced logistic regression.
- **lexical**: command-name unigram/bigram TF-IDF, sublinear term frequency, balanced logistic regression.
- **core8**: eight features, training-median imputation, standard scaler, balanced logistic regression.
- Fixed C=1, L2 penalty, liblinear solver, max_iter=2000, seed=20261101; score threshold=0.5. Fit every transformation on training rows only. No classifier-capacity search.

## Partitions and dependence

- Primary: leave one application/scenario pair out (12 possible task groups).
- Source transfer: three deterministic folds, each holding out one of GPT4.1/GPT4o/GPT5 and two of the six human aliases (sorted aliases allocated cyclically). All versions/repeats and both applications of a source remain together. This is family/configuration transfer under source labels, not pure model-architecture attribution.
- Crossed transfer: test the intersection of one source fold and one task group, training excludes both the held-out source fold and the held-out task. Pool each run's one out-of-fold prediction, not repeated test windows.
- Application transfer: train on one application and evaluate on the other; shared actors mean this is not also an unseen-source test.
- Purge training runs with an identical complete command-sequence hash to any test run; report purge counts. Audit full raw-file duplicates and exact shared execution-event fingerprints between raw intervals.
- Task-cluster bootstrap (1000 draws, seed above) describes conditional variation over the observed tasks. It does not certify uncertainty over unseen people/models. Report all predeclared views and partitions, no best-score headline selection.

## Interpretation and stopping

This pilot can support feasibility, identify parser/coverage failures, and prioritize a matched security-task collection. It cannot show that Agent participation breaks existing activity association, that mode predictions repair such errors, or that same-task traces share an operator/campaign. No association-benefit number will be fabricated from task identities.

The next association experiment needs independently assigned activity/episode labels crossed with Human/Agent execution mode. Compare association without mode information, with predicted mode information, and with oracle mode information as a diagnostic upper-bound comparator; never use oracle labels as the deployed method. Evaluate improvements in false merges/splits at matched operating points, including cross-mode true links and same-mode false links. A hard rule that Human and Agent traces cannot be linked is prohibited by the research hypothesis itself.

## Artifacts

Raw archive and any command streams remain in ignored external-data storage. Versioned protocol, extractor/checks, aggregate outcomes, pseudonymous run registry, code/input hashes, and honest failures are retained under the new pilot directories. Existing website edits and previous experiments are untouched. This batch has no automatic commit or push.
