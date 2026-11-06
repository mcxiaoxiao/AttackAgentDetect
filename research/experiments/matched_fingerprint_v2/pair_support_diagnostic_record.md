# CTH pair-support diagnostic and multiplicity record

## Status and scope

This is a development diagnostic on the existing 17-task mixed-role CTH cohort, not an independent test. The matching target remains same CTF task. It reuses the three outer task folds, the fixed 54,400 evaluation pairs per model arm, and each fold's selected XGBoost representation and parameters. Pair-support and argument-syntax analyses were pursued after the primary matched experiment; results are exploratory.

## Comparisons

Two training-pair conditions are compared: the sampled training set (eight positive pairs per task plus the protocol's sampled negatives) and all distinct available within-task positives and cross-task negatives. Both conditions have equal effective total weight by label and by each observed HH/HA/AA cell. Test query/candidate keys are checked identical across conditions and feature arms. Vocabulary, transformations, imputation, and OOF decision thresholds are fitted using training-fold data only; the threshold is recalibrated under each training-pair condition.

The four B/S increments are E+B/S versus E and E+Args+B/S versus E+Args under both training conditions. E uses command-head and action-bigram similarities. E+Args additionally uses value-redacted option syntax, coarse positional argument types, and coarse arity/option-count buckets. Raw argument values are discarded before feature construction. B/S uses the fold-selected configuration from the matched CTH learner experiment.

## Multiplicity adjustment

The reported point estimates and nominal intervals use the frozen `pair_predictions.csv.gz` and the existing paired query-task bootstrap. A follow-up max-T analysis uses 10,000 common paired task-cluster draws (seed `20261105`) across the four BA contrasts. It reports simultaneous two-sided 95% intervals within this four-contrast family. This family-wise adjustment does not cover all metrics or all exploratory analyses in the project. Query tasks are resampled, while repeated candidate tasks are not independently resampled.

Recompute the adjustment from the reproducibility bundle with:

```sh
python3 code/analyze_pair_support_multiplicity.py
```

Inputs: `results/pair_support/pair_predictions.csv.gz` and `results/pair_support/analysis.json`. Output: `results/pair_support/multiplicity_adjustment.json`.

The one nominally positive contrast is all-pair training with E+Args: B/S changes BA by +6.25 percentage points [1.17, 12.24], FSR by -13.06 points, and FMR by +0.56 points. Its simultaneous max-T interval is [-0.75, 13.24]. The result therefore motivates a targeted replication; it does not establish a cross-learner or source-independent B/S benefit.
