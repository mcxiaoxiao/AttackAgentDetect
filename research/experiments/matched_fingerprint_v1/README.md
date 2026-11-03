# Matched fingerprint association v1

The protocol is documented in ../exact_mixture_matched_fingerprint_plan.md; the experiment report is in ../../reports/matched-fingerprint-v1.md. This is an exploratory same-task proxy study, not verified activity or actor attribution.

## Reproduce

The experiment uses the project .venv, XGBoost 2.1.4 and a compatible OpenMP runtime. On macOS the wrapper searches .venv/lib for libomp.dylib. The raw AIT archive is read locally through the existing pilot adapter and is never added to the publication bundle.

    research/experiments/matched_fingerprint_v1/run.sh --dataset cth
    research/experiments/matched_fingerprint_v1/run.sh --dataset ait
    DYLD_LIBRARY_PATH="$PWD/.venv/lib" .venv/bin/python research/experiments/matched_fingerprint_v1/reproduce_baselines.py
    .venv/bin/python research/experiments/matched_fingerprint_v1/summarize.py
    DYLD_LIBRARY_PATH="$PWD/.venv/lib" .venv/bin/python research/experiments/matched_fingerprint_v1/rebuild_training_pairs.py
    .venv/bin/python research/experiments/matched_fingerprint_v1/build_cases.py
    .venv/bin/python research/experiments/matched_fingerprint_v1/build_figure.py
    .venv/bin/python research/experiments/matched_fingerprint_v1/build_appendix.py
    .venv/bin/python research/experiments/matched_fingerprint_v1/build_draft.py
    .venv/bin/python research/experiments/matched_fingerprint_v1/build_page.py
    .venv/bin/python research/experiments/matched_fingerprint_v1/build_report.py
    .venv/bin/python research/experiments/matched_fingerprint_v1/verify.py

Compile research/paper/human-agent-fingerprint-draft.tex with Tectonic from the paper directory, place the PDF next to it, and run package_draft.py.

The full test suite requires the historical research package path:

    DYLD_LIBRARY_PATH="$PWD/.venv/lib" PYTHONPATH="$PWD/research:$PWD" .venv/bin/python -m unittest discover -s tests

## Outputs and interpretation

- environment.json records source and dependency hashes; baseline_reproduction.json checks frozen legacy pair selection and thresholds.
- Each dataset has a frozen pair list, task/mode audit and three fold-selection audits; CTH additionally contains the 32-slot role allocations.
- training_validation_pairs.csv.gz records the deterministic inner training, inner validation and outer training pairs; pair_predictions.csv.gz unifies the fold-local prediction files.
- fold-*-*.csv.gz contains per-pair scores and decisions. A fallback flag identifies pairs evaluated by the same-fold evidence-only model.
- metrics.json, summary.csv and error_transitions.csv contain task-macro rates, paired intervals and error changes. AIT uses fraction -1 to indicate the mixed fixed-gallery protocol, not a negative share.
- JSON null for a selected threshold represents positive infinity (all reject); individual predictions serialize that threshold as inf.
- paired_score_cases.csv contains anonymized paired cases, not raw commands. Model scores are not calibrated probabilities.
- CTH clusters are 17 tasks; AIT clusters are 12 troubleshooting tasks. Repeated draws reuse logs. Query-task bootstrap intervals are approximate under shared candidate tasks.
- Do not compare absolute CTH and AIT rates as if their tasks, timing observations and collection interfaces were the same. CTH T is missing, AIT has launch times but no measured durations.

The main-text figure uses shortened vertical scales; every row retains the same axis across learners and the scales are stated in its caption.
