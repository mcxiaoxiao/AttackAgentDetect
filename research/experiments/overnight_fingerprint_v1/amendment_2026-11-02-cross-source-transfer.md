# Exploratory amendment: cross-source Agent task retrieval

Date: 2026-11-02. This addition was written after the within-source retrieval runs were observed. It is therefore an **exploratory follow-up**, not a preregistered or confirmatory result.

## Question

Can a pair scorer trained on same-task/different-task labels from one source rank same-source task matches in the other source, when both sources use the same normalized command/action contract?

## Fixed procedure

- Run both directions: Cyber Task Horizons (CTH) to Antieval Cybench, then Antieval Cybench to CTH.
- Use the eligible Agent-only tracks already frozen in `protocol.json`: at least 16 command names, at least two records per source-qualified task; 48/121 CTH tasks/records and 12/31 Antieval tasks/records.
- Fit the TF-IDF vocabulary, matrix column scaler, and balanced logistic pair scorer only on the training source. Positive/negative pair labels are defined only within each source-qualified task namespace.
- Test ranking candidates are sampled only from the held-out target source. Candidate counts remain 5 for CTH and 3 for Antieval; random chance is 20% and 33.3%, respectively.
- Compare command cosine with command cosine plus masked-DTW distance. Keep source-native task labels; do not match task titles across sources.
- Fixed seed 20261102; one deterministic query/gallery construction; report task-macro Hit@1, task-cluster bootstrap, pair coverage, and paired task deltas.
- No hyperparameter search, feature changes, or threshold selection. Both sources were known during development, so label the result exploratory even though the scorer does not train on target-source records.

## Interpretation boundary

This checks transfer of a task-retrieval scorer between two normalized Agent CTF sources. It cannot establish Human/Agent identification, equivalence of tasks across sources, or real campaign/operator linkage. A small target source and its source-specific action distributions limit generalization claims.
