# Query-normalized score fusion v1

Date: 2026-10-30. Status: fixed exploratory batch.

## Goal and baseline

Test whether query-normalizing and blending the command-cosine retriever with the learned behavior score can improve association error rates and retrieval ranking together. The baseline is the existing learned `behavior` score with its nested 20% training false-merge target.

The pinned input is the previously inspected 177-trace, 40-task CTH cohort across three public CTF benchmarks. Same-task relevance is only a challenge-level retrieval proxy, not actor identity, campaign, or shared attack activity. The cohort has capture-path confounding and has been reused in earlier exploratory analyses; no result here is an independent confirmation.

## Fixed candidate and evaluation

- For each query, convert `behavior` and `command_cosine` to within-gallery midrank percentiles. Blend as `alpha * behavior_percentile + (1-alpha) * command_percentile` for alpha in `{0, .25, .5, .75, 1}`. The transform uses unlabeled candidate scores only.
- On outer-training tasks, generate inner task-held-out scores. Search the fixed 31-quantile threshold grid for each blend. Select a policy only if every prespecified metric is no worse than the inner-OOF baseline and at least one is better; maximize minimum oriented gain, then total gain. Otherwise retain the baseline.
- Outer evaluation uses the same 3 benchmark-stratified, task-grouped folds and the same candidate galleries as the existing recall optimization. Outer split seed: 20261030; inner split seed: 20261730 plus outer-fold index; model seeds follow the existing runner. No test task is used for fitting, fusion-weight or threshold selection.
- Metrics: false-split rate, false-merge rate, Human-Agent false-split and false-merge rates, balanced accuracy, query coverage, Hit@1, MRR, Recall@5, and NDCG@5. Error rates should decrease; all others should increase.
- Report task-cluster paired 95% intervals. These intervals condition on the fitted folds and do not substitute for a new cohort.

## Budget and stop rule

One fixed batch, local CPU, at most 20 minutes; no paid API and no new data downloads. Code and output stay in this experiment directory. Stop after this batch. Do not add parameters or repeatedly retune against the same outer tasks. Record all folds, including fallback and negative results.

## Testable prediction and rival explanation

Prediction: command cosine contributes stronger candidate ordering, while the learned behavior score recovers more same-task pairs; an unlabeled within-query blend may improve both ranking and thresholded matching by combining complementary evidence.

Rival explanation: rank-normalization can only change candidate ordering, while the observed complementarity reflects the small, benchmark-confounded cohort; gains may be unstable across held-out tasks and disappear under independent data.
