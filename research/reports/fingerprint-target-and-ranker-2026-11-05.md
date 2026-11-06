# Ranking objective and snapshot retrieval

Date: 2026-11-05

## CTH same-challenge association

The experiment uses all 54,400 original CTH candidate decisions per arm across three task folds and five exact role shares. The pointwise and ranking logistic models use identical training pair support, task weights, and feature representations. The ranking loss compares positive and negative pairs for each query. The inner task folds choose regularization by gallery AUC and select thresholds under the original pooled FMR ≤10% rule. If one candidate lacks B/S, the corresponding E scorer supplies all scores and decisions for that gallery.

| Model | Gallery AUC | Hit@1 | BA | FMR | FSR |
|---|---:|---:|---:|---:|---:|
| Frozen XGBoost E | 58.92% | 34.40% | 56.06% | 5.97% | 81.91% |
| Frozen XGBoost E+B/S | 61.32% | 36.77% | 53.68% | 15.43% | 77.21% |
| Pointwise E | 60.56% | 39.16% | 52.50% | 21.63% | 73.37% |
| Ranking E | 60.07% | 38.77% | 53.00% | 20.29% | 73.71% |
| Pointwise E+B/S | 55.30% | 35.92% | 52.85% | 18.62% | 75.68% |
| Ranking E+B/S | 49.40% | 28.94% | 52.38% | 13.83% | 81.40% |

For E, changing the loss shifts gallery AUC by −0.49 pp [−1.11, +0.02]. For E+B/S, it shifts AUC by −5.90 pp [−12.27, −1.06] and Hit@1 by −6.97 pp [−12.71, −2.17]. In that B/S comparison, FMR drops 4.79 pp while FSR rises 5.72 pp. The four logistic arms miss the 10% held-out FMR target. The existing CTH method remains the reference for the paper. The development cohort has 17 tasks, and queries share candidate tasks.

## NYU same-snapshot retrieval

The exploratory experiment uses 111 NYU challenges and 444 runs from four known D-CIPHER submissions. Each gallery pairs a query from one held-out challenge with four candidates from a single different held-out challenge, one from each snapshot. This holds candidate task constant within each ranking. The fixed ranker trains on other challenges. Its pair features exclude submission identifiers and challenge labels. The evaluation averages 15,984 galleries per view and resamples challenge IDs jointly in both query and candidate roles.

With full traces, the length-only baseline gives 59.92% gallery AUC and 34.63% Hit@1. B/S alone gives 57.95% and 32.29%, respectively. The stronger E+length baseline gives 67.66% AUC and 44.27% Hit@1. Adding B/S raises these to 68.84% and 45.80%.

| Trace view | Challenges | Δ gallery AUC in pp [95% interval] | Δ Hit@1 in pp [95% interval] |
|---|---:|---:|---:|
| All full traces | 111 | +1.19 [+0.44, +1.97] | +1.52 [+0.42, +2.68] |
| Common cohort, full | 100 | +1.08 [+0.25, +1.96] | +1.57 [+0.37, +2.85] |
| Common cohort, first 8 | 100 | +0.12 [−0.82, +1.11] | +0.32 [−0.79, +1.44] |
| Longer cohort, full | 80 | +0.73 [−0.32, +1.80] | +1.03 [−0.42, +2.56] |
| Longer cohort, first 8 | 80 | +0.15 [−0.90, +1.21] | +0.65 [−0.60, +1.99] |
| Longer cohort, first 16 | 80 | −0.72 [−1.78, +0.34] | −0.54 [−1.87, +0.80] |

The B/S increment is clearest with full traces. Fixed prefixes control observed event count and also remove later events, so the difference cannot isolate the contribution of length from the contribution of later behavior. The 16-event comparison was added after inspecting the first eight-event result. All estimates describe known snapshot configurations within one framework. The same-challenge evaluation uses its frozen predictions.

## Reproduction

Run these commands from the repository root. The second command accepts a local checkout of the revision pinned in the NYU task manifest.

```sh
.venv/bin/python research/experiments/matched_fingerprint_v2/run_gallery_ranking.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/experiments/matched_fingerprint_v2/run_snapshot_retrieval.py --source "$NYU_CHECKOUT"
```

The [CTH protocol](../experiments/matched_fingerprint_v2/cth_gallery_ranking_protocol.md) and [NYU protocol](../experiments/matched_fingerprint_v2/nyu_snapshot_retrieval_protocol.md) define the comparisons. The [CTH metrics](../results/matched_fingerprint_v2/gallery_ranking_20261105/analysis.json) and [NYU metrics](../results/matched_fingerprint_v2/snapshot_retrieval_20261105/analysis.json) contain input hashes, splits, metric intervals, and per-fold selections where applicable.
