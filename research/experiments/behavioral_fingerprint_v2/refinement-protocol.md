# Bounded association refinement, 3 November 2026

This exploratory batch follows inspection of v2 results. Primary metric: task-macro Hit@1 on the unchanged 40-task CTH cross-mode gallery. Secondary: MRR and paired 95% task-cluster intervals. All original outer tasks, queries, and candidate IDs are reused.

Four model families are evaluated: scalar B/S fusion; cosine-conditioned B/S interactions; 48 component-wise B/S summary differences; and train-standardized local B/S masked DTW plus command evidence. Logistic C is selected from {0.1, 1, 10} using three inner task folds and task-macro Hit@1 (MRR breaks ties). A separate nested selector chooses the family using the same inner predictions. Every family's outer result is retained; no outer score selects a winner.

Adapters, eight-event windows, minimum two-command CTH gate, maximum 128 events, outer fold membership, and galleries remain fixed. Fit vocabulary, scaling, imputation, and classifiers within each training fold. Timing is unavailable. Stop after this batch (30 CPU minutes, no paid calls). Report as post-observation exploratory refinement, with all attempts and inner choices recorded.
