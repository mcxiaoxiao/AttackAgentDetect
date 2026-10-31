# v10 outer-fold missed-link audit

## Decision

The v10 aggregate point estimate lowered FSR, but outer fold 3 increased from 2.56% for command TF–IDF to 10.26% for the selected artifact-containment policy. The round therefore fails the predeclared fold-level missed-link gate. The misses were not candidate-generation filters: v10 directly scored and rejected four directed positive rows, representing two unordered Human–Agent task-proxy pairs.

## Fold 3 cases

| Task proxy | Directed rows | Selected score | Fold threshold | Command TF–IDF | Artifact containment |
|---|---:|---:|---:|---:|---:|
| `intercode-ctf_reverse/task_14` | 2 | 0.0618 | 0.0819 | 0.1372 | 0.0000 |
| `intercode-ctf_reverse/task_76` | 2 | 0.0427 | 0.0819 | 0.0948 | 0.0000 |

The task-14 Human trace contained 21 parsed commands, mostly `ls`, `cat`, and `strings`; the Agent trace contained 26 commands and distinctive tokens around `dolls` and `.jpg` files. The corrected extraction found no shared distinctive artifact token. The score was below threshold because both views provided little direct overlap.

The task-76 Human trace was very short (three commands: `ls`, `file`, `cat`) while the Agent trace had 13 commands. They shared a generic `.s` extension but no distinctive artifact-name token. The short/long trace mismatch and lack of a shared artifact left a low score.

These are same-task retrieval-proxy labels, not proof of the same activity. The cases show why lowering thresholds can recover short Human/Agent pairs, while also explaining the false-merge cost: many different-task pairs share generic shell behavior.

## Follow-up

v11 selected the lowest inner-feasible threshold. It accepted these four directed rows and had no outer-fold FSR increase, but its task-macro FMR remained 75.87%. See `results/balanced_innovation_v11/results.json` and `outer_pair_scores.csv` for the complete outer predictions.
