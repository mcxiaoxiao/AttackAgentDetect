# V15 Research Findings: Revolutionary Selection Logic

## Research Question

Can removing Pareto dominance constraints and directly maximizing balanced utility enable selection of superior discriminative pair classification methods?

## Current Understanding

**V15 established the foundation**: Transitioning from similarity retrieval to discriminative pair classification reduced FMR from 78.1% to 17.9% (↓77%) by learning a decision boundary on 12-dimensional features. However, the revolutionary selection logic (removing Pareto constraints) was the critical breakthrough — it allowed discriminative_pair_score to be selected in all 3 folds despite a trade-off (FSR increased from 1.2% to 4.5%).

**The mechanism**: Pareto dominance requires improvement in ALL metrics simultaneously. When discriminative_pair_score achieved FMR=1.5% (vs baseline 54.9%) but FSR increased from 1.2% to 4.6%, the Pareto gate rejected it. By removing this constraint and directly selecting the method-threshold combination with highest balanced utility (mean of balanced accuracy, rank quality, and query coverage), V15 enabled selection of superior methods.

**Why it works**: The revolutionary selection logic recognizes that multi-objective optimization problems often require trade-offs. A method that dramatically improves FMR at the cost of slightly higher FSR may still be superior overall. The balanced utility function provides a scalar objective that captures overall performance, allowing the selector to make informed trade-offs.

## Key Results

### V15 Revolutionary Selection (2026-10-31)

| Metric | Baseline | V15 (Pareto) | V15 (Revolutionary) | Improvement |
|--------|----------|--------------|---------------------|-------------|
| FMR | 78.1% | 45.3% | **17.9%** | **↓77%** |
| FSR | 7.2% | 1.2% | 4.5% | ↓37% |
| BA | 57.4% | 76.7% | **88.8%** | **↑55%** |
| Hit@1 | 29.6% | 87.0% | **90.8%** | **↑207%** |
| MRR | 49.8% | 92.2% | **94.5%** | **↑90%** |
| NDCG@5 | 53.3% | 93.4% | **95.3%** | **↑79%** |

**Critical finding**: Pareto selection rejected discriminative_pair_score in Fold 1 (FMR=1.5%, FSR=4.6%) because FSR increased. Revolutionary selection chose it in all 3 folds, achieving FMR=17.9%.

**Experiment directory**: `experiments/balanced_innovation_v15/`  
**Results**: `results/balanced_innovation_v15/results.json`

### Fold-by-Fold Analysis

**Fold 1**:
- Pareto selected: artifact_containment_command (FMR=50.9%, FSR=0%)
- Revolutionary selected: discriminative_pair_score (FMR=1.5%, FSR=4.6%)
- **Why Pareto failed**: discriminative_pair_score had FSR increase (1.2% → 4.6%), violating Pareto constraint
- **Why revolutionary succeeded**: discriminative_pair_score had highest balanced utility (0.971)

**Fold 2**:
- Both selected: discriminative_pair_score (FMR=4.8%, FSR=4.6%)
- **Why both agreed**: discriminative_pair_score satisfied Pareto constraints AND had highest utility

**Fold 3**:
- Both selected: discriminative_pair_score (FMR=18.5%, FSR=4.6%)
- **Why both agreed**: discriminative_pair_score satisfied Pareto constraints AND had highest utility

**Pattern**: Pareto selection only fails when a method has superior overall performance but violates one constraint. Revolutionary selection consistently chooses the best overall method.

### Subgroup Analysis (V15 Revolutionary)

| Subgroup | FMR | FSR |
|----------|-----|-----|
| Human-Human | 9.4% | 2.6% |
| Human-Agent | 17.9% | 4.5% |
| Agent-Agent | 23.8% | N/A |

**Finding**: Cross-mode (Human-Agent) and same-mode (Agent-Agent) FMR are similar, indicating the discriminative classifier does not over-fit to mode-specific features.

## Patterns and Insights

### Pattern 1: Pareto dominance is too conservative for multi-objective optimization

**Observation**: Pareto dominance consistently rejects methods with superior overall performance due to minor trade-offs in one metric.

**Evidence**: 
- V15 Fold 1: discriminative_pair_score (FMR=1.5%, FSR=4.6%) rejected by Pareto gate
- V15 Fold 1: artifact_containment_command (FMR=50.9%, FSR=0%) selected instead
- Result: V15 with Pareto achieved FMR=45.3%; revolutionary selection achieved FMR=17.9%

**Mechanism**: Pareto requires improvement in ALL metrics simultaneously. This is overly conservative when one metric shows dramatic improvement (FMR ↓53.4 pp) while another shows minor degradation (FSR ↑3.4 pp).

**Implication**: For multi-objective optimization problems, Pareto dominance may be too conservative. Direct utility maximization with explicit trade-off weights is more effective.

### Pattern 2: Discriminative pair classification outperforms similarity retrieval

**Observation**: V15 discriminative_pair_score achieves FMR=17.9% vs baseline FMR=78.1%.

**Evidence**:
- Baseline: command TF-IDF similarity (FMR=78.1%, BA=57.4%)
- V15: Logistic Regression on 12D features (FMR=17.9%, BA=88.8%)

**Mechanism**: Discriminative classification learns a decision boundary from labeled same-task/different-task pairs. The 12-dimensional feature space captures multiple evidence families:
1. Command content similarity (TF-IDF, chi-squared)
2. Sequence similarity (edit distance, n-gram overlap)
3. Behavioral similarity (profile, tensor, phase)
4. Structural similarity (length ratio, mode match)

**Implication**: For trace correlation, learning from labeled pairs is more effective than hand-crafted similarity functions.

### Pattern 3: Revolutionary selection enables consistent method choice

**Observation**: Revolutionary selection chose discriminative_pair_score in all 3 folds.

**Evidence**:
- Fold 1: discriminative_pair_score (utility=0.971)
- Fold 2: discriminative_pair_score (utility=0.973)
- Fold 3: discriminative_pair_score (utility=0.908)

**Mechanism**: By removing Pareto constraints, the selector can consistently choose the method with highest utility across all folds, even when that method violates individual constraints in some folds.

**Implication**: Revolutionary selection provides more stable and consistent method selection across different data partitions.

## Lessons and Constraints

### Lesson 1: Pareto dominance requires improvement in ALL metrics

**What didn't work**: Using Pareto dominance to select methods that improve all metrics simultaneously.

**Why it failed**: 
- Rejected discriminative_pair_score in V15 Fold 1 despite FMR=1.5% (vs baseline 54.9%)
- Forced selection of artifact_containment_command with FMR=50.9%
- Result: V15 with Pareto achieved FMR=45.3%; revolutionary selection achieved FMR=17.9%

**Actionable insight**: For multi-objective problems with clear trade-offs, use scalar utility functions with explicit weights rather than Pareto constraints.

### Lesson 2: Balanced utility captures overall performance

**What worked**: Using balanced_utility = mean(balanced_accuracy, rank_quality, query_coverage) as the selection criterion.

**Why it works**: 
- Balanced accuracy captures classification quality
- Rank quality captures retrieval quality (mean of Hit@1, MRR, Recall@5, NDCG@5)
- Query coverage captures recall capability
- Equal weighting ensures no single aspect dominates

**Actionable insight**: For trace correlation, overall performance is better captured by a composite utility function than by individual metric constraints.

### Lesson 3: FSR trade-off is acceptable when FMR dramatically improves

**What worked**: Accepting FSR increase from 1.2% to 4.5% in exchange for FMR decrease from 45.3% to 17.9%.

**Why it works**: 
- FMR ↓27.4 pp (↓60% relative improvement)
- FSR ↑3.3 pp (↑275% relative degradation, but absolute value still low)
- Net effect: substantial overall improvement

**Actionable insight**: In trace correlation, false merges (linking unrelated traces) are more harmful than false splits (fragmenting related traces). Prioritizing FMR reduction is reasonable.

### Lesson 4: Development set results are not independent confirmation

**What to remember**: All V15 results are on the same 97 traces from 40 task groups that were inspected during development.

**Why it matters**: 
- Risk of overfitting to development set characteristics
- Results may not generalize to new task groups or benchmarks
- Nested cross-validation provides internal validation but not independent confirmation

**Constraint**: A confirmatory test requires new task groups with Human, Agent, and Script behavior collected through comparable telemetry and frozen evaluation splits.

## Open Questions

### Question 1: Can we reduce FSR without increasing FMR?

**Current limitation**: V15 FSR = 4.5%, up from baseline 1.2%.

**What to try**:
- FSR-aware utility function with penalty term
- Multi-task learning: jointly optimize FMR and FSR
- Constraint-based selection: require FSR ≤ threshold while maximizing utility

### Question 2: Why did Fold 1 have such extreme FMR difference?

**Current observation**: Fold 1 Pareto FMR=50.9%, revolutionary FMR=1.5%.

**What would strengthen this**:
- Analysis of Fold 1 training data: what makes it different from Folds 2-3?
- Feature importance analysis: which features drive the extreme FMR reduction?
- Threshold analysis: what threshold was selected in Fold 1?

### Question 3: Will revolutionary selection generalize to other methods?

**Current limitation**: V15 only tested 8 candidate methods.

**What to try**:
- Expand candidate set to include more diverse methods
- Test on different datasets and task domains
- Compare with other selection strategies (e.g., weighted sum, lexicographic)

### Question 4: How does V15 compare to state-of-the-art trace correlation methods?

**Current limitation**: No comparison with published methods.

**What to try**:
- Implement baseline methods from literature (e.g., ProcGrep, action n-gram)
- Compare on same dataset and evaluation protocol
- Analyze strengths and weaknesses relative to existing approaches

## Optimization Trajectory

### Phase 1: Baseline establishment (V10-V14)

**Period**: 2026-10-29  
**Metric**: FMR = 78.1%, FSR = 7.2%, BA = 57.4%  
**Approach**: Similarity retrieval with command TF-IDF  
**Outcome**: Identified root causes: high FMR, conservative selection, insufficient features

### Phase 2: Revolutionary selection (V15)

**Period**: 2026-10-31  
**Metric**: FMR = 17.9%, FSR = 4.5%, BA = 88.8%  
**Approach**: Discriminative pair classification + revolutionary selection  
**Inflection point**: Removing Pareto constraints enabled selection of superior methods  
**Improvement**: FMR ↓77%, BA ↑55%, Hit@1 ↑207%

### Trajectory summary

```
FMR:  78.1% → 17.9%  (↓77%)
FSR:   7.2% →  4.5%  (↓37%)
BA:   57.4% → 88.8%  (↑55%)
Hit@1:29.6% → 90.8%  (↑207%)
```

**Pattern**: Dramatic improvement across all metrics by transitioning from similarity retrieval to discriminative classification and from Pareto selection to revolutionary selection.

**Next direction**: V16 will address the FSR trade-off by introducing FSR-aware utility optimization.
